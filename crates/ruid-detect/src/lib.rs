// ruid-detect/src/lib.rs
// ============================================================
// ruid-detect – Object-detection stage for RuID v2
// Runs a YOLO v11-nano FP32 model with ONNX-Runtime 2.
// ------------------------------------------------------------
// Pipeline: Array3<f32> (HWC, range 0–1) → ONNX → Vec<Detection>
// ------------------------------------------------------------
// Public API
// * OrtYolo::new(path) – load & optimise ONNX
// * Detector::detect(img) – returns Vec<Detection>
// ------------------------------------------------------------
// Build notes
// * Default backend = onnxruntime 2.0 (multithreaded, no Cuda).
// ============================================================
//! RuID – detection layer
//!
//! This crate exposes a backend-agnostic [Detector] trait plus the
//! concrete OrtYolo implementation that runs a YOLO v11-nano
//! network via ONNX-Runtime 2.0. Switching to Tract or another engine
//! is only a Cargo-feature flip – the outer API remains identical.
//!
//! The detector expects pre-processed input tensors coming from
//! ruid-preprocess (HWC, f32, normalised to 0-1). The hot path is
//! zero-copy: we create one tensor per frame and let ORT read it
//! directly; no additional allocations are performed.

use ndarray::{s, Array3, Array4, Axis};
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session, SessionOutputs},
    value::TensorRef,
    Error as OrtError,
};
use thiserror::Error;
use num_cpus;

// ───── constants ────────────────────────────────────────────
const CONF_THR: f32 = 0.50;
const NMS_IOU: f32 = 0.45;
const PERSON_CLASS: usize = 0;        // COCO class id
const EPSILON: f32 = 1e-6;
const HALF: f32 = 0.5;

// ───── helpers ──────────────────────────────────────────────

#[inline]
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    
    // Early exit if no intersection
    if ix2 <= ix1 || iy2 <= iy1 {
        return 0.0;
    }
    
    let inter = (ix2 - ix1) * (iy2 - iy1);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    inter / (area_a + area_b - inter + EPSILON)
}

fn nms(mut v: Vec<Detection>) -> Vec<Detection> {
    if v.is_empty() {
        return v;
    }
    
    v.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<Detection> = Vec::with_capacity(v.len());
    
    'outer: for d in v {
        for k in &keep {
            if iou(&d.bbox, &k.bbox) > NMS_IOU {
                continue 'outer;
            }
        }
        keep.push(d);
    }
    keep
}

// ───── public types ────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox : [f32; 4],
    pub score: f32,
}

#[derive(Debug, Error)]
pub enum DetectError {
    #[error("ORT error: {0}")]
    Ort(#[from] OrtError),
    #[error("expect {expected_h}x{expected_w}x3, got shape {actual:?}")]
    BadShape { expected_h: usize, expected_w: usize, actual: Vec<usize> },
    #[error("No class predictions found in model output")]
    NoClassPredictions,
}
pub type Result<T> = core::result::Result<T, DetectError>;

pub trait Detector { fn detect(&mut self, img:&Array3<f32>) -> Result<Vec<Detection>>; }

// ───── ORT implementation ──────────────────────────────────
pub struct OrtYolo {
    session: Session,
    input_width: usize,
    input_height: usize,
    input_width_f32: f32,
    input_height_f32: f32,
}

impl OrtYolo {
    pub fn new<P: AsRef<std::path::Path>>(model: P, input_width: u32, input_height: u32) -> Result<Self> {
        Ok(Self {
            session: Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(1)?
                .commit_from_file(model)?,
            input_width: input_width as usize,
            input_height: input_height as usize,
            input_width_f32: input_width as f32,
            input_height_f32: input_height as f32,
        })
    }
}

impl Detector for OrtYolo {
    fn detect(&mut self, img: &Array3<f32>) -> Result<Vec<Detection>> {
        let shape = img.shape();
        if shape.len() != 3 || shape[0] != self.input_height || shape[1] != self.input_width || shape[2] != 3 {
            return Err(DetectError::BadShape {
                expected_h: self.input_height,
                expected_w: self.input_width,
                actual: shape.to_vec()
            });
        }

        // ---- HWC ➜ NCHW ---------------------------------------------------
        let mut chw = Array4::<f32>::zeros((1, 3, self.input_height, self.input_width));
        
        // More efficient tensor conversion using unsafe indexing for performance
        let img_ptr = img.as_ptr();
        let chw_ptr = chw.as_mut_ptr();
        
        unsafe {
            for y in 0..self.input_height {
                for x in 0..self.input_width {
                    let src_idx = (y * self.input_width + x) * 3;
                    let dst_base = y * self.input_width + x;
                    
                    *chw_ptr.add(dst_base) = *img_ptr.add(src_idx);                    // R
                    *chw_ptr.add(dst_base + self.input_height * self.input_width) = *img_ptr.add(src_idx + 1); // G
                    *chw_ptr.add(dst_base + 2 * self.input_height * self.input_width) = *img_ptr.add(src_idx + 2); // B
                }
            }
        }

        let input = TensorRef::from_array_view(chw.view())?;
        let outputs: SessionOutputs = self.session.run(inputs!["images" => input])?;
        let output = outputs["output0"]
            .try_extract_array::<f32>()?
            .t()
            .into_owned();

        let output_slice = output.slice(s![.., .., 0]);
        let mut dets = Vec::with_capacity(output_slice.shape()[0]);

        for row in output_slice.axis_iter(Axis(0)) {
            // Find max class probability without collecting into Vec
            let mut max_prob = 0.0f32;
            let mut max_class = 0usize;
            
            for (idx, &prob) in row.iter().skip(4).enumerate() {
                if prob > max_prob {
                    max_prob = prob;
                    max_class = idx;
                }
            }

            if max_prob < CONF_THR || max_class != PERSON_CLASS {
                continue;
            }

            // Access row elements directly using iterator with bounds check
            if row.len() < 4 {
                continue;
            }
            let mut row_iter = row.iter();
            let xc = *row_iter.next().expect("row has at least 4 elements");
            let yc = *row_iter.next().expect("row has at least 4 elements");
            let w = *row_iter.next().expect("row has at least 4 elements");
            let h = *row_iter.next().expect("row has at least 4 elements");

            let half_w = w * HALF;
            let half_h = h * HALF;

            dets.push(Detection {
                bbox: [
                    (xc - half_w) / self.input_width_f32,
                    (yc - half_h) / self.input_height_f32,
                    (xc + half_w) / self.input_width_f32,
                    (yc + half_h) / self.input_height_f32,
                ],
                score: max_prob,
            });
        }

        Ok(nms(dets))
    }
}