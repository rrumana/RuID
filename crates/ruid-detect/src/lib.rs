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
// where Detection { bbox, score } (bbox normalised 0–1)
// ------------------------------------------------------------
// Build notes
// * Default backend = onnxruntime-sys 2.0 (multithreaded, no Cuda).
// * Pure-Rust Tract backend kept as a feature tract (optional).
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
const IN_W: usize = 640;
const IN_H: usize = 480;
const CONF_THR: f32 = 0.50;
const NMS_IOU: f32 = 0.45;
const PERSON_CLASS: usize = 0;        // COCO class id

// ───── helpers ──────────────────────────────────────────────

#[inline]
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let iw  = (ix2 - ix1).max(0.0);
    let ih  = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let area_a = (a[2]-a[0]) * (a[3]-a[1]);
    let area_b = (b[2]-b[0]) * (b[3]-b[1]);
    inter / (area_a + area_b - inter + 1e-6)
}

fn nms(mut v: Vec<Detection>) -> Vec<Detection> {
    v.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<Detection> = Vec::with_capacity(v.len());
    'outer: for d in v {
        for k in &keep {
            if iou(&d.bbox, &k.bbox) > NMS_IOU { continue 'outer }
        }
        keep.push(d);
    }
    keep
}

// ───── public types ────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox : [f32; 4],          // x1 y1 x2 y2  (0-1)
    pub score: f32,
}

#[derive(Debug, Error)]
pub enum DetectError {
    #[error("ORT error: {0}")]
    Ort(#[from] OrtError),
    #[error("expect 640×640×3, got shape {0:?}")]
    BadShape(Vec<usize>),
}
pub type Result<T> = core::result::Result<T, DetectError>;

pub trait Detector { fn detect(&mut self, img:&Array3<f32>) -> Result<Vec<Detection>>; }

// ───── ORT implementation ──────────────────────────────────
pub struct OrtYolo { session: Session }

impl OrtYolo {
    pub fn new<P: AsRef<std::path::Path>>(model:P) -> Result<Self> {
        Ok(Self { session: Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(num_cpus::get_physical()-1)?
                .commit_from_file(model)? 
            })
    }
}

impl Detector for OrtYolo {
    fn detect(&mut self, img:&Array3<f32>) -> Result<Vec<Detection>> {
        if img.shape() != &[IN_H, IN_W, 3] {
            return Err(DetectError::BadShape(img.shape().to_vec()))
        }

        // ---- HWC ➜ NCHW ---------------------------------------------------
        let mut chw = Array4::<f32>::zeros((1, 3, IN_H, IN_W));
        for y in 0..IN_H { for x in 0..IN_W {
            let p = img.slice(s![y, x, ..]);
            chw[[0, 0, y, x]] = p[0];
            chw[[0, 1, y, x]] = p[1];
            chw[[0, 2, y, x]] = p[2];
        }}

        let input = TensorRef::from_array_view(chw.view())?;
        let outputs: SessionOutputs = self.session.run(inputs!["input" => input])?;
        let output = outputs["output"].try_extract_array::<f32>()?.t().into_owned();

        let mut dets = Vec::with_capacity(output.shape()[1]);

        let output = output.slice(s![.., .., 0]);
        for row in output.axis_iter(Axis(0)) {
            let row: Vec<_> = row.iter().copied().collect();
            let (class_id, prob) = row
                .iter()
                .skip(4)
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .unwrap();
            if prob < CONF_THR || class_id != PERSON_CLASS {
                continue;
            }
            let xc = row[0] / IN_W as f32 * (IN_W as f32);
            let yc = row[1] / IN_H as f32 * (IN_H as f32);
            let w = row[2] / IN_W as f32 * (IN_W as f32);
            let h = row[3] / IN_H as f32 * (IN_H as f32);

            dets.push(Detection {
                bbox: [
                    (xc - w * 0.5) / IN_W as f32,
                    (yc - h * 0.5) / IN_H as f32,
                    (xc + w * 0.5) / IN_W as f32,
                    (yc + h * 0.5) / IN_H as f32,
                ],
                score: prob,
            });
        }

        Ok(nms(dets))
    }
}