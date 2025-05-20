// ruid-detect/src/lib.rs
// ============================================================
// ruid-detect  –  Object-detection stage for RuID v2
// Runs a static-quantised **YOLOv8-nano INT8** network via
// Tract (pure-Rust) or OnnxRuntime (optional feature).
// ------------------------------------------------------------
// Pipeline: Array3<f32> → Tensor → Vec<Detection>
// ------------------------------------------------------------
// Public API
//   * TractYolo::new(path)      – load & optimise ONNX
//   * Detector::detect(arr3)    – returns Vec<Detection>
//     where Detection { bbox, score, class }
// ------------------------------------------------------------
//   Build notes
//     * Default backend = Tract 0.17 (no C deps, works on Pi).
//     * `--features ort` switches to onnxruntime-sys if desired. (someday)
// ============================================================

//! RuID – detection layer
//!
//! This crate provides a backend-agnostic [`Detector`] trait plus a
//! concrete **`TractYolo`** implementation that runs an INT8 YOLOv8-nano
//! model. In the future Switching to onnxruntime or another engine is a 
//! matter of enabling a Cargo feature – the outer API stays identical.
//!
//! Input tensors come from `ruid-preprocess` (CHW, f32).  Output is a
//! vector of [`Detection`] structs containing normalised corner boxes,
//! confidence and class index.  No copies are made inside the hot path –
//! Tract reads directly from the tensor memory we allocate once per frame.

use ndarray::Array3;
use tract_onnx::prelude::*;
use tract_ndarray::{Array4, Axis, s};
use thiserror::Error;

const INPUT_SIZE: f32 = 224.0;          // we exported 224×224
const STRIDES: [i32; 3] = [8, 16, 32];  // for 224×224 input
const CONF_THR:   f32 = 0.05;           // a bit higher now

// ------------------------------------------------------------
// helpers: sigmoid • IoU • NMS
// ------------------------------------------------------------
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn iou(a: &[f32;4], b: &[f32;4]) -> f32 {
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

fn non_max_suppression(dets: Vec<Detection>, iou_thr: f32) -> Vec<Detection> {
    let mut dets = dets;
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut keep: Vec<Detection> = Vec::with_capacity(dets.len());

    'outer: for d in dets {
        for k in &keep {
            if iou(&d.bbox, &k.bbox) > iou_thr {
                continue 'outer;
            }
        }
        keep.push(d);
        if keep.len() >= 300 { break }
    }
    keep
}

#[derive(Debug, Error)]
pub enum DetectError {
    #[error("Model load or inference error: {0}")]
    Tract(#[from] TractError),
    #[error("Invalid input channels: expected 3, got {0}")]
    InvalidChannels(usize),
    #[error("Invalid output shape: expected [1, N, 6], got {0:?}")]
    InvalidOutputShape(Vec<usize>),
}

pub type Result<T> = std::result::Result<T, DetectError>;

/// A single detection: bounding box [x1,y1,x2,y2] in normalized coords plus score.
#[derive(Debug)]
pub struct Detection {
    pub bbox:  [f32; 4],
    pub score: f32,
    pub class: usize,
}

/// Trait for object detectors.
pub trait Detector {
    fn detect(&self, input: &Array3<f32>) -> Result<Vec<Detection>>;
}

/// Tract-powered YOLOv8-Nano INT8 detector.
pub struct TractYolo {
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, TypedModel>,
}

impl TractYolo {
    /// Load and optimize the ONNX model, preparing it for inference.
    pub fn new(model_path: &str) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec![1,3,224,224]))?
            .into_optimized()?
            .into_runnable()?;          

        Ok(Self { model })
    }
}

impl Detector for TractYolo {
    fn detect(&self, input: &Array3<f32>) -> Result<Vec<Detection>> {
        let h = input.shape()[0];
        let w = input.shape()[1];
        let c = input.shape()[2];
        if c != 3 {
            return Err(DetectError::InvalidChannels(c));
        }

        // 1) Build a tract_ndarray::Array4 of shape [1,3,H,W]
        let mut arr4 = Array4::<f32>::zeros((1, 3, h, w));
        for y in 0..h {
            for x in 0..w {
                for ch in 0..3 {
                    arr4[(0, ch, y, x)] = input[(y, x, ch)];
                }
            }
        }

        // 2) Into a tract Tensor, then into a TValue for the tvec! macro
        let tensor: Tensor = arr4.into_tensor();
        let inputs: TVec<TValue> = tvec![tensor.into()];

        // 3) Run the model
        let outputs = self.model.run(inputs)?;
        let view    = outputs[0].to_array_view::<f32>()?;
        let view    = view.index_axis(Axis(0), 0);          // [84, 1029]

        eprintln!("output shape {:?}", view.shape());

        let mut dets = Vec::new();
        let mut anchor_ofs = 0;
        // walk the three detection layers
        for (s_i, &stride) in STRIDES.iter().enumerate() {
            let g = (INPUT_SIZE as i32 / stride) as usize;      // grid size
            for gy in 0..g {
                for gx in 0..g {
                    let anchor = anchor_ofs + gy * g + gx;

                    // -------- raw predictions ----------
                    let x  = view[[0, anchor]];
                    let y  = view[[1, anchor]];
                    let w  = view[[2, anchor]];
                    let h  = view[[3, anchor]];
                    let obj_logit = view[[4, anchor]];

                    // -------- decode centre + size -----
                    let bx = (sigmoid(x) * 2.0 - 0.5 + gx as f32) * stride as f32;
                    let by = (sigmoid(y) * 2.0 - 0.5 + gy as f32) * stride as f32;
                    let bw = (sigmoid(w) * 2.0).powi(2) * stride as f32;
                    let bh = (sigmoid(h) * 2.0).powi(2) * stride as f32;

                    // -------- best class ---------------
                    let cls_slice = view.slice(s![5.., anchor]);
                    let (best_cls, &cls_logit) = cls_slice
                        .iter()
                        .enumerate()
                        .max_by(|a,b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap();
                    let conf = sigmoid(obj_logit) * sigmoid(cls_logit);

                    if conf >= CONF_THR && best_cls == 0 {      // keep only “person”
                        dets.push(Detection {
                            bbox: [
                                (bx - bw/2.0) / INPUT_SIZE,      // x1,y1,x2,y2  normalised 0-1
                                (by - bh/2.0) / INPUT_SIZE,
                                (bx + bw/2.0) / INPUT_SIZE,
                                (by + bh/2.0) / INPUT_SIZE,
                            ],
                            score: conf,
                            class: best_cls,
                        });
                    }
                }
            }
            anchor_ofs += g * g;
        }

        // 4. NMS exactly as before
        Ok(non_max_suppression(dets, 0.05))
    }
}