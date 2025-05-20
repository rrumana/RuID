// ruid-preprocess/src/lib.rs
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
use tract_ndarray::{Array4, Axis};
use thiserror::Error;

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
            .with_input_fact(0, InferenceFact::dt_shape(i8::datum_type(), tvec![1,3,224,224]))?
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
                    arr4[(0,ch,y,x)] = input[(y,x,ch)];
                }
            }
        }

        // 2) Into a tract Tensor, then into a TValue for the tvec! macro
        let tensor: Tensor = arr4.into_tensor();
        let inputs: TVec<TValue> = tvec![tensor.into()];

        // 3) Run the model
        let outputs = self.model.run(inputs)?;
        let output = &outputs[0];

        // 4) Get a tract-ndarray ArrayViewD<f32>; shape should be [1, N, 6]
        let view = output.to_array_view::<f32>()?;   

        let shape = view.shape();
        let shape_vec = view.shape().to_vec();
        if !(shape_vec.len() == 3 && shape_vec[0] == 1 && shape_vec[2] == 6) {
            return Err(DetectError::InvalidOutputShape(shape_vec));
        }
        let n = shape[1];

        // 5) Pull out each detection row
        let mut dets = Vec::with_capacity(n);
        for i in 0..n {
            // index_axis over axis=1 gives a 1×6 view
            let row = view.index_axis(Axis(1), i);  

            let xc    = row[0];
            let yc    = row[1];
            let wbox  = row[2];
            let hbox  = row[3];
            let score = row[4];
            let class = row[5] as usize;

            dets.push( Detection {
                bbox:  [xc - wbox/2.0, yc - hbox/2.0, xc + wbox/2.0, yc + hbox/2.0],
                score,
                class,
            });
        }

        Ok(dets)
    }
}