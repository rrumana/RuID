use std::{path::Path, sync::Arc};

use tract_onnx::prelude::*;
use tract_ndarray::{Array4, s};
use thiserror::Error;
use serde::{Serialize, Deserialize};

/// 1×128 embedding
pub type Embedding = Vec<f32>;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Detection {
    pub bbox: [i32; 4],
    pub embedding: Embedding,
    pub cam_id: String,
    pub frame_ts: u64, // microseconds
}

#[derive(Error, Debug)]
pub enum RuIdError {
    #[error("tract error: {0}")]
    Tract(#[from] TractError),
    #[error("opencv error: {0}")]
    Cv(#[from] opencv::Error),
}

/// Wraps a tract runnable ONNX model
pub struct OnnxModel {
    model: Arc<RunnableModel<TypedFact, Box<dyn TypedOp>>>,
}

impl OnnxModel {
    /// Load, optimize, and make runnable from an ONNX file.
    pub fn new(model_path: &Path) -> Result<Self, RuIdError> {
        // 1) Load the model
        let mut model = tract_onnx::onnx()
            .model_for_path(model_path)?
            // (Optional) you can set shapes if your ONNX has dynamic dims:
            // .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1,3,640,640)))?
            ;
        // 2) Optimize & make runnable
        let runnable = model
            .into_optimized()?
            .into_runnable()?;
        Ok(OnnxModel {
            model: Arc::new(runnable),
        })
    }

    /// Run the model on a single [N, C, H, W] f32 array.
    /// Returns a Vec of output tensors as `ndarray::ArrayD<f32>`.
    pub fn run(&self, input: Array4<f32>) -> Result<Vec<ndarray::ArrayD<f32>>, RuIdError> {
        // tract expects an ArrayD (dynamic-dimensional) input
        let input = input.into_dyn();  

        // 1) into a tract tensor
        let tensor = tract_ndarray::Tensor::from(input);
        // 2) run inference
        let outputs = self.model.run(tvec!(tensor))?;
        // 3) convert each tract tensor back into ndarray
        let mut arrays = Vec::with_capacity(outputs.len());
        for output in outputs {
            let arr: ndarray::ArrayD<f32> = output.to_array_view::<f32>()?.to_owned();
            arrays.push(arr);
        }
        Ok(arrays)
    }
}
