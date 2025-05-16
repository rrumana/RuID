use anyhow::Result;
use ndarray::{Array3, Array4};
use tract_onnx::prelude::*;   // brings in ModelBuilder, RunnableModel, Tensor, etc.
use tract_ndarray::ArrayD;

/// A single detection: bounding box [x1,y1,x2,y2] in normalized coords plus score.
#[derive(Debug)]
pub struct Detection {
    pub bbox:  [f32;4],
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
    input_dims: [usize; 4], // e.g. [1,3,224,224]
}

impl TractYolo {
    /// Load and optimize the ONNX model, preparing it for inference.
    pub fn new(model_path: &str) -> Result<Self> {
        // 1. Load the ONNX graph
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            // 2. Declare the input fact: 1×3×224×224 f32
            .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1,3,224,224)))?
            // 3. Optimize and compile
            .into_optimized()?
            .into_runnable()?;

        let input_dims = [1,3,224,224];
        Ok(Self { model, input_dims })
    }
}

impl Detector for TractYolo {
    fn detect(&self, input: &Array3<f32>) -> Result<Vec<Detection>> {
        let (h, w, c) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        assert_eq!(c, 3, "Expected 3 channels");

        // 1. Rearrange H×W×C → N×C×H×W
        let mut arr4 = Array4::<f32>::zeros((1,3,h,w));
        for y in 0..h {
            for x in 0..w {
                for ch in 0..3 {
                    arr4[(0,ch,y,x)] = input[(y,x,ch)];
                }
            }
        }

        // 2. Run the model
        let input_tensor: Tensor = arr4.into();
        let outputs = self.model.run(tvec![input_tensor])?;

        // 3. Convert first output to an ArrayD<f32>
        let output: ArrayD<f32> = outputs[0].to_array_view::<f32>()?.to_owned();
        // YOLOv8 output shape is usually [1, N, 6] where each row = [xc,yc,w,h,score,class]
        let dets_nd = output.into_dimensionality::<ndarray::Ix3>()?;
        let mut dets = Vec::new();

        for row in dets_nd.index_axis(ndarray::Axis(1), 0).outer_iter() {
            let xc    = row[0];
            let yc    = row[1];
            let wbox  = row[2];
            let hbox  = row[3];
            let score = row[4];
            let class = row[5] as usize;
            // Convert center→corner coords
            dets.push(Detection {
                bbox: [xc - wbox/2.0, yc - hbox/2.0, xc + wbox/2.0, yc + hbox/2.0],
                score,
                class,
            });
        }

        Ok(dets)
    }
}
