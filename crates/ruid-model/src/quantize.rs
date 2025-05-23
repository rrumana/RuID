//! Model quantization functionality using ONNX Runtime
//!
//! This module provides static quantization capabilities to convert FP32 models
//! to INT8 QOperator format for better performance and smaller model size.

use crate::ModelConfig;
use anyhow::{Context, Result};
use ndarray::Array4;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::path::{Path, PathBuf};

/// Model quantizer for converting FP32 models to INT8
pub struct ModelQuantizer<'a> {
    config: &'a ModelConfig,
}

impl<'a> ModelQuantizer<'a> {
    /// Create a new model quantizer
    pub fn new(config: &'a ModelConfig) -> Self {
        Self { config }
    }

    /// Perform static quantization on a model
    pub async fn quantize_static(
        &self,
        input_model: &Path,
        output_name: &str,
        input_name: &str,
        input_shape: &[i64],
    ) -> Result<PathBuf> {
        let output_path = self.config.output_dir.join(output_name);
        
        println!("▶ Static-quantising {} → {}", 
                 input_model.file_name().unwrap().to_string_lossy(),
                 output_path.file_name().unwrap().to_string_lossy());

        // For now, we'll use a simplified approach since ONNX Runtime's quantization
        // APIs are not directly available in the Rust bindings.
        // In a production environment, you would either:
        // 1. Call Python's onnxruntime.quantization via subprocess
        // 2. Use a different quantization library
        // 3. Implement custom quantization logic

        // As a placeholder, we'll copy the model and add a note that this needs
        // to be implemented with proper quantization
        self.quantize_via_python_subprocess(input_model, &output_path, input_name, input_shape).await
    }

    /// Quantize model by calling Python's onnxruntime.quantization via subprocess
    async fn quantize_via_python_subprocess(
        &self,
        input_model: &Path,
        output_model: &Path,
        input_name: &str,
        input_shape: &[i64],
    ) -> Result<PathBuf> {
        // Create a temporary Python script for quantization
        let script_content = self.generate_quantization_script(
            input_model,
            output_model,
            input_name,
            input_shape,
        )?;

        let temp_script = tempfile::NamedTempFile::with_suffix(".py")?;
        tokio::fs::write(temp_script.path(), script_content).await?;

        // Execute the Python script
        let output = tokio::process::Command::new("python3")
            .arg(temp_script.path())
            .output()
            .await
            .context("Failed to execute quantization script")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("Quantization failed: {}", stderr));
        }

        Ok(output_model.to_path_buf())
    }

    /// Generate Python script for quantization
    fn generate_quantization_script(
        &self,
        input_model: &Path,
        output_model: &Path,
        input_name: &str,
        input_shape: &[i64],
    ) -> Result<String> {
        let shape_str = format!("({}, {}, {}, {})", 
                               input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
        
        let script = format!(r#"
#!/usr/bin/env python3
import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    CalibrationMethod,
    quantize_static,
)

class RandomCalibrationReader(CalibrationDataReader):
    """Feeds a single random tensor into quantize_static as calibration data."""

    def __init__(self, input_name: str, shape={shape_str}):
        self.input_name = input_name
        self.shape = shape
        self._done = False

    def get_next(self):
        if self._done:
            return None
        rnd = np.random.rand(*self.shape).astype(np.float32)
        self._done = True
        return {{self.input_name: rnd}}

def main():
    reader = RandomCalibrationReader(input_name="{input_name}", shape={shape_str})
    
    quantize_static(
        model_input="{input_path}",
        model_output="{output_path}",
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,  # fully fused INT8 ops
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv", "MatMul"],  # keep Sigmoid FP32
    )
    print(f"Quantization complete: {output_path}")

if __name__ == "__main__":
    main()
"#,
            shape_str = shape_str,
            input_name = input_name,
            input_path = input_model.to_string_lossy(),
            output_path = output_model.to_string_lossy(),
        );

        Ok(script)
    }

    /// Generate calibration data for quantization
    pub fn generate_calibration_data(&self, input_shape: &[i64]) -> Array4<f32> {
        let shape = (
            input_shape[0] as usize,
            input_shape[1] as usize,
            input_shape[2] as usize,
            input_shape[3] as usize,
        );
        
        // Generate random calibration data
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Array4::from_shape_fn(shape, |_| rng.gen_range(0.0..1.0))
    }

    /// Validate quantized model by loading it
    pub fn validate_quantized_model(&self, model_path: &Path) -> Result<()> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        
        // Basic validation - check if we can create a session
        drop(session);
        println!("✓ Quantized model validation successful");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_quantizer_creation() {
        let temp_dir = tempdir().unwrap();
        let config = ModelConfig {
            output_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        
        let quantizer = ModelQuantizer::new(&config);
        assert_eq!(quantizer.config.image_size, (224, 224));
    }

    #[test]
    fn test_calibration_data_generation() {
        let temp_dir = tempdir().unwrap();
        let config = ModelConfig {
            output_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        
        let quantizer = ModelQuantizer::new(&config);
        let shape = [1, 3, 224, 224];
        let data = quantizer.generate_calibration_data(&shape);
        
        assert_eq!(data.shape(), &[1, 3, 224, 224]);
    }

    #[test]
    fn test_quantization_script_generation() {
        let temp_dir = tempdir().unwrap();
        let config = ModelConfig {
            output_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        
        let quantizer = ModelQuantizer::new(&config);
        let input_path = temp_dir.path().join("input.onnx");
        let output_path = temp_dir.path().join("output.onnx");
        let shape = [1, 3, 224, 224];
        
        let script = quantizer.generate_quantization_script(
            &input_path,
            &output_path,
            "images",
            &shape,
        );
        
        assert!(script.is_ok());
        let script_content = script.unwrap();
        assert!(script_content.contains("quantize_static"));
        assert!(script_content.contains("images"));
    }
}