//! YOLO11 model export functionality
//!
//! This module handles exporting YOLO11 models via the ultralytics CLI
//! and provides functionality for custom dimensions and quantization.

use crate::{quantize::ModelQuantizer, ExportResult, ModelConfig, get_base_models_dir};
use anyhow::{anyhow, Context, Result};
use std::path::PathBuf;
use tokio::process::Command;

/// YOLO model exporter
pub struct YoloExporter<'a> {
    config: &'a ModelConfig,
}

impl<'a> YoloExporter<'a> {
    /// Create a new YOLO exporter with the given configuration
    pub fn new(config: &'a ModelConfig) -> Self {
        Self { config }
    }

    /// Export YOLO11 nano model
    pub async fn export(&self) -> Result<ExportResult> {
        // Step 1: Export YOLO11 via ultralytics CLI
        let fp32_path = self.export_yolo11_onnx().await?;
        
        // Step 2: Move to output directory
        let fp32_output = self.config.output_dir.join("yolo11n_fp32.onnx");
        tokio::fs::copy(&fp32_path, &fp32_output).await
            .with_context(|| format!("Failed to copy YOLO model to output directory"))?;

        // Step 3: Quantize if requested
        let int8_model = if self.config.quantize {
            let quantizer = ModelQuantizer::new(self.config);
            let int8_path = quantizer.quantize_static(
                &fp32_output,
                "yolo11n_int8.onnx",
                "images",
                &[
                    self.config.batch_size as i64,
                    3,
                    self.config.image_size.0 as i64,
                    self.config.image_size.1 as i64,
                ],
            ).await?;

            // Remove FP32 model if quantization succeeded
            tokio::fs::remove_file(&fp32_output).await.ok();
            int8_path
        } else {
            fp32_output.clone()
        };

        Ok(ExportResult {
            fp32_model: if self.config.quantize { None } else { Some(fp32_output) },
            int8_model,
            input_shape: vec![
                self.config.batch_size as i64,
                3,
                self.config.image_size.0 as i64,
                self.config.image_size.1 as i64,
            ],
            input_name: "images".to_string(),
            output_names: vec!["output0".to_string()],
        })
    }

    /// Export YOLO11 model via ultralytics CLI
    async fn export_yolo11_onnx(&self) -> Result<PathBuf> {
        println!("▶ Exporting YOLO11-nano via `yolo export` ({}×{})…",
                 self.config.image_size.1, self.config.image_size.0);

        // Ensure base models directory exists
        let base_models_dir = get_base_models_dir();
        std::fs::create_dir_all(&base_models_dir)
            .with_context(|| format!("Failed to create base models directory: {:?}", base_models_dir))?;

        let output = Command::new("yolo")
            .args([
                "export",
                "model=yolo11n.pt",
                "format=onnx",
                &format!("imgsz={}", self.config.image_size.0),
                "simplify=True",
            ])
            .current_dir(&base_models_dir)
            .output()
            .await
            .context("Failed to execute yolo export command")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("YOLO export failed: {}", stderr));
        }

        // Find the exported model file
        self.find_exported_yolo_model().await
    }

    /// Find the exported YOLO model file in various possible locations
    async fn find_exported_yolo_model(&self) -> Result<PathBuf> {
        let base_models_dir = get_base_models_dir();
        
        // Candidate locations where ultralytics might save the model
        let mut candidates = vec![
            base_models_dir.join("yolo11n.onnx"),
            base_models_dir.join("runs/export/train/yolo11n.onnx"),
            base_models_dir.join("runs/export/train2/yolo11n.onnx"),
            base_models_dir.join("runs/export/train3/yolo11n.onnx"),
        ];

        // Also check for numbered export directories
        for i in 1..=10 {
            candidates.push(base_models_dir.join(format!("runs/export/train{}/yolo11n.onnx", i)));
        }

        // Find the first existing file
        for candidate in &candidates {
            if candidate.exists() {
                return Ok(candidate.clone());
            }
        }

        // If not found in standard locations, use glob to search
        let pattern = base_models_dir.join("runs/export/*/yolo11n.onnx");
        let glob_pattern = pattern.to_string_lossy();
        
        for entry in glob::glob(&glob_pattern)? {
            if let Ok(path) = entry {
                if path.exists() {
                    return Ok(path);
                }
            }
        }

        Err(anyhow!(
            "Could not find exported yolo11n.onnx in any of the expected locations: {:?}",
            candidates
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_yolo_exporter_creation() {
        let temp_dir = tempdir().unwrap();
        let config = ModelConfig {
            output_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        
        let exporter = YoloExporter::new(&config);
        assert_eq!(exporter.config.image_size, (224, 224));
    }

    #[tokio::test]
    async fn test_find_exported_model_nonexistent() {
        let temp_dir = tempdir().unwrap();
        let config = ModelConfig {
            output_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        
        let exporter = YoloExporter::new(&config);
        let result = exporter.find_exported_yolo_model().await;
        assert!(result.is_err());
    }
}