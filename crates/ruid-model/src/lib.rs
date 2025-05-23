//! # ruid-model
//!
//! A Rust crate that serves as a wrapper around ultralytics and onnxruntime for generating
//! and quantizing models with arbitrary dimensions. This crate replaces the functionality
//! of export_models.py with a more flexible Rust implementation.
//!
//! ## Features
//!
//! - Export YOLO11 models with custom dimensions via ultralytics CLI
//! - Export ResNet-50 Re-ID models with custom dimensions
//! - Static quantization using ONNX Runtime (INT8 QOperator format)
//! - Support for arbitrary input dimensions
//! - Async model export operations

use anyhow::{Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::process::Command;

pub mod export;
pub mod quantize;
pub mod resnet;
pub mod yolo;

/// Configuration for model export operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Output directory for exported models
    pub output_dir: PathBuf,
    /// Input image dimensions (height, width)
    pub image_size: (u32, u32),
    /// Whether to quantize the exported model
    pub quantize: bool,
    /// Batch size for model input
    pub batch_size: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./models"),
            image_size: (224, 224),
            quantize: true,
            batch_size: 1,
        }
    }
}

/// Get the base models directory path
pub fn get_base_models_dir() -> PathBuf {
    PathBuf::from("./models/base")
}

/// Supported model types for export
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// YOLO11 nano detection model
    Yolo11Nano,
    /// ResNet-50 Re-ID backbone
    ResNet50ReId,
}

/// Model export result containing paths to generated files
#[derive(Debug)]
pub struct ExportResult {
    /// Path to the FP32 ONNX model (if kept)
    pub fp32_model: Option<PathBuf>,
    /// Path to the quantized INT8 model
    pub int8_model: PathBuf,
    /// Model input shape
    pub input_shape: Vec<i64>,
    /// Model input name
    pub input_name: String,
    /// Model output names
    pub output_names: Vec<String>,
}

/// Main model exporter struct
pub struct ModelExporter {
    config: ModelConfig,
}

impl ModelExporter {
    /// Create a new model exporter with the given configuration
    pub fn new(config: ModelConfig) -> Result<Self> {
        // Ensure output directory exists
        std::fs::create_dir_all(&config.output_dir)
            .with_context(|| format!("Failed to create output directory: {:?}", config.output_dir))?;
        
        Ok(Self { config })
    }

    /// Create a new model exporter with default configuration
    pub fn with_defaults() -> Result<Self> {
        Self::new(ModelConfig::default())
    }

    /// Export a model of the specified type
    pub async fn export_model(&self, model_type: ModelType) -> Result<ExportResult> {
        match model_type {
            ModelType::Yolo11Nano => self.export_yolo11().await,
            ModelType::ResNet50ReId => self.export_resnet50_reid().await,
        }
    }

    /// Export YOLO11 nano model with custom dimensions
    async fn export_yolo11(&self) -> Result<ExportResult> {
        let yolo_exporter = yolo::YoloExporter::new(&self.config);
        yolo_exporter.export().await
    }

    /// Export ResNet-50 Re-ID model with custom dimensions
    async fn export_resnet50_reid(&self) -> Result<ExportResult> {
        let resnet_exporter = resnet::ResNetExporter::new(&self.config);
        resnet_exporter.export().await
    }

    /// Get the current configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: ModelConfig) -> Result<()> {
        std::fs::create_dir_all(&config.output_dir)
            .with_context(|| format!("Failed to create output directory: {:?}", config.output_dir))?;
        self.config = config;
        Ok(())
    }
}

/// Utility function to check if ultralytics CLI is available
pub async fn check_ultralytics_available() -> Result<bool> {
    let output = Command::new("yolo")
        .arg("--help")
        .output()
        .await;
    
    match output {
        Ok(output) => Ok(output.status.success()),
        Err(_) => Ok(false),
    }
}

/// Utility function to validate ONNX model
pub fn validate_onnx_model(model_path: &Path) -> Result<()> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)?;
    
    // Basic validation - check if we can create a session
    drop(session);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_model_exporter_creation() {
        let temp_dir = tempdir().unwrap();
        let config = ModelConfig {
            output_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        
        let exporter = ModelExporter::new(config);
        assert!(exporter.is_ok());
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.image_size, (224, 224));
        assert_eq!(config.batch_size, 1);
        assert!(config.quantize);
    }

    #[tokio::test]
    async fn test_ultralytics_check() {
        // This test will pass if ultralytics is installed, otherwise it will return false
        let available = check_ultralytics_available().await.unwrap();
        // We don't assert true/false since it depends on the environment
        println!("Ultralytics available: {}", available);
    }
}