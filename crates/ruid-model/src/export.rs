//! High-level export functionality
//!
//! This module provides convenient functions for exporting models with
//! various configurations and custom dimensions.

use crate::{ExportResult, ModelConfig, ModelExporter, ModelType};
use anyhow::Result;
use std::path::PathBuf;

/// Export a YOLO11 model with default configuration
pub async fn export_yolo11_default() -> Result<ExportResult> {
    let exporter = ModelExporter::with_defaults()?;
    exporter.export_model(ModelType::Yolo11Nano).await
}

/// Export a YOLO11 model with custom dimensions
pub async fn export_yolo11_custom(
    output_dir: PathBuf,
    image_size: (u32, u32),
    quantize: bool,
) -> Result<ExportResult> {
    let config = ModelConfig {
        output_dir,
        image_size,
        quantize,
        batch_size: 1,
    };
    
    let exporter = ModelExporter::new(config)?;
    exporter.export_model(ModelType::Yolo11Nano).await
}

/// Export a ResNet-50 Re-ID model with default configuration
pub async fn export_resnet50_default() -> Result<ExportResult> {
    let exporter = ModelExporter::with_defaults()?;
    exporter.export_model(ModelType::ResNet50ReId).await
}

/// Export a ResNet-50 Re-ID model with custom dimensions
pub async fn export_resnet50_custom(
    output_dir: PathBuf,
    image_size: (u32, u32),
    quantize: bool,
) -> Result<ExportResult> {
    let config = ModelConfig {
        output_dir,
        image_size,
        quantize,
        batch_size: 1,
    };
    
    let exporter = ModelExporter::new(config)?;
    exporter.export_model(ModelType::ResNet50ReId).await
}

/// Export both YOLO11 and ResNet-50 models with the same configuration
pub async fn export_all_models(config: ModelConfig) -> Result<(ExportResult, ExportResult)> {
    let exporter = ModelExporter::new(config)?;
    
    let yolo_result = exporter.export_model(ModelType::Yolo11Nano).await?;
    let resnet_result = exporter.export_model(ModelType::ResNet50ReId).await?;
    
    Ok((yolo_result, resnet_result))
}

/// Batch export models with different configurations
pub async fn batch_export(configs: Vec<(ModelType, ModelConfig)>) -> Result<Vec<ExportResult>> {
    let mut results = Vec::new();
    
    for (model_type, config) in configs {
        let exporter = ModelExporter::new(config)?;
        let result = exporter.export_model(model_type).await?;
        results.push(result);
    }
    
    Ok(results)
}

/// Export models with multiple dimensions for the same model type
pub async fn export_multi_dimension(
    model_type: ModelType,
    base_config: ModelConfig,
    dimensions: Vec<(u32, u32)>,
) -> Result<Vec<ExportResult>> {
    let mut results = Vec::new();
    
    for (height, width) in dimensions {
        let mut config = base_config.clone();
        config.image_size = (height, width);
        
        // Update output directory to include dimensions
        let dim_suffix = format!("_{}x{}", height, width);
        let base_name = config.output_dir.file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("models"))
            .to_string_lossy();
        config.output_dir = config.output_dir.parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(format!("{}{}", base_name, dim_suffix));
        
        let exporter = ModelExporter::new(config)?;
        let result = exporter.export_model(model_type).await?;
        results.push(result);
    }
    
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_batch_export_config_creation() {
        let temp_dir = tempdir().unwrap();
        
        let configs = vec![
            (ModelType::Yolo11Nano, ModelConfig {
                output_dir: temp_dir.path().join("yolo"),
                image_size: (224, 224),
                quantize: true,
                batch_size: 1,
            }),
            (ModelType::ResNet50ReId, ModelConfig {
                output_dir: temp_dir.path().join("resnet"),
                image_size: (128, 64),
                quantize: true,
                batch_size: 1,
            }),
        ];
        
        assert_eq!(configs.len(), 2);
        assert_eq!(configs[0].0, ModelType::Yolo11Nano);
        assert_eq!(configs[1].0, ModelType::ResNet50ReId);
    }

    #[test]
    fn test_multi_dimension_config() {
        let temp_dir = tempdir().unwrap();
        let _base_config = ModelConfig {
            output_dir: temp_dir.path().join("models"),
            image_size: (224, 224),
            quantize: true,
            batch_size: 1,
        };
        
        let dimensions = vec![(224, 224), (416, 416), (640, 640)];
        assert_eq!(dimensions.len(), 3);
        
        // Test dimension suffix generation
        for (height, width) in &dimensions {
            let suffix = format!("_{}x{}", height, width);
            assert!(suffix.contains(&height.to_string()));
            assert!(suffix.contains(&width.to_string()));
        }
    }
}