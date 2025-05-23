//! ResNet-50 Re-ID model export functionality
//!
//! This module handles exporting ResNet-50 Re-ID models by generating
//! a Python script that creates and exports the model using PyTorch.

use crate::{quantize::ModelQuantizer, ExportResult, ModelConfig};
use anyhow::{Context, Result};
use std::path::PathBuf;

/// ResNet model exporter
pub struct ResNetExporter<'a> {
    config: &'a ModelConfig,
}

impl<'a> ResNetExporter<'a> {
    /// Create a new ResNet exporter with the given configuration
    pub fn new(config: &'a ModelConfig) -> Self {
        Self { config }
    }

    /// Export ResNet-50 Re-ID model
    pub async fn export(&self) -> Result<ExportResult> {
        // Step 1: Generate and export ResNet model via Python script
        let fp32_path = self.export_resnet_onnx().await?;
        
        // Step 2: Quantize if requested
        let int8_model = if self.config.quantize {
            let quantizer = ModelQuantizer::new(self.config);
            let int8_path = quantizer.quantize_static(
                &fp32_path,
                "resnet50_market1501_int8.onnx",
                "images",
                &[
                    self.config.batch_size as i64,
                    3,
                    128, // ResNet Re-ID typically uses 128x64 input
                    64,
                ],
            ).await?;

            // Remove FP32 model if quantization succeeded
            tokio::fs::remove_file(&fp32_path).await.ok();
            int8_path
        } else {
            fp32_path.clone()
        };

        Ok(ExportResult {
            fp32_model: if self.config.quantize { None } else { Some(fp32_path) },
            int8_model,
            input_shape: vec![
                self.config.batch_size as i64,
                3,
                128, // Standard Re-ID input height
                64,  // Standard Re-ID input width
            ],
            input_name: "images".to_string(),
            output_names: vec!["embeddings".to_string()],
        })
    }

    /// Export ResNet model via Python script
    async fn export_resnet_onnx(&self) -> Result<PathBuf> {
        println!("▶ Building & exporting ResNet-50 Re-ID backbone…");

        // Generate Python script for ResNet export
        let script_content = self.generate_resnet_export_script()?;
        let temp_script = tempfile::NamedTempFile::with_suffix(".py")?;
        tokio::fs::write(temp_script.path(), script_content).await?;

        // Execute the Python script
        let output = tokio::process::Command::new("python3")
            .arg(temp_script.path())
            .output()
            .await
            .context("Failed to execute ResNet export script")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("ResNet export failed: {}", stderr));
        }

        let output_path = self.config.output_dir.join("resnet50_market1501_fp32.onnx");
        Ok(output_path)
    }

    /// Generate Python script for ResNet export
    fn generate_resnet_export_script(&self) -> Result<String> {
        let output_path = self.config.output_dir.join("resnet50_market1501_fp32.onnx");
        
        let script = format!(r#"
#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchvision import models
import os

def create_resnet50_reid():
    """Create ResNet-50 model adapted for Re-ID with 128-dimensional embeddings."""
    # Load pre-trained ResNet-50
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Replace the final fully connected layer for Re-ID
    # Standard Re-ID uses 128-dimensional embeddings
    resnet.fc = nn.Linear(resnet.fc.in_features, 128)
    
    # Set to evaluation mode
    resnet.eval()
    
    return resnet

def export_resnet_onnx():
    """Export ResNet-50 Re-ID model to ONNX format."""
    # Create the model
    model = create_resnet50_reid()
    
    # Create dummy input tensor (batch_size=1, channels=3, height=128, width=64)
    # This is the standard input size for person Re-ID
    dummy_input = torch.randn(1, 3, 128, 64)
    
    # Ensure output directory exists
    output_dir = "{output_dir}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export to ONNX
    output_path = "{output_path}"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={{
            "images": {{0: "batch_size"}},
            "embeddings": {{0: "batch_size"}}
        }}
    )
    
    print(f"ResNet-50 Re-ID model exported to: {{output_path}}")
    return output_path

if __name__ == "__main__":
    try:
        export_resnet_onnx()
        print("✓ ResNet export completed successfully")
    except Exception as e:
        print(f"✗ ResNet export failed: {{e}}")
        raise
"#,
            output_dir = self.config.output_dir.to_string_lossy(),
            output_path = output_path.to_string_lossy(),
        );

        Ok(script)
    }

    /// Create ResNet model with custom dimensions
    pub async fn export_with_custom_dimensions(&self, height: u32, width: u32) -> Result<ExportResult> {
        println!("▶ Building & exporting ResNet-50 Re-ID backbone with custom dimensions ({}×{})…", height, width);

        // Generate Python script with custom dimensions
        let script_content = self.generate_custom_resnet_script(height, width)?;
        let temp_script = tempfile::NamedTempFile::with_suffix(".py")?;
        tokio::fs::write(temp_script.path(), script_content).await?;

        // Execute the Python script
        let output = tokio::process::Command::new("python3")
            .arg(temp_script.path())
            .output()
            .await
            .context("Failed to execute custom ResNet export script")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("Custom ResNet export failed: {}", stderr));
        }

        let fp32_path = self.config.output_dir.join(format!("resnet50_reid_{}x{}_fp32.onnx", height, width));
        
        // Quantize if requested
        let int8_model = if self.config.quantize {
            let quantizer = ModelQuantizer::new(self.config);
            let int8_path = quantizer.quantize_static(
                &fp32_path,
                &format!("resnet50_reid_{}x{}_int8.onnx", height, width),
                "images",
                &[
                    self.config.batch_size as i64,
                    3,
                    height as i64,
                    width as i64,
                ],
            ).await?;

            tokio::fs::remove_file(&fp32_path).await.ok();
            int8_path
        } else {
            fp32_path.clone()
        };

        Ok(ExportResult {
            fp32_model: if self.config.quantize { None } else { Some(fp32_path) },
            int8_model,
            input_shape: vec![
                self.config.batch_size as i64,
                3,
                height as i64,
                width as i64,
            ],
            input_name: "images".to_string(),
            output_names: vec!["embeddings".to_string()],
        })
    }

    /// Generate Python script for custom dimension ResNet export
    fn generate_custom_resnet_script(&self, height: u32, width: u32) -> Result<String> {
        let output_path = self.config.output_dir.join(format!("resnet50_reid_{}x{}_fp32.onnx", height, width));
        
        let script = format!(r#"
#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchvision import models
import os

def export_custom_resnet_onnx():
    """Export ResNet-50 Re-ID model with custom dimensions to ONNX format."""
    # Create the model
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Linear(resnet.fc.in_features, 128)
    resnet.eval()
    
    # Create dummy input tensor with custom dimensions
    dummy_input = torch.randn(1, 3, {height}, {width})
    
    # Ensure output directory exists
    output_dir = "{output_dir}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export to ONNX
    output_path = "{output_path}"
    torch.onnx.export(
        resnet,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={{
            "images": {{0: "batch_size"}},
            "embeddings": {{0: "batch_size"}}
        }}
    )
    
    print(f"Custom ResNet-50 Re-ID model ({height}×{width}) exported to: {{output_path}}")
    return output_path

if __name__ == "__main__":
    try:
        export_custom_resnet_onnx()
        print("✓ Custom ResNet export completed successfully")
    except Exception as e:
        print(f"✗ Custom ResNet export failed: {{e}}")
        raise
"#,
            height = height,
            width = width,
            output_dir = self.config.output_dir.to_string_lossy(),
            output_path = output_path.to_string_lossy(),
        );

        Ok(script)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_resnet_exporter_creation() {
        let temp_dir = tempdir().unwrap();
        let config = ModelConfig {
            output_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        
        let exporter = ResNetExporter::new(&config);
        assert_eq!(exporter.config.image_size, (224, 224));
    }

    #[test]
    fn test_resnet_script_generation() {
        let temp_dir = tempdir().unwrap();
        let config = ModelConfig {
            output_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        
        let exporter = ResNetExporter::new(&config);
        let script = exporter.generate_resnet_export_script();
        
        assert!(script.is_ok());
        let script_content = script.unwrap();
        assert!(script_content.contains("resnet50"));
        assert!(script_content.contains("torch.onnx.export"));
        assert!(script_content.contains("embeddings"));
    }

    #[test]
    fn test_custom_resnet_script_generation() {
        let temp_dir = tempdir().unwrap();
        let config = ModelConfig {
            output_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };
        
        let exporter = ResNetExporter::new(&config);
        let script = exporter.generate_custom_resnet_script(256, 128);
        
        assert!(script.is_ok());
        let script_content = script.unwrap();
        assert!(script_content.contains("256"));
        assert!(script_content.contains("128"));
        assert!(script_content.contains("torch.randn(1, 3, 256, 128)"));
    }
}