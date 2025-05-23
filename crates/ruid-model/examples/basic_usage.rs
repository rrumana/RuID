//! Basic usage example for ruid-model crate
//!
//! This example demonstrates how to export models with different configurations
//! and custom dimensions using the ruid-model crate.

use anyhow::Result;
use ruid_model::{export, ModelConfig, ModelExporter, ModelType};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 ruid-model Basic Usage Example");
    println!("==================================");

    // Example 1: Export YOLO11 with default settings
    println!("\n📦 Example 1: Export YOLO11 with default settings");
    match export::export_yolo11_default().await {
        Ok(result) => {
            println!("✅ YOLO11 export successful!");
            println!("   INT8 model: {:?}", result.int8_model);
            println!("   Input shape: {:?}", result.input_shape);
        }
        Err(e) => {
            println!("❌ YOLO11 export failed: {}", e);
        }
    }

    // Example 2: Export ResNet-50 with custom dimensions
    println!("\n📦 Example 2: Export ResNet-50 with custom dimensions");
    let custom_output = PathBuf::from("./models/custom");
    match export::export_resnet50_custom(custom_output, (256, 128), true).await {
        Ok(result) => {
            println!("✅ ResNet-50 export successful!");
            println!("   INT8 model: {:?}", result.int8_model);
            println!("   Input shape: {:?}", result.input_shape);
        }
        Err(e) => {
            println!("❌ ResNet-50 export failed: {}", e);
        }
    }

    // Example 3: Export both models with the same configuration
    println!("\n📦 Example 3: Export both models with same configuration");
    let config = ModelConfig {
        output_dir: PathBuf::from("./models/batch"),
        image_size: (416, 416),
        quantize: true,
        batch_size: 1,
    };

    match export::export_all_models(config).await {
        Ok((yolo_result, resnet_result)) => {
            println!("✅ Batch export successful!");
            println!("   YOLO model: {:?}", yolo_result.int8_model);
            println!("   ResNet model: {:?}", resnet_result.int8_model);
        }
        Err(e) => {
            println!("❌ Batch export failed: {}", e);
        }
    }

    // Example 4: Export with multiple dimensions
    println!("\n📦 Example 4: Export YOLO with multiple dimensions");
    let base_config = ModelConfig {
        output_dir: PathBuf::from("./models/multi_dim"),
        image_size: (224, 224), // This will be overridden
        quantize: false, // Keep FP32 for this example
        batch_size: 1,
    };

    let dimensions = vec![(224, 224), (416, 416), (640, 640)];
    match export::export_multi_dimension(ModelType::Yolo11Nano, base_config, dimensions).await {
        Ok(results) => {
            println!("✅ Multi-dimension export successful!");
            for (i, result) in results.iter().enumerate() {
                println!("   Model {}: {:?} (shape: {:?})", 
                         i + 1, 
                         result.fp32_model.as_ref().unwrap_or(&result.int8_model),
                         result.input_shape);
            }
        }
        Err(e) => {
            println!("❌ Multi-dimension export failed: {}", e);
        }
    }

    // Example 5: Using ModelExporter directly for fine control
    println!("\n📦 Example 5: Using ModelExporter directly");
    let custom_config = ModelConfig {
        output_dir: PathBuf::from("./models/direct"),
        image_size: (512, 512),
        quantize: true,
        batch_size: 2,
    };

    match ModelExporter::new(custom_config) {
        Ok(exporter) => {
            println!("✅ ModelExporter created successfully");
            println!("   Configuration: {:?}", exporter.config());
            
            // You could now call exporter.export_model(ModelType::Yolo11Nano).await
            // but we'll skip the actual export to avoid redundancy
        }
        Err(e) => {
            println!("❌ ModelExporter creation failed: {}", e);
        }
    }

    println!("\n🎉 Example completed!");
    println!("\nNote: Some exports may fail if dependencies (ultralytics, PyTorch) are not installed.");
    println!("Install with: pip install ultralytics torch torchvision onnxruntime");

    Ok(())
}