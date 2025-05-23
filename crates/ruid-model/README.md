# ruid-model

A Rust crate that serves as a wrapper around ultralytics and onnxruntime for generating and quantizing models with arbitrary dimensions. This crate replaces the functionality of `export_models.py` with a more flexible Rust implementation.

## Features

- 🚀 **YOLO11 Export**: Download and export YOLO11-nano models via ultralytics CLI
- 🧠 **ResNet-50 Re-ID**: Export ResNet-50 Re-ID backbone models with custom dimensions
- ⚡ **Static Quantization**: Convert FP32 models to INT8 using ONNX Runtime (QOperator format)
- 📐 **Arbitrary Dimensions**: Support for custom input dimensions for both model types
- 🔄 **Async Operations**: All model export operations are async for better performance
- 🛠️ **Flexible API**: Multiple convenience functions and direct control options

## Dependencies

Before using this crate, ensure you have the following Python dependencies installed:

```bash
pip install ultralytics torch torchvision onnxruntime
```

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
ruid-model = { path = "path/to/ruid-model" }
tokio = { version = "1.0", features = ["macros", "rt-multi-thread"] }
```

### Basic Usage

```rust
use ruid_model::{export, ModelConfig, ModelType};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Export YOLO11 with default settings (224x224, quantized)
    let yolo_result = export::export_yolo11_default().await?;
    println!("YOLO11 exported to: {:?}", yolo_result.int8_model);

    // Export ResNet-50 with custom dimensions
    let resnet_result = export::export_resnet50_custom(
        PathBuf::from("./models"),
        (256, 128), // height x width
        true,       // quantize
    ).await?;
    println!("ResNet-50 exported to: {:?}", resnet_result.int8_model);

    Ok(())
}
```

### Advanced Usage

```rust
use ruid_model::{ModelConfig, ModelExporter, ModelType};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create custom configuration
    let config = ModelConfig {
        output_dir: PathBuf::from("./custom_models"),
        image_size: (416, 416),
        quantize: true,
        batch_size: 1,
    };

    // Create exporter with custom config
    let exporter = ModelExporter::new(config)?;
    
    // Export specific model type
    let result = exporter.export_model(ModelType::Yolo11Nano).await?;
    
    println!("Model exported successfully!");
    println!("Input shape: {:?}", result.input_shape);
    println!("Input name: {}", result.input_name);
    println!("Output names: {:?}", result.output_names);

    Ok(())
}
```

### Batch Export

```rust
use ruid_model::{export, ModelConfig, ModelType};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Export both models with the same configuration
    let config = ModelConfig {
        output_dir: PathBuf::from("./models"),
        image_size: (320, 320),
        quantize: true,
        batch_size: 1,
    };

    let (yolo_result, resnet_result) = export::export_all_models(config).await?;
    
    println!("YOLO11: {:?}", yolo_result.int8_model);
    println!("ResNet-50: {:?}", resnet_result.int8_model);

    Ok(())
}
```

### Multi-Dimension Export

```rust
use ruid_model::{export, ModelConfig, ModelType};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let base_config = ModelConfig {
        output_dir: PathBuf::from("./models"),
        image_size: (224, 224), // Will be overridden
        quantize: true,
        batch_size: 1,
    };

    // Export YOLO with multiple dimensions
    let dimensions = vec![(224, 224), (416, 416), (640, 640)];
    let results = export::export_multi_dimension(
        ModelType::Yolo11Nano,
        base_config,
        dimensions,
    ).await?;

    for (i, result) in results.iter().enumerate() {
        println!("Model {}: {:?}", i + 1, result.int8_model);
    }

    Ok(())
}
```

## API Reference

### Core Types

#### `ModelConfig`
Configuration for model export operations:
- `output_dir: PathBuf` - Directory where models will be saved
- `image_size: (u32, u32)` - Input image dimensions (height, width)
- `quantize: bool` - Whether to quantize the model to INT8
- `batch_size: u32` - Batch size for model input

#### `ModelType`
Supported model types:
- `Yolo11Nano` - YOLO11 nano detection model
- `ResNet50ReId` - ResNet-50 Re-ID backbone

#### `ExportResult`
Result of model export operation:
- `fp32_model: Option<PathBuf>` - Path to FP32 model (if kept)
- `int8_model: PathBuf` - Path to quantized INT8 model
- `input_shape: Vec<i64>` - Model input shape
- `input_name: String` - Model input name
- `output_names: Vec<String>` - Model output names

### Main API

#### `ModelExporter`
Main struct for model export operations:
- `new(config: ModelConfig) -> Result<Self>` - Create with custom config
- `with_defaults() -> Result<Self>` - Create with default config
- `export_model(model_type: ModelType) -> Result<ExportResult>` - Export specific model

#### Convenience Functions
- `export::export_yolo11_default()` - Export YOLO11 with defaults
- `export::export_yolo11_custom(output_dir, image_size, quantize)` - Export YOLO11 with custom settings
- `export::export_resnet50_default()` - Export ResNet-50 with defaults
- `export::export_resnet50_custom(output_dir, image_size, quantize)` - Export ResNet-50 with custom settings
- `export::export_all_models(config)` - Export both models with same config
- `export::export_multi_dimension(model_type, config, dimensions)` - Export with multiple dimensions

## Model Details

### YOLO11 Nano
- **Input**: RGB images with configurable dimensions
- **Default Size**: 224×224
- **Output**: Detection results (bounding boxes, classes, confidence)
- **Quantization**: Supports INT8 quantization via ONNX Runtime

### ResNet-50 Re-ID
- **Input**: RGB images, typically 128×64 for person Re-ID
- **Output**: 128-dimensional embedding vectors
- **Use Case**: Person re-identification tasks
- **Quantization**: Supports INT8 quantization via ONNX Runtime

## Quantization

The crate uses ONNX Runtime's static quantization with the following settings:
- **Format**: QOperator (fully fused INT8 operations)
- **Calibration**: Random data calibration
- **Operations**: Conv and MatMul layers are quantized, Sigmoid remains FP32
- **Method**: MinMax calibration method

## Error Handling

All functions return `Result<T, anyhow::Error>` for comprehensive error handling. Common error scenarios:
- Missing dependencies (ultralytics, Python, PyTorch)
- File system permissions
- Invalid model configurations
- Export failures

## Examples

Run the basic usage example:

```bash
cargo run --example basic_usage
```

## License

This crate is part of the RuID project. See the main project LICENSE for details.

## Contributing

Contributions are welcome! Please ensure all tests pass and follow the existing code style.

```bash
# Run tests
cargo test

# Run with all features
cargo test --all-features

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy