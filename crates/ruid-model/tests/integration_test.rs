//! Integration tests for ruid-model crate

use ruid_model::{ModelConfig, ModelExporter, ModelType};
use std::path::PathBuf;
use tempfile::tempdir;

#[tokio::test]
async fn test_model_exporter_integration() {
    let temp_dir = tempdir().unwrap();
    let config = ModelConfig {
        output_dir: temp_dir.path().to_path_buf(),
        image_size: (224, 224),
        quantize: false, // Skip quantization for faster testing
        batch_size: 1,
    };

    // Test that we can create an exporter
    let exporter = ModelExporter::new(config).unwrap();
    assert_eq!(exporter.config().image_size, (224, 224));
    assert_eq!(exporter.config().batch_size, 1);
    assert!(!exporter.config().quantize);

    // Test configuration updates
    let mut exporter = exporter;
    let new_config = ModelConfig {
        output_dir: temp_dir.path().join("new_models"),
        image_size: (416, 416),
        quantize: true,
        batch_size: 2,
    };
    
    exporter.set_config(new_config).unwrap();
    assert_eq!(exporter.config().image_size, (416, 416));
    assert_eq!(exporter.config().batch_size, 2);
    assert!(exporter.config().quantize);
}

#[tokio::test]
async fn test_ultralytics_availability_check() {
    // This test checks if the ultralytics availability check works
    // It should return either true or false without panicking
    let available = ruid_model::check_ultralytics_available().await.unwrap();
    
    // We don't assert a specific value since it depends on the environment
    // but we ensure the function completes successfully
    println!("Ultralytics available: {}", available);
}

#[test]
fn test_model_config_serialization() {
    let config = ModelConfig {
        output_dir: PathBuf::from("/tmp/models"),
        image_size: (640, 640),
        quantize: true,
        batch_size: 4,
    };

    // Test that we can serialize and deserialize the config
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: ModelConfig = serde_json::from_str(&json).unwrap();
    
    assert_eq!(config.output_dir, deserialized.output_dir);
    assert_eq!(config.image_size, deserialized.image_size);
    assert_eq!(config.quantize, deserialized.quantize);
    assert_eq!(config.batch_size, deserialized.batch_size);
}

#[test]
fn test_model_types() {
    // Test that model types are correctly defined
    let yolo = ModelType::Yolo11Nano;
    let resnet = ModelType::ResNet50ReId;
    
    assert_eq!(yolo, ModelType::Yolo11Nano);
    assert_eq!(resnet, ModelType::ResNet50ReId);
    assert_ne!(yolo, resnet);
}

#[test]
fn test_export_result_structure() {
    use ruid_model::ExportResult;
    
    // Test that we can create an ExportResult
    let result = ExportResult {
        fp32_model: Some(PathBuf::from("/tmp/model_fp32.onnx")),
        int8_model: PathBuf::from("/tmp/model_int8.onnx"),
        input_shape: vec![1, 3, 224, 224],
        input_name: "images".to_string(),
        output_names: vec!["output0".to_string()],
    };
    
    assert!(result.fp32_model.is_some());
    assert_eq!(result.input_shape, vec![1, 3, 224, 224]);
    assert_eq!(result.input_name, "images");
    assert_eq!(result.output_names.len(), 1);
}