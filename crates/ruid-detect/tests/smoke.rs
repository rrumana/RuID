use ruid_detect::{OrtYolo, Detector};


#[test]
fn yolo_smoke() -> anyhow::Result<()> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")?;
    let default = format!("{}/../../models/base/yolo11n.onnx", manifest);
    let model = std::env::var("YOLO_MODEL").unwrap_or(default);
    
    // Test with 640x480 dimensions
    let mut det = OrtYolo::new(&model, 640, 640)?;

    // Blank 640x480 tensor → no detections
    let input = ndarray::Array3::<f32>::zeros((640,640,3));
    let out = det.detect(&input)?;
    assert!(out.is_empty());
    Ok(())
}
