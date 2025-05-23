use ruid_detect::{OrtYolo, Detector};


#[test]
fn yolo_smoke() -> anyhow::Result<()> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")?;
    let default = format!("{}/../../models/yolov11.onnx", manifest);
    let model = std::env::var("YOLO_MODEL").unwrap_or(default);
    let mut det = OrtYolo::new(&model)?;

    // Blank 640x480 tensor → no detections
    let input = ndarray::Array3::<f32>::zeros((480,640,3));
    let out = det.detect(&input)?;
    assert!(out.is_empty());
    Ok(())
}