use ruid_detect::{TractYolo, Detector};

#[test]
fn yolo_smoke() -> anyhow::Result<()> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")?;
    let default = format!("{}/../../models/yolo11n_int8.onnx", manifest);
    let model = std::env::var("YOLO_MODEL").unwrap_or(default);
    let det = TractYolo::new(&model)?;

    // Blank 224×224 tensor → no detections
    let input = ndarray::Array3::<f32>::zeros((224,224,3));
    let out = det.detect(&input)?;
    assert!(out.is_empty());
    Ok(())
}