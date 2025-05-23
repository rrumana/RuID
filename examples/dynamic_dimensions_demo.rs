//! Demonstration of dynamic dimensions in RuID
//! 
//! This example shows how to use different camera, preprocessing, and detection
//! dimensions throughout the pipeline.

use anyhow::Result;
use ruid_camera::{VideoFrame, FrameBacking};
use ruid_preprocess::Preprocessor;
use ruid_detect::{OrtYolo, Detector};
use std::time::Duration;

fn main() -> Result<()> {
    println!("RuID Dynamic Dimensions Demo");
    println!("============================");
    
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let (cam_w, cam_h, prep_w, prep_h) = if args.len() >= 5 {
        let cw = args[1].parse::<u32>().unwrap_or(640);
        let ch = args[2].parse::<u32>().unwrap_or(480);
        let pw = args[3].parse::<u32>().unwrap_or(224);
        let ph = args[4].parse::<u32>().unwrap_or(224);
        (cw, ch, pw, ph)
    } else {
        println!("Usage: {} <cam_width> <cam_height> <preprocess_width> <preprocess_height>", args[0]);
        println!("Using defaults: 640x480 camera, 640x480 preprocessing");
        (640, 480, 224, 224)
    };

    println!("Camera dimensions: {}x{}", cam_w, cam_h);
    println!("Preprocessing dimensions: {}x{}", prep_w, prep_h);
    
    // 1. Create preprocessor with dynamic dimensions
    let preprocessor = Preprocessor::new(prep_w, prep_h);
    println!("✓ Created preprocessor for {}x{} output", prep_w, prep_h);
    
    // 2. Create a fake camera frame with the specified dimensions
    let frame_size = (cam_w * cam_h * 3 / 2) as usize; // NV12 format
    let mut fake_frame_data = vec![128u8; frame_size];
    // Fill Y plane with white
    fake_frame_data[..(cam_w * cam_h) as usize].fill(255);
    
    let fake_frame = VideoFrame {
        backing: FrameBacking::Cpu(fake_frame_data),
        width: cam_w,
        height: cam_h,
        stride: cam_w * 2,
        pts: Duration::ZERO,
    };
    println!("✓ Created fake {}x{} camera frame", cam_w, cam_h);
    
    // 3. Run preprocessing
    let tensor = preprocessor.run(&fake_frame)?;
    println!("✓ Preprocessed to tensor shape: {:?}", tensor.shape());
    
    // 4. For detection, we need to use the model's expected dimensions (640x480 for the current model)
    // This demonstrates that while preprocessing can be dynamic, detection is constrained by the model
    if prep_w == 224 && prep_h == 224 {
        let model_path = "models/base/yolo11n.onnx";
        if std::path::Path::new(model_path).exists() {
            let mut detector = OrtYolo::new(model_path, prep_w, prep_h)?;
            let detections = detector.detect(&tensor)?;
            println!("✓ Detection completed, found {} objects", detections.len());
        } else {
            println!("⚠ Model file not found at {}, skipping detection", model_path);
        }
    } else {
        println!("⚠ Detection skipped - current model requires 640x480 input");
        println!("  (preprocessing output is {}x{})", prep_w, prep_h);
    }
    
    println!("\nDemo completed successfully!");
    println!("The RuID pipeline now supports dynamic dimensions for:");
    println!("  • Camera capture: Camera::new(width, height, fps)");
    println!("  • Preprocessing: Preprocessor::new(dst_width, dst_height)");
    println!("  • Detection: OrtYolo::new(model_path, input_width, input_height)");
    
    Ok(())
}