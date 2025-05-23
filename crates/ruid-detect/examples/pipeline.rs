// examples/pipeline.rs
//------------------------------------------------------------
// Full camera pipeline:  OpenCV → ruid-preprocess → OrtYolo → draw
// Now uses ruid-model for 224x224 tensor size and ruid-preprocess
//------------------------------------------------------------
use anyhow::{Context, Result};
use opencv::{
    core::{Scalar, Rect},
    highgui, imgproc, prelude::*, videoio,
};
use ruid_camera::{VideoFrame, FrameBacking};
use ruid_detect::{Detector, OrtYolo};
use ruid_model::{export, ModelConfig};
use ruid_preprocess::Preprocessor;
use std::{
    collections::VecDeque,
    path::PathBuf,
    time::{Duration, Instant},
};

// ───── pipeline constants ────────────────────────────────────────────
const CAMERA_WIDTH: u32 = 640;
const CAMERA_HEIGHT: u32 = 480;
const TENSOR_WIDTH: u32 = 224;
const TENSOR_HEIGHT: u32 = 224;

// ─── tiny helpers ───────────────────────────────────────────
fn now() -> Instant { Instant::now() }

fn fps(window: &VecDeque<Instant>) -> f64 {
    if window.len() < 2 { return 0.0 }
    let dt = window
        .back()
        .unwrap()
        .duration_since(*window.front().unwrap());
    (window.len() - 1) as f64 / dt.as_secs_f64()
}

/// Setup and export YOLO model with 224x224 tensor dimensions
async fn setup_model() -> Result<PathBuf> {
    println!("🔧 Setting up YOLO11 model for {}×{} input...", TENSOR_WIDTH, TENSOR_HEIGHT);
    
    let config = ModelConfig {
        output_dir: PathBuf::from("./models/detect_pipeline"),
        image_size: (TENSOR_HEIGHT, TENSOR_WIDTH), // Note: height, width order
        quantize: false, // Use FP32 for better compatibility
        batch_size: 1,
    };

    // Check if model already exists
    let model_path = config.output_dir.join("yolo11n_fp32.onnx");
    if model_path.exists() {
        println!("✅ Found existing model at: {:?}", model_path);
        return Ok(model_path);
    }

    println!("📦 Exporting YOLO11 model...");
    match export::export_yolo11_custom(
        config.output_dir.clone(),
        config.image_size,
        config.quantize,
    ).await {
        Ok(result) => {
            let model_path = result.fp32_model.unwrap_or(result.int8_model);
            println!("✅ Model exported successfully to: {:?}", model_path);
            println!("   Input shape: {:?}", result.input_shape);
            println!("   Input name: {}", result.input_name);
            Ok(model_path)
        }
        Err(e) => {
            eprintln!("❌ Model export failed: {}", e);
            eprintln!("💡 Make sure ultralytics is installed: pip install ultralytics");
            
            // Fallback: try to use existing model in models/base directory
            let fallback_path = PathBuf::from("models/base/yolo11n.onnx");
            if fallback_path.exists() {
                println!("🔄 Using fallback model: {:?}", fallback_path);
                Ok(fallback_path)
            } else {
                Err(e)
            }
        }
    }
}

/// Convert OpenCV Mat to VideoFrame for preprocessing
fn mat_to_videoframe(mat: &Mat) -> Result<VideoFrame> {
    // Convert BGR to RGB
    let mut rgb_mat = Mat::default();
    imgproc::cvt_color(mat, &mut rgb_mat, imgproc::COLOR_BGR2RGB, 0)?;
    
    // Resize to camera dimensions
    let mut resized_mat = Mat::default();
    imgproc::resize(
        &rgb_mat,
        &mut resized_mat,
        opencv::core::Size::new(CAMERA_WIDTH as i32, CAMERA_HEIGHT as i32),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    
    // Convert to NV12 format (simplified - just using RGB data)
    let rgb_data = resized_mat.data_bytes()?;
    let frame_size = (CAMERA_WIDTH * CAMERA_HEIGHT * 3 / 2) as usize; // NV12 format
    let mut nv12_data = vec![128u8; frame_size];
    
    // Simple RGB to Y conversion (just use green channel for simplicity)
    let y_size = (CAMERA_WIDTH * CAMERA_HEIGHT) as usize;
    for i in 0..y_size {
        let rgb_idx = i * 3;
        if rgb_idx + 1 < rgb_data.len() {
            // Simple luminance conversion: Y = 0.299*R + 0.587*G + 0.114*B
            let r = rgb_data[rgb_idx] as f32;
            let g = rgb_data[rgb_idx + 1] as f32;
            let b = rgb_data[rgb_idx + 2] as f32;
            nv12_data[i] = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
        }
    }
    
    Ok(VideoFrame {
        backing: FrameBacking::Cpu(nv12_data),
        width: CAMERA_WIDTH,
        height: CAMERA_HEIGHT,
        stride: CAMERA_WIDTH * 2,
        pts: Duration::ZERO,
    })
}

// ─── main ───────────────────────────────────────────────────
fn main() -> Result<()> {
    println!("🚀 RuID Detection Pipeline Demo");
    println!("===============================");
    println!("Camera: {}×{} → Tensor: {}×{}",
             CAMERA_WIDTH, CAMERA_HEIGHT, TENSOR_WIDTH, TENSOR_HEIGHT);
    println!("Press ESC to quit");
    println!();

    // Step 1: Setup model with 224x224 tensor size
    let rt = tokio::runtime::Runtime::new()?;
    let model_path = rt.block_on(setup_model())?;
    
    // Step 2: Initialize detector with 224x224 tensor dimensions
    println!("🧠 Initializing YOLO detector...");
    let mut detector = OrtYolo::new(&model_path, TENSOR_WIDTH, TENSOR_HEIGHT)?;
    println!("✅ Detector initialized");
    
    // Step 3: Initialize preprocessor for 224x224 output
    println!("⚙️  Initializing preprocessor...");
    let preprocessor = Preprocessor::new(TENSOR_WIDTH, TENSOR_HEIGHT);
    println!("✅ Preprocessor initialized");

    //---------------- Camera config -------------------------
    let cam_index = 0;
    let mut cap = videoio::VideoCapture::new(cam_index, videoio::CAP_ANY)
        .context("OpenCV can't open camera")?;
    cap.set(videoio::CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH as f64)?;
    cap.set(videoio::CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT as f64)?;
    cap.set(videoio::CAP_PROP_FPS, 30.0)?;

    //---------------- Rolling FPS ---------------------------
    const WINDOW: usize = 30;
    let mut times: VecDeque<Instant> = VecDeque::with_capacity(WINDOW);

    //---------------- preview window ------------------------
    highgui::named_window("RuID Detection Pipeline", highgui::WINDOW_AUTOSIZE)?;

    println!("\n🎬 Starting detection pipeline...");

    //---------------- main loop -----------------------------
    loop {
        let mut frame_bgr = Mat::default();
        cap.read(&mut frame_bgr)?;
        if frame_bgr.empty() { continue }

        // --- convert to VideoFrame for preprocessing ----------------
        let video_frame = mat_to_videoframe(&frame_bgr)?;
        
        // --- preprocess using ruid-preprocess (224x224) -------------
        let tensor = preprocessor.run(&video_frame)
            .context("Failed to preprocess frame")?;

        // --- inference -----------------------------------------------
        let dets = detector.detect(&tensor)?;

        // dump detections for debug
        for d in &dets {
            eprintln!("box={:?} score={:.3}", d.bbox, d.score);
        }

        // --- draw --------------------------------------------------------
        for d in dets {
            let [x1,y1,x2,y2] = d.bbox;
            let (w,h) = (frame_bgr.cols(), frame_bgr.rows());
            let rect = Rect::new(
                (x1*w as f32) as i32, (y1*h as f32) as i32,
                ((x2-x1)*w as f32) as i32, ((y2-y1)*h as f32) as i32);
            imgproc::rectangle(&mut frame_bgr, rect,
                               Scalar::new(0.0,255.0,0.0,0.0),
                               2, imgproc::LINE_8, 0)?;
        }

        // --- FPS ---------------------------------------------------------
        times.push_back(now());
        if times.len() > WINDOW { times.pop_front(); }
        let fps_text = format!("FPS: {:.1}", fps(&times));
        let info_text = format!("Tensor: {}x{}", TENSOR_WIDTH, TENSOR_HEIGHT);
        
        imgproc::put_text(&mut frame_bgr, &fps_text,
                          opencv::core::Point::new(10,30),
                          imgproc::FONT_HERSHEY_SIMPLEX, 0.8,
                          Scalar::new(0.0,255.0,0.0,0.0),
                          2, imgproc::LINE_8, false)?;
                          
        imgproc::put_text(&mut frame_bgr, &info_text,
                          opencv::core::Point::new(10,60),
                          imgproc::FONT_HERSHEY_SIMPLEX, 0.6,
                          Scalar::new(0.0,255.0,0.0,0.0),
                          2, imgproc::LINE_8, false)?;

        highgui::imshow("RuID Detection Pipeline", &frame_bgr)?;
        if highgui::wait_key(1)? == 27 { break }          // ESC to quit
    }
    
    println!("\n🎉 Detection pipeline completed!");
    Ok(())
}
