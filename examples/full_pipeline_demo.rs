//! Full RuID Pipeline Demo
//!
//! This example demonstrates the complete RuID pipeline:
//! 1. Camera capture at 640x480 using ruid-camera
//! 2. Preprocessing to 224x224 using ruid-preprocess  
//! 3. Model export with custom dimensions using ruid-model
//! 4. Object detection using ruid-detect
//! 5. Visual display of results with OpenCV
//!
//! Usage: cargo run --example full_pipeline_demo

use anyhow::{Context, Result};
use ndarray::Array3;
use opencv::{
    core::{Scalar, Rect, Point},
    highgui, imgproc, prelude::*, videoio,
};
use ruid_camera::{Camera, frame_stream};
use ruid_detect::{Detector, OrtYolo};
use ruid_model::{export, ModelConfig};
use ruid_preprocess::Preprocessor;
use std::{
    collections::VecDeque,
    path::PathBuf,
    time::Instant,
};
use tokio_stream::StreamExt;

// Pipeline configuration constants
const CAMERA_WIDTH: u32 = 640;
const CAMERA_HEIGHT: u32 = 480;
const TENSOR_WIDTH: u32 = 224;
const TENSOR_HEIGHT: u32 = 224;
const CAMERA_FPS: u32 = 30;
const FPS_WINDOW_SIZE: usize = 30;

/// FPS calculation helper
fn calculate_fps(window: &VecDeque<Instant>) -> f64 {
    if window.len() < 2 { 
        return 0.0;
    }
    let duration = window.back().unwrap().duration_since(*window.front().unwrap());
    (window.len() - 1) as f64 / duration.as_secs_f64()
}

/// Setup and export YOLO model with the specified tensor dimensions
async fn setup_model() -> Result<PathBuf> {
    println!("🔧 Setting up YOLO11 model for {}×{} input...", TENSOR_WIDTH, TENSOR_HEIGHT);
    
    let config = ModelConfig {
        output_dir: PathBuf::from("./models/pipeline_demo"),
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

/// Initialize camera capture pipeline
async fn setup_camera() -> Result<impl tokio_stream::Stream<Item = Result<ruid_camera::VideoFrame, ruid_camera::CameraError>>> {
    println!("📷 Initializing camera at {}x{}@{}fps...", CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS);
    
    let camera = Camera::new(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
        .context("Failed to initialize camera")?;
    
    println!("✅ Camera initialized successfully");
    Ok(frame_stream(camera))
}

/// Setup OpenCV fallback camera (for systems without GStreamer/libcamera)
fn setup_opencv_camera() -> Result<videoio::VideoCapture> {
    println!("🔄 Falling back to OpenCV camera...");
    
    let mut cap = videoio::VideoCapture::new(0, videoio::CAP_ANY)
        .context("Failed to open OpenCV camera")?;
    
    cap.set(videoio::CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH as f64)?;
    cap.set(videoio::CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT as f64)?;
    cap.set(videoio::CAP_PROP_FPS, CAMERA_FPS as f64)?;
    
    println!("✅ OpenCV camera initialized");
    Ok(cap)
}

/// Convert OpenCV Mat to Array3 for preprocessing
fn mat_to_array3(mat: &Mat) -> Result<Array3<f32>> {
    let data = mat.data_bytes()?;
    let (height, width) = (mat.rows() as usize, mat.cols() as usize);
    
    let mut array = Array3::<f32>::zeros((height, width, 3));
    for (idx, &byte) in data.iter().enumerate() {
        array.as_slice_mut().unwrap()[idx] = byte as f32 / 255.0;
    }
    
    Ok(array)
}

/// Convert VideoFrame (NV12) to OpenCV Mat (BGR) with proper color conversion
fn videoframe_to_mat(frame: &ruid_camera::VideoFrame) -> Result<Mat> {
    match &frame.backing {
        ruid_camera::FrameBacking::Cpu(bytes) => {
            // NV12 format: Y plane followed by interleaved UV plane
            let y_size = (frame.width * frame.height) as usize;
            let uv_size = y_size / 2;
            
            // Create BGR Mat directly
            let mut bgr_mat = Mat::new_rows_cols_with_default(
                frame.height as i32,
                frame.width as i32,
                opencv::core::CV_8UC3,
                opencv::core::Scalar::all(0.0),
            )?;
            
            // Manual NV12 to BGR conversion (similar to preprocessor but BGR output)
            let y_plane = &bytes[..y_size];
            let uv_plane = &bytes[y_size..y_size + uv_size];
            let bgr_data = bgr_mat.data_bytes_mut()?;
            
            for j in 0..frame.height as usize {
                for i in 0..frame.width as usize {
                    let y_val = y_plane[j * frame.width as usize + i] as f32;
                    let uv_idx = (j / 2) * frame.width as usize + (i & !1);
                    let u = uv_plane[uv_idx] as f32 - 128.0;
                    let v = uv_plane[uv_idx + 1] as f32 - 128.0;

                    let r = (y_val + 1.402 * v).clamp(0.0, 255.0) as u8;
                    let g = (y_val - 0.344_13 * u - 0.714_14 * v).clamp(0.0, 255.0) as u8;
                    let b = (y_val + 1.772 * u).clamp(0.0, 255.0) as u8;

                    let base = (j * frame.width as usize + i) * 3;
                    bgr_data[base] = b;     // BGR order for OpenCV
                    bgr_data[base + 1] = g;
                    bgr_data[base + 2] = r;
                }
            }
            
            Ok(bgr_mat)
        }
        ruid_camera::FrameBacking::DmaBuf(_fd) => {
            // For DMA-Buf, we'd need more complex GPU-based conversion
            eprintln!("⚠️  DMA-Buf frames not yet supported for display, using placeholder");
            let placeholder = Mat::new_rows_cols_with_default(
                frame.height as i32,
                frame.width as i32,
                opencv::core::CV_8UC3,
                opencv::core::Scalar::all(0.0),
            )?;
            Ok(placeholder)
        }
    }
}

/// Draw detection results on the frame
fn draw_detections(frame: &mut Mat, detections: &[ruid_detect::Detection]) -> Result<()> {
    let (frame_width, frame_height) = (frame.cols() as f32, frame.rows() as f32);
    
    for detection in detections {
        let [x1, y1, x2, y2] = detection.bbox;
        
        // Convert normalized coordinates to pixel coordinates
        let pixel_x1 = (x1 * frame_width) as i32;
        let pixel_y1 = (y1 * frame_height) as i32;
        let pixel_x2 = (x2 * frame_width) as i32;
        let pixel_y2 = (y2 * frame_height) as i32;
        
        // Draw bounding box
        let rect = Rect::new(
            pixel_x1,
            pixel_y1,
            pixel_x2 - pixel_x1,
            pixel_y2 - pixel_y1,
        );
        
        imgproc::rectangle(
            frame,
            rect,
            Scalar::new(0.0, 255.0, 0.0, 0.0), // Green color
            2,
            imgproc::LINE_8,
            0,
        )?;
        
        // Draw confidence score
        let label = format!("Person: {:.2}", detection.score);
        imgproc::put_text(
            frame,
            &label,
            Point::new(pixel_x1, pixel_y1 - 10),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            false,
        )?;
    }
    
    Ok(())
}

/// Main pipeline execution
#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 RuID Full Pipeline Demo");
    println!("==========================");
    println!("Camera: {}×{} → Tensor: {}×{}", 
             CAMERA_WIDTH, CAMERA_HEIGHT, TENSOR_WIDTH, TENSOR_HEIGHT);
    println!();

    // Step 1: Setup model
    let model_path = setup_model().await?;
    
    // Step 2: Initialize detector
    println!("🧠 Initializing YOLO detector...");
    let mut detector = OrtYolo::new(&model_path, TENSOR_WIDTH, TENSOR_HEIGHT)
        .context("Failed to initialize YOLO detector")?;
    println!("✅ Detector initialized");
    
    // Step 3: Initialize preprocessor
    println!("⚙️  Initializing preprocessor...");
    let preprocessor = Preprocessor::new(TENSOR_WIDTH, TENSOR_HEIGHT);
    println!("✅ Preprocessor initialized");
    
    // Step 4: Setup display window
    println!("🖥️  Setting up display window...");
    highgui::named_window("RuID Full Pipeline Demo", highgui::WINDOW_AUTOSIZE)?;
    println!("✅ Display window ready");
    
    // Step 5: Initialize camera (try GStreamer first, fallback to OpenCV)
    let use_gstreamer = match setup_camera().await {
        Ok(mut camera_stream) => {
            println!("✅ Using GStreamer camera pipeline");
            
            // FPS tracking
            let mut fps_times: VecDeque<Instant> = VecDeque::with_capacity(FPS_WINDOW_SIZE);
            let mut frame_count = 0u64;
            
            println!("\n🎬 Starting pipeline... Press ESC to quit");
            
            // Main processing loop with GStreamer
            while let Some(frame_result) = camera_stream.next().await {
                let frame = frame_result.context("Failed to get camera frame")?;
                
                // Preprocess frame to tensor
                let tensor = preprocessor.run(&frame)
                    .context("Failed to preprocess frame")?;
                
                // Run detection
                let detections = detector.detect(&tensor)
                    .context("Failed to run detection")?;
                
                // Convert camera frame to OpenCV Mat for display
                let mut display_frame = videoframe_to_mat(&frame)
                    .context("Failed to convert VideoFrame to Mat")?;
                
                // Draw detection results on the frame
                draw_detections(&mut display_frame, &detections)?;
                
                // Update FPS counter
                fps_times.push_back(Instant::now());
                if fps_times.len() > FPS_WINDOW_SIZE {
                    fps_times.pop_front();
                }
                
                // Draw FPS and frame info
                let fps_text = format!("FPS: {:.1}", calculate_fps(&fps_times));
                let info_text = format!("Frame: {} | Detections: {}", frame_count, detections.len());
                
                imgproc::put_text(
                    &mut display_frame,
                    &fps_text,
                    Point::new(10, 30),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    Scalar::new(255.0, 255.0, 255.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    false,
                )?;
                
                imgproc::put_text(
                    &mut display_frame,
                    &info_text,
                    Point::new(10, 60),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    Scalar::new(255.0, 255.0, 255.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    false,
                )?;
                
                // Display frame
                highgui::imshow("RuID Full Pipeline Demo", &display_frame)?;
                
                // Check for exit key
                if highgui::wait_key(1)? == 27 { // ESC key
                    break;
                }
                
                frame_count += 1;
                
                // Print detection info every 30 frames
                if frame_count % 30 == 0 {
                    println!("Frame {}: {} detections, {:.1} FPS", 
                             frame_count, detections.len(), calculate_fps(&fps_times));
                }
            }
            
            true
        }
        Err(e) => {
            eprintln!("⚠️  GStreamer camera failed: {}", e);
            false
        }
    };
    
    // Fallback to OpenCV camera if GStreamer failed
    if !use_gstreamer {
        let mut opencv_cap = setup_opencv_camera()?;
        let mut fps_times: VecDeque<Instant> = VecDeque::with_capacity(FPS_WINDOW_SIZE);
        let mut frame_count = 0u64;
        
        println!("\n🎬 Starting OpenCV pipeline... Press ESC to quit");
        
        loop {
            let mut frame_bgr = Mat::default();
            opencv_cap.read(&mut frame_bgr)?;
            
            if frame_bgr.empty() {
                continue;
            }
            
            // Convert BGR to RGB
            let mut frame_rgb = Mat::default();
            imgproc::cvt_color(&frame_bgr, &mut frame_rgb, imgproc::COLOR_BGR2RGB, 0)?;
            
            // Resize to camera dimensions if needed
            let mut frame_resized = Mat::default();
            imgproc::resize(
                &frame_rgb,
                &mut frame_resized,
                opencv::core::Size::new(CAMERA_WIDTH as i32, CAMERA_HEIGHT as i32),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;
            
            // For OpenCV fallback, we need to manually resize and normalize
            // since the preprocessor expects VideoFrame from the camera
            let mut tensor_resized = Mat::default();
            imgproc::resize(
                &frame_resized,
                &mut tensor_resized,
                opencv::core::Size::new(TENSOR_WIDTH as i32, TENSOR_HEIGHT as i32),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;
            
            // Convert to Array3 for detection
            let tensor = mat_to_array3(&tensor_resized)?;
            
            // Run detection
            let detections = detector.detect(&tensor)
                .context("Failed to run detection")?;
            
            // Draw detection results on original BGR frame
            draw_detections(&mut frame_bgr, &detections)?;
            
            // Update FPS counter
            fps_times.push_back(Instant::now());
            if fps_times.len() > FPS_WINDOW_SIZE {
                fps_times.pop_front();
            }
            
            // Draw FPS and frame info
            let fps_text = format!("FPS: {:.1}", calculate_fps(&fps_times));
            let info_text = format!("Frame: {} | Detections: {}", frame_count, detections.len());
            
            imgproc::put_text(
                &mut frame_bgr,
                &fps_text,
                Point::new(10, 30),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.8,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                false,
            )?;
            
            imgproc::put_text(
                &mut frame_bgr,
                &info_text,
                Point::new(10, 60),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                false,
            )?;
            
            // Display frame
            highgui::imshow("RuID Full Pipeline Demo", &frame_bgr)?;
            
            // Check for exit key
            if highgui::wait_key(1)? == 27 { // ESC key
                break;
            }
            
            frame_count += 1;
            
            // Print detection info every 30 frames
            if frame_count % 30 == 0 {
                println!("Frame {}: {} detections, {:.1} FPS", 
                         frame_count, detections.len(), calculate_fps(&fps_times));
            }
        }
    }
    
    println!("\n🎉 Pipeline demo completed!");
    println!("Thank you for trying the RuID full pipeline demo!");
    
    Ok(())
}