use anyhow::{Context, Result};
use opencv::{
    core::Scalar,
    highgui, imgproc, prelude::*,
};
use ruid_camera::{Camera, frame_stream, FrameBacking};
use ruid_preprocess::Preprocessor;
use tokio_stream::StreamExt;
use std::{
    collections::VecDeque,
    time::Instant,
};

const FPS_WINDOW_SIZE: usize = 30;

/// Convert VideoFrame (NV12) to OpenCV Mat (BGR) for display
fn videoframe_to_mat(frame: &ruid_camera::VideoFrame) -> Result<Mat> {
    match &frame.backing {
        FrameBacking::Cpu(bytes) => {
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
            
            // Manual NV12 to BGR conversion
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
        FrameBacking::DmaBuf(_fd) => {
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

/// Calculate FPS from timing window
fn calculate_fps(window: &VecDeque<Instant>) -> f64 {
    if window.len() < 2 {
        return 0.0;
    }
    let duration = window.back().unwrap().duration_since(*window.front().unwrap());
    (window.len() - 1) as f64 / duration.as_secs_f64()
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments for camera and preprocessing dimensions
    let args: Vec<String> = std::env::args().collect();
    let (cam_width, cam_height, preprocess_width, preprocess_height) = if args.len() >= 5 {
        let cw = args[1].parse::<u32>().unwrap_or(640);
        let ch = args[2].parse::<u32>().unwrap_or(480);
        let pw = args[3].parse::<u32>().unwrap_or(224);
        let ph = args[4].parse::<u32>().unwrap_or(224);
        (cw, ch, pw, ph)
    } else {
        (640, 480, 224, 224)
    };
    
    println!("🚀 RuID Camera Preview");
    println!("======================");
    println!("Camera: {}x{}, Preprocessing: {}x{}", cam_width, cam_height, preprocess_width, preprocess_height);
    println!("Press ESC to quit");
    println!();

    // Initialize camera
    let cam = Camera::new(cam_width, cam_height, 30)?;
    let mut s = frame_stream(cam);

    // Setup display window
    highgui::named_window("RuID Camera Preview", highgui::WINDOW_AUTOSIZE)?;

    // FPS tracking
    let mut fps_times: VecDeque<Instant> = VecDeque::with_capacity(FPS_WINDOW_SIZE);
    let mut count = 0u64;

    // Move Preprocessor creation _out_ of the loop
    let pp = Preprocessor::new(preprocess_width, preprocess_height);

    println!("🎬 Starting camera preview...");

    // Single loop: pull → preprocess → display → count → repeat
    while let Some(frame_res) = s.next().await {
        let frame = frame_res?;

        // preprocess into a tensor
        let _tensor = pp.run(&frame)?;
        // TODO: inference on `tensor`

        // Convert camera frame to OpenCV Mat for display
        let mut display_frame = videoframe_to_mat(&frame)
            .context("Failed to convert VideoFrame to Mat")?;
        
        // Update FPS counter
        fps_times.push_back(Instant::now());
        if fps_times.len() > FPS_WINDOW_SIZE {
            fps_times.pop_front();
        }
        
        // Draw FPS info
        let fps_text = format!("FPS: {:.1}", calculate_fps(&fps_times));
        let info_text = format!("Frame: {} | Preprocessing: {}x{}", count, preprocess_width, preprocess_height);
        
        imgproc::put_text(
            &mut display_frame,
            &fps_text,
            opencv::core::Point::new(10, 30),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.8,
            Scalar::new(0.0, 255.0, 0.0, 0.0), // Green text
            2,
            imgproc::LINE_8,
            false,
        )?;
        
        imgproc::put_text(
            &mut display_frame,
            &info_text,
            opencv::core::Point::new(10, 60),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            Scalar::new(0.0, 255.0, 0.0, 0.0), // Green text
            2,
            imgproc::LINE_8,
            false,
        )?;
        
        // Display frame
        highgui::imshow("RuID Camera Preview", &display_frame)?;
        
        // Check for exit key
        if highgui::wait_key(1)? == 27 { // ESC key
            break;
        }

        // FPS counting
        count += 1;
        if count % 150 == 0 {
            let fps = calculate_fps(&fps_times);
            eprintln!("avg {:.1} FPS (last 150 frames)", fps);
        }
    }

    println!("\n🎉 Camera preview completed!");
    Ok(())
}