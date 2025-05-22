// examples/pipeline.rs
//------------------------------------------------------------
// Full camera pipeline:  OpenCV → preprocess → OrtYolo → draw
//------------------------------------------------------------
use anyhow::{Context, Result};
use ndarray::Array3;
use opencv::{
    core::{Scalar, Rect},
    highgui, imgproc, prelude::*, videoio,
};
use ruid_detect::{Detector, OrtYolo};
use std::{
    collections::VecDeque,
    time::Instant,
};

// ───── constants ────────────────────────────────────────────
const IN_W: usize = 640;
const IN_H: usize = 480;

// ─── tiny helpers ───────────────────────────────────────────
fn now() -> Instant { Instant::now() }

fn fps(window: &VecDeque<Instant>) -> f64 {
    if window.len() < 2 { return 0.0 }
    let dt = window.back().unwrap().duration_since(*window.front().unwrap());
    (window.len() - 1) as f64 / dt.as_secs_f64()
}

// ─── main ───────────────────────────────────────────────────
fn main() -> Result<()> {
    //---------------- Camera config -------------------------
    let cam_index = 0;
    let mut cap = videoio::VideoCapture::new(cam_index, videoio::CAP_ANY)
        .context("OpenCV can't open camera")?;
    cap.set(videoio::CAP_PROP_FRAME_WIDTH,  IN_W as f64)?;
    cap.set(videoio::CAP_PROP_FRAME_HEIGHT, IN_H as f64)?;
    cap.set(videoio::CAP_PROP_FPS, 30.0)?;

    //---------------- ORT detector --------------------------
    let model = "models/yolov11.onnx";
    let mut detector = OrtYolo::new(model)?;

    //---------------- Rolling FPS ---------------------------
    const WINDOW: usize = 30;
    let mut times: VecDeque<Instant> = VecDeque::with_capacity(WINDOW);

    //---------------- preview window ------------------------
    highgui::named_window("RuID pipeline", highgui::WINDOW_AUTOSIZE)?;

    //---------------- main loop -----------------------------
    loop {
        let mut frame_bgr = Mat::default();
        cap.read(&mut frame_bgr)?;
        if frame_bgr.empty() { continue }

        // --- preprocess --------------------------------------------------
        // BGR → RGB, resize+pad to 640×640, normalise 0-1  (quick & dirty)
        let mut rgb = Mat::default();
        imgproc::cvt_color(&frame_bgr, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;
        let mut rgb640 = Mat::default();
        imgproc::resize(&rgb, &mut rgb640,
                        opencv::core::Size::new(640, 480),
                        0.0, 0.0, imgproc::INTER_LINEAR)?;

        let data = rgb640.data_bytes()?;
        let mut arr = Array3::<f32>::zeros((IN_H, IN_W, 3));
        for (idx, &b) in data.iter().enumerate() {
            arr.as_slice_mut().unwrap()[idx] = b as f32 / 255.0;
        }

        // --- inference ---------------------------------------------------
        let dets = detector.detect(&arr)?;

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
        let txt = format!("FPS: {:.1}", fps(&times));
        imgproc::put_text(&mut frame_bgr, &txt,
                          opencv::core::Point::new(10,30),
                          imgproc::FONT_HERSHEY_SIMPLEX, 0.8,
                          Scalar::new(0.0,255.0,0.0,0.0),
                          2, imgproc::LINE_8, false)?;

        highgui::imshow("RuID pipeline", &frame_bgr)?;
        if highgui::wait_key(1)? == 27 { break }          // ESC to quit
    }
    Ok(())
}
