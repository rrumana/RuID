// examples/visualize.rs
// ------------------------------------------------------------
// Visual smoke-test: run YOLO, draw bounding boxes, show window.
// cargo run -p ruid-detect --example visualize -- <model> <image>
// ------------------------------------------------------------
use anyhow::{Context, Result};
use opencv::{
    core::{Scalar, Rect},
    highgui, imgcodecs, imgproc,
    prelude::*,
};
use ndarray::Array3;
use ruid_detect::{Detector, TractYolo};

const TARGET: f32 = 224.0;

fn main() -> Result<()> {
    // ---------------------------------------------------------------------
    // CLI args
    // ---------------------------------------------------------------------
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: visualize <model.onnx> <image.jpg>");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let image_path = &args[2];

    // ---------------------------------------------------------------------
    // Load image with OpenCV (BGR u8)
    // ---------------------------------------------------------------------
    let mut mat = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)
        .with_context(|| format!("reading {}", image_path))?;
    let (orig_h, orig_w) = (mat.rows() as f32, mat.cols() as f32);

    // ---------------------------------------------------------------------
    // Pre-process: BGR ➜ RGB 
    // RGB u8; just convert to f32, resize 224×224, normalise 0-1
    // ---------------------------------------------------------------------
    let mut rgb = Mat::default();
    imgproc::cvt_color(&mat, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;

    let scale   = (TARGET / orig_h).min(TARGET / orig_w);       // keep aspect
    let new_w   = (orig_w * scale).round() as i32;
    let new_h   = (orig_h * scale).round() as i32;

    // 1. resize keeping ratio
    let mut resized = Mat::default();
    imgproc::resize(
        &rgb, &mut resized,
        opencv::core::Size::new(new_w, new_h),
        0.0, 0.0, imgproc::INTER_LINEAR)?;

    let pad_x = ((TARGET as i32) - new_w) / 2;
    let pad_y = ((TARGET as i32) - new_h) / 2;

    // 2. pad to 224×224 with 114
    let mut letter = Mat::new_rows_cols_with_default(
        224, 224, opencv::core::CV_8UC3,
        Scalar::all(114.0))?;                           // 114 = UL padding

    {
        let mut roi = letter.roi_mut(Rect::new(pad_x, pad_y, new_w, new_h))?;
        resized.copy_to(&mut roi)?;
    }

    // 3. convert to f32 ndarray 0-1
    let total = (224 * 224 * 3) as usize;
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(letter.data(), total)
    };
    let mut arr = Array3::<f32>::zeros((224,224,3));
    for (idx, px) in bytes.iter().enumerate() {
        arr.as_slice_mut().unwrap()[idx] = *px as f32 / 255.0;
    }

    // ------------------------------------------------------------
    // Run detector
    // ------------------------------------------------------------
    let detector = TractYolo::new(model_path)?;
    let dets     = detector.detect(&arr)?;

    for det in &dets {
        eprintln!("det box={:?} score={:.3}", det.bbox, det.score);
    }
    eprintln!("visualize: network returned {} detections", dets.len());

    // ---------------------------------------------------------------------
    // Draw detections back on original-size image
    // ---------------------------------------------------------------------
    for det in dets {
        let [x1, y1, x2, y2] = det.bbox;
        let sx1 = ((x1 * TARGET - pad_x as f32) / scale) as i32;
        let sy1 = ((y1 * TARGET - pad_y as f32) / scale) as i32;
        let sx2 = ((x2 * TARGET - pad_x as f32) / scale) as i32;
        let sy2 = ((y2 * TARGET - pad_y as f32) / scale) as i32;
        let rect = Rect::new(sx1, sy1, (sx2 - sx1) as i32, (sy2 - sy1) as i32);
        imgproc::rectangle(
            &mut mat,
            rect,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;
        // putText: class id + score
        imgproc::put_text(
            &mut mat,
            &format!("c{} {:.2}", det.class, det.score),
            opencv::core::Point::new(sx1, sy1 - 5),
            highgui::QT_FONT_NORMAL,
            0.5,
            Scalar::new(255.0, 255.0, 0.0, 0.0),
            1,
            imgproc::LINE_8,
            false,
        )?;
    }

    // ---------------------------------------------------------------------
    // Show window
    // ---------------------------------------------------------------------
    highgui::imshow("YOLO visualize", &mat)?;
    highgui::wait_key(0)?;
    Ok(())
}
