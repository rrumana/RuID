use anyhow::{anyhow, Result};
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender};
use num_cpus;
use opencv::{
    core::{self, Mat, Rect, Scalar, Vec3b},
    highgui,
    imgproc,
    prelude::*,
    videoio::{VideoCapture, CAP_V4L2},
};
use std::{
    borrow::Borrow,
    cell::RefCell,
    cmp::Ordering,
    collections::VecDeque,
    fmt::{Debug, Display},
    process,
    thread,
    time::Instant,
    collections::HashMap,
};
use tract_ndarray::{Array4, s};
use tract_onnx::prelude::*;

///////////////////////////////////////

#[derive(Debug, Copy, Clone)]
pub struct Bbox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
}

impl Bbox {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, confidence: f32) -> Self {
        Self { x1, y1, x2, y2, confidence }
    }
}

#[derive(Debug, Clone)]
struct TaggedFrame {
    seq_id: usize,
    mat: Mat,
}

#[derive(Parser)]
struct CliArgs {
    /// Path to ONNX or .tract model weights
    #[arg(long)]
    weights: String,

    /// How many frames to process
    #[arg(long, default_value = "10000")]
    num_frames: usize,

    /// "tract" or "onnx"
    #[arg(long, default_value = "tract")]
    engine: String,
}

thread_local! {
    static MODEL_SCRATCH: RefCell<Array4<f32>> = RefCell::new(Array4::zeros((1, 3, 480, 640)));
}

fn run_model_tract<F, O, M>(
    model: &RunnableModel<F, O, M>,
    mat: &Mat
) -> Result<Vec<Bbox>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    M: Borrow<Graph<F, O>>
{
    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;

    MODEL_SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        for y in 0..rows {
            for x in 0..cols {
                let px = *mat.at_2d::<Vec3b>(y as i32, x as i32).unwrap();
                scratch[[0, 0, y, x]] = px[2] as f32 / 255.0;
                scratch[[0, 1, y, x]] = px[1] as f32 / 255.0;
                scratch[[0, 2, y, x]] = px[0] as f32 / 255.0;
            }
        }
    });

    let image_tensor: Tensor = MODEL_SCRATCH.with(|scratch_cell| {
        let scratch = scratch_cell.borrow();
        scratch.to_owned().into()
    });

    let forward = model.run(tvec![image_tensor.into()])?;
    let results = forward[0].to_array_view::<f32>()?.view().t().into_owned();

    let mut bboxes = Vec::new();
    for i in 0..results.len_of(tract_ndarray::Axis(0)) {
        let row = results.slice(s![i, .., ..]);
        let confidence = row[[4, 0]];
        if confidence >= 0.5 {
            let x_c = row[[0, 0]];
            let y_c = row[[1, 0]];
            let w_  = row[[2, 0]];
            let h_  = row[[3, 0]];
            let x1 = x_c - w_ / 2.0;
            let y1 = y_c - h_ / 2.0;
            let x2 = x_c + w_ / 2.0;
            let y2 = y_c + h_ / 2.0;
            bboxes.push(Bbox::new(x1, y1, x2, y2, confidence));
        }
    }

    Ok(non_maximum_suppression(bboxes, 0.5))
}

fn draw_bboxes(mat: &mut Mat, bboxes: &[Bbox]) -> opencv::Result<()> {
    for bbox in bboxes {
        let x1 = bbox.x1.max(0.0) as i32;
        let y1 = bbox.y1.max(0.0) as i32;
        let x2 = bbox.x2.max(0.0) as i32;
        let y2 = bbox.y2.max(0.0) as i32;

        let pt1 = core::Point::new(x1, y1);
        let pt2 = core::Point::new(x2, y2);

        imgproc::rectangle(
            mat,
            Rect::new(
                pt1.x,
                pt1.y,
                (pt2.x - pt1.x).max(0),
                (pt2.y - pt1.y).max(0),
            ),
            Scalar::new(0.0, 0.0, 255.0, 1.0),
            1,
            imgproc::LINE_8,
            0,
        )?;

        let label = format!("{:.2}", bbox.confidence);
        imgproc::put_text(
            mat,
            &label,
            pt1,
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::new(255.0, 255.0, 255.0, 1.0), // white
            1,
            imgproc::LINE_8,
            false,
        )?;
    }
    Ok(())
}

fn non_maximum_suppression(mut boxes: Vec<Bbox>, iou_threshold: f32) -> Vec<Bbox> {
    boxes.sort_unstable_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(Ordering::Equal));
    let mut keep = Vec::with_capacity(boxes.len());
    'candidate: for current in boxes {
        for &kept in &keep {
            if calculate_iou(&current, &kept) > iou_threshold {
                continue 'candidate;
            }
        }
        keep.push(current);
    }
    keep
}

fn calculate_iou(a: &Bbox, b: &Bbox) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let intersection = inter_w * inter_h;

    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

    let union = area_a + area_b - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn inference_thread_tract(
    rx_infer: Receiver<TaggedFrame>,
    tx_out: Sender<TaggedFrame>,
    model_path: &str,
    width: usize,
    height: usize,
) -> Result<()> {
    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .with_input_fact(0, f32::fact([1, 3, height, width]).into())?
        .into_optimized()?
        .into_runnable()?;

    while let Ok(tagged) = rx_infer.recv() {
        let seq_id = tagged.seq_id;
        let mut mat = tagged.mat.clone();

        match run_model_tract(&model, &mat) {
            Ok(bboxes) => {
                // Draw bounding boxes
                if let Err(e) = draw_bboxes(&mut mat, &bboxes) {
                    eprintln!("Error drawing bboxes: {:?}", e);
                }
                let output = TaggedFrame { seq_id, mat };
                tx_out.send(output)?;
            },
            Err(e) => eprintln!("Error in Tract inference: {:?}", e),
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    // Parse CLI
    let args = CliArgs::parse();
    let width = 640;
    let height = 480;

    ctrlc::set_handler(move || {
        eprintln!("Received Ctrl+C! Exiting now...");
        process::exit(0);
    })?;

    // Initialize an OpenCV VideoCapture that references libcamera (or another suitable backend)
    let mut cap = VideoCapture::new(0, CAP_V4L2)?; 

    if !cap.is_opened()? {
        return Err(anyhow!("Could not open camera with OpenCV. Check your pipeline/device."));
    }

    // Create OpenCV window
    highgui::named_window("Camera Inference", highgui::WINDOW_AUTOSIZE)?;

    // Bounded channels for inference
    let (tx_infer, rx_infer) = bounded::<TaggedFrame>(4);
    let (tx_annotated, rx_annotated) = bounded::<TaggedFrame>(4);

    // Spawn inference threads
    match args.engine.as_str() {
        "tract" => {
            let concurrency = num_cpus::get();
            for _ in 0..concurrency {
                let rx_infer_clone = rx_infer.clone();
                let tx_annotated_clone = tx_annotated.clone();
                let model_path = args.weights.clone();
                thread::spawn(move || {
                    if let Err(e) = inference_thread_tract(
                        rx_infer_clone,
                        tx_annotated_clone,
                        &model_path,
                        width,
                        height,
                    ) {
                        eprintln!("Tract inference thread error: {:?}", e);
                    }
                });
            }
        },
        _ => {
            eprintln!("Unsupported engine: {}", args.engine);
            return Ok(());
        }
    };

    // So threads can eventually shut down
    drop(tx_annotated);

    let mut frame_count = 0usize;
    let mut annotated_count = 1usize;
    let mut pending_map = HashMap::new();

    // We'll keep a small queue of timestamps for a rolling average:
    const ROLLING_WINDOW_SIZE: usize = 30;
    let mut frame_times: VecDeque<Instant> = VecDeque::with_capacity(ROLLING_WINDOW_SIZE);

    // Main loop
    while frame_count < args.num_frames {
        // Grab a frame
        let mut frame = Mat::default();
        cap.read(&mut frame)?;
        if frame.empty() {
            eprintln!("Empty frame from camera!");
            continue;
        }
        frame_count += 1;

        // Send frame to inference
        let tagged = TaggedFrame {
            seq_id: frame_count,
            mat: frame,
        };
        if tx_infer.send(tagged).is_err() {
            eprintln!("Inference channel closed");
            break;
        }

        // Check for annotated frames
        if let Ok(annotated) = rx_annotated.try_recv() {
            pending_map.insert(annotated.seq_id, annotated.mat);
        }

        // Display frames in correct sequence
        while let Some(mat) = pending_map.remove(&annotated_count) {
            highgui::imshow("Camera Inference", &mat)?;
            highgui::wait_key(1)?;

            // Update rolling timestamp queue
            let now = Instant::now();
            frame_times.push_back(now);
            // If we exceed the capacity, pop the oldest
            if frame_times.len() > ROLLING_WINDOW_SIZE {
                frame_times.pop_front();
            }

            // Compute rolling FPS if we have at least 2 timestamps
            let fps = if frame_times.len() >= 2 {
                let duration = frame_times.back().unwrap().duration_since(*frame_times.front().unwrap()).as_secs_f64();
                // The difference in frames is (len - 1), because the queue has e.g. 5 timestamps for 4 intervals
                (frame_times.len() - 1) as f64 / duration
            } else {
                0.0 // or some placeholder while we fill the buffer
            };

            println!(
                "Displayed frame {} (rolling FPS: {:.2})",
                annotated_count, fps
            );
            annotated_count += 1;
        }
    }

    // Cleanup
    drop(rx_infer);
    drop(tx_infer);

    // Drain any remaining annotated frames
    while let Ok(annotated) = rx_annotated.try_recv() {
        pending_map.insert(annotated.seq_id, annotated.mat);
    }
    while let Some(mat) = pending_map.remove(&annotated_count) {
        highgui::imshow("Camera Inference", &mat)?;
        highgui::wait_key(1)?;

        // Update rolling timestamp queue
        let now = Instant::now();
        frame_times.push_back(now);
        if frame_times.len() > ROLLING_WINDOW_SIZE {
            frame_times.pop_front();
        }

        let fps = if frame_times.len() >= 2 {
            let duration = frame_times.back().unwrap().duration_since(*frame_times.front().unwrap()).as_secs_f64();
            (frame_times.len() - 1) as f64 / duration
        } else {
            0.0
        };

        println!(
            "Displayed frame {} (rolling FPS: {:.2})",
            annotated_count, fps
        );
        annotated_count += 1;
    }

    Ok(())
}
