use anyhow::{Error, Result};
use anyhow::anyhow;
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender};
use image::{DynamicImage, ImageFormat, Rgb, RgbImage, ImageBuffer};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use libcamera::{
    camera::CameraConfigurationStatus,
    camera_manager::CameraManager,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    geometry::Size,
    pixel_format::PixelFormat,
    properties,
    request::ReuseFlag,
    stream::StreamRole,
};
use num_cpus;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::{
    process,
    cell::RefCell,
    sync::mpsc, 
    cmp::Ordering,
    thread,
    time::Duration,
    borrow::Borrow,
    fmt::{Debug, Display},
};
use sdl2::{
    pixels::PixelFormatEnum,
    rect::Rect as SdlRect,
    render::Canvas,
    video::Window,
};
use tract_onnx::prelude::*;
use tract_ndarray::{Array4, s};

const PIXEL_FORMAT_MJPEG: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'M', b'J', b'P', b'G']), 0);

#[derive(Parser)]
struct CliArgs {
    #[arg(long)]
    weights: String,

    #[arg(long, default_value = "10000")]
    num_frames: usize,

    #[arg(long, default_value = "tract")]
    engine: String, 
}

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
    image: RgbImage,
} 

fn run_model_tract<F, O, M>(
    model: &RunnableModel<F, O, M>,
    decoded_image: DynamicImage
) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> 
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    M: Borrow<Graph<F, O>> 
{
    let rgba = decoded_image.to_rgba8();
    MODEL_SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        for (y, row) in rgba.rows().enumerate() {
            for (x, pixel) in row.enumerate() {
                scratch[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
                scratch[[0, 1, y, x]] = pixel[1] as f32 / 255.0;
                scratch[[0, 2, y, x]] = pixel[2] as f32 / 255.0;
            }
        }
    });

    let image_tensor: Tensor = MODEL_SCRATCH.with(|scratch_cell| {
        let scratch = scratch_cell.borrow();
        scratch.to_owned().into()
    });

    let forward = model.run(tvec![image_tensor.into()]).unwrap();
    let results = forward[0].to_array_view::<f32>().unwrap().view().t().into_owned();

    let mut bbox_vec = vec![];
    for i in 0..results.len_of(tract_ndarray::Axis(0)) {
        let row = results.slice(s![i, .., ..]);
        let confidence = row[[4, 0]];
        if confidence >= 0.5 {
            let x_c = row[[0, 0]];
            let y_c = row[[1, 0]];
            let w_   = row[[2, 0]];
            let h_   = row[[3, 0]];
            let x1 = x_c - w_ / 2.0;
            let y1 = y_c - h_ / 2.0;
            let x2 = x_c + w_ / 2.0;
            let y2 = y_c + h_ / 2.0;
            bbox_vec.push(Bbox::new(x1, y1, x2, y2, confidence));
        }
    }

    let final_bboxes = non_maximum_suppression(bbox_vec, 0.5);

    let mut annotated = decoded_image.into_rgb8();
    draw_bboxes_on_image(&mut annotated, &final_bboxes);

    Ok(annotated)
}

pub fn non_maximum_suppression(mut boxes: Vec<Bbox>, iou_threshold: f32) -> Vec<Bbox> {
    boxes.sort_unstable_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(Ordering::Equal)
    });

    let mut keep = Vec::with_capacity(boxes.len());

    'candidate: for current_box in boxes {
        for &kept_box in &keep {
            if calculate_iou(&current_box, &kept_box) > iou_threshold {
                continue 'candidate;
            }
        }
        keep.push(current_box);
    }

    keep
}

#[inline]
fn calculate_iou(box1: &Bbox, box2: &Bbox) -> f32 {
    let x1 = box1.x1.max(box2.x1);
    let y1 = box1.y1.max(box2.y1);
    let x2 = box1.x2.min(box2.x2);
    let y2 = box1.y2.min(box2.y2);

    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    let area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    let union = area1 + area2 - intersection;
    if union <= 0.0 {
        return 0.0;
    }
    intersection / union
}

fn draw_bboxes_on_image(image: &mut RgbImage, bboxes: &[Bbox]) {
    for bbox in bboxes {
        let x = bbox.x1 as i32;
        let y = bbox.y1 as i32;
        let width = (bbox.x2 - bbox.x1).max(0.0) as u32;
        let height = (bbox.y2 - bbox.y1).max(0.0) as u32;

        let rect = Rect::at(x, y).of_size(width, height);
        draw_hollow_rect_mut(image, rect, Rgb([255, 0, 0]));
    }
}

fn init_sdl2(width: u32, height: u32) -> Result<Canvas<Window>> {
    let sdl_context = sdl2::init().map_err(|e| anyhow!("Failed to initialize SDL4: {}", e))?;
    let video_subsystem = sdl_context.video().map_err(|e| anyhow!("Failed to get SDL2 video subsystem: {}", e))?;
    let window = video_subsystem
        .window("Camera Inference", width, height)
        .position_centered()
        .resizable()
        .build()
        .map_err(|e| anyhow!("Failed to build SDL2 window: {}", e))?;

    let canvas = window
        .into_canvas()
        .accelerated()
        .build()
        .map_err(|e| anyhow!("Failed to build SDL2 canvas: {}", e))?;

    Ok(canvas)
}

fn display_frame(
    canvas: &mut Canvas<Window>,
    rgb_image: &RgbImage,
) -> Result<()> {
    let (width, height) = (rgb_image.width(), rgb_image.height());
    let creator = canvas.texture_creator();
    let mut texture = creator
        .create_texture_streaming(PixelFormatEnum::RGB24, width, height)?;

    let _ = texture.with_lock(None, |buffer: &mut [u8], pitch: usize| {
        for (y, row) in rgb_image.rows().enumerate() {
            let start = y * pitch;
            let end = start + (3 * width as usize);
            let dest = &mut buffer[start..end];
            for (x, pixel) in row.enumerate() {
                let i = x * 3;
                dest[i] = pixel[0];
                dest[i + 1] = pixel[1];
                dest[i + 2] = pixel[2];
            }
        }
    });

    canvas.clear();
    let (win_w, win_h) = canvas.window().size();
    let dest_rect = SdlRect::new(0, 0, win_w, win_h);
    let _ = canvas.copy(&texture, None, dest_rect);
    canvas.present();
    Ok(())
}

thread_local! {
    static MODEL_SCRATCH: RefCell<Array4<f32>> = RefCell::new(Array4::zeros((1, 3, 480, 640)));
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
        let rgb = tagged.image;
        let result = run_model_tract(&model, DynamicImage::ImageRgb8(rgb));
        if let Ok(annotated) = result {
            let output = TaggedFrame { seq_id, image: annotated };
            tx_out.send(output)?;
        }
    }

    Ok(())
}

fn inference_thread_onnx(
    rx_infer: Receiver<TaggedFrame>,
    _tx_out: Sender<TaggedFrame>,
    model_path: &str,
    _width: usize,
    _height: usize,
) -> Result<()> {
    let _model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(num_cpus::get())?
        .commit_from_file(model_path)?;

    while let Ok(tagged) = rx_infer.recv() {
        let seq_id = tagged.seq_id;
        let rgb = tagged.image;
        //let result = run_model_onnx(&model, DynamicImage::ImageRgb8(rgb));
        //if let Ok(annotated) = result {
        //    let output = TaggedFrame { seq_id, image: annotated };
        //    tx_out.send(output).unwrap();
        //}
    }

    Ok(())
}



fn main() -> Result<()> {
    let args = CliArgs::parse();
    let width = 640;
    let height = 480;

    ctrlc::set_handler(move || {
        eprintln!("Received Ctrl+C! Exiting now...");
        process::exit(0);
    }).expect("Error setting Ctrl-C handler"); 

    let mgr = CameraManager::new()?;
    let cams = mgr.cameras();
    let cam = match cams.get(0) {
        Some(cam) => cam,
        None => {
            eprintln!("No cameras found");
            return Ok(());
        }
    };

    match cam.properties().get::<properties::Model>() {
        Ok(model) => println!("Using camera: {}", *model),
        Err(e) => {
            eprintln!("Failed to get camera model: {}", e);
            return Ok(());
        }
    }

    let mut cam = match cam.acquire() {
        Ok(cam) => cam,
        Err(e) => {
            eprintln!("Failed to acquire camera: {}", e);
            return Ok(());
        }
    }; 

    let mut cfgs = match cam.generate_configuration(&[StreamRole::VideoRecording]) {
        Some(cfgs) => cfgs,
        None => {
            eprintln!("Failed to generate camera configuration.");
            return Ok(());
        }
    };

    match cfgs.get_mut(0) {
        Some(mut cfg) => {
            cfg.set_pixel_format(PIXEL_FORMAT_MJPEG);
            cfg.set_size(Size { width, height });
        },
        None => {
            eprintln!("Failed to get camera configuration");
            return Ok(());
        }
    }

    //println!("Generated config: {:#?}", cfgs);

    match cfgs.validate() {
        CameraConfigurationStatus::Valid => println!("Camera configuration valid!"),
        CameraConfigurationStatus::Adjusted => println!("Camera configuration was adjusted: {:#?}", cfgs),
        CameraConfigurationStatus::Invalid => panic!("Invalid camera configuration"),
    }

    cam.configure(&mut cfgs).expect("Unable to configure camera");

    let mut alloc = FrameBufferAllocator::new(&cam);
    let cfg = cfgs.get(0).unwrap();
    let stream = cfg.stream().unwrap();
    let buffers = alloc.alloc(&stream).unwrap();
    println!("Allocated {} buffers", buffers.len());

    let buffers = buffers
        .into_iter()
        .map(MemoryMappedFrameBuffer::new)
        .collect::<Result<Vec<_>, _>>()?;

    let reqs: Vec<_> = buffers
        .into_iter()
        .enumerate()
        .map(|(i, buf)| {
            let mut req = cam.create_request(Some(i as u64)).expect("Failed to create request");
            req.add_buffer(&stream, buf).expect("Failed to add buffer to request");
            req
        })
        .collect();

    let (tx_req, rx_req) = mpsc::channel();
    cam.on_request_completed(move |req| {
        tx_req.send(req).expect("Failed to send completed request");
    });

    cam.start(None)?;
    for req in reqs {
        println!("Request queued for execution: {req:#?}");
        cam.queue_request(req).unwrap();
    }
   
    // 9) Initialize SDL2 for display (640x640)
    let mut canvas = init_sdl2(width, height)
        .map_err(|e| Error::msg(format!("Failed to init SDL2: {e}")))?;
    
    // 7) Create inference channels
    let (tx_infer, rx_infer) = bounded::<TaggedFrame>(4);
    let (tx_annotated, rx_annotated) = bounded::<TaggedFrame>(4);

    // 2) Start inference threads 
    // Note, tract is not inheretly multithreaded, hence the different handling
    match args.engine.as_str() {
        "tract" => {
            let concurrency = num_cpus::get();
            for _ in 0..concurrency {
                let rx_infer_clone = rx_infer.clone();
                let tx_annotated_clone = tx_annotated.clone();
                let model_path = args.weights.clone();
                thread::spawn(move || {
                    inference_thread_tract(rx_infer_clone, tx_annotated_clone, &model_path, width.try_into().unwrap(), height.try_into().unwrap()).unwrap();
                });
            }
        },
        "onnx" => {
            let rx_infer_clone = rx_infer.clone();
            let tx_annotated_clone = tx_annotated.clone();
            thread::spawn(move || {
                inference_thread_onnx(rx_infer_clone, tx_annotated_clone, &args.weights, width.try_into().unwrap(), height.try_into().unwrap());
            });
        },
        _ => {
            eprintln!("Unsupported engine: {}", args.engine);
            return Ok(());
        }
    };

    drop(tx_annotated); // to let them shut down eventually

    // 9) Main loop: handle camera requests + display
    let mut frame_count = 0;
    let mut annotated_count = 1;
    let mut pending_map = std::collections::HashMap::new();
    while frame_count < args.num_frames {
        // Wait for next completed request
        let mut req = match rx_req.recv_timeout(Duration::from_secs(2)) {
            Ok(r) => r,
            Err(_) => {
                eprintln!("Timed out waiting for camera request");
                break;
            }
        };

        // Decode MJPEG
        let fb: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();
        let planes = fb.data();
        let frame_data = planes.get(0).unwrap();
        let bytes_used = fb.metadata().unwrap().planes().get(0).unwrap().bytes_used as usize;
        let decoded = match image::load_from_memory_with_format(
            &frame_data[..bytes_used],
            ImageFormat::Jpeg,
        ) {
            Ok(img) => {
                frame_count += 1;
                img.to_rgb8()
                },
            Err(e) => {
                eprintln!("Failed to decode MJPEG: {e}");
                cam.queue_request(req)?;
                continue;
            }
        };

        let tagged = TaggedFrame {
            seq_id: frame_count,
            image: decoded,
        };

        // Send to inference
        match tx_infer.send(tagged) {
            Ok(_) => (),
            Err(_) => {
                eprintln!("Inference channel closed");
                break;
            }
        }

        // Re-queue request for next capture
        req.reuse(ReuseFlag::REUSE_BUFFERS); 
        cam.queue_request(req)?;

        // If an annotated image is available, display it
        if let Ok(annotated) = rx_annotated.try_recv() {
            pending_map.insert(annotated.seq_id, annotated.image);
        }

        // Display any pending frames
        while let Some(image) = pending_map.remove(&annotated_count) {
            display_frame(&mut canvas, &image)?;
            println!("Displayed frame {}", annotated_count);
            annotated_count += 1;
        } 
    }

    // Cleanup
    // Let the inference threads exit
    drop(rx_infer);
    drop(tx_infer);

    // Drain any remaining annotated images
    if let Ok(annotated) = rx_annotated.try_recv() {
        pending_map.insert(annotated.seq_id, annotated.image);
    }

    // Display any pending frames
    while let Some(image) = pending_map.remove(&annotated_count) {
        display_frame(&mut canvas, &image)?;
        println!("Displayed frame {}", annotated_count);
        annotated_count += 1;
    } 

    // Stop camera
    cam.stop()?;
    println!("Done capturing. Exiting.");

    Ok(())
}
