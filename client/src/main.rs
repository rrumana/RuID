use anyhow::{Error, Result};
use anyhow::anyhow;
use clap::Parser;
use image::{DynamicImage, ImageFormat, Rgb, RgbImage, ImageBuffer};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use libcamera::{
    camera::ActiveCamera,
    camera::CameraConfigurationStatus,
    camera_manager::CameraManager,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    geometry::Size,
    pixel_format::PixelFormat,
    properties,
    request::Request,
    request::ReuseFlag,
    stream::Stream,
    stream::StreamRole,
};
use std::{
    sync::mpsc, 
    cmp::Ordering,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
    borrow::Borrow,
    fmt::{Debug, Display},
};
use tract_onnx::prelude::*;
use tract_ndarray::s;
use sdl2::{
    pixels::PixelFormatEnum,
    rect::Rect as SdlRect,
    render::Canvas,
    video::Window,
};
use num_cpus;
use crossbeam_channel::{bounded, Receiver, Sender};

// ================ PIPELINE STAGES ================== //

/// 0) Parse CLI arguments
#[derive(Parser)]
struct CliArgs {
    #[arg(long)]
    weights: String,

    #[arg(long, default_value = "200")]
    num_frames: usize,
}

/// 2) Run inference on the frame
#[derive(Debug, Clone)]
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

fn run_model<F, O, M>(
    model: &RunnableModel<F, O, M>,
    decoded_image: &DynamicImage
) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> 
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    M: Borrow<Graph<F, O>> 
{
    let resized = image::imageops::resize(
        decoded_image,
        640, 640,
        image::imageops::FilterType::Triangle,
    );
    let dynamic_resized = DynamicImage::from(resized);
    let rgba = dynamic_resized.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    let image_tensor: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, h as usize, w as usize), |(_, c, y, x)| {
        rgba.get_pixel(x as u32, y as u32)[c] as f32 / 255.0
    })
    .into();

    let forward = model.run(tvec![image_tensor.to_owned().into()]).unwrap();
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

    let final_bboxes = non_maximum_suppression(&mut bbox_vec, 0.5);

    let mut annotated = dynamic_resized.to_rgb8();
    draw_bboxes_on_image(&mut annotated, &final_bboxes);

    Ok(annotated)
}


/// 3) Cull extra bounding boxes and Annotate the frame in-place

pub fn non_maximum_suppression(boxes: &mut Vec<Bbox>, iou_threshold: f32) -> Vec<Bbox> {
    boxes.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(Ordering::Equal)
    });

    let mut keep = Vec::new();
    while !boxes.is_empty() {
        let current = boxes.remove(0);
        keep.push(current.clone());
        boxes.retain(|other| calculate_iou(&current, other) <= iou_threshold);
    }
    keep
}

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

/// 4) Display the frame

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

// ================ THREAD STARTERS ================== //

/// Inference thread: read frames from camera, run inference, annotate, send to display

fn inference_thread<F, O, M>(
    rx_infer: Receiver<RgbImage>,
    tx_out: Sender<RgbImage>,
    model: RunnableModel<F, O, M>,
) where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    M: Borrow<Graph<F, O>> + Clone + 'static,
{
    while let Ok(rgb) = rx_infer.recv() {
        match run_model(&model, &DynamicImage::ImageRgb8(rgb)) {
            Ok(annotated) => {
                // send annotated image back
                if tx_out.send(annotated).is_err() {
                    break;
                }
            }
            Err(e) => eprintln!("Inference error: {e}"),
        }
    }
    eprintln!("inference_thread shutting down.");
}

const PIXEL_FORMAT_MJPEG: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'M', b'J', b'P', b'G']), 0);

fn main() -> Result<()> {
    // 1) Parse CLI arguments
    let args = CliArgs::parse();
    let width = 640;
    let height = 640;

    // 2) Load and optimize the ONNX model
    let model = tract_onnx::onnx()
        .model_for_path(&args.weights)?
        .with_input_fact(0, f32::fact([1, 3, width as i32, height as i32]).into())?
        .into_optimized()?
        .into_runnable()?;

    // 3) Initialize libcamera + find camera 
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
        Err(e) => println!("Unable to get camera model: {}", e),
    }

    let mut cam = match cam.acquire() {
        Ok(cam) => cam,
        Err(e) => {
            eprintln!("Failed to acquire camera: {}", e);
            return Ok(());
        }
    }; 

    // 4. Configure the camera for MJPEG (since RG24 is not supported)
    let mut cfgs = cam.generate_configuration(&[StreamRole::VideoRecording]).unwrap();
    cfgs.get_mut(0).unwrap().set_pixel_format(PIXEL_FORMAT_MJPEG);
    //cfgs.get_mut(0).unwrap().set_size(Size { width, height });

    println!("Generated config: {:#?}", cfgs);

    match cfgs.validate() {
        CameraConfigurationStatus::Valid => println!("Camera configuration valid!"),
        CameraConfigurationStatus::Adjusted => println!("Camera configuration was adjusted: {:#?}", cfgs),
        CameraConfigurationStatus::Invalid => panic!("Invalid camera configuration"),
    }

    cam.configure(&mut cfgs).expect("Unable to configure camera");

    // 5. Allocate buffers
    let mut alloc = FrameBufferAllocator::new(&cam);
    let cfg = cfgs.get(0).unwrap();
    let stream = cfg.stream().unwrap();
    let buffers = alloc.alloc(&stream).unwrap();
    println!("Allocated {} buffers", buffers.len());

    let buffers = buffers
        .into_iter()
        .map(MemoryMappedFrameBuffer::new)
        .collect::<Result<Vec<_>, _>>()?;

    // 6. Create requests
    let reqs = buffers
        .into_iter()
        .enumerate()
        .map(|(i, buf)| {
            let mut req = cam.create_request(Some(i as u64)).unwrap();
            req.add_buffer(&stream, buf).unwrap();
            req
        })
        .collect::<Vec<_>>();

    // 7. Setup request-completed callback channel
    let (tx_req, rx_req) = mpsc::channel();
    cam.on_request_completed(move |req| {
        tx_req.send(req).unwrap();
    });

    // 8. Start capturing
    cam.start(None).unwrap();
    for req in reqs {
        println!("Request queued for execution: {req:#?}");
        cam.queue_request(req).unwrap();
    }
   
    // 9) Initialize SDL2 for display (640x640)
    let mut canvas = init_sdl2(width, height)
        .map_err(|e| Error::msg(format!("Failed to init SDL2: {e}")))?;
    
    // 7) Create inference channels
    let (tx_infer, rx_infer) = bounded::<RgbImage>(4);
    let (tx_annotated, rx_annotated) = bounded::<RgbImage>(4);

    // 8) Spawn inference threads
    let concurrency = num_cpus::get()-1;
    for _ in 0..concurrency {
        let rx_infer_clone = rx_infer.clone();
        let tx_annotated_clone = tx_annotated.clone();
        let model = model.clone();
        thread::spawn(move || {
            inference_thread(rx_infer_clone, tx_annotated_clone, model);
        });
    }
    drop(tx_annotated); // to let them shut down eventually

    // 9) Main loop: handle camera requests + display
    let mut frame_count = 0;
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
            Ok(img) => img.to_rgb8(),
            Err(e) => {
                eprintln!("Failed to decode MJPEG: {e}");
                // re-queue for next time
                cam.queue_request(req)?;
                continue;
            }
        };

        // Send to inference
        match tx_infer.send(decoded) {
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
            display_frame(&mut canvas, &annotated)?;
            println!("Frame {frame_count}");
            frame_count += 1;
        }
    }

    // Cleanup
    // Let the inference threads exit
    drop(rx_infer);
    drop(tx_infer);
    // Drain any remaining annotated images
    while let Ok(annotated) = rx_annotated.try_recv() {
        display_frame(&mut canvas, &annotated)?;
    }

    // Stop camera
    cam.stop()?;
    println!("Done capturing. Exiting.");

    Ok(())
}
