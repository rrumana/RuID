//! Minimal single‑Pi camera node: grabs frames, creates dummy detections,
//! and serves   /frame        → image/jpeg
//!               /detections  → JSON Vec<Detection>

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use opencv::{prelude::*, videoio};
use ru_common::{Detection, OnnxModel, RuIdError};
use std::{
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::RwLock;

#[derive(Clone)]
struct AppState {
    det_model: Arc<OnnxModel>,
    reid_model: Arc<OnnxModel>,
    cam_id: String,
    latest_jpeg: Arc<RwLock<Vec<u8>>>,
    latest_dets: Arc<RwLock<Vec<Detection>>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Build shared state
    let env = ort::Environment::builder().with_name("ruid").build()?;
    let state = AppState {
        det_model: Arc::new(OnnxModel::new(
            &env,
            &PathBuf::from("../models/yolov8n_int8.onnx"),
        )?),
        reid_model: Arc::new(OnnxModel::new(
            &env,
            &PathBuf::from("../models/resnet50_market1501_int8.onnx"),
        )?),
        cam_id: hostname::get()?.into_string().unwrap_or("cam".into()),
        latest_jpeg: Arc::new(RwLock::new(Vec::new())),
        latest_dets: Arc::new(RwLock::new(Vec::new())),
    };

    // Spawn the frame‑producer thread
    tokio::task::spawn_blocking({
        let st = state.clone();
        move || camera_loop(st).expect("camera loop exited")
    });

    // HTTP routes
    let app = Router::new()
        .route(
            "/frame",
            get({
                let st = state.clone();
                async move |State(st): State<AppState>| -> Response {
                    let jpeg = st.latest_jpeg.read().await.clone();
                    if jpeg.is_empty() {
                        StatusCode::SERVICE_UNAVAILABLE.into_response()
                    } else {
                        ([("Content-Type", "image/jpeg")], jpeg).into_response()
                    }
                }
            }),
        )
        .route(
            "/detections",
            get({
                let st = state;
                async move |State(st): State<AppState>| {
                    let dets = st.latest_dets.read().await.clone();
                    Json(dets)
                }
            }),
        )
        .with_state(state);

    axum::Server::bind(&"0.0.0.0:12345".parse()?)
        .serve(app.into_make_service())
        .await?;
    Ok(())
}

fn camera_loop(st: AppState) -> Result<(), RuIdError> {
    let mut cam = videoio::VideoCapture::new_default(0)?;
    cam.set(videoio::CAP_PROP_FRAME_WIDTH, 640.0)?;
    cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 480.0)?;

    loop {
        let mut frame = opencv::core::Mat::default();
        if !cam.read(&mut frame)? {
            continue;
        }

        // TODO: real YOLO + ReID.  For now, dummy detection at (0,0,32,32)
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        let det = Detection {
            bbox: [0, 0, 32, 32],
            embedding: vec![0.0; 128],
            cam_id: st.cam_id.clone(),
            frame_ts: ts,
        };

        // Encode JPEG
        let mut jpeg_buf = opencv::types::VectorOfu8::new();
        opencv::imgcodecs::imencode(
            ".jpg",
            &frame,
            &mut jpeg_buf,
            &opencv::types::VectorOfi32::new(),
        )?;
        *st.latest_jpeg.write().unwrap() = jpeg_buf.to_vec();
        *st.latest_dets.write().unwrap() = vec![det];
    }
}
