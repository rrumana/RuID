// ruid-camera/src/lib.rs
// ============================================================
// Zero‑copy camera capture crate for RuID v2
// Uses GStreamer + libcamera to grab NV12 frames backed by
// DMA‑Buf FDs so downstream GPU code can import without copies.
// ------------------------------------------------------------
// Public API:
//   * Camera::new() – build and start a capture pipeline
//   * Camera::next_frame() – async stream of DmaBufFrame
// ------------------------------------------------------------
// Build notes
//   * Compile on ARM & x86_64 (libcamera optional on desktops).
// ============================================================

//! RuID – camera capture layer
//!
//! This crate exposes a minimal, zero‑copy camera API around a
//! `gstreamer` pipeline composed of `libcamerasrc` → `appsink`.
//! Frames are delivered as [`DmaBufFrame`], which holds the raw
//! DMA‑Buf file‑descriptor plus metadata (width, height, stride,
//! timestamp).  Downstream GPU preprocess code can import the fd
//! directly via EGL/Vulkan without an intermediate memcpy.

use thiserror::Error;
use gst::prelude::*;
use std::os::unix::prelude::RawFd;
use std::time::Duration;

mod stream;
pub use stream::frame_stream;

// Custom error types for this crate, useful as this project grows
#[derive(Error, Debug)]
pub enum CameraError {
    #[error("GStreamer init failed: {0}")]
    GstInit(#[source] gst::glib::Error),
    #[error("Failed to parse pipeline: {0}")]
    ParsePipeline(#[source] gst::glib::Error),
    #[error("Pipeline is not a gst::Pipeline")]
    NotPipeline,
    #[error("AppSink element not found")]
    AppSinkNotFound,
    #[error("AppSink element downcast failed")]
    AppSinkDowncastFailed,
    #[error("Failed to set pipeline to Playing: {0}")]
    StateChange(#[source] gst::StateChangeError),
    #[error("Failed to pull sample: {0}")]
    PullSample(#[source] gst::glib::BoolError),
    #[error("Sample has no buffer")]
    MissingBuffer,
    #[error("Sample has no caps")]
    MissingCaps,
    #[error("Caps missing struct")]
    MissingStructure,
    #[error("Failed to get field value: {0}")]
    FieldError(String),
    #[error("Buffer map failed: {0}")]
    BufferMap(String),
}

pub type Result<T> = std::result::Result<T, CameraError>;

/// A captured frame.
///
/// * `DmaBuf(fd)`  – zero‑copy path (fast on Pi, some webcams)
/// * `Cpu(bytes)`  – fallback heap copy (always works)
#[derive(Debug)]
pub enum FrameBacking {
    DmaBuf(RawFd),
    Cpu(Vec<u8>),      // NV12 bytes, stride == width*2
}

/// Metadata + FD for a single captured frame.
#[derive(Debug)]
pub struct VideoFrame {
    pub backing: FrameBacking,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub pts: Duration,
}

/// Camera handle – owns the pipeline and *appsink*.
pub struct Camera {
    pipeline: gst::Pipeline,
    appsink: gst_app::AppSink,
}

impl Camera {
    /// Build and *Playing* a libcamerasrc pipeline that delivers NV12 frames.
    ///
    /// ```no_run
    /// use ruid_camera::Camera;
    /// let cam = Camera::new(640, 480, 30).unwrap();
    /// for _ in 0..10 {
    ///     let frame = cam.next_frame_blocking().unwrap();
    ///     println!("Got backing {:?} ({}×{})", frame.backing, frame.width, frame.height);
    /// }
    /// ```
    pub fn new(width: u32, height: u32, fps: u32) -> Result<Self> {
        gst::init().map_err(CameraError::GstInit)?;

        let src = if gst::ElementFactory::find("libcamerasrc").is_some() {
            // Pi (libcamera) stack
            "libcamerasrc"
        } else {
            // PC webcam
            "v4l2src device=/dev/video0 io-mode=4"
        };

        let pipe_str = format!(
            "{src} ! videoconvert ! video/x-raw,format=NV12,width={w},height={h},framerate={f}/1 \
            ! queue leaky=2 max-size-buffers=8 ! appsink name=sink emit-signals=true sync=false",
            src = src, w = width, h = height, f = fps
        );

        let pipeline = gst::parse::launch(&pipe_str)
            .map_err(CameraError::ParsePipeline)?
            .downcast::<gst::Pipeline>()
            .map_err(|_| CameraError::NotPipeline)?;

        let appsink = pipeline
            .by_name("sink")
            .ok_or(CameraError::AppSinkNotFound)?
            .downcast::<gst_app::AppSink>()
            .map_err(|_| CameraError::AppSinkDowncastFailed)?;

        pipeline
            .set_state(gst::State::Playing)
            .map_err(CameraError::StateChange)?;

        Ok(Self { pipeline, appsink })
    }

    /// Asynchronous retrieval – using GStreamer's async API.
    pub async fn next_frame(&self) -> Result<VideoFrame> {
        todo!();       
        // let sample = self
        //     .appsink
        //     .pull_sample_async()
        //     .await
        //     .map_err(Error::PullSample)?;
        //
        // Self::sample_to_frame(sample)
    }

    /// Blocking retrieval – Hate this with a passion but gst is hard and this is a prototype.
    pub fn next_frame_blocking(&self) -> Result<VideoFrame> {
        let sample = self
            .appsink
            .pull_sample()
            .map_err(CameraError::PullSample)?;

        Self::sample_to_frame(sample)
    }

    /// Convert a `gst::Sample` into our [`VideoFrame`] wrapper.
    fn sample_to_frame(sample: gst::Sample) -> Result<VideoFrame> {
        let buffer = sample.buffer().ok_or(CameraError::MissingBuffer)?;
        let caps   = sample.caps().ok_or(CameraError::MissingCaps)?;
        let s      = caps.structure(0).ok_or(CameraError::MissingStructure)?;
        let width  = s.get::<i32>("width").map_err(|e| CameraError::FieldError(e.to_string()))? as u32;
        let height = s.get::<i32>("height").map_err(|e| CameraError::FieldError(e.to_string()))? as u32;
        let stride = (width * 2) as u32;                // NV12 bytes/px

        let pts = buffer
            .pts()
            .and_then(|t| Some(Duration::from_nanos(t.nseconds())))
            .unwrap_or_else(|| Duration::ZERO);

        // Try fast DMA‑Buf path
        if buffer.n_memory() > 0 {
            let mem = buffer.peek_memory(0);
            if let Some(dmabuf) = mem.downcast_memory_ref::<gst_allocators::DmaBufMemory>() {
                return Ok(VideoFrame {
                    backing: FrameBacking::DmaBuf(dmabuf.fd()),
                    width, height, stride, pts,
                });
            }
        }

        // Fallback: map + copy into Vec<u8>
        let map = buffer.map_readable().map_err(|e| CameraError::BufferMap(e.to_string()))?;          // &[u8] for entire NV12 image
        let mut bytes = Vec::with_capacity(map.size());
        bytes.extend_from_slice(map.as_slice());   // one memcpy
        drop(map);                                 // unmap

        Ok(VideoFrame {
            backing: FrameBacking::Cpu(bytes),
            width, height, stride, pts,
        })
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

// ---------------------------------------------------------------------------
// Integration test (cargo test -- --nocapture) – skipped on CI without camera
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn capture_one() {
        let cam = Camera::new(640, 480, 30).expect("create");
        let frame = cam.next_frame_blocking().expect("frame");
        println!("Received fd {:?} ({}x{}) stride {} bytes", frame.backing, frame.width, frame.height, frame.stride);
        assert_eq!(frame.width, 640);
    }
}