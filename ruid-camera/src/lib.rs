// ruid-camera/src/lib.rs
// ============================================================
// Zero‑copy camera capture crate for RuID v2
// Uses GStreamer + libcamera to grab NV12 frames backed by
// DMA‑Buf FDs so downstream GPU code can import without copies.
// ------------------------------------------------------------
// Public API:
//   * Camera::new() – build and start a capture pipeline
//   * Camera::next_frame() – async stream of DmaBufFrame
//
// Compile on ARM & x86_64 (libcamera optional on desktops).
// ============================================================

//! RuID – camera capture layer
//!
//! This crate exposes a minimal, zero‑copy camera API around a
//! `gstreamer` pipeline composed of `libcamerasrc` → `appsink`.
//! Frames are delivered as [`DmaBufFrame`], which holds the raw
//! DMA‑Buf file‑descriptor plus metadata (width, height, stride,
//! timestamp).  Downstream GPU preprocess code can import the fd
//! directly via EGL/Vulkan without an intermediate memcpy.

use anyhow::{anyhow, Result};
use gst::prelude::*;
use std::os::unix::prelude::RawFd;
use std::time::Duration;

mod stream;
pub use stream::frame_stream;

/// Metadata + FD for a single captured frame.
#[derive(Debug)]
pub struct DmaBufFrame {
    /// DMA‑Buf file descriptor (owning).
    pub fd: RawFd,
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
    ///     println!("Got fd {} ({}×{})", frame.fd, frame.width, frame.height);
    /// }
    /// ```
    pub fn new(width: u32, height: u32, fps: u32) -> Result<Self> {
        gst::init().map_err(|e| anyhow!("GStreamer init failed: {e}"))?;

        let src = if gst::ElementFactory::find("libcamerasrc").is_some() {
            // Pi / libcamera stack
            "libcamerasrc"
        } else {
            // Laptop or USB webcam
            "v4l2src device=/dev/video0 io-mode=2"
        };

        let pipe_str = format!(
            "{src} ! videoconvert ! video/x-raw,format=NV12,width={w},height={h},framerate={f}/1 \
            ! queue leaky=2 max-size-buffers=8 ! tee name=t \
            t. ! queue leaky=2 max-size-buffers=1 ! videoconvert ! autovideosink sync=false \
            t. ! queue leaky=2 max-size-buffers=8 ! appsink name=sink emit-signals=true \
            sync=false",
            src = src, w = width, h = height, f = fps
        );
        let pipeline = gst::parse::launch(&pipe_str)?
            .downcast::<gst::Pipeline>()
            .map_err(|_| anyhow!("Not a gst::Pipeline"))?;

        let appsink = pipeline
            .by_name("sink")
            .ok_or_else(|| anyhow!("appsink not found"))?
            .downcast::<gst_app::AppSink>().unwrap();

        pipeline.set_state(gst::State::Playing)?;
        Ok(Self { pipeline, appsink })
    }

    /// Blocking retrieval – Hate this with a passion gst is hard.
    pub fn next_frame_blocking(&self) -> Result<DmaBufFrame> {
        let sample = self
            .appsink
            .pull_sample()
            .map_err(|_| anyhow!("Failed to pull sample"))?;
        Self::sample_to_dmabuf(sample)
    }

    /// Convert a `gst::Sample` into our [`DmaBufFrame`] wrapper.
    fn sample_to_dmabuf(sample: gst::Sample) -> Result<DmaBufFrame> {
        // Extract buffer + caps.
        let buffer = sample.buffer().ok_or_else(|| anyhow!("Sample has no buffer"))?;
        let caps = sample.caps().ok_or_else(|| anyhow!("Sample has no caps"))?;
        let s = caps.structure(0).ok_or_else(|| anyhow!("Caps missing struct"))?;
        let width = s.get::<i32>("width")? as u32;
        let height = s.get::<i32>("height")? as u32;
        let stride = s
            .get::<i32>("stride")
            .unwrap_or((width * 2) as i32) as u32;           // NV12 ≈ 2 bytes/px

        if buffer.n_memory() == 0 {
            return Err(anyhow!("Buffer has no memory"));
        }

        // Down‑cast the first memory block to DmaBufMemoryRef
        let mem = buffer.peek_memory(0);
        let dmabuf =
            mem.downcast_memory_ref::<gst_allocators::DmaBufMemory>()
                .ok_or_else(|| anyhow!("Memory is not DMA‑Buf; zero‑copy path broken"))?;

        let fd: RawFd = dmabuf.fd();

        // PTS → Duration (or zero if missing)
        let pts = buffer
            .pts()
            .map(|t| Duration::from_nanos(t.nseconds()))
            .unwrap_or(Duration::ZERO);

        Ok(DmaBufFrame {
            fd,
            width,
            height,
            stride,
            pts,
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
        println!("Received fd {} ({}×{}) stride {} bytes", frame.fd, frame.width, frame.height, frame.stride);
        assert_eq!(frame.width, 640);
    }
}

// ---------------------------------------------------------------------------
// End of file
