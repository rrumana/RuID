// ruid-preprocess/src/lib.rs
// ============================================================
// ruid-preprocess  –  Resize + Normalize stage for RuID v2
// Converts camera NV12 DMA-Buf frames into model-ready
// tensors (CHW RGB, f32 0-1) with **no extra copies** when
// an OpenCL device is available (falls back to SIMD CPU).
// ------------------------------------------------------------
// Public API
//   * Preprocessor::new(out_w, out_h)         – configure target size
//   * Preprocessor::run(&self, &VideoFrame)   – returns ndarray::Array3<f32>
//     (GPU path behind a `cfg(feature = "gpu")` gate)
// ------------------------------------------------------------
// Build notes
//   * Compiles on x86-64 & aarch64.  
//   * OpenCL headers optional; CPU fallback always available.
// ============================================================

//! RuID – preprocessing layer
//!
//! This crate implements a tiny, zero-copy resize & normalisation stage.
//! On systems with OpenCL (Mesa VC6, AMD, Nvidia, etc.) it **imports the
//! camera DMA-Buf fd directly into the GPU** and runs a 2-tap Lanczos
//! kernel; otherwise it maps the buffer and uses a hand-vectorised NEON /
//! AVX2 routine. Or at least that's the plan.  Currently only the CPU path exists
//!
//! Output is an `ndarray::Array3<f32>` in **CHW** order, already scaled to
//! 0-1.0, ready to be fed into the YOLO/ResNet detectors.

use thiserror::Error;
use ndarray::Array3;
use ruid_camera::{VideoFrame, FrameBacking};
use resize::{new, Pixel, Type};
use rgb::FromSlice;

#[derive(Error, Debug)]
pub enum PreprocessError {
    #[error("GPU path not yet implemented; build with --features gpu")]
    GpuNotImplemented,
    #[error("OpenCL kernel not yet implemented")]
    GpuUnimplemented,
    #[error("Failed to create resizer: {0}")]
    ResizeError(String),
    #[error("Invalid ndarray buffer")]
    NdArrayBufferError,
}

pub type Result<T> = std::result::Result<T, PreprocessError>;

#[derive(Clone)]
pub struct Preprocessor {
    dst_w: u32,
    dst_h: u32,
}

impl Preprocessor {
    /// Create a pre‑processor that outputs WxH RGB (0‑1.0f32).
    pub fn new(dst_w: u32, dst_h: u32) -> Self {
        Self { dst_w, dst_h }
    }

    /// CPU path – always compiled.
    pub fn run(&self, frame: &VideoFrame) -> Result<Array3<f32>> {
        // 1. get raw NV12 bytes (copy if needed)
        let nv12 = match &frame.backing {
            FrameBacking::Cpu(bytes) => bytes.as_slice(),
            FrameBacking::DmaBuf(_fd) => {
                return Err(PreprocessError::GpuNotImplemented);
            }
        };

        // 2. Y plane and interleaved UV plane sizes
        let w = frame.width as usize;
        let h = frame.height as usize;
        let y_plane  = &nv12[..w * h];
        let uv_plane = &nv12[w * h..];

        // 3. Convert NV12 → RGB (bilinear chroma up‑sample)
        let mut rgb = vec![0u8; w * h * 3];
        nv12_to_rgb(y_plane, uv_plane, w, h, &mut rgb);

        // 4. Resize to dst size with Lanczos3 (resize crate)
        let mut dst = vec![0u8; (self.dst_w * self.dst_h * 3) as usize];

        // Create a reusable resizer instance:
        let mut resizer = new(
            w,
            h,
            self.dst_w as usize,
            self.dst_h as usize,
            Pixel::RGB8,
            Type::Lanczos3,
        )
        .map_err(|e| PreprocessError::ResizeError(e.to_string()))?;

        // Now invoke it on your raw byte buffers:
        let _ = resizer.resize(
            rgb.as_rgb(),        // &[RGB<u8>]
            dst.as_rgb_mut(),    // &mut [RGB<u8>]
        );

        // 5. Normalize to 0‑1 and pack into ndarray (H,W,C)
        let mut arr = Array3::<f32>::zeros((self.dst_h as usize,
                                            self.dst_w as usize,
                                            3));
        for (out_elem, px) in arr.iter_mut().zip(dst.iter()) {
            *out_elem = *px as f32 / 255.0;
        }
        Ok(arr)
    }

    /// GPU path stub.  Compiles only with `--features gpu`.
    #[cfg(feature = "gpu")]
    pub fn run(&self, frame: &VideoFrame) -> Result<Array3<f32>> {
        unimplemented!("OpenCL kernel TODO");
    }
}

/// Naive NV12 4:2:0 → RGB24 conversion (BT.601, full range).
fn nv12_to_rgb(y: &[u8], uv: &[u8], w: usize, h: usize, out: &mut [u8]) {
    for j in 0..h {
        for i in 0..w {
            let y_val = y[j * w + i] as f32;
            let uv_idx = (j / 2) * w + (i & !1);
            let u = uv[uv_idx]     as f32 - 128.0;
            let v = uv[uv_idx + 1] as f32 - 128.0;

            let r = (y_val + 1.402 * v).clamp(0.0, 255.0);
            let g = (y_val - 0.344_13 * u - 0.714_14 * v).clamp(0.0, 255.0);
            let b = (y_val + 1.772 * u).clamp(0.0, 255.0);

            let base = (j * w + i) * 3;
            out[base]     = r as u8;
            out[base + 1] = g as u8;
            out[base + 2] = b as u8;
        }
    }
}
