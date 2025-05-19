// ruid-camera/src/stream.rs
use crate::{Camera, VideoFrame, Result};
use futures_core::Stream;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream};

// back‑pressure: appsink → channel → consumer
const DEPTH: usize = 4;

pub fn frame_stream(cam: Camera) -> impl Stream<Item = Result<VideoFrame>> {
    let (tx, rx) = mpsc::channel(DEPTH);

    std::thread::spawn(move || {
        loop {
            match cam.next_frame_blocking() {
                Ok(f) => {
                    if tx.blocking_send(Ok(f)).is_err() {
                        break; // consumer dropped
                    }
                }
                Err(e) => {
                    let _ = tx.blocking_send(Err(e));
                    break;
                }
            }
        }
    });

    ReceiverStream::new(rx)
}
