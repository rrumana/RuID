// ruid-camera/src/stream.rs
use crate::{Camera, DmaBufFrame};
use anyhow::Result;
use futures_core::Stream;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream};

const DEPTH: usize = 4; // back‑pressure: appsink → channel → consumer

pub fn frame_stream(cam: Camera) -> impl Stream<Item = Result<DmaBufFrame>> {
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
