// examples/preview.rs
use ruid_camera::{Camera, frame_stream};
use tokio_stream::StreamExt;
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cam   = Camera::new(640, 480, 60)?;
    let mut s = frame_stream(cam);

    let mut count = 0u64;
    let mut start = Instant::now();

    while let Some(frame) = s.next().await {
        let _f = frame?;
        count += 1;

        // simple FPS counter
        if count % 150 == 0 {
            let fps = count as f64 / start.elapsed().as_secs_f64();
            eprintln!("{count} frames  |  avg {fps:.1} FPS");
            start = Instant::now();
            count = 0;
        }

        // TODO: hand off `f` to preprocess + inference here...
    }
    Ok(())
}
