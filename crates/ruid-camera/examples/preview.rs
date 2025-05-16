use ruid_camera::{Camera, frame_stream};
use ruid_preprocess::Preprocessor;
use tokio_stream::StreamExt;
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cam   = Camera::new(640, 480, 30)?;
    let mut s = frame_stream(cam);

    let mut count = 0u64;
    let mut start = Instant::now();

    // Move Preprocessor creation _out_ of the loop
    let pp = Preprocessor::new(224, 224);

    // Single loop: pull → preprocess → count → repeat
    while let Some(frame_res) = s.next().await {
        let frame = frame_res?;

        // preprocess into a tensor
        let _tensor = pp.run(&frame)?;
        // TODO: inference on `tensor`

        // FPS counting
        count += 1;
        if count % 150 == 0 {
            let elapsed = start.elapsed();
            let fps = 150.0 / elapsed.as_secs_f64();
            eprintln!("avg {:.1} FPS (last 150 frames)", fps);
            start = Instant::now();
            count = 0;
        }
    }

    Ok(())
}