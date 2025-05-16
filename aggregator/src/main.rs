//! Aggregator that polls N ≥ 1 camera nodes provided as CLI args
//!   $ aggregator http://pi1:8080 http://pi2:8080
//! Defaults to http://localhost:8080 if no args supplied.

use linfa::clustering::KMeans;
use ndarray::Array2;
use reqwest::Client;
use ru_common::Detection;
use std::{env, time::Duration};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut cams: Vec<String> = env::args().skip(1).collect();
    if cams.is_empty() {
        cams.push("http://localhost:12345".into());
    }
    println!("Polling cameras: {:?}", cams);

    let client = Client::builder()
        .timeout(Duration::from_millis(750))
        .build()?;

    loop {
        let mut all: Vec<Detection> = Vec::new();

        // fan‑out / fan‑in
        for cam in &cams {
            match client
                .get(format!("{cam}/detections"))
                .send()
                .await?
                .json::<Vec<Detection>>()
                .await
            {
                Ok(mut v) => all.append(&mut v),
                Err(e) => eprintln!("WARN {cam}: {}", e),
            }
        }

        if all.is_empty() {
            eprintln!("No detections this tick");
            tokio::time::sleep(Duration::from_millis(500)).await;
            continue;
        }

        // Build embedding matrix
        let m: Array2<f32> = Array2::from_shape_vec(
            (all.len(), 128),
            all.iter().flat_map(|d| d.embedding.clone()).collect(),
        )?;

        let k = (all.len() / 3).max(1);
        let model = KMeans::params(k).fit(&m)?;
        let clusters = model.predict(&m);

        println!("\n{} detections, {} clusters", all.len(), k);
        for (idx, cl) in clusters.iter().enumerate() {
            println!(
                "  ID {:>2}: cam {:<15} bbox {:?}",
                cl, all[idx].cam_id, all[idx].bbox
            );
        }

        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}
