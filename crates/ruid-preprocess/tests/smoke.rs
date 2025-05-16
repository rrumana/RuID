use ruid_camera::{VideoFrame, FrameBacking};
use ruid_preprocess::Preprocessor;

#[test]
fn cpu_smoke() {
    // Fake gray NV12 640×480
    let w = 640; let h = 480;
    let mut bytes = vec![128u8; (w * h * 3 / 2) as usize];
    // Y plane = 255 (white)
    bytes[..(w*h) as usize].fill(255);

    let frame = VideoFrame {
        backing: FrameBacking::Cpu(bytes),
        width: w, height: h, stride: w*2, pts: std::time::Duration::ZERO,
    };

    let pp = Preprocessor::new(224, 224);
    let out = pp.run(&frame).unwrap();
    assert_eq!(out.shape(), &[224, 224, 3]);
}