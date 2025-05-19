## -1 | Reality

Test This project is just in it's infancy, I do it in my spare time for fun. Understand that this is the where I hope it will end up, not where the project is today. I'll update this occasionally with progress, but don't expect this document to represent the actual program for quire some time.

---

## 0  |  Big‑picture goals

| Constraint             | Target                                                                                  | Why it matters                                                  |
| ---------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Edge HW**            | Raspberry Pi 4/5 (4× A72 + V3D/VC6) **or** generic x86‑64 dev PC                        | Covers both low‑power deployment and developer laptops          |
| **Throughput**         | ≥ 10 FPS @ 640×480 with **Pi 5 8GB**                                                    | Smooth enough for tracking; leaves headroom for other workloads |
| **Copies on hot‑path** | **Zero** from camera sensor → GPU inference                                             | Copies kill cache and battery                                   |
| **Latency budg.**      | ≤ 120 ms edge→cluster identify round‑trip                                               | Matches interactive use cases (door unlock, AR overlay)         |                              |
| **Modularity**         | Each stage = standalone crate; swap & upgrade independently                             | Re‑train models, change trackers without kernel recompile       |
| **Repro builds**       | `cargo build --release -Zbuild-std` cross‑compiled; GH Actions matrix + `docker buildx` | Deterministic binaries for ARM & x86                            |
| **Local preview**      | Optional tee to GUI; headless by default                                                | Debug in one binary, ship the same binary                       |

---

## 1  |  High‑level component map

```text
               ┌─────────────────────────┐
               │  Camera sensor / ISP    │
               └──────────┬──────────────┘
                          │  DMA‑Buf (NV12)
               ┌──────────▼──────────────┐
               │  ruid‑camera  (edge)    │
               │  ─── capture thread ─── │
               └──────────┬──────────────┘
          async Stream<DmaBufFrame>
                          │
               ┌──────────▼──────────────┐
               │ ruid‑preprocess (CL)    │
               │  resize + norm kernel   │
               └──────────┬──────────────┘
          OpenCL/VK buffer│
                          │
               ┌──────────▼──────────────┐
               │  ruid‑detect            │
               │  (YOLOv8‑n INT8 ORT)    │
               └──────────┬──────────────┘
                detections│
                          │
               ┌──────────▼──────────────┐
               │  ruid‑reid             │
               │  (ResNet‑IBN embed)    │
               └──────────┬──────────────┘
             det+embed+box│
                          │
               ┌──────────▼──────────────┐
               │  ruid‑track (BYTE)      │
               └──────────┬──────────────┘
             traj/track id│
                          │  gRPC
               ┌──────────▼──────────────┐
               │     hub‑server          │
               │  cluster + DB + UI      │
               └─────────────────────────┘
```

---

## 2  |  Crate workspace layout

```
ruid/
│
├─ crates/
│   ├─ ruid-camera/      -- GStreamer/libcamera capture → Stream<DmaBufFrame>
│   ├─ ruid-preprocess/  -- OpenCL+NEON resize+normalize (CPU fallback)
│   ├─ ruid-detect/      -- YOLOv8‑n inference behind a Trait (OnnxRuntime)
│   ├─ ruid-reid/        -- ResNet‑IBN embedding extractor (ORT)
│   ├─ ruid-track/       -- DeepSORT / BYTETrack impl; no ML deps
│   ├─ ruid-proto/       -- tonic‑build generated .proto (gRPC)
│   └─ ruid-utils/       -- tracing, anyhow, small helpers
│
├─ edge-client/          -- CLI binary; ties crates together, CLI flags
└─ hub-server/           -- gRPC + REST + Web UI; stores embeddings & clusters
```

*Each crate ships its own benches under `benches/`, use Criterion.*

---

## 3  |  Hot‑path timing budget (per 640×480 frame)

| Stage                | target ms              | Notes                                      |
| -------------------- | ---------------------- | ------------------------------------------ |
| **Camera → appsink** | 0 (DMA)                | ISR + ISP hidden; no DRAM copy             |
| **Pre‑process**      | **1.0 ms**             | OpenCL kernel (resize+norm)                |
| **YOLOv8‑n INT8**    | 8× 0.8 ms = **6.5 ms** | Batch=1; ORT NNAPI/ACL EP                  |
| **ReID embed**       | **0.9 ms**             | 256‑D vector                               |
| **Tracker**          | **0.2 ms**             | BYTETrack O(N log N)                       |
| **gRPC tx**          | 0.2 ms LAN             | protobuf w/ compression                    |
| **Total**            | **\~9 ms**             | ⇒ 111 FPS theoretical; we cap at FPS=10–30 |

Plenty of margin for occasional GC pauses or kernel misses.

---

## 4  |  Threading & async story

```
+-------------------------------------------------------------- tokio runtime (multi‑thread)
|   async tasks ──────────────────────────────────────────────┐
|   edge-client main()                                        │
|                                                             │
|   CamStream (ReceiverStream) ← mpsc ← capture thread  --\   │
|                                                     DMA‑Buf │
|   preprocess_task  (spawn_blocking for OpenCL) -------------+─ hashes/embeds
|   detector_task    (spawn_blocking for ORT) ---------------/  detections
|   tracker_task     (pure async) ---------------------------→   track msgs
|   grpc_sink        (tonic::channel) ----------------------→   hub‑server
+-------------------------------------------------------------- OS threads
        capture thread (Camera::next_frame_blocking)
        OpenCL driver threads
```

* capture thread = **one** per sensor; constant low CPU (condvar wait)
* heavy compute stages wrapped in `spawn_blocking`: doesn’t starve reactor
* bounded `mpsc` channels everywhere: natural back‑pressure
  (if infer slows, capture stalls upstream queue → sensor free‑runs at vsync)

---

## 5  |  Zero‑copy contracts

| Boundary              | Ownership rule                                                                        |
| --------------------- | ------------------------------------------------------------------------------------- |
| **Camera → Pre‑proc** | `DmaBufFrame` owns FD, dropped after GPU import; no CPU mmap.                         |
| **Pre‑proc → ORT**    | Use `OrtIoBinding` with `OrtMemType::Gpu` when GPU EP available; else staging buffer. |
| **ORT → Tracker**     | Small structs (`Detection{bbox,score,embed}`) – trivially copied.                     |

Every stage *must* document if it duplicates or maps the FD; default is
**no copies**.

---

## 6  |  Observability stack

* `tracing` crate with layer = stdout **+** OTLP gRPC
* Edge exports Prometheus `/metrics` (FPS, queue depth, GPU %)
* Hub scrapes metrics + hosts Grafana dashboard JSON in repo
* Failure modes: queue depth > DEPTH/2 raises WARN; gRPC RTT > 100 ms raises WARN

---

## 7  |  CI / Release flow

```yaml
name: CI

on: [push, pull_request]

jobs:
  build-test:
    strategy:
      matrix:
        target: [x86_64-unknown-linux-gnu, aarch64-unknown-linux-musl]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get install --yes libgstreamer1.0-dev ...
      - run: cargo test --release --workspace --target ${{ matrix.target }}
      - run: cargo build --release --workspace --target ${{ matrix.target }}
      - run: docker buildx build --platform linux/${{ matrix.target }} .
```

Push to `main` triggers:

1. cross‑build static `edge-client`
2. build multi‑arch Docker images:
   `ghcr.io/rrumana/ruid-edge:<git‑sha>` and `:latest`

---

## 8  |  Extensibility scenarios

* **Swap YOLO model** → Add a `rid‑detect‑<name>` crate; feature‑gate via Cargo features.
* **Add Jetson target** → Provide new pre‑process kernel using CUDA; implement `ExecPlan` trait.
* **Edge offline** → Tracker can persist `.rec` files to SD; server replays later.
* **Multiple sensors** → Instantiate `Camera` per `/dev/video*`, merge streams with per‑cam UUID.
* **Privacy mode** → Enable `ruid-preprocess --scramble` that hashes embeddings locally, never sends raw frames.

---

## 9  |  ASCII sequence for one frame

```
Sensor ISR  →  ISP writes NV12 into DMABUF#42  ───────────────┐
                                                         (enqueued to appsink)
capture thread
    pull_sample() → DmaBufFrame{fd=42} ─▶ mpsc ─▶ preprocess task
preprocess
    clImportDMABUF(42) → CLImage         ─▶ kernel → CLBuffer<float32>
    clEnqueueReadBuffer → Host tensor    ─▶ detector task
detector (ORT)
    IO binding in GPU mem → inference → Detection{bbox} list
                                     ─▶ tracker
tracker (BYTE)
    assign ids  → TrackMsg{id,box,embed}  ─▶ gRPC
hub‑server
    match / cluster → REST / WebSocket UI
```

*No DRAM copies until **after** inference if GPU EP used.*

---

## 10  |  How to run end‑to‑end locally (developer checklist)

```bash
# 1. camera preview
gst-launch-1.0 v4l2src ! videoconvert ! autovideosink

# 2. run edge pipeline
cargo run -p edge-client -- --preview --model yolov8n-int8.onnx

# 3. run hub
cargo run -p hub-server -- --ui-port 8080

# 4. open UI
firefox http://localhost:8080
```

Flags:

* `--preview` : enables tee branch
* `--gpu=off` : force CPU fallback for debugging
* `--trace=info` : set global tracing filter

---
