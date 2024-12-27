# RuID
A Rust-based extension of a research project I took part in during EE292D Edge ML class at Stanford

My goals for this project are the following:
  - Recreate Raspberry Pi friendly quantization of SOTA vision model like Yolo.
  - Train Model for person Re-identifiaction using ReID dataset like Market1501.
  - Figure out how to run model efficiently on Rust, cross-compile with optimizations for Raspberry Pi
  - Create a client program to extract feature embedding for each identified person.
  - Create a server program to recieve all feature embeddings and keep track of individuals.
  -   Applying techniques like clustering to generate an approximate representation of each individual
  - Scale server program to work with an arbitrary number of clients.
  - Iterate training, quantization, and clustering techniques to achieve better performance.
  - Explore SIMD NEON optimizations for Raspberry Pi.
  - Containerize for easy deployment.
  - Add features as needed.

# Completion
I have made a program that takes camera input using libcamera and feeds those frames into Yolov11-nano. Since the Tract runtime is not threaded, I have added a basic threading implementation using some threads and channels. Lots more can be done to optimize this.

I am getting about 20FPS on my laptop at about 100% CPU utilization before quantization or any other optiizations. This seems promising, but I am planning on adding support for onnx runtime to comapre. I will add a command line flag to select your runtime, that way I can compare Trace + threading to Onnxruntime. If it turns out that I cannot beat Onnxruntime, I will end up using that.

Once I have that figured out I will repeat the process for Resnet-50 and then look into quantization.

# Things that need fixing
Pretty much everything. I need to look into more efficient ways to do image transformations. Hopefully there is an easy way to do it on the Raspberry Pi GPU. I also need to ensure my model and my camera input match. My laptop camera does not seem to support 640x640 so I am scaling a 1920x1080 image down to 640x640. This is not only inefficient but scaling away from native will hurt visual fidelity and therefore accuracy.

# Crates to read:
  - Machine Learning: burn, tflite-rs or onnxruntime-rs depending on which is better at the time.
  - Image Processing: v4l, image, ndarray
  - Config + Args: clap, serde, toml
  - Error Handling: anyhow, thiserror
  - Logging: log, env_logger, log4rs
  - Web Server: hypr or warp

# FAQ

### Why am I doing this project?
  - I want to
  - That is all

### When can I expect updates?
  - I currently have no set timeline for this project.
  - Keeping my day job and touching grass come first.

### Why Rust specifically?
  - I want to challenge myself to learn Rust better, this is a great way to force myself out of my comfort zone.
  - I want to evaluate the state of Rust ML libraries to see what is good, what is missing, and where I could potentially contribute in the future.
  - Rust is a fantastic language for making performant, memory safe programs.
  - I like it.

### Anything else?
  - No
  - If something comes up I'll add an addendum somewhere in this README so people can see. I doubt that this will be getting much traffic but oh well.

### Can I use this project?
  - I, personally, do not care. You can fork, clone, or modify my code in this repository as you see fit.
  - I will create a licensing section for this project when the time comes since I will be using a premade model, and that will likely bring some restrictions of its own. Yolov8 (The model used in the original incarnation of this project) through Ultralytics is under AGPL3.0 licensing, which is somewhat limiting.
