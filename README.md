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
  - Containerize for easy deployment.
  - Add features as needed.

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
