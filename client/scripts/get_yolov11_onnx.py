import torch
from ultralytics import YOLO
import os

def download_and_convert_yolov11_to_onnx(
        output_dir="../models", 
        onnx_filename="yolov11.onnx"
):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Download the YOLOv11 model
    print("Downloading YOLOv11 model...")
    model = YOLO("../models/yolo11n.pt")  # Automatically downloads YOLOv11 weights from Ultralytics

    # Convert the model to ONNX format
    onnx_path = os.path.join(output_dir, onnx_filename)
    print("Converting YOLOv11 to ONNX format...")
    
    dummy_input = torch.randn(1, 3, 480, 640)  # Example input size for YOLO models
    model.model.eval()  # Ensure the model is in evaluation mode
    
    torch.onnx.export(
        model.model,              # PyTorch model
        dummy_input,              # Dummy input for tracing
        onnx_path,                # Output ONNX file path
        export_params=True,       # Store trained parameters in the ONNX model
        do_constant_folding=True, # Optimize constant folding
        input_names=["input"],   # Input tensor name
        output_names=["output"], # Output tensor name
        opset_version=19,         # ONNX opset version
    )

    print(f"ONNX model saved to {onnx_path}")

if __name__ == "__main__":
    download_and_convert_yolov11_to_onnx()
