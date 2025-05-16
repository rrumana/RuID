#!/usr/bin/env python3
"""
Download YOLOv8-nano and a ResNet-50 ReID backbone, export the detector
to ONNX via the Ultralytics CLI (default run folder), then copy+quantize
both YOLO and ResNet into ../models/.
"""
import pathlib
import shutil
import subprocess
import glob
import torch
from torchvision import models
from onnxruntime.quantization import quantize_dynamic, QuantType

BASE = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

def find_and_move_yolo():
    # run CLI export
    print("▶ Exporting YOLOv8-nano via `yolo export`…")
    subprocess.run([
        "yolo", "export",
        "model=yolov8n.pt",
        "format=onnx",
        "imgsz=640",
        "simplify=True",
    ], check=True)

    # first, check if yolo placed the ONNX in cwd
    direct = BASE / "yolov8n.onnx"
    if direct.exists():
        src = direct
    else:
        # fallback: runs/export/<run_id>/yolov8n.onnx
        pattern = BASE / "runs" / "export" / "*" / "yolov8n.onnx"
        matches = glob.glob(str(pattern))
        if not matches:
            raise FileNotFoundError(f"no ONNX found matching {pattern} or {direct}")
        src = pathlib.Path(matches[-1])

    dst = MODELS_DIR / "yolov8n_fp32.onnx"
    print(f"  ✓ Moving {src} → {dst}")
    shutil.copy(src, dst)
    # optional cleanup:
    # if direct.exists(): direct.unlink()
    # else: shutil.rmtree(BASE / "runs" / "export")
    return dst

def quantize(in_fp32: pathlib.Path, out_name: str):
    out_int8 = MODELS_DIR / out_name
    print(f"▶ Quantizing {in_fp32.name} → {out_int8.name}")
    quantize_dynamic(str(in_fp32), str(out_int8), weight_type=QuantType.QInt8)
    return out_int8

def export_yolov8():
    fp32 = find_and_move_yolo()
    quantize(fp32, "yolov8n_int8.onnx")
    fp32.unlink()  # remove fp32 if you prefer

def export_resnet_reid():
    print("▶ Building & exporting ResNet-50 ReID backbone…")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 128)
    resnet.eval()

    dummy = torch.randn(1, 3, 128, 64)
    fp32 = MODELS_DIR / "resnet50_market1501_fp32.onnx"
    torch.onnx.export(
        resnet, dummy, fp32,
        export_params=True, opset_version=17,
        input_names=['images'], output_names=['embeddings']
    )
    quantize(fp32, "resnet50_market1501_int8.onnx")
    fp32.unlink()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    print("Starting model export…")
    export_yolov8()
    export_resnet_reid()
    print("\n✅ Done — INT8 ONNX models in", MODELS_DIR)
