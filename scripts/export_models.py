#!/usr/bin/env python3
"""
export_models.py  –  RuID v2 model fetch + quantise

1. Downloads **YOLO 11-nano** (`yolo11n.pt`) via the Ultralytics CLI and
   exports a 224×224 ONNX graph.
2. Builds a static-quant (**QOperator format**) INT8 model that Tract can load.
3. Exports a ResNet-50 Re-ID backbone and quantises it likewise.

Resulting models are written to  ./models/
"""

from __future__ import annotations
import subprocess, pathlib, shutil, glob, numpy as np, torch
from torchvision import models
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    CalibrationMethod,
    quantize_static,
    QuantizationMode,
)

BASE        = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR  = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ──────────────────────────  Calibration helper  ──────────────────────────
class RandomCalibrationReader(CalibrationDataReader):
    """Feeds a single random tensor into quantize_static as calibration data."""

    def __init__(self, input_name: str, shape=(1, 3, 640, 640)):
        self.input_name = input_name
        self.shape      = shape
        self._done      = False

    def get_next(self):
        if self._done:
            return None
        rnd = np.random.rand(*self.shape).astype(np.float32)
        self._done = True
        return {self.input_name: rnd}


# ──────────────────────────  YOLO 11 export  ──────────────────────────────
def find_and_export_yolo11() -> pathlib.Path:
    print("▶ Exporting YOLO11-nano via `yolo export` (224×224)…")
    subprocess.run(
        [
            "yolo",
            "export",
            "model=yolo11n.pt",
            "format=onnx",
            "imgsz=640x480",
            "simplify=True",
        ],
        check=True,
    )

    # Candidate 1: written into the script dir (common when you run from there)
    cand1 = SCRIPTS_DIR / "yolo11n.onnx"
    if cand1.exists():
        return cand1

    # Candidate 2: workspace root
    cand2 = BASE / "yolo11n.onnx"
    if cand2.exists():
        return cand2

    # Fallback: runs/export/*/yolo11n.onnx
    pattern = BASE / "runs" / "export" / "*" / "yolo11n.onnx"
    matches = glob.glob(str(pattern))
    if matches:
        return pathlib.Path(matches[-1])

    raise FileNotFoundError(
        f"Could not find yolo11n.onnx in {cand1}, {cand2}, or {pattern}"
    )


def quantise_static(
    in_model: pathlib.Path,
    out_name: str,
    input_name: str,
    calib_shape=(1, 3, 224, 224),
):
    out_model = MODELS_DIR / out_name
    print(f"▶ Static-quantising {in_model.name} → {out_model.name}")

    reader = RandomCalibrationReader(input_name=input_name, shape=calib_shape)

    quantize_static(
        model_input=str(in_model),
        model_output=str(out_model),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,  # fully fused INT8 ops
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv", "MatMul"],  # ← keep Sigmoid FP32
    )
    return out_model


def export_yolo11():
    fp32_src = find_and_export_yolo11()
    fp32_dst = MODELS_DIR / "yolo11n_fp32.onnx"
    shutil.copy(fp32_src, fp32_dst)

    quantise_static(fp32_dst, "yolo11n_int8.onnx", input_name="images")
    fp32_dst.unlink()


# ──────────────────────────  ResNet-50 Re-ID  ─────────────────────────────
def export_resnet_reid():
    print("▶ Building & exporting ResNet-50 Re-ID backbone…")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 128)
    resnet.eval()

    dummy = torch.randn(1, 3, 128, 64)
    fp32 = MODELS_DIR / "resnet50_market1501_fp32.onnx"
    torch.onnx.export(
        resnet,
        dummy,
        fp32,
        export_params=True,
        opset_version=17,
        input_names=["images"],
        output_names=["embeddings"],
    )

    quantise_static(fp32, "resnet50_market1501_int8.onnx", input_name="images", calib_shape=(1, 3, 128, 64))
    fp32.unlink()


# ──────────────────────────  main  ─────────────────────────────
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    print("Starting model export…")
    export_yolo11()
    export_resnet_reid()
    print(f"\n✅ Done — INT8 ONNX models in {MODELS_DIR}")
