"""
Export the trained slang detector to ONNX Runtime format.

Run this only after the PyTorch detector meets accuracy targets:
  python scripts/export_detector_onnx.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer


def export_detector(model_dir: Path, output_dir: Path, quantize: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    ort_model = ORTModelForSequenceClassification.from_pretrained(model_dir, export=True)
    ort_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if quantize:
        quantizer = ORTQuantizer.from_pretrained(output_dir)
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)

    print(f"Exported detector ONNX artifacts to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export slang detector to ONNX.")
    parser.add_argument("--model-dir", default="models/slang_detector")
    parser.add_argument("--output-dir", default="models/slang_detector_onnx")
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()
    export_detector(Path(args.model_dir), Path(args.output_dir), not args.no_quantize)


if __name__ == "__main__":
    main()
