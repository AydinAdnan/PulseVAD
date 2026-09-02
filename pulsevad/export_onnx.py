"""ONNX export (spec phase-06 step 6.2): FP32 + INT8 QDQ, parity-verified."""

from pathlib import Path

import numpy as np
import torch


def export_onnx(model, path: Path, opset: int = 17) -> Path:
    """Export a folded (or fake-quant) model to static ONNX, batch-dynamic."""
    model = model.eval()
    torch.onnx.export(
        model,
        torch.randn(1, 64, 21),
        str(path),
        input_names=["log_mel"],
        output_names=["logits"],
        dynamic_axes={"log_mel": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,  # legacy TorchScript exporter: no onnxscript dep
    )
    return path


def onnx_session(path: Path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def session_logits(sess, features: np.ndarray, batch: int = 4096) -> np.ndarray:
    out = [sess.run(["logits"], {"log_mel": features[i:i + batch]})[0]
           for i in range(0, len(features), batch)]
    return np.concatenate(out)


class _CalibReader:
    """Minimal CalibrationDataReader over precomputed feature windows."""

    def __init__(self, arrays, batch: int = 512):
        self._it = None
        self._arrays, self._batch = arrays, batch
        self.rewind()

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        def gen():
            for feats in self._arrays:
                for i in range(0, len(feats), self._batch):
                    yield {"log_mel": feats[i:i + self._batch].astype(np.float32)}
        self._it = iter(gen())


def quantize_onnx_int8(fp32_path: Path, int8_path: Path, calib_features) -> Path:
    """Static QDQ quantization via ORT: per-channel int8 weights, per-tensor
    symmetric int8 activations, calibrated on real features."""
    from onnxruntime.quantization import (QuantFormat, QuantType, quantize_static)
    quantize_static(
        str(fp32_path), str(int8_path),
        _CalibReader(calib_features),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        extra_options={"ActivationSymmetric": True},
    )
    return int8_path


def verify_parity(torch_model, onnx_path: Path, x: torch.Tensor, atol: float = 1e-4) -> float:
    """Max |torch - onnxruntime| logits; spec gate: atol=1e-4."""
    with torch.no_grad():
        ref = torch_model(x).numpy()
    got = onnx_session(onnx_path).run(["logits"], {"log_mel": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    assert diff < atol, f"ONNX parity failed: max diff {diff:.2e} >= {atol:.0e}"
    return diff
