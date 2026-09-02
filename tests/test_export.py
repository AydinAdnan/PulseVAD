"""Spec phase-06 gates: BN-fold exactness, INT8 RTN, ONNX parity, C header."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pulsevad.model import PulseVAD
from pulsevad.quantize import (Int8PulseVAD, fake_quant_weights, fold_batchnorm,
                               weight_scales, write_c_header)
from pulsevad.export_onnx import export_onnx, verify_parity


def _folded_pair(seed=0):
    """Model with realistic trained BN buffers (non-default mean/var) — default
    0/1 buffers let fold bugs hide because random-init logits are tiny."""
    torch.manual_seed(seed)
    model = PulseVAD().eval()
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.running_mean.uniform_(-0.5, 0.5)
                mod.running_var.uniform_(0.05, 4.0)
                mod.weight.uniform_(0.5, 1.5)
                mod.bias.uniform_(-0.2, 0.2)
    return model, fold_batchnorm(model)


def test_fold_is_logit_exact():
    model, folded = _folded_pair()
    x = torch.randn(8, 64, 21)
    with torch.no_grad():
        diff = (model(x) - folded(x)).abs().max().item()
    assert diff < 1e-4, f"BN folding changed logits by {diff:.2e}"


def test_int8_weights_and_calibration():
    _, folded = _folded_pair()
    q = Int8PulseVAD().eval()
    q.load_state_dict(folded.state_dict())
    # calibrate on synthetic data, then fake-quant weights
    batches = [torch.randn(16, 64, 21) for _ in range(6)]
    scales = q.calibrate(batches)
    assert set(scales) == set(q.act_scales)
    assert all(v > 0 for v in scales.values())
    fake_quant_weights(q, weight_scales(q))
    # every surviving weight must be an exact multiple of its channel scale
    for name, mod in q.named_modules():
        if isinstance(mod, (torch.nn.Conv1d, torch.nn.Linear)):
            s = weight_scales(q)[name].view(-1, *([1] * (mod.weight.dim() - 1)))
            ratio = mod.weight / s
            assert torch.allclose(ratio, ratio.round(), atol=1e-4), name


def test_int8_passthrough_is_logit_exact():
    """With activation Q/DQ off and unquantized weights, Int8PulseVAD's
    hand-written forward must reproduce FoldedPulseVAD exactly."""
    _, folded = _folded_pair()
    q = Int8PulseVAD().eval()
    q.load_state_dict(folded.state_dict())
    q._qact = False
    x = torch.randn(8, 64, 21)
    with torch.no_grad():
        diff = (folded(x) - q(x)).abs().max().item()
    assert diff < 1e-6, f"Int8 forward graph diverges from folded: {diff:.2e}"


def test_int8_logits_close_to_fp32():
    _, folded = _folded_pair()
    q = Int8PulseVAD().eval()
    q.load_state_dict(folded.state_dict())
    q.calibrate([torch.randn(32, 64, 21) for _ in range(4)])
    fake_quant_weights(q, weight_scales(q))
    x = torch.randn(32, 64, 21)
    with torch.no_grad():
        diff = (folded(x) - q(x)).abs().max().item()
    assert diff < 0.5, f"INT8 fake-quant drifted {diff:.3f} logits on random net"


def test_onnx_fp32_parity(tmp_path):
    _, folded = _folded_pair()
    path = export_onnx(folded, tmp_path / "p.onnx")
    diff = verify_parity(folded, path, torch.randn(4, 64, 21))
    assert diff < 1e-4


def test_c_header_generated(tmp_path):
    _, folded = _folded_pair()
    q = Int8PulseVAD().eval()
    q.load_state_dict(folded.state_dict())
    q.calibrate([torch.randn(8, 64, 21) for _ in range(2)])
    out = tmp_path / "pulsevad_weights.h"
    write_c_header(q, out)
    text = out.read_text(encoding="ascii")
    assert text.startswith("#ifndef PULSEVAD_WEIGHTS_H")
    assert "static const int8_t W_ADAPTER" in text
    assert "static const float S_ADAPTER" in text
    assert text.rstrip().endswith("#endif /* PULSEVAD_WEIGHTS_H */")
    # adapter weight array holds 64*128 values
    import re
    m = re.search(r"W_ADAPTER\[(\d+)\]", text)
    assert int(m.group(1)) == 64 * 128
