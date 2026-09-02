"""INT8 post-training quantization (spec phase-06): BN folding + symmetric RTN.

Order per spec: fold BatchNorm1d into the preceding conv weights/biases, then
quantize weights per-channel-symmetric and activations per-tensor-symmetric
(Round-To-Nearest, min-max calibration). The paper (arXiv:2607.25870, Fig. 4)
shows INT8 RTN is lossless for the pruned model: AUC 0.851 -> 0.851.

The INT8 model is kept as a fake-quant float forward (identical numerics to a
true INT8 kernel up to fp32 accumulation); the int8 weights + scales saved
here are exactly what the C header / phase-8 engine consumes.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Activation quant points in forward order (per-tensor, symmetric). Each
# name is the input to the next weight-carrying op; "add" is the residual
# sum feeding conv4_dw, "gap" feeds the classifier.
ACT_POINTS = (
    "x", "adapter", "conv0_dw", "conv0_pw", "block1", "block2",
    "subA_dw", "subA_pw", "subC_dw", "subC_pw", "skip", "add",
    "conv4_dw", "conv4_pw", "conv5", "gap",
)
QMAX = 127  # symmetric range [-127, 127]


class FoldedPulseVAD(nn.Module):
    """PulseVAD with BatchNorm folded into conv weights+bias.

    Logits-identical to the BN version in eval mode; BN-free graph exports to
    plain Conv/Relu/Mean/Gemm ops (TFLM-compatible, no folding at runtime).
    """

    def __init__(self, dims: dict | None = None) -> None:
        """dims: layer widths; defaults are the unpruned 81k model."""
        super().__init__()
        d = {"adapter": 128, "conv0_pw": 128, "b1": 64, "b2": 64, "b3": 64,
             "c4": 64, "p4": 128, "c5": 128} | (dims or {})
        self.adapter = nn.Conv1d(64, d["adapter"], 1, bias=True)
        self.conv0_dw = nn.Conv1d(d["adapter"], d["adapter"], 11, padding=5,
                                  groups=d["adapter"], bias=False)
        self.conv0_pw = nn.Conv1d(d["adapter"], d["conv0_pw"], 1, bias=True)
        self.block1 = nn.Conv1d(d["conv0_pw"], d["b1"], 1, bias=True)
        self.block2 = nn.Conv1d(d["b1"], d["b2"], 1, bias=True)
        self.subA_dw = nn.Conv1d(d["b2"], d["b3"], 17, padding=8,
                                 groups=d["b2"], bias=False)
        self.subA_pw = nn.Conv1d(d["b3"], d["b3"], 1, bias=True)
        self.subC_dw = nn.Conv1d(d["b3"], d["b3"], 17, padding=8,
                                 groups=d["b3"], bias=False)
        self.subC_pw = nn.Conv1d(d["b3"], d["b3"], 1, bias=True)
        self.skip = nn.Conv1d(d["b2"], d["b3"], 1, bias=True)
        self.conv4_dw = nn.Conv1d(d["b3"], d["c4"], 29, dilation=2, padding=28,
                                  groups=d["b3"], bias=False)
        self.conv4_pw = nn.Conv1d(d["c4"], d["p4"], 1, bias=True)
        self.conv5 = nn.Conv1d(d["p4"], d["c5"], 1, bias=True)
        self.classifier = nn.Linear(d["c5"], 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.adapter(x))
        x = self.conv0_dw(x)
        x = F.relu(self.conv0_pw(x))
        x = F.relu(self.block1(x))
        x = F.relu(self.block2(x))
        main = F.relu(self.subA_pw(self.subA_dw(x)))
        main = self.subC_pw(self.subC_dw(main))
        x = F.relu(main + self.skip(x))
        x = self.conv4_dw(x)
        x = F.relu(self.conv4_pw(x))
        x = F.relu(self.conv5(x))
        x = x.mean(dim=-1)  # global average pool == AdaptiveAvgPool1d(1) squeeze
        return self.classifier(x)


_BN_FOLD_PAIRS = [
    ("adapter", "adapter"), ("conv0_pw", "conv0_pw"), ("block1", "block1"),
    ("block2", "block2"), ("subA_pw", "block3.subA_pw"),
    ("subC_pw", "block3.subC_pw"), ("skip", "block3.skip"),
    ("conv4_pw", "conv4_pw"), ("conv5", "conv5"),
]


def fold_batchnorm(model) -> FoldedPulseVAD:
    """W' = W * gamma/sqrt(var+eps), b' = beta - mu * gamma/sqrt(var+eps).

    Works on any width (unpruned 81k or pruned 2.1k student); the folded
    model's dims are read from the source model's modules.
    """
    dims = {
        "adapter": model.adapter.conv.out_channels,
        "conv0_pw": model.conv0_pw.conv.out_channels,
        "b1": model.block1.conv.out_channels,
        "b2": model.block2.conv.out_channels,
        "b3": model.block3.subA_dw.out_channels,
        "c4": model.conv4_dw.out_channels,
        "p4": model.conv4_pw.conv.out_channels,
        "c5": model.conv5.conv.out_channels,
    }
    folded = FoldedPulseVAD(dims).eval()
    with torch.no_grad():
        for dst, src in _BN_FOLD_PAIRS:
            cbn = model.get_submodule(src)
            conv, bn = cbn.conv, cbn.bn
            gain = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            getattr(folded, dst).weight.copy_(conv.weight * gain.view(-1, 1, 1))
            getattr(folded, dst).bias.copy_(bn.bias - bn.running_mean * gain)
        folded.conv0_dw.weight.copy_(model.conv0_dw.weight)
        folded.conv4_dw.weight.copy_(model.conv4_dw.weight)
        folded.classifier.weight.copy_(model.classifier.weight)
        folded.classifier.bias.copy_(model.classifier.bias)
    return folded


def qdq(t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest quantize-dequantize (fake quant)."""
    return torch.clamp(torch.round(t / scale), -QMAX, QMAX) * scale


def weight_scales(model: nn.Module, bits: int = 8) -> dict[str, torch.Tensor]:
    """Symmetric per-output-channel weight scales: s = max|w| / (2^(b-1) - 1)."""
    scales = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, (nn.Conv1d, nn.Linear)):
                w = mod.weight
                dims = tuple(range(1, w.dim()))
                scales[name] = w.abs().amax(dim=dims) / (2 ** (bits - 1) - 1)
    return scales


def fake_quant_weights(model: nn.Module, scales: dict[str, torch.Tensor]) -> None:
    """Destructively replace weights with their int8-roundtripped values."""
    with torch.no_grad():
        for name, mod in model.named_modules():
            if name in scales:
                s = scales[name].view(-1, *([1] * (mod.weight.dim() - 1)))
                mod.weight.copy_(qdq(mod.weight, s))


class Int8PulseVAD(FoldedPulseVAD):
    """Folded model with int8 fake-quant weights + per-tensor activation Q/DQ."""

    def __init__(self) -> None:
        super().__init__()
        self.act_scales: dict[str, torch.Tensor] = {}
        self._qact = False
        self._record: dict[str, torch.Tensor] | None = None

    def calibrate(self, batches) -> dict[str, float]:
        """Min-max activation calibration over `batches` (train data)."""
        assert not self._qact, "calibrate before quantizing activations"
        self._record = {}
        with torch.no_grad():
            for x in batches:
                self(x)
        scales = {k: v.item() for k, v in self._record.items()}
        missing = set(ACT_POINTS) - set(scales)
        assert not missing, f"calibration missed activation points: {missing}"
        self.act_scales = {k: torch.tensor(v) for k, v in scales.items()}
        self._record = None
        return scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def q(t: torch.Tensor, name: str) -> torch.Tensor:
            if self._record is not None:
                prev = self._record.get(name)
                m = t.detach().abs().max()
                self._record[name] = m if prev is None else torch.maximum(prev, m)
            if self._qact:
                t = qdq(t, self.act_scales[name])
            return t

        a = q(x, "x")
        a = q(F.relu(self.adapter(a)), "adapter")
        a = q(self.conv0_dw(a), "conv0_dw")
        a = q(F.relu(self.conv0_pw(a)), "conv0_pw")
        a = q(F.relu(self.block1(a)), "block1")
        a = q(F.relu(self.block2(a)), "block2")
        res = a  # skip branch consumes the block2 output (as in PulseVAD.block3)
        a = q(self.subA_dw(a), "subA_dw")
        a = q(F.relu(self.subA_pw(a)), "subA_pw")
        a = q(self.subC_dw(a), "subC_dw")
        main = q(self.subC_pw(a), "subC_pw")
        skip = q(self.skip(res), "skip")
        a = q(F.relu(main + skip), "add")
        a = q(self.conv4_dw(a), "conv4_dw")
        a = q(F.relu(self.conv4_pw(a)), "conv4_pw")
        a = q(F.relu(self.conv5(a)), "conv5")
        a = q(a.mean(dim=-1), "gap")
        return self.classifier(a)


def save_int8(model: Int8PulseVAD, path: Path, meta: dict) -> None:
    """Persist true int8 weights + scales (what the C engine loads)."""
    layers = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv1d, nn.Linear)):
            s = weight_scales(model)[name]
            w = mod.weight.detach()
            q = torch.clamp(torch.round(w / s.view(-1, *([1] * (w.dim() - 1)))),
                            -QMAX, QMAX).to(torch.int8)
            layers[name] = {"weight": q, "scale": s}
    torch.save({"layers": layers, "act_scales": model.act_scales, "meta": meta}, path)


def write_c_header(model: Int8PulseVAD, path: Path, macro: str = "PULSEVAD_WEIGHTS_H") -> None:
    """Generate pulsevad_weights.h: int8 weight arrays + scales as C literals."""
    lines = [
        f"#ifndef {macro}", f"#define {macro}", "",
        "/* Auto-generated by pulsevad.quantize -- do not edit. */",
        "#include <stdint.h>", "",
    ]
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.Conv1d, nn.Linear)):
            continue
        w = mod.weight.detach()
        s = weight_scales(model)[name]
        cname = name.replace(".", "_").upper()
        flat = torch.clamp(torch.round(w / s.view(-1, *([1] * (w.dim() - 1)))),
                           -QMAX, QMAX).to(torch.int8).reshape(-1).tolist()
        lines.append(f"/* {name}: shape {list(w.shape)}, per-channel scale */")
        lines.append(f"static const int8_t W_{cname}[{len(flat)}] = {{")
        for i in range(0, len(flat), 20):
            lines.append("  " + ",".join(str(v) for v in flat[i:i + 20]) + ",")
        lines.append("};")
        lines.append(
            f"static const float S_{cname}[{s.numel()}] = {{"
            + ",".join(f"{v:.8g}f" for v in s.tolist()) + "};"
        )
        lines.append("")
    lines.append(f"#endif /* {macro} */")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
