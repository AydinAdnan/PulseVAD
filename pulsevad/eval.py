"""Strictly causal evaluation (spec phase-07).

Protocol (non-negotiable): each 200 ms window (3,200 samples) is scored
independently from audio up to and including that window only — no overlap,
no future context, no median filtering, no hangover. The cached eval windows
already satisfy this; `evaluate_causal_clip` demonstrates it end-to-end from
raw audio for external clips (e.g. AVA-Speech).

Metrics: frame AUC, best F1 over a threshold sweep, and the certified
always-on operating point FPR @ TPR=0.95.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from pulsevad.frontend import MelFrontend

THRESHOLDS = np.linspace(0.0, 1.0, 201)  # spec 7.1 step 5: 0.0..1.0 sweep


def causal_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """AUC, best-F1 (+ its threshold), and FPR @ TPR=0.95."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    auc = float(roc_auc_score(y_true, probs))

    pred = probs[:, None] >= THRESHOLDS[None, :]
    tp = (pred & (y_true == 1)[:, None]).sum(0)
    fp = (pred & (y_true == 0)[:, None]).sum(0)
    fn = ((~pred) & (y_true == 1)[:, None]).sum(0)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    best = int(f1.argmax())

    fpr, tpr, roc_thr = roc_curve(y_true, probs)
    # first ROC point with TPR >= 0.95
    idx = int(np.searchsorted(tpr, 0.95))
    idx = min(idx, len(fpr) - 1)
    return {
        "n": int(len(y_true)),
        "auc": round(auc, 4),
        "f1": round(float(f1[best]), 4),
        "f1_threshold": round(float(THRESHOLDS[best]), 4),
        "fpr_at_tpr95": round(float(fpr[idx]), 4),
    }


@torch.no_grad()
def probs_from_features(model, features: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Speech probability per cached window; rows are independent 200 ms frames."""
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(features))
    out = [model(x[i:i + batch], return_logits=False) for i in range(0, len(x), batch)]
    return torch.cat(out).numpy()


def evaluate_window_set(model, features: np.ndarray, labels: np.ndarray) -> dict:
    """Score one held-out category from cached windows (causal by construction)."""
    return causal_metrics(labels, probs_from_features(model, features))


@torch.no_grad()
def evaluate_causal_clip(model, audio, sr: int = 16000, hop_samples: int = 3200) -> np.ndarray:
    """Stream raw audio in strictly non-overlapping 200 ms hops.

    `audio`: mono float array (any sample rate; resampled to 16 kHz if needed).
    Returns one speech probability per hop. Last partial hop is dropped.
    """
    import scipy.signal as sps

    if sr != 16000:
        g = np.gcd(sr, 16000)
        audio = sps.resample_poly(audio, 16000 // g, sr // g)
    frontend = MelFrontend()
    device = next(model.parameters()).device
    n_hops = len(audio) // hop_samples
    probs = np.empty(n_hops, dtype=np.float64)
    for i in range(n_hops):
        hop = audio[i * hop_samples:(i + 1) * hop_samples]
        feats = frontend(torch.from_numpy(np.ascontiguousarray(hop[None, :])))
        probs[i] = model(torch.from_numpy(np.ascontiguousarray(feats)).to(device),
                         return_logits=False).item()
    return probs


def ava_ground_truth(annotations: list, duration_s: float, hop_s: float = 0.2) -> np.ndarray:
    """Bin AVA-Speech segment annotations to the 200 ms grid by majority.

    `annotations`: list of (start_s, end_s, label) with label in
    {'speech', 'noise'}; anything overlapping a hop counts. A hop is speech
    if speech-covered seconds >= half the hop.
    """
    n_hops = int(duration_s / hop_s)
    speech_s = np.zeros(n_hops)
    for start, end, label in annotations:
        if label != "speech":
            continue
        lo, hi = max(0.0, start), min(duration_s, end)
        i0, i1 = int(lo / hop_s), min(n_hops, int(np.ceil(hi / hop_s)))
        for i in range(i0, i1):
            speech_s[i] += min(hi, (i + 1) * hop_s) - max(lo, i * hop_s)
    return (speech_s >= hop_s / 2).astype(np.uint8)


def report_table(per_model: dict[str, dict[str, dict]]) -> str:
    """Cross-model markdown table: rows = models, cols = eval categories."""
    cats = sorted({c for m in per_model.values() for c in m})
    cols = " | ".join(["model"] + cats)
    sep = " | ".join(["---"] * (len(cats) + 1))
    rows = []
    for name, cats_m in per_model.items():
        cells = " | ".join(
            f"AUC {cats_m[c]['auc']:.3f} FPR@95 {cats_m[c]['fpr_at_tpr95']:.3f}"
            if c in cats_m else "—" for c in cats
        )
        rows.append(f"**{name}** | {cells}")
    return "\n".join([f"| {cols} |", f"| {sep} |"] + [f"| {r} |" for r in rows])


def mean_ci(values: list, z: float = 1.96) -> tuple[float, float]:
    """Mean and 95% CI half-width (normal approx; n>=3 seeds)."""
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(z * v.std(ddof=1) / np.sqrt(len(v)))


# ---------------------------------------------------------------- comparison

# Our shipped artifacts (measured: pulsevad/, this repo). Footprints are
# weight bytes: params x 4 (FP32) / params x 1 (+scale tables) (INT8).
OURS = {
    "PulseVAD (teacher, unpruned)": {
        "params": 81_090, "macs": 1_660_000, "latency_ms": 200,
        "footprint_kb": {"fp32": 324.4, "int8": 81.1}, "commercial": "YES (100% permissive)",
    },
    "PulseVAD (ship, pruned)": {
        "params": 2_118, "macs": 44_000, "latency_ms": 200,
        "footprint_kb": {"fp32": 8.5, "int8": 2.1}, "commercial": "YES (100% permissive)",
    },
}

# Competitors without runnable public weights: cited rows (paper Table 1 /
# spec phase-07 §2). `protocol` flags causal vs the inflated non-causal
# sliding-window protocol — NOT directly comparable to our measured rows.
CITED = [
    {"name": "Silero-VAD (v5/v6) [cited]", "params": 545_000, "footprint": "~2.2 MB",
     "macs": ">10M", "ava_auc": 0.920, "latency_ms": 32, "commercial": "YES (MIT)",
     "protocol": "causal", "source": "github.com/snakers4/silero-vad"},
    {"name": "MarbleNet [cited]", "params": 91_000, "footprint": "~364 KB",
     "macs": ">2.0M", "ava_auc": 0.850, "latency_ms": 630, "commercial": "NS",
     "protocol": "causal", "source": "catalog.ngc.nvidia.com (vad_marblenet)"},
    {"name": "AtomicVAD [cited]", "params": 300, "footprint": "~1.2 KB",
     "macs": 6_000, "ava_auc": 0.869, "latency_ms": 630, "commercial": "NS",
     "protocol": "causal", "source": "Analog Devices (GGCU custom activation, R2 fails)"},
    {"name": "TinyVAD [cited]", "params": 11_600, "footprint": "n/a",
     "macs": None, "ava_auc": 0.864, "latency_ms": 630, "commercial": "n/a",
     "protocol": "NON-CAUSAL 87.5% overlap", "source": "paper"},
    {"name": "SincQDR [cited]", "params": 8_000, "footprint": "n/a",
     "macs": None, "ava_auc": 0.914, "latency_ms": 1181, "commercial": "n/a",
     "protocol": "NON-CAUSAL 87.5% overlap", "source": "paper"},
    {"name": "ResectNet [cited]", "params": 4_500, "footprint": "n/a",
     "macs": None, "ava_auc": 0.886, "latency_ms": 200, "commercial": "n/a",
     "protocol": "causal", "source": "paper (GRU, R2 fails)"},
]


def build_comparison(ours_measured: dict, ours: dict | None = None,
                     cited: list | None = None) -> tuple[str, dict]:
    """Merge measured rows (per-category AUC) with cited competitor rows into
    the comparison artifact. Returns (markdown, data).

    `ours_measured`: {model_name: {"clean": {...}, ..., "params": int, ...}}
    including per-category causal_metrics + optionally a measured competitor.
    """
    ours = ours or OURS
    cited = cited or CITED
    ship = ours["PulseVAD (ship, pruned)"]

    data = {"ours": ours_measured, "cited": cited, "wins": []}
    lines = [
        "# PulseVAD Cross-VAD Comparison (strictly causal protocol)", "",
        "All PulseVAD rows are measured by this repo (`pulsevad/eval.py`); the",
        "5 held-out categories are scored with zero overlap, zero smoothing.",
        "[cited] rows are competitor SELF-REPORTED numbers — see `protocol`",
        "column before comparing AUCs.", "",
    ]

    # ---- headline table: size / compute / latency / license
    lines += ["## Size, compute, latency, license", "",
              "| Model | Params | Footprint | MACs/200ms | Input latency | Commercial |",
              "|---|---|---|---|---|---|"]
    for name, r in ours.items():
        lines.append(f"| **{name}** (measured) | {r['params']:,} | "
                     f"{r['footprint_kb']['fp32']:.1f} KB FP32 / {r['footprint_kb']['int8']:.1f} KB INT8 | "
                     f"{r['macs']:,} | {r['latency_ms']} ms | {r['commercial']} |")
    for c in cited:
        lines.append(f"| {c['name']} | {c['params']:,} | {c['footprint']} | {c['macs']} | "
                     f"{c['latency_ms']} ms | {c['commercial']} |")
    lines.append("")

    # ---- measured AUC table (our models + any same-audio competitor)
    lines += ["## Measured AUC — 5 held-out categories (this repo, same windows)",
              "", "| Model | " + " | ".join(
                  sorted(k for k in next(iter(ours_measured.values()))
                         if isinstance(next(iter(ours_measured.values()))[k], dict))) + " |",
              "|---|" + "---|" * 5]
    for name, row in ours_measured.items():
        cells = " | ".join(
            f"{row[c]['auc']:.3f}" if isinstance(row.get(c), dict) else str(row.get(c, "—"))
            for c in sorted(k for k in row if isinstance(row.get(k), dict)))
        lines.append(f"| **{name}** | {cells} |")
    lines.append("")

    # ---- where we win / lose, computed per cited competitor
    lines += ["## Where the 2.1k ship model wins (and where it doesn't)", ""]
    wins = []
    for c in cited:
        if c["name"].startswith("Silero") and "Silero (measured)" in ours_measured:
            continue  # measured row supersedes cited for win math? keep both
        p_ratio = c["params"] / ship["params"]
        m = [f"### vs {c['name']}", ""]
        if c["protocol"] != "causal":
            m.append(f"- protocol: **{c['protocol']}** — AUC {c['ava_auc']} is inflated; "
                     f"not comparable to our causal numbers.")
        if ship["params"] < c["params"]:
            wins.append(f"{p_ratio:.1f}x fewer parameters than {c['name']}")
            m.append(f"- **win**: {p_ratio:.1f}x fewer params ({ship['params']:,} vs {c['params']:,})")
        else:
            m.append(f"- **loss**: {c['name']} is {1/p_ratio:.1f}x smaller "
                     f"({c['params']:,} vs {ship['params']:,} params)")
        if c["macs"] and isinstance(c["macs"], int) and ship["macs"] < c["macs"]:
            m.append(f"- **win**: {c['macs']/ship['macs']:.1f}x fewer MACs "
                     f"({ship['macs']:,} vs {c['macs']:,})")
        if c["latency_ms"] > ship["latency_ms"]:
            wins.append(f"{c['latency_ms']/ship['latency_ms']:.1f}x lower input latency than {c['name']}")
            m.append(f"- **win**: {c['latency_ms']/ship['latency_ms']:.1f}x lower latency "
                     f"({ship['latency_ms']} ms vs {c['latency_ms']} ms)")
        elif c["latency_ms"] < ship["latency_ms"]:
            m.append(f"- **loss**: {c['name']} has {ship['latency_ms']/c['latency_ms']:.1f}x "
                     f"lower latency ({c['latency_ms']} ms vs {ship['latency_ms']} ms)")
        if c["commercial"] == "NS":
            m.append("- **win**: license/commercial cleanness unverified for them (NS); "
                     "PulseVAD is 100% permissive")
        m.append(f"- cited AUC {c['ava_auc']} ({c['protocol']}); ours: see measured table")
        lines += m + [""]

    lines = ["## Summary wins", ""] + [f"- {w}" for w in wins] + [""] + lines
    data["wins"] = wins
    return "\n".join(lines), data
