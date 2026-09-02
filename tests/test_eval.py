"""Spec phase-07 gates: causal metric math, hop logic, AVA binning."""

import numpy as np
import torch

from pulsevad.eval import (ava_ground_truth, causal_metrics, evaluate_causal_clip,
                           evaluate_window_set, mean_ci)
from pulsevad.model import PulseVAD


def test_causal_metrics_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = causal_metrics(y, p)
    assert m["auc"] == 1.0 and m["f1"] == 1.0
    assert m["fpr_at_tpr95"] <= 0.5
    r = np.random.default_rng(0)
    m = causal_metrics(r.integers(0, 2, 4000), r.random(4000))
    assert 0.45 < m["auc"] < 0.55  # random scores -> chance AUC


def test_fpr_at_tpr95_interpretation():
    # scores = labels + small noise: TPR 95% reachable with small FPR
    r = np.random.default_rng(1)
    y = r.integers(0, 2, 20000)
    p = np.clip(y + r.normal(0, 0.1, 20000), 0, 1)
    m = causal_metrics(y, p)
    assert m["fpr_at_tpr95"] < 0.2


def test_window_set_end_to_end():
    torch.manual_seed(0)
    model = PulseVAD().eval()
    feats = np.random.randn(200, 64, 21).astype(np.float32)
    labels = (feats.mean(axis=(1, 2)) > 0).astype(np.uint8)
    m = evaluate_window_set(model, feats, labels)
    assert m["n"] == 200 and 0 <= m["auc"] <= 1


def test_causal_clip_hop_count():
    torch.manual_seed(0)
    model = PulseVAD().eval()
    sr = 16000
    audio = np.random.randn(sr * 2).astype(np.float32)  # 2 s = 10 hops of 200 ms
    probs = evaluate_causal_clip(model, audio, sr=sr)
    assert len(probs) == 10
    assert np.all((probs >= 0) & (probs <= 1))
    # resampling path: 8 kHz input -> same hop count
    probs8 = evaluate_causal_clip(model, audio[::2], sr=8000)
    assert len(probs8) == 10


def test_ava_binning_majority():
    # 1 s clip, 5 hops. Speech 0.15..0.55 s: hop0 overlap 0.05 (<0.1 -> noise),
    # hop1 full (speech), hop2 overlap 0.15 (>=0.1 -> speech), hops 3-4 none.
    ann = [(0.15, 0.55, "speech")]
    gt = ava_ground_truth(ann, duration_s=1.0)
    assert gt.tolist() == [0, 1, 1, 0, 0]
    # noise annotations are ignored
    gt = ava_ground_truth([(0.0, 1.0, "noise")], duration_s=1.0)
    assert gt.sum() == 0


def test_mean_ci():
    mean, ci = mean_ci([0.84, 0.85, 0.86])
    assert abs(mean - 0.85) < 1e-9 and 0 < ci < 0.02


def test_comparison_artifact():
    from pulsevad.eval import build_comparison
    ours = {
        "PulseVAD (teacher, unpruned)": {
            "params": 81090,
            **{c: {"auc": 0.97, "fpr_at_tpr95": 0.05} for c in
               ("clean", "windy", "dns_synthetic", "speech_noise", "pure_noise")},
        },
        "PulseVAD (ship, pruned)": {
            "params": 2118,
            **{c: {"auc": 0.93, "fpr_at_tpr95": 0.04} for c in
               ("clean", "windy", "dns_synthetic", "speech_noise", "pure_noise")},
        },
    }
    md, data = build_comparison(ours)
    assert "vs MarbleNet [cited]" in md
    assert "43.0x fewer parameters" in md  # 91k / 2,118
    assert "257.3x fewer parameters" in md  # 545k / 2,118 vs Silero
    assert "**loss**" in md  # AtomicVAD smaller; Silero lower latency — honest losses
    assert "Summary wins" in md and data["wins"]
    # measured AUC table lists both models with 5 categories
    assert md.count("**PulseVAD (ship, pruned)**") >= 2
