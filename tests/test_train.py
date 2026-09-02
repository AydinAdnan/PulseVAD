import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pulsevad.model import PulseVAD
from pulsevad.scheduler import CyclicWarmupCosineScheduler, lr_at_epoch
from pulsevad.train import CachedWindows, TRAIN_HP, evaluate, train

WINDOW = 3_200


# ---------------------------------------------------------------- scheduler

def test_lr_warmup_is_linear():
    # 4-epoch warmup 0 -> 3.5e-3
    assert lr_at_epoch(0) == 0.0
    assert lr_at_epoch(2) == 3.5e-3 * 2 / 4
    assert abs(lr_at_epoch(4) - 3.5e-3) < 1e-12


def test_lr_hold_plateau():
    for e in range(4, 20):
        assert lr_at_epoch(e) == 3.5e-3


def test_lr_cosine_decay_endpoints_and_midpoint():
    # epoch 20: cos(0)=1 -> max; epoch 30: cos(pi/2)=0 -> midpoint; epoch 40: cos(pi)=-1 -> min
    assert abs(lr_at_epoch(20) - 3.5e-3) < 1e-9
    mid = 1e-5 + 0.5 * (3.5e-3 - 1e-5)
    assert abs(lr_at_epoch(30) - mid) < 1e-9
    assert abs(lr_at_epoch(40) - 1e-5) < 1e-9


def test_lr_decay_is_monotone_and_bounded():
    lrs = [lr_at_epoch(e) for e in range(20, 41)]
    assert all(b <= a + 1e-12 for a, b in zip(lrs, lrs[1:]))
    assert min(lrs) >= 1e-5 and max(lrs) <= 3.5e-3 + 1e-12


def test_scheduler_sets_optimizer_lr():
    model = PulseVAD()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = CyclicWarmupCosineScheduler(opt)
    assert sched.step(2) == opt.param_groups[0]["lr"]
    assert opt.param_groups[0]["lr"] == 3.5e-3 * 2 / 4
    sched.step(19)
    assert opt.param_groups[0]["lr"] == 3.5e-3


def test_hyperparameters_match_spec():
    # spec phase-04 table
    assert TRAIN_HP["epochs"] == 40
    assert TRAIN_HP["batch_size"] == 512
    assert TRAIN_HP["momentum"] == 0.9
    assert TRAIN_HP["nesterov"] is True
    assert TRAIN_HP["weight_decay"] == 8.75e-4
    assert TRAIN_HP["label_smoothing"] == 0.09


# ---------------------------------------------------------------- dataset

def test_cached_windows_alignment(tmp_path):
    n = 32
    feats = np.random.randn(n, 64, 21).astype(np.float32)
    labels = np.random.randint(0, 2, n).astype(np.uint8)
    np.save(tmp_path / "f.npy", feats)
    np.save(tmp_path / "y.npy", labels)
    ds = CachedWindows(tmp_path / "f.npy", tmp_path / "y.npy")
    assert len(ds) == n
    x, y = ds[7]
    assert x.shape == (64, 21) and x.dtype == torch.float32
    assert y == int(labels[7]) and np.allclose(x.numpy(), feats[7])


# ---------------------------------------------------------------- e2e micro-train

@pytest.fixture()
def synthetic_cache(tmp_path):
    """Linearly separable toy cache: label drives the mean of every mel bin."""
    rng = np.random.default_rng(11)
    n = 256
    labels = rng.integers(0, 2, n).astype(np.uint8)
    feats = rng.standard_normal((n, 64, 21)).astype(np.float32) * 0.5
    feats[labels == 1] += 0.8
    feats[labels == 0] -= 0.8
    for tag in ("train", "val"):
        np.save(tmp_path / f"{tag}_features.npy", feats)
        np.save(tmp_path / f"{tag}_labels.npy", labels)
    return tmp_path


def test_micro_train_converges_and_saves(synthetic_cache, tmp_path):
    out = tmp_path / "runs"
    best = train(
        cache_dir=synthetic_cache, out_dir=out, seed=0,
        epochs=3, batch_size=64, num_workers=0, device="cpu",
    )
    # separable task: 3 epochs must beat chance comfortably
    assert best["auc"] > 0.9, f"AUC {best['auc']} after 3 epochs"
    # loss must decrease over training
    history = json.loads((out / "history.json").read_text())
    assert len(history) == 3
    assert history[-1]["train_loss"] < history[0]["train_loss"]

    # checkpoints: one per epoch + best
    assert (out / "checkpoint_epoch_0.pth").exists()
    assert (out / "checkpoint_epoch_2.pth").exists()
    assert (out / "best_model.pth").exists()

    # best checkpoint reloads into a fresh model and matches best AUC recompute
    ckpt = torch.load(out / "best_model.pth", weights_only=False)
    model = PulseVAD()
    model.load_state_dict(ckpt["model"])
    ds = CachedWindows(synthetic_cache / "val_features.npy", synthetic_cache / "val_labels.npy")
    loader = torch.utils.data.DataLoader(ds, batch_size=64)
    metrics = evaluate(model, loader, "cpu", TRAIN_HP["label_smoothing"])
    assert abs(metrics["auc"] - best["auc"]) < 1e-6


