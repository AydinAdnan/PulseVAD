"""Training engine (spec phase-04): SGD+Nesterov over cached features.

Reads the Phase-3 memmapped cache (never the raw audio), trains the 81,090
baseline with the paper recipe, tracks validation AUC/F1, saves every epoch
plus the best checkpoint by validation AUC (released best was epoch 38/40).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pulsevad.model import PulseVAD
from pulsevad.scheduler import CyclicWarmupCosineScheduler

TRAIN_HP = dict(
    epochs=40,
    batch_size=512,
    momentum=0.9,
    nesterov=True,
    weight_decay=8.75e-4,
    label_smoothing=0.09,
    **{"warmup_epochs": 4, "hold_epochs": 16, "decay_epochs": 20,
       "max_lr": 3.5e-3, "min_lr": 1e-5},
)


class CachedWindows(Dataset):
    """Index view over the memmapped (N, 64, 21) features + (N,) labels."""

    def __init__(self, features, labels) -> None:
        self.features = np.load(features, mmap_mode="r")
        self.labels = np.load(labels)
        assert len(self.features) == len(self.labels), "feature/label misalignment"

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(np.ascontiguousarray(self.features[i]))
        return x, int(self.labels[i])


@torch.no_grad()
def evaluate(model, loader, device, label_smoothing: float) -> dict:
    """Validation loss / frame AUC / best-F1 (spec phase-04 step 4.2)."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    probs, ys, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += loss_fn(logits, y).item() * len(y)
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
        ys.append(y.cpu())
    probs = torch.cat(probs).numpy()
    ys = torch.cat(ys).numpy()

    auc = float(roc_auc_score(ys, probs))
    thresholds = np.linspace(0.05, 0.95, 19)
    pred = probs[:, None] >= thresholds[None, :]
    tp = (pred & (ys == 1)[:, None]).sum(0)
    fp = (pred & (ys == 0)[:, None]).sum(0)
    fn = ((~pred) & (ys == 1)[:, None]).sum(0)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return {
        "loss": loss_sum / len(ys),
        "auc": auc,
        "f1": float(f1.max()),
        "f1_threshold": float(thresholds[int(f1.argmax())]),
    }


def train(
    cache_dir,
    out_dir,
    seed: int = 0,
    epochs: int | None = None,
    batch_size: int | None = None,
    num_workers: int = 8,
    device: str | None = None,
) -> dict:
    hp = dict(TRAIN_HP)
    if epochs is not None:
        hp["epochs"] = epochs
    if batch_size is not None:
        hp["batch_size"] = batch_size

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir, out_dir = Path(cache_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CachedWindows(cache_dir / "train_features.npy", cache_dir / "train_labels.npy")
    val_ds = CachedWindows(cache_dir / "val_features.npy", cache_dir / "val_labels.npy")
    gen = torch.Generator().manual_seed(seed)  # reproducible shuffling
    train_loader = DataLoader(
        train_ds, batch_size=hp["batch_size"], shuffle=True, generator=gen,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=hp["batch_size"],
        num_workers=max(2, num_workers // 2), pin_memory=(device == "cuda"),
    )
    print(f"[train] {len(train_ds)} train / {len(val_ds)} val windows on {device}", flush=True)

    model = PulseVAD().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=hp["max_lr"], momentum=hp["momentum"],
        nesterov=hp["nesterov"], weight_decay=hp["weight_decay"],
    )
    scheduler = CyclicWarmupCosineScheduler(
        optimizer,
        warmup_epochs=hp["warmup_epochs"], hold_epochs=hp["hold_epochs"],
        decay_epochs=hp["decay_epochs"], max_lr=hp["max_lr"], min_lr=hp["min_lr"],
    )
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=hp["label_smoothing"])

    history, best_auc = [], -1.0
    for epoch in range(hp["epochs"]):
        lr = scheduler.step(epoch)
        model.train()
        t0, run_loss = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(y)
        metrics = evaluate(model, val_loader, device, hp["label_smoothing"])
        metrics.update(
            epoch=epoch, lr=round(lr, 6),
            train_loss=round(run_loss / len(train_ds), 5),
            secs=round(time.time() - t0, 1),
        )
        history.append(metrics)
        print(json.dumps(metrics), flush=True)

        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "metrics": metrics},
            out_dir / f"checkpoint_epoch_{epoch}.pth",
        )
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "metrics": metrics,
                 "hp": hp, "seed": seed},
                out_dir / "best_model.pth",
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    best = max(history, key=lambda m: m["auc"])
    print(f"[train] best epoch {best['epoch']}: AUC {best['auc']:.4f}", flush=True)
    return best
