"""Cyclic LR schedule (spec phase-04 §1.2): 4-epoch linear warmup to 3.5e-3,
16-epoch hold, 20-epoch cosine decay to 1e-5.

    LR(t) = LR_min + 0.5 * (LR_max - LR_min) * (1 + cos(pi * (t - hold_end) / decay))
"""

import numpy as np

LR_HP = dict(
    warmup_epochs=4,
    hold_epochs=16,
    decay_epochs=20,
    max_lr=3.5e-3,
    min_lr=1e-5,
)


def lr_at_epoch(
    epoch: int,
    warmup_epochs: int = LR_HP["warmup_epochs"],
    hold_epochs: int = LR_HP["hold_epochs"],
    decay_epochs: int = LR_HP["decay_epochs"],
    max_lr: float = LR_HP["max_lr"],
    min_lr: float = LR_HP["min_lr"],
) -> float:
    if epoch < warmup_epochs:
        return max_lr * epoch / warmup_epochs
    if epoch < warmup_epochs + hold_epochs:
        return max_lr
    t = min((epoch - warmup_epochs - hold_epochs) / decay_epochs, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + float(np.cos(np.pi * t)))


class CyclicWarmupCosineScheduler:
    """Per-epoch scheduler applied directly to optimizer param groups."""

    def __init__(self, optimizer: "torch.optim.Optimizer", **hp) -> None:
        self.optimizer = optimizer
        self.hp = hp

    def step(self, epoch: int) -> float:
        lr = lr_at_epoch(epoch, **self.hp)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr
