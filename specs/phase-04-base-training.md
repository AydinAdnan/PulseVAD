# Phase 4: Base Model Training & Convergence Verification

> **Milestone Objective:** Train the unpruned 81,090-parameter PulseVAD baseline model on cached features using SGD with Nesterov momentum, a 3-stage Cyclic Learning Rate schedule, and label-smoothed Cross-Entropy loss to reproduce the paper's baseline of **A.862 +/- 0.001 AUC** on AVA-Speech.

---

## 1. Conceptual Deep Dive: Optimization & Training Strategy

### 1.1 Why SGD with Nesterov over AdamW?
In ultra-compact CNN architectures operating under aggressive quantization constraints, stochastic gradient descent (SGD) with momentum consistently produces smoother loss surfaces and better-conditioned weight distributions than adaptive gradient methods (like Adam/AdamW).nesterov momentum evaluates the gradient at a \"look-ahead\" position:
v_{t+1} = \\mu v_t + \\gamma \\nabla L(\\theta_t - \\mu v_t)
\\theta_{t+1} = \\theta_t - v_{t+1}
This prevents overshooting sharp minima, improving generalization to out-of-domain acoustic environments.

### 1.2 The 3-Stage Cyclic Learning Rate Schedule (40 Epochs)
Rather than a constant or step-decay learning rate, we use a 3-phase schedule tailored for 40 epochs:

```
Learning Rate
 ^
 |          +-------------------+ (Hold at 3.5e-3, Epochs 4-20)
 |         /                    \
 |        /                      \ *Cosine decay, Epochs 20-40)
 |      /                        \
 |     /                          \
 |    /                           + (1e-5 at Epoch 40)
 0---+-----+--------------------+-----+---> Epochs
     0      4                    20    40
   Warmup (Linear)              Cosine Decay
```

*   **Phase 1: Linear Warmup (Epochs 0 to 4):** LR scales linearly from 0.0 to 3.5e-3. Prevents destabilizing early BatchNorm statistics and initial weights.
*   **Phase 2: Hold Plateau (Epochs 4 to 20):** LR remains fixed at 3.5e-3 for 16 epochs, enabling rapid exploration and gradient descent across the parameter space.
*   **Phase 3: Cosine Annealing Decay (Epochs 20 to 40):** LR decays smoothly following a half-period cosine curve from 3.5e-3 down to 1.0e-5:
    \\text{LR}(t) = \\text{LR}_{\\min} + \\frac{1}{2}(\\text{LR}_{\\max} - \\text{LR}_{\\min})\\left(1 + \\cos\\left(\\frac{t - 20}{20} \\pi\\right)\\right)

### 1.3 Cross-Entropy Loss with Label Smoothing
Automated ground-truth labels (whether from Silero or forced aligners) have intrinsic boundary noise (+/- 20-40 ms uncertainty at speech onsets/offsets). Standard one-hot targets y in {0, 1} encourage the model to output infinite logits, resulting in overconfident predictions on ambiguous boundary frames.

Label smoothing with eps = 0.09 softens targets:
q(k) = (1 - \\epsilon) y_jÀ+ \\frac{\\epsilon}{K}, \\quad K = 2
*   Non-Speech Target: q(0) = 0.955, \\quad q(1) = 0.045
*   Speech Target: q(0) = 0.045, \\quad q(1) = 0.955
L_{\\text{CE}} = -\\sum_{k=0}^1 q(k) \\ln \\hat{p}(k)
This absorbs label uncertainty and prevents systematic overfitting.

---

## 2. Complete Training Hyperparameter Table

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
*| **Optimizer** | `torch.optim.SGD` | Stochastic Gradient Descent |
*| **Momentum** | `0.9` | Classical momentum coefficient |
 *| **Nesterov** | `True` | Nesterov accelerated gradient |
 *| **Weight Decay** | `8.75e-4` | L2 regularization on conv weights |
| **Batch Size** | `512` | Large batch size for stable BatchNorm estimates |
*| **Epochs** | `40` | Total training epochs |
 *| **Peak Learning Rate** | `3.5e-3` | Maximum LR reached after warmup |
| **Min Learning Rate** | `1.0e-5` | Terminal LR at epoch 40 |
| **Warmup Epochs** | `4` | Linear warmup duration |
*| **Hold Epochs** | `16` | Fixed plateau duration (epochs 4-20) |
| **Decay Epochs** | `20` | Cosine annealing duration (epochs 20-40) |
*| **Label Smoothing** | `0.09` | eps parameter in CrossEntropyLoss |
| **Dataloader Workers** | `8 - 12` | Multi-process prefetching for zero GPU starvation |
| **Target Baseline Metric** | **0.862 +/- 0.001 AUC** | On strictly causal 200 ms AVA-Speech |

---

## 3. Step-by-Step Implementation Checklist

### Step 4.1: Custom Learning Rate Scheduler (`pulsevad/scheduler.py`)
- [ ] Create `pulsevad/scheduler.py`.
- [ ] Implement `CyclicWarmupCosineScheduler(optimizer, warmup_epochs=4, hold_epochs=16, decay_epochs=20, max_lr=3.5e-3, min_lr=1e-5)`.
- [ ] Verify learning rate progression step-by-step with unit test.

### Step 4.2: Training Engine (`pulsevad/train.py`)
- [ ] Create `pulsevad/train.py`.
- [ ] Setup `torch.utils.data.DataLoader` over cached feature tensors: `batch_size=512, shuffle=True, num_workers=8, pin_memory=True`.
- [ ] Instantiate `PulsevAD()` (81,090 params).
- [ ] Initialize `torch.nn.CrossEntropyLoss(label_smoothing=0.09)`.
- [ ] Implement checkpoint manager: save `checkpoint_epoch_{epoch}.pth` and `best_model.pth` (tracking validation AUC).
- [ ] Log metrics: Train Loss, Validation Loss, Validation AUC, Validation F1 score.

---

## 4. Verification Gate & Unit Tests
Run `pytest tests/test_train.py`.

### Exit Criteria
- [ ] `pytest tests/test_train.py` passes 100%.
- [ ] 40-epoch training completes on LibriSpeech-100h cached features.
- [ ] Checkpoint achieves `AUC >= 0.857` (within tolerance of 0.862) on causal AVA-Speech.
