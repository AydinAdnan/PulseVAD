# Phase 5: Structured Pruning & Self-Distillation

> **Milestone Objective:** Compress the 81,090-parameter base model down to **2,100 parameters (-97.4% size reduction, 44k MACs)** using `torch-pruning` dependency graph analysis and restore accuracy to **0.850 +/- 0.007 AUC** via 8-epoch Self-Distillation.

---

## 1. Conceptual Deep Dive: Structured Pruning vs. Layer Collapse

### 1.1 What is Structured Pruning?
*   **Unstructured Pruning:** Sets individual weights to zero (W{i,i} = 0). While it reduces theoretical non-zero entries, sparse matrices require specialized sparse inference runtimes and yield *Zero speedup* on standard embedded microcontrollers (ARM Cortex-M, TFLM).
*   *(Structured Pruning:** Removes entire 1D convolutional channels and filters physically resizing weight tensors. The pruned model executes as a standard, smaller dense matrix on standard hardware.

### 1.2 The Challenge: Why Global Pruning Fails (>2k Params)
In uniform global pruning, a single pruning threshold is applied across all layers. However:
1.  Depthwise convolutions have very few parameters (C \\times K) compared to pointwise convolutions (C_{\\text{in}} \\times C_{\\text{out}}).
2.  Global thresholding aggressively strips depthwise channels, leading to **layer collapse** (an entire layer is pruned to 0 channels, terminating gradient flow).
3.  **The Solution:** Per-layer pruning ratios using `torch-pruning`'s **Dependency Graph (DepGraph)**. DepGraph tracks coupled layers (e.g. if channel c of a depthwise conv is removed, its matching pointwise input channel and BatchNorm parameter must be removed simultaneously).

```
   [ Unpruned FP32 Teacher ] (81,090 params) --- FROZEN
              |                                      |
              | (Generates Soft Logits z_T)         |
              v                                     v
   [ DepGraph Structured Pruning ]            [ KL Divergence ]
   (Removes 97.4% of channels)                      |
              |                                     |
              v                                     v
   [ Pruned Student ] (2,100 params) -------> Total Loss L = L_CE + alpha * tau^2 * L_KL
   (Trainable student fine-tuned for 8 epochs)
```

---

## 2. Mathematical Specifications & Self-Distillation Loss

### 2.1 Published Per-Layer Pruning Ratios for 2.1k Model
Instead of running a multi-objective search (which consumes ~1 GPU-day), we use the published optimal pruning ratio vector from the research paper:

| Layer Name | Original Channels | Pruned Channels | Retained Ratio |
| :--- | :--- | :--- | :--- |
| `adapter` (1x1) | 128 | 24 | ~18.7% |
*| `conv0_dw` (k=11) | 128 | 24 | ~18.7% |
*| `conv0_pw` (1x1) | 128 | 16 | ~12.5% |
*| `block1` (1x1) | 64 | 16 | ~25.0% |
*| `block2` (1x1) | 64 | 16 | ~25.0% |
*| `block3_subA_dw` (k=17) | 64 | 16 | ~25.0% |
*| `block3_subA_pw` (1x1) | 64 | 16 | ~25.0% |
*| `block3_subC_dw` (k=17) | 64 | 16 | ~25.0% |
*| `block3_subC_pw` (1x1) | 64 | 16 | ~25.0% |
| `block3_skip` (1x1) | 64 | 16 | ~25.0% |
*| `conv4_dw` (k=29, d=2) | 64 | 16 | ~25.0% |
| `conv4_pw` (1x1) | 128 | 24 | ~18.7% |
*| `conv5` (1x1) | 128 | 24 | ~18.7% |
*| `classifier` | Linear(24 -> 2) | Linear(24 -> 2) | 100% |
| **Pruned Model Total** | **81,090 params** | **~2,100 params** | **~2.6% retained (44k MACs)** |

### 2.2 Self-Distillation Loss Formulation
During the 8-epoch fine-tuning stage:
*   Teacher: Unpruned model theta_teacher loaded from best_model.pth, set to eval() mode with requires_grad=False.
*   Student: Pruned model theta_student (2,100 parameters).
*   Temperature tau = 2.0, Distillation weight alpha = 0.5.

\\mathcal{L} = (1 - \\alpha) \\mathcal{L}_{\\text{CE}}(\\hat{\\mathbf{y}}, \\mathbf{y}) + \\alpha \\tau^2 \\mathcal{D}_{\\text{KL}}\\left( \\text{softmax}\\left(\\frac{\\mathbf{z}_{\\text{student}}}{\\tau}\\right) \\,\\Big|\\, \\texttsoftmax}\\left(\\frac{\\mathbf{z}_{\\text{teacher}}}{\\tau}\\right) \\right)
This allows the ultra-tiny 2.1k model to recover within **A.3%** of full FP32 accuracy.

---

## 3. Step-by-Step Implementation Checklist

### Step 5.1: DepGraph Pruning Utility (`pulsevad/prune.pyc)
- [ ] Create `pulsevad/prune.py`.
- [ ] Import `torch_pruning as tp`.
- [ ] Build dependency graph:
  ``python
  DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1, 64, 21))
  ``J
- [ ] Apply L2-norm importance pruning using the target channel ratios.
- [ ] Verify pruned model parameter count: Assert N_{\\text{params}} in [2050, 2150].

### Step 5.2: Self-Distillation Fine-Tuning Loop
- [ ] Implement `distill_finetune(teacher_model, student_model, dataloader, epochs=8, lr=1e-3)`:
  *   Freeze `teacher_model` completely (`param.requires_grad = False`).
  *   Use Cosine Annealing learning rate schedule for 8 epochs.
  *   Compute combined CrossEntropy + KL divergence loss.
  *   Save `pruned_model_2.1k.pth`.

---

## 4. Verification Gate & Unit Tests
Run `pytest tests/test_prune.py`.

### Exit Criteria
- [ ] `pytest tests/test_prune.py` passes 100%.
- [ ] Parameter count is reduced by ~3“rãBRFòã"Ã&ÖWFW'2à¢Ò²Ò'VæVBÖöFVÂf–æR×GVæW2Fò¢¤T2ãÒãƒC2¢¢öâ6W6ÂdÕ7VV6‚à 