# Phase 2: Model Architecture & Parameter Verification

> **Milestone Objective:** Implement the full convolutional neural network backbone (PulseVAD / FlexibleVAD) in PyTorch, verify its exact layer-by-layer parameter count (strictly **81,090 parameters**), and confirm forward/backward compute graph validity.

---

## 1. Conceptual Deep Dive: Architecture Design Principles

The PulseVAD architecture is designed under four strict edge-deployment constraints:

1.  **CNN-Only Backbone:** No recurrent units (LSTM/GRU) which suffer from sequential execution dependencies, variable-length hidden states, and poor quantization behavior on microcontrollers. No specialized activations like cos() (GGCU) which require hardware floating-point trigonometric units.
2.  **1x1 Adapter Layer ( \to 128$):** Decouples the 64-channel Mel spectrogram input from internal feature widths. This allows internal channels to be aggressively pruned without altering the external audio interface.
3.  **Depthwise-Separable & Dilated Convolutions:**
    *   *Depthwise Conv:* Applies a spatial/temporal filter independently to each channel ({\text{in}} \times 1 \times K$ weights), capturing time-frequency dynamics.
    *   *Pointwise Conv:* Applies a  \times 1$ convolution across channels ({\text{in}} \times C_{\text{out}} \times 1$ weights), mixing feature representations.
    *   *Dilation:* In Layer 7 (=29, d=2$), receptive field expands to  \times 2 - 1 = 57$ time frames, covering multiple syllables across time without increasing parameters.
4.  **Global Average Pooling (GAP) Classifier:** Instead of flattening the temporal dimension before the linear head, features are pooled across time:
    \mathbf{z} = \frac{1}{21} \sum_{t=1}^{21} \mathbf{h}(t) \in \mathbb{R}^{128}
    This makes the parameter count completely independent of input context length (allowing the same model to run at 60 ms, 200 ms, or 360 ms).

---

## 2. Complete Layer-by-Layer Parameter & MAC Specification

Every convolution is **bias-free** (ias=False).  
Every BatchNorm1d layer uses **eps=1e-3, momentum=0.1** (critical: not PyTorch's default 1e-5).  
Activations are standard **ReLU()**.  
Padding is **same** (stride=1) so the temporal length remains **21 frames** through all layers until GAP.

`
Input: (B, 64, 21)
  |
  v
[1. Adapter Conv] 1x1 Conv(64 -> 128) + BN + ReLU
  | (B, 128, 21)
  v
[2. conv0_dw] DW-Conv(128 -> 128, k=11, pad=5, groups=128) (No BN, No ReLU)
  | (B, 128, 21)
  v
[3. conv0_pw] PW-Conv(128 -> 128, k=1) + BN + ReLU
  | (B, 128, 21)
  v
[4. block1] 1x1 Conv(128 -> 64) + BN + ReLU
  | (B, 64, 21)
  v
[5. block2] 1x1 Conv(64 -> 64) + BN + ReLU
  | (B, 64, 21)
  v
[6. block3: Residual Block (k=17)]
  |-- Main Branch:
  |     DW-Conv(64 -> 64, k=17, pad=8, g=64) -> PW-Conv(64 -> 64, k=1) + BN + ReLU + Dropout(0.1)
  |     -> DW-Conv(64 -> 64, k=17, pad=8, g=64) -> PW-Conv(64 -> 64, k=1) + BN (NO activation)
  |-- Skip Branch:
  |     1x1 Conv(64 -> 64) + BN
  |-- Add Main + Skip -> ReLU -> Dropout(0.1)
  | (B, 64, 21)
  v
[7. conv4_dw] Dilated DW-Conv(64 -> 64, k=29, dilation=2, pad=28, groups=64) (No BN/ReLU)
  | (B, 64, 21)
  v
[8. conv4_pw] PW-Conv(64 -> 128, k=1) + BN + ReLU
  | (B, 128, 21)
  v
[9. conv5] 1x1 Conv(128 -> 128, k=1) + BN + ReLU
  | (B, 128, 21)
  v
[10. Global Average Pooling] mean over 21 frames -> (B, 128)
  |
  v
[11. Classifier Head] Linear(128 -> 2, bias=True)
  |
  v
Output Logits: (B, 2)
`

### Exact Parameter Table

| # | Layer Name | Operation Details | Shape / Dimensions | Param Count | MACs (21 frames) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | dapter | Conv1d(64, 128, k=1, bias=False) + BN |  \times 128 + 2(128)$ | **8,448** | 172,032 |
| **2** | conv0_dw | Conv1d(128, 128, k=11, g=128, pad=5, bias=False) |  \times 1 \times 11$ | **1,408** | 29,568 |
| **3** | conv0_pw | Conv1d(128, 128, k=1, bias=False) + BN |  \times 128 + 2(128)$ | **16,640** | 344,064 |
| **4** | lock1 | Conv1d(128, 64, k=1, bias=False) + BN |  \times 64 + 2(64)$ | **8,320** | 172,032 |
| **5** | lock2 | Conv1d(64, 64, k=1, bias=False) + BN |  \times 64 + 2(64)$ | **4,224** | 86,016 |
| **6** | lock3 (Res) | SubA (\times 17 + 64\times 64 + 128$) + SubB (\times 17 + 64\times 64 + 128$) + Skip (\times 64 + 128$) |  + 5312 + 4224$ | **14,848** | 303,744 |
| **7** | conv4_dw | Conv1d(64, 64, k=29, d=2, g=64, pad=28, bias=False) |  \times 1 \times 29$ | **1,856** | 38,976 |
| **8** | conv4_pw | Conv1d(64, 128, k=1, bias=False) + BN |  \times 128 + 2(128)$ | **8,448** | 172,032 |
| **9** | conv5 | Conv1d(128, 128, k=1, bias=False) + BN |  \times 128 + 2(128)$ | **16,640** | 344,064 |
| **10** | gap | Global Average Pooling over time | No parameters | **0** | 2,688 |
| **11** | classifier | Linear(128, 2, bias=True) |  \times 2 + 2$ | **258** | 256 |
| **TOTAL** | | **Full PulseVAD Unpruned Model** | | **81,090** | **~1.66 M** |

---

## 3. Step-by-Step Implementation Checklist

### Step 2.1: Model Architecture Implementation (pulsevad/model.py)
- [ ] Create pulsevad/model.py.
- [ ] Define helper module ConvBNReLU(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, eps=1e-3, momentum=0.1).
- [ ] Define DepthwiseSeparableBlock(in_channels, out_channels, kernel_size, dilation=1).
- [ ] Define ResidualBlock(channels=64, kernel_size=17, dropout=0.1):
  *   Main branch sub-block 1: DW(k=17, pad=8) -> PW(1x1) + BN + ReLU + Dropout(0.1).
  *   Main branch sub-block 2: DW(k=17, pad=8) -> PW(1x1) + BN (NO ReLU, NO Dropout).
  *   Skip branch: Conv1d(64, 64, k=1, bias=False) + BN.
  *   Combined: out = F.relu(main + skip), followed by Dropout(0.1).
- [ ] Define PulseVAD(nn.Module) assembling all 11 stages.
- [ ] Implement orward(x, return_logits=True):
  *   Input tensor: (B, 64, 21)
  *   Feature output before classifier: (B, 128)
  *   If 
eturn_logits=True: return raw (B, 2) logits.
  *   If 
eturn_logits=False: return softmax(logits, dim=-1)[:, 1] (speech probability).

---

## 4. Verification Gate & Unit Tests

Create 	ests/test_model.py:

`python
import torch
import pytest
from pulsevad.model import PulseVAD

def test_exact_parameter_count():
    model = PulseVAD()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 81090, f'Expected exactly 81,090 params, got {total_params}'
    assert trainable_params == 81090, f'Expected all 81,090 params trainable, got {trainable_params}'

def test_forward_pass_shapes():
    model = PulseVAD()
    model.eval()
    x = torch.randn(8, 64, 21)
    
    # Test logits output
    logits = model(x, return_logits=True)
    assert logits.shape == (8, 2), f'Expected (8, 2), got {logits.shape}'
    
    # Test probability output
    probs = model(x, return_logits=False)
    assert probs.shape == (8,), f'Expected (8,), got {probs.shape}'
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

def test_backward_gradient_flow():
    model = PulseVAD()
    model.train()
    x = torch.randn(4, 64, 21)
    y = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f'Gradient missing for {name}'
        assert not torch.isnan(param.grad).any(), f'NaN gradient in {name}'
`

### Exit Criteria
*   [ ] pytest tests/test_model.py passes 100%.
*   [ ] Parameter count is verified as exactly **81,090**.
