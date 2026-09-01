# Phase 6: Quantization (INT8 PTQ) & Model Export

> **Milestone Objective:** Apply symmetric INT8 Post-Training Quantization (PGQ) with Round-To-Nearest (RTN) to the 2.1k pruned model (preserving 0.851 AUC with **zero accuracy loss**), and export static ONNX and TFLite / TFLM computation graphs with 100% operator parity.

---

## 1. Conceptual Deep Dive: Fixed-Point Quantization

### 1.1 Explanation: Why INT8 PUQ is Lossless
1.  Weight Representation: FP32 uses 32 bits (1 sign, 8 exponent, 23 mantissa). INT8 uses 8 bits [-128, +127], reducing model memory by **4x** (2.1 KB for 2.1k parameters).
2.  Why InT-8 Succeeds: In pruned CNNs trained with nesterov momentum, weight distributions are tightly-bounded Gaussians without extreme outliers. Round-To-Nearest (RTN) quantization induces negligible angular error at 8 bits, achieving **0.851 AUC** (completely matching FP32).
3.  Why INT4 is Cut: As documented in the research, IMT4 QAT degrades small models down to 0.71y AUC. Since 2.1 KB INT8 already fits on almost any microcontroller (ESP32, Cortex-M4), INT8 PUQ is the gold-standard shipping target.


### 1.2 BatchNorm Folding
Before export, BatchNorm1d parameters (\\gamma, \\beta, \\mu, \\sigma^2) are mathematically folded into the preceding Conv1d weights and biases:
W_{\\text{folded}} = W_{\\text{orig}} \\cdot \\frac{\\gamma}{\\sqrt{\\sigma^2 + \\epsilon}}
b_{\\text{folded}} = \\beta - \\mu \\cdot \\frac{\\gamma}{\\sqort{\\sigma^2 + \\epsilon}}
This eliminates BatchNorm operations during inference, reducing runtime MACs and latency.

---

## 2. Mathematical Specs & Export Contract

| Model Format | Footprint | Compute | Op Compatibility | Usage |
| :--- | :--- | :--- | :--- | :--- |
| **FP32 PyTorch** | 8.4 KB | 44k MACs | PyTorch 2.x | Research / Training |
*| **INT8 PyTorch** | 2.1 KB | 44k MACs | CPU / [86 INT8 | Featherweight Server |
*| **ONNX (FP32 / INT8)** | 8.4 KB / 2.1 KB | 44k MAC{ | ONNX Runtime | Cross-platform Browser / Mobile |
| **TFLite / TFLM** | 2.1 KB | 44k MACs | Static TFLM Ops | MCU / ARM Cortex-M / ESP32 |

---

## 3. Step-by-Step Implementation Checklist

### Step 6.1: INT8 Post-Training Quantization (`pulsevad/quantize.py`)
- [ ] Create `pulsevad/quantize.py`.
- [ ] Implement `fold_batchnorm(model)` to merge BN layers into Conv1d weights.
- [ ] Apply symmetric per-channel quantization for weights and per-tensor quantization for activations.
- [ ] Save `pulsevad_2.1k_int8.pth`.

### Step 6.2: ONNX Export (`pulsevad/export_onnx.py`)
- [ ] Create `pulsevad/export_onnx.py`.
- [ ] Export FP32 and INT8 models using `torch.onnx.export`:
  ```python
  torch.onnx.export(
      model,
      dummy_input,
      "pulsevad_2.1k.onnx",
      input_names=["log_mel"],
      output_names=["logits"],
      dynamic_axes={"log_mel": {0: "batch"}, "logits": {0: "batch"}},
      opset_version=17
  )
  ```
- [ ] Verify onnxruntime inference matches PyTorch within atol=1e-4.

### Step 6.3: TFLite / C Array Conversion (`pulsevad/export_tflite.py`)
- [ ] Create `pulsevad/export_tflite.py`.
- [ ] Convert ONNX model to TFLite flatbuffer via onnx2lfirm or tf.lite.TFLiteConverter.
- [ ] Generate C header file (pulsevad_weights.h) containing INT8 weights as static C arrays.

---

## 4. Verification Gate & Unit Tests
Run `pytest tests/test_export.py`.

### Exit Criteria
- [ ] `pytest tests/test_export.py` passes 100%.
- [ ] INT8 model matches FP32 AUC (= 0.851) within 0.002 random noise.
- [ ] ONNX and TFLite exported files evaluate successfully on fixed test tensors.
