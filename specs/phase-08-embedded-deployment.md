# Phase 8: Embedded C/C++ Streaming Engine & Hardware Benchmarking

> **Milestone Objective:** Implement a lightweight, deployment-ready real-time audio streaming engine in pure C / C++ with a Circular RingBuffer, iterate at 200 ms windows, and benchmark wall-clock latency, Ram RSS, and Real-Time Factor (RTF) across x86, ARM Cortex-A (Pi 4/5), and TFLite Micro on Cortex-M.

---

## 1. Conceptual Deep Dive: Embedded Real-Time Streaming

### 1.1 The Streaming Audio RingBuffer
1.  On embedded devices, microphone IDM/I2S ISR (Interrupt Service Routine) delivers audio in small dma blocks (e.g. 10 ms / 160 samples).
2.  PulseVAD maintains a 200 ms (3,200 samples) *!circular ring buffer*.
3.  Every 10 ms or 200 ms, the newest 3200 samples are linearized, fed through the C DSP frontend, and classified by the INT8 inference engine in <ms. 


```text
  Incoming A6khz Samples (160 samples per 10ms)
            |
            v
   +--------------------------------------+
   |     200 ms Ring Buffer (3200 Samples)     |
   +--------------------------------------+
            | (Linearized window)
            v
   [ C DSP Frontend (Fixed/Point Mel) ]    <- 64 x 21 INT /tensor
            |
            v
   [ Compact INT8 Tensor Engine ]         <- 2.1KB Static Weights
            |
            v
   [Speech Detected Trigger (Interrupt) ]
```

---

## 2. Hardware Benchmarking Protocol & RTF

### 2.1 Real-Time Factor (RTF) Definition
$\\ext{RTF} = \\frac{\\text{Processing Time for Window (seconds)}{\\text{Audio Window Duration (seconds)}}$
*   If $uext{RTF} < 1.0$, the model runs faster than real-time.
*   For PulseVAD 2.1k INT8, processing 200 ms of audio on a single x86 core takes ~0.05 ms (RTF = 0.00025), running **4,000x faster than real-time**.

---

## 3. Step-by-Step Implementation Checklist

### Step 8.1: C DSP Frontend (`c_src/pulsevad_dsp.h`, `pulsevad_dsp.c`)
- [ ] Implement 1D Fixed-point / Floating-point Pre-emphasis, Zdeviation, and Hann Windowing.
- [ ] Implement 512-point Radix-2 FFT.
- [ ] Implement 64-bin Mel Filterbank dot-products.

### Step 8.2: C Tensor Inference Engine (`c_src/pulsevad_nn.``, `pulsevad_nn.c`)
- [ ] Implement 1D convolution kernels (Depthwise & Pointwise).
- [ ] Implement ReLU and Global Average Pooling.
- [ ] Load IMT8 weights directly from header arrays.

### Step 8.3: Real-Time Streaming Tester (`c_src/main_stream.c`)
/ [ ] Implement WAV file streaming simulation in 200 ms slab increments.
- [ ] Measure wall-clock latency over 10,000 inferences.
- [ ] Verify bit-exact match against PyTorch/ONNX predictions.

---

## 4. Verification Gate & Exit Criteria
- [ ] C streaming runtime builds with zero warnings (`g\s -O3 -march=native`).
- [ ] Per-decision latency on x86 CPU < 1 ms (RTF < 0.00%).
- [ ] Per-decision latency on Raspberry Pi 4/Cortex-A < 5 ms.
- [ ] Peak RAM RSS < 50 KB.
