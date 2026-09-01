# PulseVAD: Master Specification & Development Roadmap

> **An Ultra-Tiny, Commercially Clean, Hardware-Friendly Voice Activity Detector (VAD)**  
> *Based on the kiloVAD research paper (Analog Devices & UCLA, INTERSPEECH 2026) and the PulseVAD Build Plan v2.*

---

## 1. Executive Summary & Purpose

**PulseVAD** is a ground-up, commercially clean implementation of an ultra-tiny Voice Activity Detection (VAD) neural network. Designed specifically for always-on edge devices, microcontrollers (MCUs), and embedded DSPs, PulseVAD achieves state-of-the-art streaming accuracy under strictly causal conditions while maintaining an ultra-compact footprint:

*   **Unpruned Baseline:** **81,090 parameters** (~1.66M MACs per 200 ms inference, **0.862 AUC** on AVA-Speech).
*   **Pruned Target (Primary Ship):** **2,100 parameters** (~44k MACs per 200 ms inference, **0.850 AUC** on AVA-Speech, **2.1 KB INT8** footprint).
*   **Latency:** **200 ms** algorithmic latency (3x faster response than standard 630 ms models like MarbleNet, TinyVAD, and AtomicVAD).
*   **Architecture:** 1D Depthwise-Separable Convolutional Neural Network (CNN-only) + Global Average Pooling (GAP) + Linear Head. **Zero recurrent layers (no LSTMs/GRUs), zero custom transcendental activations (no cos/sin/GGCU), zero learnable sinc filters.**
*   **Portability:** 100% compatible with static computation graphs, TensorFlow Lite for Microcontrollers (TFLM), ONNX Runtime, and pure C fixed-point inference engines.

---

## 2. Core Mental Model: How VAD Works

Voice Activity Detection is a binary classification problem operating on continuous audio streams:
\\text{Audio Signal } x(t) \\xrightarrow{\\text{Audio Frontend}} \\text{Spectrogram Features } \\mathbf{X} \\in \\mathbb{R}^{64 \\times 21} \\xrightarrow{\\text{Neural Backbone}} \\text{Probability } P(\\text{Speech} \\mid \\mathbf{X})

`
+---------------------------------------------------------------------------------------+
|                                  PULSEVAD END-TO-END PIPELINE                         |
+---------------------------------------------------------------------------------------+
                                                                                         
   Continuous 16 kHz Audio Stream                                                       
               |                                                                        
               v                                                                        
   [ Phase 1: Audio Frontend ]                                                          
      - 200 ms sliding window (3,200 samples @ 16 kHz)                                  
      - Pre-emphasis filter: y[n] = x[n] - 0.97 * x[n-1]                                
      - Waveform Z-Score Normalization (mean=0, std=1 across 3200 samples)             
      - Mel-Spectrogram: 64 bins, n_fft=512, win=400 (Hann), hop=160 (10ms)             
      - Log-compression: log(Mel + 1e-5)                                                
      - Per-bin Time Normalization: Z-norm across 21 time frames                        
               |                                                                        
               v  Tensor: (Batch, 64 mel bins, 21 time frames)                          
   [ Phase 2: Neural Backbone (PulseVAD) ]                                              
      - 1x1 Adapter Conv (64 -> 128 channels)                                           
      - Depthwise-Separable Conv Block (kernel=11, channels=128)                        
      - 1x1 Projection Layers (128 -> 64 -> 64)                                         
      - Dilated Residual Block (kernel=17, skip connection)                             
      - Dilated Depthwise Block (kernel=29, dilation=2, channels 64 -> 128)             
      - Global Average Pooling (GAP across 21 frames -> 128-dim vector)                
      - Linear Classifier (128 -> 2 classes: [Non-Speech, Speech])                      
               |                                                                        
               v                                                                        
   [ Output Decision ]                                                                  
      - Logits: [s_0, s_1] -> Softmax -> P(Speech) in [0.0, 1.0]                         
      - Causal Thresholding: P(Speech) >= threshold -> Speech Trigger                   
`

---

## 3. Spec-Driven Development (SDD) Roadmap

This project is organized into **9 structured phases**. Each phase contains:
1. **Theoretical Foundations:** Deep explanations of the digital signal processing (DSP), neural network mechanics, and optimization algorithms.
2. **Mathematical Specs & Exact Tensor Dimensions:** Exact equations and shapes at every step.
3. **Step-by-Step Implementation Checklist:** Actionable, checkable tasks for handwritten code.
4. **Unit Tests & Numerical Parity Gates:** Hard assertions to prove correctness before moving forward.
5. **Common Pitfalls & Failure Modes:** Traps that cause silent performance degradation.

### Roadmap Overview

`
+---------------------------------------------------------------------------------------+
|  Phase 0: Environment Setup, Dependencies & Legal Footing                             |
|  [specs/phase-00-environment-and-legal.md]                                            |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  Phase 1: Deterministic Audio Frontend & Feature Extraction                           |
|  [specs/phase-01-audio-frontend.md]                                                   |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  Phase 2: Model Architecture & Parameter Verification (81,090 Params)                 |
|  [specs/phase-02-model-architecture.md]                                               |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  Phase 3: Dataset Ingestion, Silero Self-Labeling & Feature Cache                     |
|  [specs/phase-03-dataset-and-pipeline.md]                                             |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  Phase 4: Base Model Training & Convergence Verification                              |
|  [specs/phase-04-base-training.md]                                                    |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  Phase 5: Structured DepGraph Pruning & Self-Distillation (2.1k Params)               |
|  [specs/phase-05-structured-pruning-and-distillation.md]                              |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  Phase 6: Quantization (INT8 PTQ) & Model Export (ONNX / TFLite)                      |
|  [specs/phase-06-quantization-and-export.md]                                          |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  Phase 7: Strictly Causal Evaluation & Benchmark Suite                                |
|  [specs/phase-07-causal-evaluation-and-benchmarking.md]                               |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|  Phase 8: Embedded C/C++ Streaming Engine & Hardware Benchmarking                     |
|  [specs/phase-08-embedded-deployment.md]                                              |
+---------------------------------------------------------------------------------------+
`

---

## 4. Directory & Codebase Layout

When implemented, the repository will follow this modular layout:

`
PulseVAD/
├── docs/                                  # Research papers and reference documents
│   ├── 2607.25870v1.pdf                   # kiloVAD Interspeech 2026 paper
│   └── PulseVAD_build_plan_v2.pdf         # Primary build plan and architecture doc
├── specs/                                 # Complete Spec-Driven Development specs
│   ├── README.md                          # Master roadmap (this file)
│   ├── phase-00-environment-and-legal.md  # Env, setup, licensing & data legality
│   ├── phase-01-audio-frontend.md         # DSP frontend and Mel spectrogram specs
│   ├── phase-02-model-architecture.md     # CNN backbone and exact parameter specs
│   ├── phase-03-dataset-and-pipeline.md   # Data download, Silero labeling, cache
│   ├── phase-04-base-training.md          # SGD, Cyclic LR, training engine
│   ├── phase-05-structured-pruning-and-distillation.md     # Torch-pruning & self-distillation
│   ├── phase-06-quantization-and-export.md    # INT8 PTQ, ONNX, TFLite export
│   ├── phase-07-causal-evaluation-and-benchmarking.md      # AVA-Speech causal evaluation suite
│   └── phase-08-embedded-deployment.md    # Streaming buffer & C runtime
├── data/                                  # Audio corpora and cached feature tensors
│   ├── raw/                               # Raw downloaded wavs (LibriSpeech, MUSAN, etc.)
│   ├── labels/                            # Silero-generated 10ms frame label manifests
│   └── cache/                             # Precomputed (64, 21) float32 feature tensors
├── pulsevad/                              # Core Python package
│   ├── __init__.py
│   ├── frontend.py                        # DSP Audio Frontend (PyTorch / NumPy)
│   ├── model.py                           # FlexibleVAD / PulseVAD PyTorch Module
│   ├── dataset.py                         # Audio loading, mixing, SNR scaling & RIR
│   ├── train.py                           # Training loop with cyclic LR & label smoothing
│   ├── prune.py                           # DepGraph pruning & distillation fine-tuning
│   ├── quantize.py                        # INT8 Post-Training Quantization
│   ├── export.py                          # ONNX & TFLite export helpers
│   └── eval.py                            # Causal per-frame evaluator (AUC, F1, ROC)
├── c_src/                                 # Embedded C / C++ streaming implementation
│   ├── pulsevad_dsp.h / .c                # Fixed/float DSP frontend in pure C
│   ├── pulsevad_nn.h / .c                 # Lightweight tensor inference engine
│   └── main_stream.c                      # Real-time microphone / WAV streaming tester
├── tests/                                 # Automated unit test suite
│   ├── test_frontend.py                   # DSP frontend numeric tests
│   ├── test_model.py                      # Parameter count, shape & backward tests
│   ├── test_dataset.py                    # Label alignment & SNR mixing tests
│   ├── test_prune.py                      # Shape preservation & sparsity tests
│   └── test_export.py                     # Bit-exact parity tests across runtimes
├── Makefile                               # Standard build automation commands
├── requirements.txt                       # Locked dependencies
└── pyproject.toml                         # Project metadata
`

---

## 5. Non-Negotiable Engineering Rules & Failure Checklist

Before writing any code, memorize these 12 critical principles:

1. **Strictly Causal Evaluation:** Never use future frames, non-causal bidirectional lookaheads, sliding-window median filters, or hangover smoothing during benchmark evaluation. Each 200 ms frame must be judged independently using only audio up to that moment.
2. **Frame Dimension Exactness:** With  = 3200$ samples (200 ms @ 16 kHz), $\\text{hop} = 160$ (10 ms), and center=True, output frame count is strictly:
   \\text{frames} = \\left\\lfloor \\frac{3200}{160} \\right\\rfloor + 1 = 20 + 1 = \\mathbf{21}
   *(Off-by-one errors here break downstream linear heads).*
3. **Frontend Normalization Order:**
   \\text{Waveform} \\xrightarrow{\\text{Pre-emphasis}} \\xrightarrow{\\text{Waveform Z-norm}} \\xrightarrow{\\text{Mel Spectrogram}} \\xrightarrow{\\text{Log Transform}} \\xrightarrow{\\text{Per-Bin Temporal Z-norm}}
   *Swapping waveform Z-norm and per-bin Z-norm silently destroys audio robustness!*
4. **BatchNorm Hyperparameters:** All BatchNorm1d layers must have $\\epsilon = 10^{-3}$ and $\\text{momentum} = 0.1$. PyTorch defaults to $\\epsilon = 10^{-5}$; using default $\\epsilon$ produces systematic logit drift.
5. **Bias-Free Convolutions:** All convolution layers must have ias=False because they are immediately followed by BatchNorm1d (which has its own affine bias $\\beta$).
6. **Structured Pruning via DepGraph:** Never prune pointwise convolutions independently without pruning their coupled depthwise and residual partner channels. Use dependency graphs (	orch-pruning).
7. **Per-Layer Pruning Ratios:** Uniform global pruning causes catastrophic layer collapse below 2k parameters. Per-layer ratio vectors must be maintained.
8. **Self-Distillation Teacher Freezing:** When fine-tuning the pruned student, the unpruned teacher model weights must be strictly frozen (eval() mode with 
equires_grad=False).
9. **Label Smoothing in Loss:** Use cross-entropy with $\\epsilon = 0.09$ label smoothing to absorb boundary noise from automated labeling.
10. **Commercially Clean Data:** Never use CC BY-NC-SA datasets (like LibriVAD or the Silero pre-labeled dataset) for training commercial weights. Use our clean self-labeling pipeline.
11. **Seed Replicates:** Report mean $\\pm$ 95% confidence intervals over multiple random seeds (minimum 3 seeds; paper uses 10). Pruned models have a variance of $\\pm 0.007$ AUC.
12. **Pure-Noise-Free Training:** The training set consists of 25% clean speech, 25% wind-mixed speech, and 50% DNS/MUSAN noise-mixed speech (with 50% simulated room reverberation). Do not inject pure-noise-only files into the training set (matches official config libri_dns_full_no_pure_noise_v2).
