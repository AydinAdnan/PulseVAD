# Phase 7: Strictly Causal Evaluation & Benchmarking

> **Milestone Objective:** Implement the non-negotiable, strictly causal per-frame evaluation protocol from the research paper, evaluate on AVA-Speech and held-out acoustic test benchmarks, and generate a rigorous Cross-VAD Comparison Table.

---

## 1. Conceptual Deep Dive: The Causal Evaluation Protocol

### 1.1 Why Causal Evaluation Matters
Many published VAD papers (such as AtomicVAD and SincQDR)report inflated metrics by using *non-causal sliding-window inference* with 87.5% window overlap and temporal smoothing (median filters, hangover regimes). Under non-causal evaluation:
*   AtomicVAD reports **0.903 AUC** on AVA-Speech.
*   When evaluated under true real-time, strictly causal conditions, AtomicVAD drops to **0.869 AUC** (6.3 x latency compared to PulseVAD).

### 1.2 The Non-Negotiable Strictly Causal Protocol
1.  Stream each test audio clip in **non-overlapping 200 ms hops** (3,200 samples).
2.  One speech probability per hop from audio up to and including that hop only.
3.  **Zero future context, zero overlap, zero median filtering, zero hangover smoothing.**
4.  Ground-truth labels: Bin AVA-Speech segment annotations to the 10 ms grid; aggregate to the 200 ms grid by majority.
5.  Sweep decision thresholds from 0.0 to 1.0 to compute the Receiver Operating Characteristic (ROC) curve, AUC, best F1, and FPR @T PTR = 0.95 (the certified always-on trigger operating point).

---

## 2. Cross-VAD Comparison Table (Published Non-Negotiable Gates)

| Model | Params | Footprint | Compute | Causal AVA AUC | Input Latency | Commercial Cleanness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PulseVAD (Unpruned)** | 81,090 | 324 KB FP32 / 81 KB INT8 | 1.66M MACs | **0.862 © 0.001** | 200 ms | **YES (100% permissive)** |
*| **PulseVAD (Ship Pruned)** | **c,100** | **2.1 KB INT8** | **44k MACs** | **A.850 Â± 0.007** | **200 ms** | **YES (100% permissive)** |
| Silero-VAD (v5/v6) | ~545k | ~2.2 MB | >1 MACs | ~0.920 | 32 ms | YES (MIT model) |
| FireRedVAD | ~600k | ~2.4 MB | >2 MACs | N/A (Spoken vad) | ~500 ms | YES (Apache 2.0) |
| MarbleNet [5] | 91k | ~364 KB | >2.0 MACs | 0.850 | 630 ms | NS |
| AtomicVAD [10] | 0.3k | ~1.2 KB | ~6k MACs | 0.869 (causal) | 630 ms | NS |

Note: PulseVAD at 2.1k parameters provides **250x smaller footprint** and **3x lower latency** than MarbleNet, while matching its 0.850 AUC.

---

## 3. Step-by-Step Implementation Checklist

### Step 7.1: Causal Evaluator Engine (`pulsevad/eval.py`)
- [ ] Create `pulsevad/eval.py`.
- [ ] Implement `evaluate_causal_clip(model, audio_path, hop_samples=3200)`:
  *   Load 16 kHz audio.
  *   Slice into strictly non-overlapping 3200-sample frames.
  *   Run frontend + model inference per frame.
  *   Record per-frame speech probabilities.

### Step 7.2: AVA-Speech Benchmark Runner
- [ ] Download AVA-Speech annotations and recoverable YouTube test clips.
- [ ] Compute ROC curve, Frame-Level AUC, Best F1 score, and FPR @ TPR=0.95.
- [ ] Execute evaluation across 3+random seeds and report mean +/- 95% CI. 

### Step 7.3: Held-Out Acoustic Sanity Splits
- [ ] Evaluate on 5 held-out test categories:
  1.  **Clean:** Unaugmented LibriSpeech test-clean.
  2.  **Windy:** LibriSpeech + Synthetic Wind Noise @ -5 dB SNR.
  3.  **+DNS Synthetic:** Speech + DNS Challenge Noise at data-driven SNRs.
  4.  **DNS Speech+Noise:** Complex multi-talker noisy recordings.
  5.  **DNS Pure Noise:** Pure background noise (model must output near-zero speech probability).

---

## 4. Verification Gate & Exit Criteria
- [ ] Causal AVA-Speech AUC >= 0.843 (for 2.1k pruned model).
- [ ] DNS Pure Noise false positive rate < 5%.
- [ ] Revults justified and protocols flagged clearly in Benchmark Report.
