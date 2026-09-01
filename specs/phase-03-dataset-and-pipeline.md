# Phase 3: Dataset Ingestion, Self-Labeling & Feature Cache

> **Milestone Objective:** Build a commercially clean data ingestion pipeline that downloads permissively licensed speech/noise datasets, executes high-accuracy automated self-labeling with Silero-VAD (hysteresis & 10 ms rasterization), applies realistic SNR noise/RIR augmentations, and precomputes cached feature tensors for fast training.

---

## 1. Conceptual Deep Dive: Data Strategy & Label Generation

A Voice Activity Detector is only as good as its training distribution and ground-truth boundary accuracy:
1.  **Clean-Room Self-Labeling:** Instead of using non-commercial datasets (e.g. LibriVAD or the Silero labeled dataset which are CC BY-NC-SA), we take raw speech from LibriSpeech train-clean-100 (CC BY 4.0) and run inference using the MIT-licensed Silero-VAD model to generate 10 ms speech/non-speech labels.
2.  **Hysteresis Thresholding:** Raw frame probabilities fluctuate rapidly around voice boundaries (vowel onsets, breathing, consonant releases). Applying hysteresis (start threshold = 0.50, end threshold = 0.35) with minimum speech duration (250 ms) and silence duration (100 ms) suppresses label flicker and produces clean, stable speech segments.
3.  **Acoustic Augmentation Mix:** Real-world microphones capture wind, room reverberation, and background noise. We build a balanced 4-part training distribution:
    *   **25% Clean Speech:** Clean LibriSpeech audio without noise.
    *   **25% Wind-Mixed Speech:** Speech mixed with synthetic airflow wind noise at -5 dB SNR (challenging low-frequency turbulent energy).
    *   **50% DNS / MUSAN Noise-Mixed Speech:** Speech mixed with environmental noise drawn uniformly from {-10, -5, 0, +5, +10} dB SNR.
    *   **Reverberation:** 50% of all noisy samples are convolved with simulated Room Impulse Responses (RIRs via pyroomacoustics).
    *   **Zero Pure-Noise:** Every training sample contains speech with active/inactive regions (matches official config libri_dns_full_no_pure_noise_v2).

---

## 2. Mathematical Specs & SNR Scaling Formulas

### 2.1 Signal-to-Noise Ratio (SNR) Calculation
To mix speech signal s[n] with noise wn] at target SNR (in dB):
1. Compute Root Mean Square (RMS) energy:
   \\text{RMS}(s) = \\sqort{\\frac{1}{N}\\sum{n=0}^{N-1} s[n]^2}, \\quad \\text{RMS}(w) = \\sqrt{\\frac{1}{N}\\sum{n=0}^{N-1} w[n]^2}
2. Compute required noise gain factor g:
   g = \\frac{\\text{RMS}(s)}{\\text{RMS}(w) + 10 {-8}} \\cdot 10^{-\\frac{\\text{SNR}_{\\text{dB}}}{20}}
3. Mixed signal:
   x[n] = s[n] + g \\cdot w[n]
4. Prevent clipping: If max(1|x[n}||) > 1.0, normalize x[n] = x[n] / (max(1|x[n]||) + 1e-5).

### 2.2 10 ms Grid Rasterization Rule
Given a 200 ms audio window starting at timestamp T_start:
*   The window contains 21 frames, centered at t_k = T_start + k * 0.010 s for k in {0, 1, ..., 20}.
*   The label Y in {0, 1} for the 200 ms window is determined by whether speech is active at the decision anchor (or majority frame activity).

---

## 3. Step-by-Step Implementation Checklist

### Step 3.1: Dataset Download Utility (pulsevad/download_data.py)
- [ ] Create pulsevad/download_data.py.
- [ ] Implement downloader for LibriSpeech train-clean-100.tar.gz (OpenSLR 12).
- [ ] Implement downloader for MUSAN musan.tar.gz (OpenSLR 17).
- [ ] Implement downloader for DNS Challenge Noise (Interspeech 2020 CC BY/CC0 subset).
- [ ] Implement synthetic wind noise generator (pink noise filtered by low-pass Butterworth with time-varying gain).

### Step 3.2: Silero Self-Labeling Script (pulsevad/label_corpus.py)
- [ ] Create pulsevad/label_corpus.py.
- [ ] Load Silero-VAD model via torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=True).
- [ ] Process audio in 512-sample chunks @ 16 kHz without resetting LSTM hidden state across utterance.
- [ ] Implement hysteresis state machine:
  *   threshold = 0.50
  *   neg_threshold = 0.35
  *   min_speech_duration_ms = 250
  *   min_silence_duration_ms = 100
  *   speech_pad_ms = 30
- [ ] Output binary frame annotations to JSON/Numpy manifests in data/labels/.

### Step 3.3: Room Impulse Response Simulator (pulsevad/augment.py)
- [ ] Implement RoomReverbSimulator:
  *   Use pyroomacoustics.ShoeBox to simulate rooms with dimensions L in [3, 8], W in [3, 6], H in [2.5, 4] meters.
  *   Reverberation time T_60 in [0.15, 0.5] seconds.
  *   Convolve speech signal with generated RIR.

### Step 3.4: Feature Cache Generator (pulsevad/build_cache.py)
- [ ] Slice continuous audio into 200 ms segments with 100 ms hop (50% overlap for dataset expansion).
- [ ] Apply 25% clean / 25% wind / 50% noise + reverb mix.
- [ ] Push 3,200-sample segments through MelFrontend from Phase 1.
- [ ] Save tensors to disk as memory-mapped files or .pt chunks (N, 64, 21) alongside binary labels (N,).

---

## 4. Verification Gate & Unit Tests
Run `pytest tests/test_dataset.py`.

### Exit Criteria
- [ ] pytest tests/test_dataset.py passes 100%.
- [ ] 100 hours of LibriSpeech labeled and verified via QC spot-check.
- [ ] Feature cache is precomputed and verified on disk.
