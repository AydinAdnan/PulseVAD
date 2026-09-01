# Phase 1: Audio Frontend & Feature Extraction

> **Milestone Objective:** Implement a deterministic, amplitude-agnostic Audio Frontend in PyTorch/NumPy that transforms raw 16 kHz audio waveforms into normalized Log-Mel spectrograms of exact shape (Batch, 64, 21) matching the paper's mathematical specification to floating-point precision.

---

## 1. Conceptual Deep Dive: Audio Signal Processing for VAD

### 1.1 Why 16 kHz Mono & 200 ms Window?
*   **16 kHz Sampling Rate:** Standard sampling rate for modern speech applications (covers up to 8 kHz Nyquist frequency, containing all critical human speech formants, vowels, and fricatives).
*   **200 ms Window (3,200 samples):** At conversational speech rates (4-5 Hz syllable rate), one syllable lasts ~200-250 ms. Capturing a 200 ms window provides the neural network with sufficient spectral-temporal context to identify vowel formants and consonant bursts while slashing algorithmic latency by **3x** compared to older 630 ms models (MarbleNet, TinyVAD, AtomicVAD).

### 1.2 Step-by-Step DSP Transformation Chain

`
Raw Audio x[n] (3200 samples @ 16 kHz)
   |
   v
[ 1. Pre-emphasis Filter ] ----> y[n] = x[n] - 0.97 * x[n-1]
   |
   v
[ 2. Waveform Z-Normalization ] -> z[n] = (y[n] - mu_y) / (sigma_y + 1e-5)
   |
   v
[ 3. STFT & Mel Filterbank ] ---> 64 Mel triangular filters (0 to 8000 Hz)
   |                              n_fft=512, win=400 (Hann), hop=160, center=True
   v
[ 4. Log Compression ] ---------> log(Mel_power + 1e-5)
   |
   v
[ 5. Per-Bin Window Z-Norm ] ---> For each of the 64 bins, normalize across 21 time frames
   |
   v
Final Normalized Tensor: (Batch, 64, 21)
`

#### Step 1: Pre-emphasis Filtering
Human speech has a natural spectral tilt: higher frequencies roll off at approximately -6 dB/octave due to glottal pulse radiation. Pre-emphasis applies a first-order high-pass Finite Impulse Response (FIR) filter:
y[n] = x[n] - \alpha \cdot x[n-1], \quad \alpha = 0.97
*Purpose:* Boosts high-frequency components (consonants, fricatives like /s/, /f/, /t/) relative to low-frequency vowel energy, ensuring balanced gradient flow across all frequency bands.

#### Step 2: Amplitude-Agnostic Waveform Normalization
Before computing the spectrogram, normalize the 200 ms pre-emphasized waveform window:
\mu_y = \frac{1}{N} \sum_{n=0}^{N-1} y[n], \quad \sigma_y = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} (y[n] - \mu_y)^2}
\tilde{y}[n] = \frac{y[n] - \mu_y}{\sigma_y + 10^{-5}}
*Purpose:* Makes the feature representation invariant to recording volume, microphone sensitivity, and distance from speaker.

#### Step 3: Short-Time Fourier Transform (STFT) & Mel Filterbank
*   
_fft: **512** (frequency resolution = 16000 / 512 = 31.25 Hz per bin)
*   win_length: **400 samples** (25 ms Hann window, zero-padded to 512)
*   hop_length: **160 samples** (10 ms frame step)
*   center: **True** (reflect-pads 256 samples on each side)
*   power: **2.0** (power spectrogram $|X(f)|^2$)
*   
_mels: **64 triangular filterbanks** spaced linearly along the Mel frequency scale between 0 Hz and 8000 Hz.

**Exact Frame Calculation:**
\text{frames} = \left\lfloor \frac{N}{\text{hop}} \right\rfloor + 1 = \left\lfloor \frac{3200}{160} \right\rfloor + 1 = 20 + 1 = \mathbf{21}

#### Step 4: Logarithmic Compression
L[c, t] = \ln\left(M[c, t] + 10^{-5}\right)
*Purpose:* Human perception of loudness is logarithmic. Adding ^{-5}$ prevents $\ln(0)$ numerical instability.

#### Step 5: Per-Bin Window Normalization (Temporal Z-Score)
For each frequency channel  \in \{0, 1, \dots, 63\}$, compute mean and standard deviation across the 21 time frames:
\mu_c = \frac{1}{21} \sum_{t=0}^{20} L[c, t], \quad \sigma_c = \sqrt{\frac{1}{21} \sum_{t=0}^{20} (L[c, t] - \mu_c)^2}
\hat{L}[c, t] = \frac{L[c, t] - \mu_c}{\sigma_c + 10^{-5}}
*Purpose:* Normalizes stationary spectral noise (e.g., constant HVAC hum or fan noise) down to zero mean in each bin, allowing the CNN to detect dynamic speech onset and formant transitions.

---

## 2. Mathematical Specs & Exact Tensor Dimensions

| Parameter | Value | Note |
| :--- | :--- | :--- |
| Audio Sample Rate | 16,000 Hz | Single channel (mono) |
| Input Audio Duration | 0.200 s (200 ms) | Exactly 3,200 audio samples |
| Pre-emphasis coefficient | 0.97 | [n] = x[n] - 0.97 x[n-1]$ |
| STFT FFT Size | 512 | Zero-padded from 400 |
| Window Function | Hann (400 samples / 25 ms) | Periodic |
| Hop Length | 160 samples (10 ms) | 100 frames/sec |
| Mel Bins | 64 | Spaced 0 to 8000 Hz |
| Log offset | ^{-5}$ | Natural logarithm $\ln$ |
| Per-bin norm epsilon | ^{-5}$ | Added to standard deviation |
| **Output Tensor Shape** | **(Batch, 64, 21)** | **(B, Channels=64, Time=21)** |

---

## 3. Step-by-Step Implementation Checklist

### Step 1.1: Audio Preprocessing Module (pulsevad/frontend.py)
- [ ] Create pulsevad/frontend.py.
- [ ] Implement pre_emphasis(waveform: torch.Tensor, alpha: float = 0.97) -> torch.Tensor:
  *   Apply filter along the last dimension.
- [ ] Implement waveform_z_norm(waveform: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
  *   Subtract mean and divide by (std + eps) along the last dimension.
- [ ] Implement MelFrontend(torch.nn.Module):
  *   Initialize 	orchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000.0, n_mels=64, power=2.0, center=True, pad_mode='reflect', norm='slaney', mel_scale='htk').
  *   Ensure window is 	orch.hann_window(400).
- [ ] Implement per-bin temporal standardization:
  *   mean = log_mel.mean(dim=-1, keepdim=True)
  *   std = log_mel.std(dim=-1, keepdim=True, unbiased=False)
  *   
ormalized = (log_mel - mean) / (std + 1e-5)
- [ ] Ensure input assertion: Raise ValueError if waveform.shape[-1] != 3200.

---

## 4. Verification Gate & Unit Tests

Create 	ests/test_frontend.py:

`python
import torch
import pytest
from pulsevad.frontend import MelFrontend, pre_emphasis

def test_frontend_output_shape():
    frontend = MelFrontend()
    batch_size = 4
    audio_3200 = torch.randn(batch_size, 3200)
    features = frontend(audio_3200)
    assert features.shape == (batch_size, 64, 21), f'Expected (4, 64, 21), got {features.shape}'

def test_frontend_zero_input():
    frontend = MelFrontend()
    silent_audio = torch.zeros(1, 3200)
    features = frontend(silent_audio)
    assert not torch.isnan(features).any(), 'NaN detected in silent input'
    assert not torch.isinf(features).any(), 'Inf detected in silent input'

def test_frontend_statistics():
    frontend = MelFrontend()
    audio = torch.randn(2, 3200)
    features = frontend(audio)
    mean = features.mean(dim=-1)
    std = features.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-4)
`

### Exit Criteria
*   [ ] pytest tests/test_frontend.py passes 100%.
*   [ ] Output tensor shape is strictly (Batch, 64, 21) with zero NaNs.
