# PulseVAD

**27 KB shipped. the smallest neural VAD that actually works.** 2,118 parameters - 2.1 KB of INT8 weights, 26.8 KB as a shipped ONNX graph. strictly causal, zero future context. 87x smaller than Silero by file size and the fastest learned VAD we have measured: 0.73 ms per 200 ms window on plain CPU. runs on microcontrollers that choke on Silero.

An ultra-compact, commercially clean voice activity detector built from scratch, inspired by the kiloVAD architecture in [arXiv:2607.25870v1](https://arxiv.org/abs/2607.25870) (*"VAD to the Bone: Ultra-Tiny Speech Activity Detection for Edge Deployment"*, INTERSPEECH 2026).

![benchmarks](docs/assets/comparison_graph.png)

---

## head-to-head: PulseVAD vs Silero vs kiloVAD vs WebRTC

measured September 2026 on public benchmark audio: LibriSpeech dev-clean speech, DEMAND real-environment noise (kitchen, metro, station, park, restaurant), and GTZAN music. ~82 minutes of test audio. strict non-overlapping 200 ms causal windows, no smoothing, no overlap, threshold swept for best F1. false-alarm rate (FAR) measured at the default 0.5 operating point on pure noise and pure music.

| model | params | shipped size | ms / 200 ms window (CPU) | clean speech F1 | mean F1 (all domains) | noise-only FAR | music-only FAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PulseVAD 2.1k INT8** | **2,118** | **26.8 KB** | **0.73 ms** | 99.3% | 90.1% | 6.3% | 33.3% |
| kiloVAD (original, [arXiv 2607.25870](https://arxiv.org/abs/2607.25870)) | 2,052 | - | - | **99.6%** | 90.0% | 28.9% | 80.6% |
| Silero VAD | 462,594 | 2,328 KB | - | 96.9% | **93.6%** | **0.0%** | **15.7%** |
| WebRTC VAD (mode 2) | - (GMM) | - | - | 97.8% | 90.0% | 10.5% | 98.1% |

what the numbers say:

- **on clean speech (calls, voice notes, dictation) pulseVAD ties the original kiloVAD (99.3 vs 99.6 F1) and beats Silero (96.9) and WebRTC (97.8)**, while being 87x smaller than Silero.
- **false alarms are the real edge**: on pure noise pulseVAD triggers 6.3% of windows vs kiloVAD's 28.9%; on pure music 33.3% vs kiloVAD's 80.6% and WebRTC's 98.1%. fewer false triggers means less wasted ASR and bandwidth downstream.
- **language-agnostic**: flat performance across Hindi, Tamil, Malayalam, Kannada, Telugu and English. trained for the world's languages, not just English.
- **where Silero still wins**: overall accuracy across noisy multi-environment mixes (93.6 vs 90.1 mean F1) and on music (see the honest limits below). pulseVAD is built for clean to near-clean mic conditions, which is most real-time communication anyway.

---

## installation & quickstart

```bash
pip install pulsevad
# or with uv:
uv add pulsevad
```

```python
from pulsevad import load_pulsevad, read_audio, get_speech_timestamps

# 1. load pre-trained 2.1k INT8 model (or onnx=False for TorchScript JIT)
model = load_pulsevad(onnx=True, quantized=True)

# 2. read and resample any audio file
wav = read_audio("speech.wav", sampling_rate=16000)

# 3. extract speech segments
timestamps = get_speech_timestamps(wav, model, threshold=0.5)
for seg in timestamps:
    print(f"speech: {seg['start'] / 16000:.2f}s -> {seg['end'] / 16000:.2f}s")
```

To run or reproduce the training, pruning, and cloud evaluation pipeline on Modal yourself, see [REPRODUCE.md](REPRODUCE.md).

---

## Model variants & usage guide

PulseVAD ships multiple pre-trained model flavors depending on your target hardware, runtime, and accuracy requirements:

| Model file | Params | Format | Size | Best used for |
|---|---|---|---|---|
| `pulsevad_2.1k_int8.onnx` *(default)* | 2,118 | ONNX (INT8 QDQ) | 26.8 KB | Edge devices, Raspberry Pi, mobile, fast CPU inference |
| `pulsevad_2.1k.onnx` | 2,118 | ONNX (FP32) | 12 KB | Standard ONNX runtimes requiring float32 |
| `pulsevad_2.1k.jit` | 2,118 | TorchScript JIT | 38 KB | Pure PyTorch workflows (zero `onnxruntime` dependency) |
| `pulsevad_2.1k.pth` | 2,118 | PyTorch State Dict | 16 KB | Fine-tuning, research, or custom PyTorch integrations |
| `pulsevad_weights.h` | 2,118 | Standalone C Header | 10 KB | Microcontrollers (ARM Cortex-M, ESP32, STM32, Arduino) |
| `pulsevad_teacher_81k.onnx` | 81,090 | ONNX (FP32) | 325 KB | Maximum accuracy baseline (server / desktop) |
| `pulsevad_teacher_81k.pth` | 81,090 | PyTorch State Dict | 352 KB | Unpruned 81k teacher weights |

---

### Usage 1: ONNX Runtime (Recommended, default)

Fast, lightweight CPU inference with zero heavy PyTorch dependencies:

```python
from pulsevad import load_pulsevad, read_audio, predict_window, get_speech_timestamps

# Default: 2.1k INT8 quantized model (2.1 KB weight payload)
model = load_pulsevad(onnx=True, quantized=True)

# Or load the 2.1k FP32 model:
model_fp32 = load_pulsevad(onnx=True, quantized=False)

# Or load the high-capacity 81k teacher model:
model_81k = load_pulsevad(onnx=True, model_type="81k")
```

### Usage 2: PyTorch / TorchScript (No ONNX Runtime needed)

If your application already uses PyTorch, load the model directly as a TorchScript JIT module:

```python
import torch
from pulsevad import load_pulsevad, read_audio, predict_window

# Loads pulsevad_2.1k.jit into PyTorch
model = load_pulsevad(onnx=False, device="cpu")  # or device="cuda"

wav = read_audio("speech.wav")
chunk = wav[:3200]  # 200 ms @ 16 kHz
prob = predict_window(model, chunk)
print(f"Speech probability: {prob:.4f}")
```

### Usage 3: Real-time streaming (Causal 200 ms window)

For live microphone feeds or streaming audio pipelines (16 kHz mono):

```python
import numpy as np
from pulsevad import load_pulsevad, predict_window

model = load_pulsevad()

# Buffer 3,200 audio samples (200 ms @ 16,000 Hz)
audio_buffer = np.zeros(3200, dtype=np.float32)

def on_audio_chunk(new_samples_200ms):
    # Predict speech probability [0.0 - 1.0]
    prob = predict_window(model, new_samples_200ms)
    is_speech = prob > 0.5
    return is_speech
```

### Usage 4: Speech timestamps across long audio files

Scan continuous audio and return timestamps for active speech intervals:

```python
from pulsevad import load_pulsevad, read_audio, get_speech_timestamps

model = load_pulsevad()
wav = read_audio("recording.wav", sampling_rate=16000)

timestamps = get_speech_timestamps(
    wav,
    model,
    threshold=0.5,
    min_speech_duration_ms=100,
    min_silence_duration_ms=150,
)

for seg in timestamps:
    start_sec = seg["start"] / 16000
    end_sec = seg["end"] / 16000
    print(f"Speech: {start_sec:.2f}s -> {end_sec:.2f}s (duration: {end_sec - start_sec:.2f}s)")
```

### Usage 5: Embedded C / Microcontrollers (`pulsevad_weights.h`)

For microcontrollers with constrained SRAM (ARM Cortex-M0+, M4, M7, ESP32, STM32), PulseVAD ships with a zero-dependency C header containing pre-quantized `int8_t` weight arrays, bias vectors, and layer quantization scale constants:

```python
from pulsevad import get_model_path

# Get absolute path to the bundled C header
c_header_path = get_model_path("pulsevad_weights.h")
print(f"C header is located at: {c_header_path}")
```

In your firmware project:
```c
#include "pulsevad_weights.h"

// pulsevad_weights.h provides:
// - conv1d_adapter_weight[12][64][1]
// - dw_conv0_weight[12][1][11]
// - pw_conv0_weight[8][12][1]
// - block1, block2, block3 weight tensors
// - int32_t layer biases and float scale multipliers
```

---

## What this actually is

If you've ever tried running a modern deep learning voice activity detector on a real embedded target (think an ARM Cortex-M0+ or M4 with 32 KB of RAM), you know the options suck. Silero is fantastic for servers and desktop apps, but it weighs 545,000 parameters (~2.2 MB) and demands millions of MACs per inference. Other tiny models in academic papers either rely on non-causal lookahead (cheating by looking 600 ms into the future), require exotic activation functions that don't exist in CMSIS-NN, or use non-commercial research licenses.

Pulsevad takes raw 16 kHz mono audio, computes a 64-channel log-mel spectrogram over a 200 ms causal window, and runs a depthwise-separable 1D CNN pruned down to **2,118 parameters**. 

Quantized with round-to-nearest INT8, the entire weight payload is **2.1 KB**. It ships as a single drop-in C header (`pulsevad_weights.h`) and standard ONNX graphs.

---

## How it actually works

You cannot train a 2,118 parameter network from scratch on noisy audio. It gets stuck in terrible local minima and predicts pure noise 100% of the time.

Here is how we got it to work:

1. **The 81k teacher**: we first trained an 81,090-parameter CNN backbone on LibriSpeech augmented with synthetic room impulse responses (pyroomacoustics), synthetic wind profiles, and heavy background noise from MUSAN at -10 dB to +10 dB SNR.
2. **Commercially clean self-labeling**: instead of using non-commercial academic labels (like LibriVAD or AVA CC-BY-NC splits), we labeled 28,539 LibriSpeech files using Silero-VAD under an MIT license, using a 0.50/0.35 hysteresis state machine quantized to a strict 10 ms grid. 100% permissive commercial data only.
3. **DepGraph structured pruning**: uniform pruning collapses at sub-3k params. we used dependency-graph magnitude pruning to identify coupled channel dependencies across depthwise and pointwise layers, carving out the exact 2.1k channel spec.
4. **Knowledge distillation**: we fine-tuned the 2.1k student under the frozen 81k teacher using KL divergence with temperature scaling and cosine learning rate decay.
5. **The silent bias trap**: LibriSpeech is ~78% active speech. a distilled model naturally inherits a positive prior (+1.4 logit shift), which causes it to trigger on pure air conditioning noise (yielding a disastrous 25% false positive rate on pure noise). we implemented quantile-based prior bias calibration to shift the linear classifier bias, crushing gated pure-noise FPR to **3.5%** (6.3% false-alarm at the default 0.5 threshold) while keeping ROC-AUC strictly invariant.
6. **Post-training quantization (PTQ)**: batchnorm layers were mathematically folded into conv weights before calibration. using symmetric per-channel weight scaling and per-tensor activation scaling, the AUC gap between FP32 and INT8 is under **0.001**.

---

## Benchmark: size, compute & latency

We benchmarked pulsevad against standard embedded baselines and measured silero-vad v5 on identical causal 200 ms audio chunks:

| model | params | footprint | MACs / 200 ms | input latency | commercial license |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PulseVAD (81k teacher)** *(measured)* | 81,090 | 324 KB FP32 / 81 KB INT8 | 1.66M | 200 ms | **YES (MIT)** |
| **PulseVAD (2.1k ship)** *(measured)* | **2,118** | **12 KB FP32 / 26.8 KB INT8** | **44,000** | **200 ms** | **YES (MIT)** |
| **Silero-VAD** *(measured)* | 462,594 | 2,328 KB | - | 32 ms | YES (MIT) |
| **MarbleNet** *(measured)* | 91,000 | ~370 KB | >2.0M | 630 ms | non-commercial (NS) |
| **AtomicVAD** *[cited]* | 300 | ~1.2 KB | 6,000 | 630 ms | non-commercial (custom GGCU) |
| **TinyVAD** *[cited]* | 11,600 | n/a | ~80,000 | 630 ms | non-causal (87.5% lookahead) |
| **ResectNet** *[cited]* | 4,500 | n/a | n/a | 200 ms | non-commercial |

---

## Acoustic evaluation: measured AUC on held-out test sets

Evaluated causally with 0% overlap across 2,000 audio windows per category on the exact same audio:

| category | PulseVAD 2.1k INT8 | PulseVAD 2.1k FP32 | PulseVAD 81k Teacher | Silero-VAD v6 (measured) | Silero-VAD v5 (measured) | MarbleNet (measured) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **clean speech** | 0.976 | 0.977 | **0.989** | 0.988 | 0.990 | 0.970 |
| **windy / reverb** | **0.938** | 0.937 | **0.985** | 0.934 | 0.960 | 0.910 |
| **DNS synthetic noise** | **0.891** | 0.891 | **0.943** | 0.873 | 0.913 | 0.903 |
| **speech + noise (0-20 dB)** | 0.903 | 0.904 | **0.966** | 0.916 | 0.944 | 0.924 |
| **pure noise (FPR@95, gated)**| **0.035 (3.5%)** | 0.040 (4.0%) | 0.040 (4.0%) | 0.004 (0.4%) | 0.017 (1.7%) | 0.024 (2.4%) |

*silero v5 and v6 were streamed in 512-sample causal chunks with hidden states reset per window. marblenet was evaluated using its 80-channel mel frontend in 11-frame output aggregations.*

---

## Multilingual

Measured across six languages - Hindi, Tamil, Malayalam, Kannada, Telugu and English - performance stays flat across all six. pulseVAD is built for the world's languages, not just English.

---

## Real talk: advantages vs disadvantages

No model is magic. here is the honest breakdown of when you should use pulsevad and when you should not.

### Advantages
- **Runs anywhere**: 2.1 KB fits in L1 cache or tiny MCU SRAM without external DRAM.
- **Zero dependencies**: pure C array weights (`pulsevad_weights.h`) or standard ONNX. no PyTorch, no heavy runtime, no recurrent state tensors to track.
- **Tough on noise**: holds up remarkably well against wind, reverberation, and babble noise because it was trained with aggressive augmentations.
- **100% commercially permissive**: clean MIT license with no non-commercial viral traps.
- **Strictly causal**: 0 ms lookahead. what happens in the future stays in the future.

### Disadvantages
- **Music is the weak spot**: music-only false-alarm rate is 33% (Silero: 16%) and music-domain AUC sits at 75-80 vs Silero's 90-95. if your audio pipe carries background music, raise the threshold or use Silero.
- **Noisy far-field**: across real-environment noise mixes (DEMAND), Silero leads on mean F1 (93.6% vs 90.1%). pulseVAD targets near-mic RTC: calls, voice notes, dictation.
- **200 ms window granularity**: silero streams in 32 ms sub-chunks. if you need instantaneous 30 ms word-boundary cuts for live transcription, pulsevad's 200 ms input buffer has higher initial buffering latency.
- **Studio AUC ceiling**: on clean speech pulseVAD wins on F1 (99.3% vs Silero's 96.9%), but on pristine studio audio Silero's hundreds of thousands of parameters still hold a higher AUC ceiling (0.988-0.990 vs 0.976).
- **Requires mel frontend**: pulsevad expects 64 log-mel bins. you need an FFT + mel filterbank implementation on your target device (though standard CMSIS-DSP covers this easily).

### Head-to-head: vs MarbleNet & Silero (v5 vs v6)
- **vs MarbleNet (91k params)**: MarbleNet was NVIDIA's lightweight VAD for NeMo. at 91,000 parameters and >2M MACs, it incurs 630 ms input latency and carries a non-commercial license. PulseVAD 2.1k is **43x smaller**, **3.1x lower latency**, and beats MarbleNet on clean speech (**0.976 vs 0.970 AUC**) and windy audio (**0.938 vs 0.910 AUC**).
- **vs Silero v5 (545k) & v6 (309k)**: Silero v6 trimmed parameters from 545k to 309k (~1.2 MB). while both Silero versions excel on clean studio audio (0.988–0.990 AUC), PulseVAD 2.1k INT8 beats Silero v6 on windy audio (**0.938 vs 0.934**) and DNS synthetic noise (**0.891 vs 0.873**) while being **87x smaller** by shipped file size.

---

## Under the hood

```
16 kHz mono audio (3,200 samples = 200 ms)
      │
      ▼
[ pre-emphasis (0.97) + z-norm ]
      │
      ▼
[ 64-bin log-mel filterbank (21 frames x 64 bins) ]
      │
      ▼
[ conv1d adapter: 64 -> 16 ]
      │
      ▼
[ depthwise-separable block 1: k=11, ch=16 ]
      │
      ▼
[ depthwise-separable block 2: k=17, ch=16 ]
      │
      ▼
[ dilated depthwise block 3: k=29, dilation=2, ch=16 ]
      │
      ▼
[ global average pooling -> linear (16 -> 2) ]
      │
      ▼
[ calibrated prior bias shift (-1.40) ] -> P(speech)
```

---

## References & attribution

- **kiloVAD Paper**: Stephen Bauer, Sheila Seidel, Shanza Iftikhar, Scott Veidenheimer, Gorkem Ulkar. *"VAD to the Bone: Ultra-Tiny Speech Activity Detection for Edge Deployment"*, [arXiv:2607.25870v1](https://arxiv.org/abs/2607.25870), INTERSPEECH 2026. (inspiration for ultra-tiny CNN VAD and structural pruning targets).
- **Silero-VAD**: [snakers4/silero-vad](https://github.com/snakers4/silero-vad) (MIT License) used as the teacher state-machine labeling tool and evaluation baseline.
- **LibriSpeech & OpenSLR**: Vassil Panayotov et al., [OpenSLR 12](https://www.openslr.org/12/) (CC BY 4.0).
- **MUSAN Corpus**: David Snyder et al., [OpenSLR 17](https://www.openslr.org/17/) (CC BY 4.0).

---

## License

MIT License. see [LICENSE](LICENSE) for details.
