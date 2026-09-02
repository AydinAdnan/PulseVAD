# PulseVAD

An ultra-tiny, commercially clean, hardware-friendly Voice Activity Detector (VAD) — a from-scratch, spec-driven reimplementation of the **kiloVAD** paper (*"VAD to the Bone: Ultra-Tiny Speech Activity Detection for Edge Deployment"*, Analog Devices + UCLA, INTERSPEECH 2026) following the **PulseVAD Build Plan v2**.

| | |
|---|---|
| **Unpruned baseline** | 81,090 params · ~1.66M MACs / 200 ms · 0.862 AUC (AVA-Speech, causal) |
| **Pruned ship target** | 2,100 params · ~44k MACs / 200 ms · 0.850 AUC · **2.1 KB INT8** |
| **Latency** | 200 ms algorithmic (3× faster than 630 ms models: MarbleNet, TinyVAD, AtomicVAD) |
| **Architecture** | CNN-only: depthwise-separable + dilated convs, GAP head, no LSTMs/GRUs, no custom activations |
| **Portability** | static graph → TFLite / TFLM / ONNX / pure-C fixed-point |
| **License posture** | trained only on permissive data (CC BY 4.0 / CC0) + Silero-VAD (MIT) as a labeling tool — 100% commercially clean weights |

---

## How it works

```
16 kHz mono audio
      │  200 ms window (3,200 samples)
      ▼
┌─────────────────────────────────────────────┐
│ Phase 1 · Audio Frontend (pulsevad/frontend.py)
│  pre-emphasis (0.97) → waveform z-norm      │
│  → Mel spectrogram (64 bins, 21 frames)     │
│  → log(mel + 1e-5) → per-bin z-norm         │
└──────────────┬──────────────────────────────┘
               ▼  (B, 64, 21)
┌─────────────────────────────────────────────┐
│ Phase 2 · CNN Backbone (pulsevad/model.py)  │
│  adapter 1×1 (64→128)                       │
│  DW-sep k=11 → 1×1 projections (128→64→64)  │
│  residual block k=17 → dilated DW k=29 d=2  │
│  pointwise 1×1 → GAP → Linear(128→2)        │
└──────────────┬──────────────────────────────┘
               ▼  logits (B, 2)
        P(speech) ≥ threshold → trigger
```

**Commercially clean by construction:** instead of non-commercial labeled datasets (LibriVAD, Silero's labeled set — CC BY-NC-SA), raw LibriSpeech (CC BY 4.0) is self-labeled with MIT-licensed Silero-VAD (hysteresis 0.50/0.35, 10 ms rasterization). Noise comes from MUSAN (CC BY 4.0); wind is synthesized locally; reverb via pyroomacoustics. See `ATTRIBUTION.md`.

## Repository layout

```
pulsevad/
├── frontend.py       # deterministic DSP: waveform → (B, 64, 21) log-Mel
├── model.py          # PulseVAD backbone — exactly 81,090 params
├── scheduler.py      # cyclic warmup/hold/cosine LR (4/16/20 epochs)
├── train.py          # SGD+Nesterov training engine over cached features
├── download_data.py  # LibriSpeech / MUSAN / DNS-noise downloaders
├── label_corpus.py   # Silero-VAD self-labeling + hysteresis + 10 ms rasterization
├── augment.py        # SNR mixing math, synthetic wind, RIR simulation
├── build_cache.py    # 25% clean / 25% wind / 50% noise mixing → memmapped features
modal_app.py          # Modal cloud pipeline (download → label → cache → train)
specs/                # 9-phase spec-driven development documents
tests/                # pytest suite (37 tests)
data/                 # raw audio, labels, feature cache (gitignored)
docs/                 # kiloVAD paper + build plan PDFs
```

## Setup

```bash
uv sync          # installs the package editable + locked deps
uv run pytest    # 37 tests must pass
```

## Cloud pipeline (Modal)

Data analysis, dataset prep, and the training loop run on [Modal](https://modal.com) — datasets land on a named volume (`pulsevad-data`), training reads the cache in-cloud, and only small artifacts (manifests, eval sets, checkpoints) get pulled back locally. **Every cloud command below is triggered manually.**

```bash
uv run modal setup                          # one-time auth

# Phase 3 — dataset pipeline (one-time)
uv run modal run modal_app.py::download     # ~17 GB → volume (~20-40 min)
uv run modal run modal_app.py::label        # Silero self-labeling, 28,539 files (~30 min)
uv run modal run modal_app.py::verify       # QC gate: labeled_files == 28539, missing == 0
uv run modal run modal_app.py::build_cache  # mix/augment/featurize → cache (~1-2 h)

# pull small artifacts back
uv run modal volume get pulsevad-data cache/manifest.json ./data/cache/
uv run modal volume get pulsevad-data cache/eval_sets ./data/cache/eval_sets
uv run modal volume get pulsevad-data labels/_summary.json ./data/labels/

# Phase 4 — base training (3 seeds; smoke test first)
uv run modal run modal_app.py::train --seed 0 --epochs 2   # smoke test (~5 min)
uv run modal run modal_app.py::train --seed 0              # full 40-epoch run (~2-3 h on T4)
uv run modal run modal_app.py::train --seed 1
uv run modal run modal_app.py::train --seed 2

# pull checkpoints
uv run modal volume get pulsevad-data runs ./runs/
```

**Key checkpoints during the pipeline:**
- `verify` must print `missing: 0` before `build_cache`
- `build_cache` manifest: train ≈ 3.3M windows, categories ≈ 25/25/50, 5 held-out eval sets (`pure_noise` speech fraction = 0.0)
- `train` per-epoch JSON logs: loss ↓, val AUC → ~0.86–0.88 in-domain (paper gate: causal AVA AUC within ~0.005 of 0.862)

## Training recipe (paper-faithful)

| Hyperparameter | Value |
|---|---|
| Optimizer | SGD, momentum 0.9, Nesterov |
| Weight decay | 8.75e-4 |
| LR schedule | 4-epoch warmup → 3.5e-3, 16-epoch hold, 20-epoch cosine → 1e-5 |
| Loss | CrossEntropy, label smoothing ε = 0.09 |
| Epochs / batch | 40 / 512 |
| Data mix | 25% clean / 25% wind @ −5 dB / 50% DNS+MUSAN noise @ {−10..+10} dB, 50% reverb, 0% pure-noise |

## Status

- [x] Phase 0 — environment, licensing, scaffolding
- [x] Phase 1 — audio frontend (9 tests)
- [x] Phase 2 — model architecture, exact 81,090 params (7 tests)
- [x] Phase 3 — dataset ingestion, Silero self-labeling, feature cache (12 tests)
- [x] Phase 4 — base training engine + cyclic LR (8 tests)
- [ ] Phase 5 — DepGraph structured pruning + self-distillation (→ 2.1k params)
- [ ] Phase 6 — INT8 PTQ + ONNX/TFLite export
- [ ] Phase 7 — strictly causal evaluation + cross-VAD benchmarking
- [ ] Phase 8 — embedded C streaming engine + hardware profiling

## Engineering rules baked in

1. Strictly causal evaluation — no overlap, no smoothing, no future context
2. Frame exactness: `floor(3200/160) + 1 = 21` frames
3. Normalization order: pre-emphasis → waveform z-norm → mel → log → per-bin z-norm
4. BatchNorm `eps=1e-3` (not PyTorch's 1e-5 default), momentum 0.1
5. Bias-free convolutions (BN supplies the bias)
6. Structured pruning only via DepGraph (coupled channel groups)
7. Per-layer pruning ratios (global uniform collapses below ~2k params)
8. Distillation teacher strictly frozen
9. Label smoothing ε = 0.09 absorbs auto-label boundary noise
10. Commercially clean data only — see `ATTRIBUTION.md`
11. ≥3 seeds, mean ± 95% CI (pruned variance ±0.007 AUC)
12. Zero pure-noise training samples (config `libri_dns_full_no_pure_noise_v2`)
