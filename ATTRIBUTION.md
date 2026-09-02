# Attribution

PulseVAD is trained exclusively on commercially clean, permissively licensed datasets and
tools. This document acknowledges the sources used.

## Datasets

- **LibriSpeech train-clean-100** — CC BY 4.0 — Vassil Panayotov, Guoguo Chen, Daniel Povey,
  Sanjeev Khudanpur. Primary speech corpus.
- **Common Voice** — CC0 (Public Domain) — Mozilla. Multilingual speech expansion.
- **Multilingual LibriSpeech (MLS)** — CC BY 4.0. Multilingual speech expansion.
- **VoxLingua107** — CC BY 4.0. Multilingual speech expansion.
- **MUSAN (Noise / Music)** — CC BY 4.0 — David Snyder, Guoguo Chen, Daniel Povey. Noise
  augmentation and background noise.
- **DNS Challenge Noise (Interspeech 2020)** — CC BY 4.0 / CC0 subset — Chandan K. A. Reddy et
  al. Acoustic interference augmentation.
- **Synthetic Wind Noise (Mirabilii 2022)** — Open source algorithm, synthesized locally.
  Airflow simulation at -5 dB SNR.

## Tools

- **Silero-VAD (v5/v6)** — MIT License — Silero Team. Used only as an automated self-labeling
  engine; no Silero-released labeled dataset or weights are used or shipped.

## Explicitly Excluded (Non-Commercial, Do NOT Use)

- Silero Released Labeled Dataset — CC BY-NC-SA 4.0 — never downloaded or used.
- LibriVAD Dataset — CC BY-NC-SA 4.0 — never downloaded or used.
- Official kiloVAD Checkpoints — CC BY-NC 4.0 — referenced only as a numerical unit-test
  reference, never shipped or trained upon.
