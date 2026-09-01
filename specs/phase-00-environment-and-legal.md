# Phase 0: Environment Setup, Dependencies & Legal Footing

> **Milestone Objective:** Establish a clean development environment, verify all dependencies, lock legal/licensing boundaries, and set up project scaffolding before writing core logic.

---

## 1. Conceptual Deep Dive: Commercial Cleanliness & Clean-Room Rules

In deep learning for audio, dataset licenses dictate commercial viability. Many popular speech datasets and existing VAD checkpoints carry restrictive licenses:
*   **CC BY-NC-SA 4.0 (Non-Commercial, Share-Alike):** Prohibits commercial use and requires downstream models to adopt the same restrictive license.
*   **CC BY-NC 4.0:** The official kiloVAD reference code and weights on HuggingFace are licensed under CC BY-NC 4.0. **You cannot ship weights trained directly on or derived from their code/weights.**
*   **Clean-Room Reimplementation:** Ideas, mathematical equations, and published architectures are not protected by copyright; their specific code expression is. By implementing PulseVAD from the paper's mathematical specifications and training solely on permissive data, our weights are **100% commercially clean**.

### The Clean Dataset Matrix (Verified 2026)

| Dataset | Size | License | Commercial Status | Usage in PulseVAD |
| :--- | :--- | :--- | :--- | :--- |
| **LibriSpeech train-clean-100** | 100 h (6.3 GB) | CC BY 4.0 | **APPROVED** (Keep attribution) | Primary speech corpus (Run A & Run B) |
| **Common Voice (Mozilla)** | 3,000+ h | CC0 | **APPROVED** (Public Domain) | Multilingual speech expansion (Run B) |
| **Multilingual LibriSpeech (MLS)** | ~44k h | CC BY 4.0 | **APPROVED** (Keep attribution) | Multilingual speech expansion (Run B) |
| **VoxLingua107** | ~6.6k h | CC BY 4.0 | **APPROVED** (Keep attribution) | Multilingual speech expansion (Run B) |
| **MUSAN (Noise / Music)** | ~109 h | CC BY 4.0 | **APPROVED** (Keep attribution) | Noise augmentation & background noise |
| **DNS Challenge Noise (Interspeech 2020)** | Tens of GB | Mixed (CC BY 4.0 / CC0) | **APPROVED** (CC BY/CC0 subset) | Acoustic interference augmentation |
| **Synthetic Wind Noise (Mirabilii 2022)** | Algorithmic | Open source code | **APPROVED** (Synthesized locally) | Airflow simulation @ -5 dB SNR |
| **Silero-VAD (v5/v6 Model)** | Pretrained Model | MIT License | **APPROVED** (Model outputs clean) | Automated self-labeling engine |
| *Silero Released Labeled Dataset* | Dataset | CC BY-NC-SA 4.0 | **STRICTLY PROHIBITED** | Do NOT download or use |
| *LibriVAD Dataset* | 15 GB | CC BY-NC-SA 4.0 | **STRICTLY PROHIBITED** | Do NOT download or use |
| *Official kiloVAD Checkpoints* | Checkpoint .pth | CC BY-NC 4.0 | **STRICTLY PROHIBITED TO SHIP** | Numerical unit-test reference only |

---

## 2. Technical Requirements & Environment Specifications

### Python & System Stack
*   **Python:** >= 3.10 (Recommended: 3.10 or 3.11)
*   **OS Support:** Windows 10/11, Linux (Ubuntu 20.04/22.04+), macOS (Apple Silicon supported via MPS/CPU)
*   **Audio I/O:** soundfile, sox_io backend via 	orchaudio
*   **Deep Learning:** 	orch >= 2.1.0, 	orchaudio >= 2.1.0
*   **Compression & Export:** 	orch-pruning >= 1.4.0, onnx >= 1.15.0, onnxruntime >= 1.16.0
*   **Audio DSP & Simulation:** scipy >= 1.11.0, 
umpy >= 1.24.0, pyroomacoustics >= 0.7.0
*   **Evaluation & Metrics:** scikit-learn >= 1.3.0, matplotlib >= 3.8.0, 	qdm >= 4.66.0

### Compute Budget & Hardware Planning
*   **Local Development:** Any modern 4-core+ CPU with 16 GB RAM can develop, test frontend DSP, run unit tests, and perform INT8 inference.
*   **GPU Training (Run A - 40 Epochs, LibriSpeech 100h):**
    *   On cached features (Phase 3), 1 seed on an NVIDIA T4 / RTX 3060 takes **2 to 3 hours**.
    *   3 seeds cost ** - ** on serverless cloud platforms (Modal, RunPod, or Lambda Labs).
    *   Total project training compute fits well within a  budget.

---

## 3. Step-by-Step Implementation Checklist

### Step 0.1: Project Directory Scaffolding
- [ ] Create core folder structure:
  `ash
  mkdir -p pulsevad data/raw data/labels data/cache tests specs c_src docs
  `
- [ ] Create .gitignore to prevent committing heavy binary checkpoints and audio datasets:
  `	ext
  __pycache__/
  *.pyc
  data/raw/*
  data/labels/*
  data/cache/*
  *.pth
  *.onnx
  *.tflite
  .venv/
  dist/
  build/
  `

### Step 0.2: Dependency Specification (
equirements.txt)
- [ ] Create 
equirements.txt with pinned versions:
  `	ext
  torch>=2.1.0
  torchaudio>=2.1.0
  torch-pruning>=1.4.0
  numpy>=1.24.0,<2.0.0
  scipy>=1.11.0
  soundfile>=0.12.1
  pyroomacoustics>=0.7.5
  scikit-learn>=1.3.0
  matplotlib>=3.8.0
  tqdm>=4.66.0
  onnx>=1.15.0
  onnxruntime>=1.16.0
  pytest>=7.4.0
  `
- [ ] Create virtual environment and install dependencies:
  `ash
  python -m venv .venv
  # Windows:
  .venv\Scripts\activate
  # Linux/macOS:
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  `

### Step 0.3: Legal Compliance & Attribution Document
- [ ] Create ATTRIBUTION.md at project root acknowledging:
  *   LibriSpeech (CC BY 4.0, Vassil Panayotov et al.)
  *   MUSAN (CC BY 4.0, David Snyder et al.)
  *   Interspeech 2020 DNS Challenge (CC BY 4.0 / CC0, Chandan K. A. Reddy et al.)
  *   Silero-VAD (MIT License, Silero Team - used as self-labeling utility)

---

## 4. Verification Gate & Unit Test

Create and execute 	ests/test_env.py:

`python
import torch
import torchaudio
import numpy as np
import scipy
import torch_pruning
import onnx
import onnxruntime

def test_environment():
    print('=== Environment Verification ===')
    print(f'PyTorch: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})')
    print(f'Torchaudio: {torchaudio.__version__}')
    print(f'NumPy: {np.__version__}')
    print(f'Torch-Pruning: {torch_pruning.__version__}')
    print(f'ONNX: {onnx.__version__}')
    print(f'ONNX Runtime: {onnxruntime.__version__}')
    
    x = torch.randn(2, 64, 21)
    assert x.shape == (2, 64, 21), 'Tensor shape mismatch'
    print('[PASS] Environment is ready!')

if __name__ == '__main__':
    test_environment()
`

### Exit Criteria
*   [ ] python tests/test_env.py executes without errors.
*   [ ] ATTRIBUTION.md is present and verified.
*   [ ] No non-commercial datasets are referenced in download scripts.
