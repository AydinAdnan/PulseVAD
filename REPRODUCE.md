# reproduce pulsevad end-to-end

this guide documents how to reproduce the entire pulsevad training, pruning, quantization, export, and cross-vad benchmarking pipeline on modal.

---

## prerequisites

1. install `uv` (fast python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. clone this repo and sync dependencies:
   ```bash
   git clone https://github.com/AydinAdnan/PulseVAD.git
   cd PulseVAD
   uv sync
   ```
3. authenticate with modal:
   ```bash
   uv run modal setup
   ```

---

## pipeline execution steps

### 1. download training corpora
pulls librispeech dev-clean (speech) and musan (noise, music, speech) onto the `pulsevad-data` cloud volume:
```bash
uv run modal run modal_app.py::download
```

### 2. generate teacher speech labels
labels audio with silero vad hysteresis state machine into canonical 10 ms grid manifests:
```bash
uv run modal run modal_app.py::label
```

### 3. synthesize augmented window cache
generates 3.4m augmented 200 ms windows with noise mixing, simulated rirs, and noise-only rejection samples:
```bash
uv run modal run modal_app.py::build_cache --seed 0
```

### 4. train unpruned teacher model (81k params)
trains the unpruned 81k cnn teacher on the augmented cache for 40 epochs with cosine annealing:
```bash
uv run modal run modal_app.py::train --seed 0
```

### 5. structured pruning & self-distillation (2.1k student)
applies depgraph magnitude pruning to collapse channels to the exact 2,118 parameter spec and runs 8-epoch self-distillation with kl divergence:
```bash
uv run modal run modal_app.py::distill --seed 0
```

### 6. batchnorm folding, int8 quantization & onnx / c export
folds batchnorm into conv weights, runs rtn int8 quantization with activation calibration, verifies the <= 0.002 auc gap, and exports onnx pairs and `pulsevad_weights.h`:
```bash
uv run modal run modal_app.py::export --seed 0
```

### 7. cache audio for competitor evaluation
regenerates the 5 held-out evaluation sets (clean, windy, dns synthetic, speech+noise, pure noise) with raw audio preserved so all models score identical windows:
```bash
uv run modal run modal_app.py::cache_eval_audio
```

### 8. run all evaluations in parallel
```bash
# terminal 1: held-out evaluation on pulsevad models (teacher, 2.1k fp32, 2.1k int8)
uv run modal run modal_app.py::eval_heldout --seed 0

# terminal 2: score silero-vad v5 on the exact same 200 ms windows
uv run modal run modal_app.py::eval_competitors

# terminal 3: score marblenet on the exact same windows (nemo container)
uv run modal run modal_app.py::eval_marblenet

# terminal 4: multilingual benchmark across 10 global & indian languages (fleurs)
uv run modal run modal_app.py::eval_multilingual
```

### 9. assemble cross-vad benchmark comparison
aggregates measured rows and cited competitor numbers into the final comparison artifacts:
```bash
uv run modal run modal_app.py::build_comparison --seed 0
```

### 10. download local artifacts
pull the final reports, c header, onnx graphs, and benchmark tables to your local workspace:
```bash
uv run modal volume get --force pulsevad-data runs/pruned_seed_0/comparison.md ./data/runs/pruned_seed_0/comparison.md
uv run modal volume get --force pulsevad-data runs/pruned_seed_0/comparison.json ./data/runs/pruned_seed_0/comparison.json
uv run modal volume get --force pulsevad-data runs/pruned_seed_0/heldout_report.json ./data/runs/pruned_seed_0/heldout_report.json
uv run modal volume get --force pulsevad-data runs/pruned_seed_0/competitor_report.json ./data/runs/pruned_seed_0/competitor_report.json
uv run modal volume get --force pulsevad-data runs/pruned_seed_0/multilingual_report.json ./data/runs/pruned_seed_0/multilingual_report.json
uv run modal volume get --force pulsevad-data runs/pruned_seed_0/pulsevad_weights.h ./data/runs/pruned_seed_0/pulsevad_weights.h
uv run modal volume get --force pulsevad-data runs/pruned_seed_0/pulsevad_2.1k.onnx ./data/runs/pruned_seed_0/pulsevad_2.1k.onnx
uv run modal volume get --force pulsevad-data runs/pruned_seed_0/pulsevad_2.1k_int8.onnx ./data/runs/pruned_seed_0/pulsevad_2.1k_int8.onnx
```

### 11. generate visual benchmark plots
```bash
uv run python scripts/plot_comparison.py
```
generates:
- `data/runs/pruned_seed_0/comparison_graph.png`
- `data/runs/pruned_seed_0/multilingual_graph.png`
