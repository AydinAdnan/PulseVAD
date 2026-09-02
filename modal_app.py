"""Modal cloud pipeline (Phase 3). YOU execute these; nothing runs by itself.

    uv run modal run modal_app.py::download        # ~17 GB -> volume (add --with-dns true for DNS noise)
    uv run modal run modal_app.py::label           # Silero-VAD -> data/labels on volume
    uv run modal run modal_app.py::build-cache     # mix + features -> data/cache on volume

Fetch artifacts back locally (optional; training reads the volume in-cloud):

    uv run modal volume get pulsevad-data labels ./data/labels
    uv run modal volume get pulsevad-data cache/manifest.json ./
"""

import json
from pathlib import Path

import modal

app = modal.App("pulsevad")
volume = modal.Volume.from_name("pulsevad-data", create_if_missing=True)
DATA_ROOT = "/data"
# Single source of truth for the label tree: manifests under LABELS_ROOT mirror
# the speech tree under SPEECH_ROOT (e.g. 103/1240/103-1240-0000.json).
SPEECH_ROOT = f"{DATA_ROOT}/raw/LibriSpeech/train-clean-100"
LABELS_ROOT = f"{DATA_ROOT}/labels/train-clean-100"
PYTORCH_CPU = "https://download.pytorch.org/whl/cpu"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", index_url=PYTORCH_CPU)
    .pip_install(
        "numpy<2", "scipy", "soundfile", "pyroomacoustics", "tqdm", "onnxruntime"
    )
    .add_local_python_source("pulsevad")
)

# CUDA build for the training function (default pypi torch ships cu124 wheels)
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1")
    .pip_install("numpy<2", "scikit-learn", "tqdm")
    .add_local_python_source("pulsevad")
)


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=4 * 3600)
def download(with_dns: bool = False):
    from pulsevad.download_data import download_all

    info = download_all(Path(DATA_ROOT) / "raw", with_dns=with_dns)
    print(json.dumps(info, indent=2))
    volume.commit()


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=2 * 3600)
def label_batch(rel_paths: list[str]):
    """Silero-label a batch of LibriSpeech files -> JSON manifests on the volume."""
    from pulsevad.label_corpus import label_wav, load_silero, read_mono

    speech, labels = Path(SPEECH_ROOT), Path(LABELS_ROOT)
    model, get_ts = load_silero()
    for rel in rel_paths:
        wav = read_mono(speech / rel)
        manifest = label_wav(wav, model, get_ts)
        out = labels / Path(rel).with_suffix(".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest))
    volume.commit()


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=4 * 3600)
def label():
    """Fan LibriSpeech clips out over CPU containers (~30 min for 100 h)."""
    import numpy as np

    speech = Path(SPEECH_ROOT)
    rels = sorted(
        str(p.relative_to(speech))
        for p in speech.rglob("*")
        if p.suffix.lower() in (".flac", ".wav")
    )
    if not rels:
        raise FileNotFoundError(f"no audio under {speech} — run ::download first")
    batches = [rels[i : i + 50] for i in range(0, len(rels), 50)]
    print(f"{len(rels)} files -> {len(batches)} batches")
    list(label_batch.map(batches))
    volume.commit()

    # QC gate (build plan §1.3 step 4): every raw file must have a manifest.
    # Metadata-only count + sampled speech-fractions (reading all 28k jsons
    # over the volume would stall for many silent minutes).
    labels = Path(LABELS_ROOT)
    jsons = list(labels.rglob("*.json")) if labels.exists() else []
    missing = len(rels) - len(jsons)
    rng = np.random.default_rng(0)
    probe = rng.choice(len(jsons), size=min(500, len(jsons)), replace=False) if jsons else []
    fracs = [json.loads(jsons[i].read_text())["speech_fraction"] for i in probe]
    summary = {
        "raw_files": len(rels),
        "labeled_files": len(jsons),
        "missing": max(0, missing),
        "mean_speech_fraction": round(float(np.mean(fracs)), 3) if fracs else 0.0,
        "min_speech_fraction": round(float(np.min(fracs)), 3) if fracs else 0.0,
    }
    (Path(DATA_ROOT) / "labels" / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if missing > 0:
        raise RuntimeError(f"{missing} label manifests missing — re-run ::label")
    volume.commit()


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=1 * 3600)
def verify(sample_size: int = 500):
    """QC-only: counts labeled files vs raw, writes data/labels/_summary.json.
    Run any time with: uv run modal run modal_app.py::verify

    ponytail: file COUNTS are metadata-only (cheap); speech-fraction stats read
    only `sample_size` random manifests — reading all 28k over the network
    volume takes 10+ silent minutes.
    """
    import numpy as np

    volume.reload()  # pick up latest committed state from other runs
    speech = Path(SPEECH_ROOT)
    labels = Path(LABELS_ROOT)

    print(f"[verify 1/4] listing audio under {SPEECH_ROOT} (metadata only)…")
    expected = {
        str(p.relative_to(speech).with_suffix(".json"))
        for p in speech.rglob("*")
        if p.suffix.lower() in (".flac", ".wav")
    }
    print(f"[verify 2/4] listing manifests under {LABELS_ROOT} (metadata only)…")
    manifests = set(labels.rglob("*.json")) if labels.exists() else set()
    manifest_rel = {str(p.relative_to(labels)) for p in manifests}

    summary = {
        "raw_files": len(expected),
        "labeled_files": len(expected & manifest_rel),
        "missing": len(expected - manifest_rel),
        "mean_speech_fraction": None,
    }
    print(f"[verify 3/4] audio={summary['raw_files']} manifests={len(manifest_rel)} "
          f"missing={summary['missing']}")

    json_list = sorted(manifest_rel)
    if json_list:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(json_list), size=min(sample_size, len(json_list)), replace=False)
        fracs = [
            json.loads((labels / json_list[i]).read_text())["speech_fraction"]
            for i in idx
        ]
        summary["mean_speech_fraction"] = round(float(np.mean(fracs)), 3)

    (Path(DATA_ROOT) / "labels" / "_summary.json").write_text(json.dumps(summary, indent=2))
    volume.commit()
    print(f"[verify 4/4] done")
    print(json.dumps(summary, indent=2))
    sample = sorted(expected)[:3]
    print("sample expected manifests:", [(s, (labels / s).exists()) for s in sample])
    if labels.exists():
        print("top-level entries in labels dir:", sorted(p.name for p in labels.iterdir())[:10])
    if summary["missing"]:
        print("-> re-run ::label to fill gaps")
    return summary


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=6 * 3600, cpu=4)
def build_cache(seed: int = 0, n_eval_windows: int = 2000):
    """Mix, augment, featurize -> train/val memmaps + 5 held-out eval sets."""
    from pulsevad.build_cache import build_cache, build_eval_sets

    volume.reload()  # ensure we see the label manifests committed by ::label

    data = Path(DATA_ROOT)
    manifest = build_cache(
        speech_dir=Path(SPEECH_ROOT),
        labels_dir=Path(LABELS_ROOT),
        noise_dirs=[
            data / "raw" / "musan" / "noise",
            data / "raw" / "DNS-Challenge" / "datasets" / "full" / "no_noise" / "free_sound",
        ],
        out_dir=data / "cache",
        seed=seed,
    )
    eval_summary = build_eval_sets(
        speech_dir=Path(SPEECH_ROOT),
        labels_dir=Path(LABELS_ROOT),
        noise_dirs=[
            data / "raw" / "musan" / "noise",
            data / "raw" / "DNS-Challenge" / "datasets" / "full" / "no_noise" / "free_sound",
        ],
        out_dir=data / "cache" / "eval_sets",
        n_windows=n_eval_windows,
    )
    print(json.dumps(manifest, indent=2))
    print(json.dumps(eval_summary, indent=2))
    volume.commit()


@app.function(
    image=gpu_image, volumes={DATA_ROOT: volume}, timeout=8 * 3600,
    gpu="T4", cpu=8,
)
def train(seed: int = 0, epochs: int = 40, num_workers: int = 8):
    """40-epoch base-training run (spec phase-04) on one T4.

        uv run modal run modal_app.py::train --seed 0
    """
    from pulsevad.train import train as run_train

    volume.reload()  # see the cache committed by ::build_cache
    best = run_train(
        cache_dir=Path(DATA_ROOT) / "cache",
        out_dir=Path(DATA_ROOT) / "runs" / f"seed_{seed}",
        seed=seed,
        epochs=epochs,
        num_workers=num_workers,
    )
    print(json.dumps({"best": best, "seed": seed}, indent=2), flush=True)
    volume.commit()
