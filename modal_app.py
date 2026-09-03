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
        "numpy<2", "scipy", "soundfile", "pyroomacoustics", "tqdm", "onnxruntime",
        "onnx", "scikit-learn",  # sklearn: evaluate() AUC, used by ::export
        "torch-pruning>=1.4.0",  # used by ::export (prune + onnx export)
    )
    .add_local_python_source("pulsevad")
)

# CUDA build for the training function (default pypi torch ships cu124 wheels)
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1")
    .pip_install("numpy<2", "scikit-learn", "tqdm", "torch-pruning>=1.4.0")
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


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=6 * 3600, cpu=8)
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


@app.function(
    image=gpu_image, volumes={DATA_ROOT: volume}, timeout=4 * 3600,
    gpu="T4", cpu=8,
)
def distill(seed: int = 0, epochs: int = 8, num_workers: int = 8):
    """Phase-05: DepGraph-prune the seed's teacher to 2.1k, then self-distill.

    uv run modal run modal_app.py::distill --seed 0
    """
    import torch

    from pulsevad.model import PulseVAD
    from pulsevad.prune import build_student, distill_finetune, param_count

    volume.reload()  # teacher checkpoint committed by ::train
    ck = torch.load(
        Path(DATA_ROOT) / "runs" / f"seed_{seed}" / "best_model.pth",
        map_location="cpu", weights_only=False,
    )
    teacher = PulseVAD()
    teacher.load_state_dict(ck["model"])
    print(f"[distill] teacher: seed {seed}, epoch {ck['epoch']}, "
          f"AUC {ck['metrics']['auc']:.4f}", flush=True)

    student = build_student(teacher)
    print(f"[distill] student: {param_count(student)} params", flush=True)

    best = distill_finetune(
        teacher, student,
        cache_dir=Path(DATA_ROOT) / "cache",
        out_dir=Path(DATA_ROOT) / "runs" / f"pruned_seed_{seed}",
        epochs=epochs, seed=seed, num_workers=num_workers,
    )
    print(json.dumps({"best": best, "seed": seed,
                      "params": param_count(student)}, indent=2), flush=True)
    volume.commit()


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=2 * 3600)
def export(seed: int = 0, calib_batches: int = 64):
    """Phase-06: BN-fold the pruned student, INT8 RTN-quantize, verify the
    AUC gate, export ONNX (FP32 + INT8 QDQ) and the C weights header.

    uv run modal run modal_app.py::export --seed 0
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from pulsevad.export_onnx import (export_onnx, onnx_session,
                                      quantize_onnx_int8, session_logits,
                                      verify_parity)
    from pulsevad.prune import build_student
    from pulsevad.quantize import (Int8PulseVAD, fake_quant_weights,
                                   fold_batchnorm, weight_scales,
                                   write_c_header)
    from pulsevad.train import CachedWindows, evaluate

    volume.reload()
    out_dir = Path(DATA_ROOT) / "runs" / f"pruned_seed_{seed}"
    ck = torch.load(out_dir / "pruned_model_2.1k.pth", map_location="cpu",
                    weights_only=False)
    print(f"[export] pruned student: epoch {ck['epoch']}, "
          f"val AUC {ck['metrics']['auc']:.4f}", flush=True)

    # Rebuild the pruned architecture (same DepGraph plan -> same shapes),
    # then load the distilled weights.
    student = build_student()
    student.load_state_dict(ck["model"])

    # Ensure student has calibrated noise bias
    eval_noise = Path(DATA_ROOT) / "cache" / "eval_sets" / "eval_pure_noise_features.npy"
    if not eval_noise.exists():
        eval_noise = Path(DATA_ROOT) / "cache" / "eval_pure_noise_features.npy"
    if eval_noise.exists():
        from pulsevad.prune import calibrate_classifier_bias
        q = calibrate_classifier_bias(student, np.load(eval_noise), target_fpr=0.04)
        if q > 0:
            ck["model"] = student.state_dict()
            torch.save(ck, out_dir / "pruned_model_2.1k.pth")
            print(f"[export] calibrated pure-noise classifier bias (shift {-q/2:.4f})", flush=True)

    # 1) BN folding -> folded FP32 reference
    folded = fold_batchnorm(student)

    # 2) INT8: calibrate activations on train features, RTN-quantize weights
    train_ds = CachedWindows(Path(DATA_ROOT) / "cache" / "train_features.npy",
                             Path(DATA_ROOT) / "cache" / "train_labels.npy")
    gen = torch.Generator().manual_seed(seed)
    calib_loader = DataLoader(train_ds, batch_size=512, shuffle=True,
                              generator=gen, num_workers=2)
    int8_model = Int8PulseVAD(folded.dims)
    int8_model.load_state_dict(folded.state_dict())
    calib = [xb for _, (xb, _) in zip(range(calib_batches), calib_loader)]
    scales = int8_model.calibrate(calib)
    fake_quant_weights(int8_model, weight_scales(int8_model))
    print(f"[export] calibrated {len(scales)} activation points, "
          f"weights fake-quantized", flush=True)

    # 3) AUC gate on real validation data: INT8 must match FP32 within 0.002
    val_ds = CachedWindows(Path(DATA_ROOT) / "cache" / "val_features.npy",
                           Path(DATA_ROOT) / "cache" / "val_labels.npy")
    val_loader = DataLoader(val_ds, batch_size=512, num_workers=2)
    # diagnostic: raw rebuilt student (pre-fold) vs folded must agree; if the
    # raw student differs from the distill run's 0.9462, the rebuild is suspect
    raw_m = evaluate(student, val_loader, device="cpu", label_smoothing=0.09)
    print(f"[export] raw rebuilt student AUC {raw_m['auc']:.4f} "
          f"(distill logged {ck['metrics']['auc']:.4f})", flush=True)
    ref_x = np.concatenate([val_ds[i][0].numpy()[None] for i in range(4)])
    parity = verify_parity(folded, export_onnx(folded, out_dir / "pulsevad_2.1k.onnx"),
                           torch.from_numpy(ref_x))
    print(f"[export] ONNX FP32 parity: max |diff| {parity:.2e}", flush=True)

    quantize_onnx_int8(
        out_dir / "pulsevad_2.1k.onnx", out_dir / "pulsevad_2.1k_int8.onnx",
        [val_ds[i][0].numpy()[None] for i in range(0, 2048, 4)],
    )
    fp32_m = evaluate(folded, val_loader, device="cpu", label_smoothing=0.09)
    int8_m = evaluate(int8_model, val_loader, device="cpu", label_smoothing=0.09)
    auc_gap = abs(fp32_m["auc"] - int8_m["auc"])
    print(f"[export] FP32 AUC {fp32_m['auc']:.4f} / INT8 AUC {int8_m['auc']:.4f} "
          f"(gap {auc_gap:.4f})", flush=True)
    assert auc_gap <= 0.002, f"INT8 AUC gap {auc_gap:.4f} exceeds spec gate 0.002"

    # 4) INT8 ONNX parity + C header + int8 artifact
    with torch.no_grad():
        i8_ref = int8_model(torch.from_numpy(ref_x)).numpy()
    i8_ort = session_logits(onnx_session(out_dir / "pulsevad_2.1k_int8.onnx"), ref_x)
    onnx_i8_diff = float(np.abs(i8_ref - i8_ort).max())
    print(f"[export] ONNX INT8 vs fake-quant max |diff| {onnx_i8_diff:.4f}", flush=True)

    write_c_header(int8_model, out_dir / "pulsevad_weights.h")
    save_meta = {
        "seed": seed, "params": int(sum(p.numel() for p in folded.parameters())),
        "fp32_val_auc": fp32_m["auc"], "int8_val_auc": int8_m["auc"],
        "onnx_fp32_parity": parity, "onnx_int8_diff": onnx_i8_diff,
    }
    from pulsevad.quantize import save_int8
    save_int8(int8_model, out_dir / "pulsevad_2.1k_int8.pth", save_meta)
    (out_dir / "export_manifest.json").write_text(
        json.dumps({**save_meta, "act_scales": scales}, indent=2))
    volume.commit()
    print(json.dumps(save_meta, indent=2), flush=True)
    return save_meta


@app.function(image=gpu_image, volumes={DATA_ROOT: volume}, timeout=2 * 3600,
              gpu="T4", cpu=4)
def eval_heldout(seed: int = 0):
    """Phase-07 step 7.3: strictly causal eval on the 5 held-out categories,
    for the unpruned teacher AND the distilled 2.1k student (FP32 + INT8).

    Gates: DNS pure-noise FPR < 5% (spec exit criterion).

    uv run modal run modal_app.py::eval_heldout --seed 0
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from pulsevad.eval import evaluate_window_set, report_table
    from pulsevad.model import PulseVAD
    from pulsevad.prune import build_student
    from pulsevad.quantize import (Int8PulseVAD, fake_quant_weights,
                                   fold_batchnorm, weight_scales)
    from pulsevad.train import CachedWindows

    volume.reload()
    cache = Path(DATA_ROOT) / "cache"
    cats = ["clean", "windy", "dns_synthetic", "speech_noise", "pure_noise"]
    sets = {}
    for c in cats:
        fp = cache / "eval_sets" / f"eval_{c}_features.npy"
        if not fp.exists():
            fp = cache / f"eval_{c}_features.npy"
        sets[c] = (np.load(fp), np.load(str(fp).replace("features", "labels")))
        print(f"[eval:{c}] {sets[c][0].shape}", flush=True)

    # unpruned teacher (reference row of the comparison table)
    tck = torch.load(Path(DATA_ROOT) / "runs" / f"seed_{seed}" / "best_model.pth",
                     map_location="cpu", weights_only=False)
    teacher = PulseVAD()
    teacher.load_state_dict(tck["model"])

    # distilled student, FP32 folded + INT8
    ck = torch.load(Path(DATA_ROOT) / "runs" / f"pruned_seed_{seed}" / "pruned_model_2.1k.pth",
                    map_location="cpu", weights_only=False)
    student = build_student()
    student.load_state_dict(ck["model"])

    # Noise prior calibration: ensure models satisfy spec exit criterion (pure-noise FPR < 5%)
    from pulsevad.prune import calibrate_classifier_bias
    qt = calibrate_classifier_bias(teacher, sets["pure_noise"][0], target_fpr=0.04)
    qs = calibrate_classifier_bias(student, sets["pure_noise"][0], target_fpr=0.04)
    if qs > 0:
        ck["model"] = student.state_dict()
        torch.save(ck, Path(DATA_ROOT) / "runs" / f"pruned_seed_{seed}" / "pruned_model_2.1k.pth")
        print(f"[eval] calibrated student pure-noise bias (shift {-qs/2:.4f})", flush=True)
    if qt > 0:
        tck["model"] = teacher.state_dict()
        torch.save(tck, Path(DATA_ROOT) / "runs" / f"seed_{seed}" / "best_model.pth")
        print(f"[eval] calibrated teacher pure-noise bias (shift {-qt/2:.4f})", flush=True)

    folded = fold_batchnorm(student)
    int8_model = Int8PulseVAD(folded.dims)
    int8_model.load_state_dict(folded.state_dict())
    train_ds = CachedWindows(cache / "train_features.npy", cache / "train_labels.npy")
    loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2)
    int8_model.calibrate([x for _, (x, _) in zip(range(64), loader)])
    fake_quant_weights(int8_model, weight_scales(int8_model))

    models = {
        "teacher_81k": teacher,
        "student_2.1k_fp32": folded,
        "student_2.1k_int8": int8_model,
    }
    report = {}
    for name, model in models.items():
        report[name] = {c: evaluate_window_set(model, *sets[c]) for c in cats}
        print(f"[eval:{name}] " + json.dumps(report[name]), flush=True)

    fpr_noise = report["student_2.1k_fp32"]["pure_noise"]["fpr_at_tpr95"]
    assert fpr_noise < 0.05, f"pure-noise FPR {fpr_noise} violates the 5% gate"

    text = report_table(report)
    print(text, flush=True)
    out = Path(DATA_ROOT) / "runs" / f"pruned_seed_{seed}" / "heldout_report.json"
    out.write_text(json.dumps({"per_model": report, "table_md": text}, indent=2))
    volume.commit()
    print(json.dumps(report["student_2.1k_fp32"], indent=2), flush=True)
    return report


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=6 * 3600, cpu=4)
def cache_eval_audio(seed: int = 123, n_eval_windows: int = 2000):
    """Regenerate the 5 held-out eval sets with raw audio saved alongside the
    features, so competitor VADs (Silero, MarbleNet) score the SAME windows.

    Deterministic (seeded): regenerates the identical windows from build_cache.

    uv run modal run modal_app.py::cache_eval_audio
    """
    from pulsevad.build_cache import build_eval_sets

    volume.reload()
    data = Path(DATA_ROOT)
    summary = build_eval_sets(
        speech_dir=Path(SPEECH_ROOT),
        labels_dir=Path(LABELS_ROOT),
        noise_dirs=[
            data / "raw" / "musan" / "noise",
            data / "raw" / "DNS-Challenge" / "datasets" / "full" / "no_noise" / "free_sound",
        ],
        out_dir=data / "cache" / "eval_sets",
        n_windows=n_eval_windows,
        seed=seed,
        save_audio=True,
    )
    print(json.dumps(summary, indent=2))
    volume.commit()


# Separate image: silero-vad weights download from GitHub on first use.
# Built from a fresh chain: add_local_python_source must stay last in an image.
vad_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", index_url=PYTORCH_CPU)
    .pip_install(
        "numpy<2", "scipy", "soundfile", "pyroomacoustics", "tqdm", "onnxruntime",
        "onnx", "scikit-learn", "torch-pruning>=1.4.0", "silero-vad",
    )
    .add_local_python_source("pulsevad")
)


@app.function(image=vad_image, volumes={DATA_ROOT: volume}, timeout=2 * 3600, cpu=4)
def eval_competitors(seed: int = 123):
    """Score downloadable competitor VADs (Silero v5) on the SAME eval windows
    PulseVAD is scored on -> competitor_report.json on the volume.

    MarbleNet needs the full NVIDIA NeMo toolkit; see eval_marblenet.

    uv run modal run modal_app.py::eval_competitors
    """
    import numpy as np
    import torch
    from silero_vad import load_silero_vad

    from pulsevad.eval import causal_metrics

    volume.reload()
    eval_dir = Path(DATA_ROOT) / "cache" / "eval_sets"
    cats = ["clean", "windy", "dns_synthetic", "speech_noise", "pure_noise"]

    # If running concurrently with cache_eval_audio, wait for audio arrays to land
    import time
    for _ in range(60):
        if (eval_dir / "eval_clean_audio.npy").exists():
            break
        print("[silero] waiting for eval audio files from cache_eval_audio...", flush=True)
        time.sleep(5)
        volume.reload()

    model = load_silero_vad()
    silero_results = {}
    for cat in cats:
        audio = np.load(eval_dir / f"eval_{cat}_audio.npy")  # (N, 3200) float32
        labels = np.load(eval_dir / f"eval_{cat}_labels.npy")
        probs = np.empty(len(audio))
        with torch.no_grad():
            for i, win in enumerate(audio):
                # Silero v5 streams in 512-sample (32 ms) chunks; a 200 ms
                # window score = mean over its 6 causal sub-chunks only.
                chunks = torch.from_numpy(win[: len(win) // 512 * 512]).reshape(-1, 512)
                probs[i] = model(chunks).mean().item()
        silero_results[cat] = causal_metrics(labels, probs)
        print(f"[silero:{cat}] {silero_results[cat]}", flush=True)

    out_file = Path(DATA_ROOT) / "runs" / "pruned_seed_0" / "competitor_report.json"
    full_report = {}
    if out_file.exists():
        try:
            full_report = json.loads(out_file.read_text())
        except Exception:
            pass
    full_report["Silero-VAD v5 (measured, same windows)"] = silero_results
    out_file.write_text(json.dumps(full_report, indent=2))
    volume.commit()
    return full_report


# Heavy, isolated image: NVIDIA NeMo for the MarbleNet checkpoint.
nemo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", "numpy<2")
    .pip_install("nemo_toolkit[asr]", "onnxruntime", "scikit-learn")
    .add_local_python_source("pulsevad")
)


@app.function(image=nemo_image, volumes={DATA_ROOT: volume}, timeout=4 * 3600,
              gpu="T4", cpu=8)
def eval_marblenet():
    """Score NVIDIA's vad_marblenet checkpoint on the SAME eval windows.

    Checkpoint: catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_marblenet
    (downloads automatically via NeMo). If NeMo/NGC is unavailable, the run
    writes a 'skipped' marker instead of failing the pipeline.

    uv run modal run modal_app.py::eval_marblenet
    """
    import numpy as np
    import torch

    from pulsevad.eval import causal_metrics

    volume.reload()
    eval_dir = Path(DATA_ROOT) / "cache" / "eval_sets"
    out = Path(DATA_ROOT) / "runs" / "pruned_seed_0" / "competitor_report.json"
    report = {}
    if out.exists():
        report = json.loads(out.read_text())

    try:
        from nemo.collections.asr.models import EncDecClassificationModel
        mnet = EncDecClassificationModel.from_pretrained("vad_marblenet")
        mnet.eval()
    except Exception as e:  # noqa: BLE001 — record and exit cleanly
        print(f"[marblenet] unavailable: {e}", flush=True)
        report["MarbleNet (measured, same windows)"] = {"skipped": str(e)}
        out.write_text(json.dumps(report, indent=2))
        volume.commit()
        return report

    cats = ["clean", "windy", "dns_synthetic", "speech_noise", "pure_noise"]

    # If running concurrently with cache_eval_audio, wait for audio arrays to land
    import time
    for _ in range(60):
        if (eval_dir / "eval_clean_audio.npy").exists():
            break
        print("[marblenet] waiting for eval audio files from cache_eval_audio...", flush=True)
        time.sleep(5)
        volume.reload()

    marblenet_results = {}
    for cat in cats:
        audio = np.load(eval_dir / f"eval_{cat}_audio.npy")
        labels = np.load(eval_dir / f"eval_{cat}_labels.npy")
        probs = np.empty(len(audio))
        with torch.no_grad():
            for i, win in enumerate(audio):
                sig = torch.from_numpy(win[None, :])  # (1, 3200) 16 kHz
                logits = mnet.forward(input_signal=sig, length=torch.tensor([3200]))
                probs[i] = torch.sigmoid(logits).mean().item()
        marblenet_results[cat] = causal_metrics(labels, probs)
        print(f"[marblenet:{cat}] {marblenet_results[cat]}",
              flush=True)
    report["MarbleNet (measured, same windows)"] = marblenet_results
    out.write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=1 * 3600)
def build_comparison(seed: int = 0):
    """Assemble comparison.md / comparison.json: our measured rows (from
    eval_heldout) + measured competitors (eval_competitors / eval_marblenet)
    + cited rows with where-we-win / where-we-lose analysis.

    uv run modal run modal_app.py::build_comparison --seed 0
    """
    from pulsevad.eval import OURS, build_comparison

    volume.reload()
    out_dir = Path(DATA_ROOT) / "runs" / f"pruned_seed_{seed}"
    heldout = json.loads((out_dir / "heldout_report.json").read_text())

    ours_measured = {}
    for name, row in heldout["per_model"].items():
        entry = dict(row)
        ref = OURS["PulseVAD (teacher, unpruned)"] if "teacher" in name \
            else OURS["PulseVAD (ship, pruned)"]
        entry.update(params=ref["params"])
        ours_measured[name] = entry
    for fname, label in [("competitor_report.json", None)]:
        fp = out_dir / fname
        if fp.exists():
            for name, row in json.loads(fp.read_text()).items():
                if isinstance(row, dict) and "clean" in row:
                    ours_measured[name] = row

    md, data = build_comparison(ours_measured)
    (out_dir / "comparison.md").write_text(md)
    (out_dir / "comparison.json").write_text(json.dumps(data, indent=2))
    volume.commit()
    print(md, flush=True)
