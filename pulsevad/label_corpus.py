"""Silero-VAD self-labeling (spec phase-03 §3.2, build plan §1.3).

Silero-VAD (MIT) is used ONLY as an automated labeling tool. Its released
labeled dataset (CC BY-NC-SA) is never downloaded or used.

The hysteresis state machine (start 0.50 / end 0.35, min speech 250 ms,
min silence 100 ms, pad 30 ms) is Silero's get_speech_timestamps — the exact
parameters the build plan mandates. We pass them explicitly and own the
10 ms-grid rasterization, which is where label noise is made or avoided.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

AUDIO_EXTS = (".flac", ".wav")

LABEL_PARAMS = dict(
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=100,
    speech_pad_ms=30,
)


def load_silero():
    """Returns (model, get_speech_timestamps). onnx=True needs onnxruntime."""
    model, utils = torch.hub.load(
        "snakers4/silero-vad", model="silero_vad", onnx=True, trust_repo=True
    )
    return model, utils[0]


def label_wav(wav: np.ndarray, model, get_ts, sr: int = 16_000) -> dict:
    """Run Silero over one clip (512-sample chunks, hidden state carried forward
    inside get_speech_timestamps — never reset mid-clip) -> segment manifest."""
    ts = get_ts(
        torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)),
        model,
        sampling_rate=sr,
        **LABEL_PARAMS,
    )
    segments = [[t["start"] / sr, t["end"] / sr] for t in ts]
    duration = len(wav) / sr
    speech_sec = sum(e - s for s, e in segments)
    return {
        "duration_sec": duration,
        "segments": segments,
        "speech_fraction": speech_sec / duration if duration > 0 else 0.0,
    }


def segments_to_frame_flags(
    segments: list[list[float]], n_frames: int, frame_sec: float = 0.01
) -> np.ndarray:
    """Rasterize segments to the 10 ms grid: frame k is speech iff its CENTER
    (k * frame_sec, matching STFT center=True) falls inside a segment."""
    flags = np.zeros(n_frames, dtype=bool)
    centers = np.arange(n_frames) * frame_sec
    for s, e in segments:
        flags |= (centers >= s) & (centers <= e)
    return flags


def _iter_audio(root: Path):
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in AUDIO_EXTS:
            yield p


def read_mono(path: Path, sr: int = 16_000) -> np.ndarray:
    wav, file_sr = sf.read(path, always_2d=True, dtype="float32")
    wav = wav[:, 0]
    if file_sr != sr:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(sr, file_sr)
        wav = resample_poly(wav, sr // g, file_sr // g).astype(np.float32)
    return wav


def label_corpus(raw_dir: Path, labels_dir: Path) -> dict:
    """Label every audio file under raw_dir; write one JSON manifest per file,
    mirroring the relative path, into labels_dir. Returns a QC summary."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    files = list(_iter_audio(raw_dir))
    if not files:
        raise FileNotFoundError(f"no audio files under {raw_dir}")
    model, get_ts = load_silero()

    fractions = []
    for path in files:
        wav = read_mono(path)
        manifest = label_wav(wav, model, get_ts)
        fractions.append(manifest["speech_fraction"])
        out = labels_dir / path.relative_to(raw_dir).with_suffix(".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest))

    summary = {
        "files": len(files),
        "mean_speech_fraction": float(np.mean(fractions)),
        "min_speech_fraction": float(np.min(fractions)),
    }
    (labels_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    # QC flag (build plan §1.3 step 4c): corpus that is 95%+ speech has no negatives
    if summary["mean_speech_fraction"] > 0.95:
        print("WARNING: corpus is >95% speech; append synthetic silences")
    return summary
