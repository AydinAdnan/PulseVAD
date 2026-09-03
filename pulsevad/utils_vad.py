"""User-facing API for loading pre-trained PulseVAD models and processing audio.
Inspired by the Silero-VAD interface for seamless drop-in edge deployment.
"""

from pathlib import Path
from typing import List, Dict, Union, Optional
import numpy as np

DATA_DIR = Path(__file__).parent / "data"


def get_model_path(filename: str) -> Path:
    """Return absolute path to a bundled model artifact in pulsevad/data/."""
    p = DATA_DIR / filename
    if not p.exists():
        raise FileNotFoundError(f"Model file '{filename}' not found in {DATA_DIR}")
    return p


def load_pulsevad(
    onnx: bool = True,
    quantized: bool = True,
    model_type: str = "2.1k",
    device: str = "cpu",
):
    """Load a pre-trained PulseVAD model.

    Args:
        onnx: If True, returns an ONNX Runtime InferenceSession.
              If False, returns a PyTorch TorchScript JIT model.
        quantized: If True, loads the 2.1 KB INT8 QDQ quantized model.
                   If False, loads the FP32 model.
        model_type: "2.1k" (default ship model, 2,118 params) or "81k" (teacher model, 81,090 params).
        device: "cpu" or "cuda" (for ONNX / TorchScript execution).

    Returns:
        Callable model object (ONNX InferenceSession or torch.jit.ScriptModule).
    """
    if model_type == "2.1k":
        if onnx:
            fname = "pulsevad_2.1k_int8.onnx" if quantized else "pulsevad_2.1k.onnx"
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ort.InferenceSession(str(get_model_path(fname)), providers=providers)
        else:
            import torch
            m = torch.jit.load(str(get_model_path("pulsevad_2.1k.jit")), map_location=device)
            m.eval()
            return m
    elif model_type in ("81k", "teacher"):
        if onnx:
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ort.InferenceSession(str(get_model_path("pulsevad_teacher_81k.onnx")), providers=providers)
        else:
            import torch
            from pulsevad.model import PulseVAD
            ck = torch.load(get_model_path("pulsevad_teacher_81k.pth"), map_location=device, weights_only=False)
            m = PulseVAD()
            m.load_state_dict(ck["state_dict"])
            m.eval()
            return m
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected '2.1k' or '81k'.")


def read_audio(path: Union[str, Path], sampling_rate: int = 16000) -> np.ndarray:
    """Read an audio file from disk and return a 16 kHz mono float32 numpy array."""
    from math import gcd
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono
    if sr != sampling_rate:
        g = gcd(sr, sampling_rate)
        audio = resample_poly(audio, sampling_rate // g, sr // g).astype(np.float32)
    return audio


def _extract_log_mel(audio_window: np.ndarray) -> np.ndarray:
    """Compute 64-channel log-mel features for a 3,200 sample (200 ms) window.
    Strictly causal: pre-emphasis -> waveform norm -> 64-mel -> per-bin norm.
    """
    import torch
    from pulsevad.frontend import MelFrontend

    frontend = MelFrontend()
    x = torch.from_numpy(audio_window).unsqueeze(0).float()
    with torch.no_grad():
        feats = frontend(x)
    return feats.numpy().astype(np.float32)


def predict_window(model, audio_window: np.ndarray) -> float:
    """Predict speech probability for a single 200 ms (3,200 samples @ 16 kHz) audio chunk.

    Args:
        model: ONNX InferenceSession or PyTorch JIT model from `load_pulsevad()`.
        audio_window: 1D numpy array of 3,200 float32 audio samples.

    Returns:
        Speech probability float in [0.0, 1.0].
    """
    if len(audio_window) != 3200:
        raise ValueError(f"Audio window must be exactly 3,200 samples (200 ms @ 16 kHz), got {len(audio_window)}")

    feats = _extract_log_mel(audio_window)  # (1, 64, 21)
    if hasattr(model, "run"):  # ONNX Runtime
        input_name = model.get_inputs()[0].name
        logits = model.run(None, {input_name: feats})[0]
    else:  # PyTorch / TorchScript
        import torch
        with torch.no_grad():
            out = model(torch.from_numpy(feats))
            logits = out.numpy() if hasattr(out, "numpy") else np.asarray(out)

    # Logits: [non_speech, speech]
    logit_diff = logits[0, 1] - logits[0, 0]
    prob = 1.0 / (1.0 + np.exp(-logit_diff))
    return float(prob)


def get_speech_timestamps(
    audio: np.ndarray,
    model,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    window_size_samples: int = 3200,
    hop_size_samples: int = 1600,
    min_speech_duration_ms: int = 100,
    min_silence_duration_ms: int = 100,
) -> List[Dict[str, int]]:
    """Scan continuous audio and return timestamps for active speech segments.

    Args:
        audio: 1D float32 numpy array at 16 kHz.
        model: Loaded PulseVAD model.
        threshold: Speech probability threshold (default 0.5).
        sampling_rate: Expected 16,000 Hz.
        window_size_samples: 3,200 (200 ms).
        hop_size_samples: Hop size between evaluations (default 1,600 = 100 ms).
        min_speech_duration_ms: Minimum active speech duration to keep.
        min_silence_duration_ms: Minimum silence duration to split segments.

    Returns:
        List of dicts: [{'start': sample_start, 'end': sample_end}, ...]
    """
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < window_size_samples:
        # Pad short audio to 200 ms
        padded = np.pad(audio, (0, window_size_samples - len(audio)))
        p = predict_window(model, padded)
        return [{"start": 0, "end": len(audio)}] if p >= threshold else []

    min_speech_samples = int(sampling_rate * min_speech_duration_ms / 1000)
    min_silence_samples = int(sampling_rate * min_silence_duration_ms / 1000)

    # Slide window
    speech_frames = []
    for start_idx in range(0, len(audio) - window_size_samples + 1, hop_size_samples):
        win = audio[start_idx : start_idx + window_size_samples]
        p = predict_window(model, win)
        speech_frames.append((start_idx, p >= threshold))

    # Aggregate frames into intervals
    segments = []
    cur_start = None
    last_speech_time = None

    for idx, is_speech in speech_frames:
        if is_speech:
            if cur_start is None:
                cur_start = idx
            last_speech_time = idx + window_size_samples
        else:
            if cur_start is not None and (idx - last_speech_time) >= min_silence_samples:
                if (last_speech_time - cur_start) >= min_speech_samples:
                    segments.append({"start": cur_start, "end": min(last_speech_time, len(audio))})
                cur_start = None

    if cur_start is not None and (last_speech_time - cur_start) >= min_speech_samples:
        segments.append({"start": cur_start, "end": min(last_speech_time, len(audio))})

    return segments
