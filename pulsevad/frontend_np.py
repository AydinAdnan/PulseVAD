"""PulseVAD pure NumPy audio frontend (zero PyTorch dependency).

Bit-exact equivalent to pulsevad.frontend.MelFrontend:
  pre-emphasis (0.97) -> waveform z-norm -> Mel spectrogram (64 bins) -> log -> per-bin z-norm
"""

from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
_MEL_FB_PATH = DATA_DIR / "mel_filterbank.npy"
_MEL_FB = None

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 3_200  # 200 ms
N_FFT = 512
WIN_LENGTH = 400  # 25 ms Hann
HOP_LENGTH = 160  # 10 ms
N_MELS = 64
PREEMPHASIS_ALPHA = 0.97
EPS = 1e-5

# Periodic Hann window of length 400 (matches torch.hann_window(400, periodic=True))
_HANN_WINDOW = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(WIN_LENGTH) / WIN_LENGTH)).astype(np.float32)
_PAD_WIN = (N_FFT - WIN_LENGTH) // 2  # 56 zeros on each side


def get_mel_filterbank() -> np.ndarray:
    """Load or return cached (257, 64) float32 mel filterbank."""
    global _MEL_FB
    if _MEL_FB is None:
        if not _MEL_FB_PATH.exists():
            raise FileNotFoundError(f"Mel filterbank not found at {_MEL_FB_PATH}")
        _MEL_FB = np.load(_MEL_FB_PATH).astype(np.float32)
    return _MEL_FB


def extract_log_mel_np(waveform: np.ndarray) -> np.ndarray:
    """Extract normalized log-Mel spectrogram from a 3200-sample (200 ms @ 16 kHz) audio chunk.

    Args:
        waveform: 1D numpy array of 3200 float32 samples, or 2D (B, 3200).

    Returns:
        float32 numpy array of shape (B, 64, 21).
    """
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim == 1:
        if x.shape[0] != WINDOW_SAMPLES:
            raise ValueError(f"Expected exactly {WINDOW_SAMPLES} samples, got {x.shape[0]}")
        x = x[np.newaxis, :]
    elif x.ndim == 2:
        if x.shape[1] != WINDOW_SAMPLES:
            raise ValueError(f"Expected shape (B, {WINDOW_SAMPLES}), got {x.shape}")
    else:
        raise ValueError(f"Expected 1D or 2D audio array, got ndim={x.ndim}")

    batch_size = x.shape[0]
    fb = get_mel_filterbank()  # (257, 64)

    # 1. Pre-emphasis: y[n] = x[n] - 0.97 * x[n-1], y[0] = x[0]
    x_pre = np.empty_like(x)
    x_pre[:, 0] = x[:, 0]
    x_pre[:, 1:] = x[:, 1:] - PREEMPHASIS_ALPHA * x[:, :-1]

    # 2. Waveform z-norm: (x - mean) / (std + EPS) with ddof=0
    mean = x_pre.mean(axis=-1, keepdims=True)
    std = x_pre.std(axis=-1, keepdims=True, ddof=0)
    x_norm = (x_pre - mean) / (std + EPS)

    # 3. Centered STFT: pad 256 on each side with reflect
    pad = N_FFT // 2  # 256
    x_padded = np.pad(x_norm, ((0, 0), (pad, pad)), mode="reflect")

    # 4. Extract 21 frames and compute power spectrum
    batch_out = []
    for b in range(batch_size):
        frames = []
        for i in range(21):
            start = i * HOP_LENGTH
            chunk = np.zeros(N_FFT, dtype=np.float32)
            # Torchaudio centers win_length=400 in n_fft=512 with 56 zeros on each side
            chunk[_PAD_WIN : _PAD_WIN + WIN_LENGTH] = (
                x_padded[b, start + _PAD_WIN : start + _PAD_WIN + WIN_LENGTH] * _HANN_WINDOW
            )
            spec = np.fft.rfft(chunk, n=N_FFT)
            power = np.abs(spec) ** 2
            frames.append(power)

        power_matrix = np.stack(frames, axis=0)  # (21, 257)
        mel = power_matrix @ fb                  # (21, 64)
        mel = mel.T                              # (64, 21)
        log_mel = np.log(mel + EPS)

        # 5. Per-bin z-norm across the 21 time frames
        b_mean = log_mel.mean(axis=-1, keepdims=True)
        b_std = log_mel.std(axis=-1, keepdims=True, ddof=0)
        norm_mel = (log_mel - b_mean) / (b_std + EPS)
        batch_out.append(norm_mel)

    return np.stack(batch_out, axis=0).astype(np.float32)
