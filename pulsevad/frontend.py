"""PulseVAD deterministic audio frontend.

Transforms a raw 200 ms (3200-sample @ 16 kHz) mono window into a
normalized log-Mel spectrogram of shape (B, 64, 21), per spec phase-01:

    pre-emphasis -> waveform z-norm -> Mel spectrogram -> log -> per-bin z-norm

No learned parameters; the same chain must be reproducible bit-exactly in
the embedded C runtime (c_src/pulsevad_dsp.c).
"""

import torch
import torchaudio

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 3_200  # 200 ms
N_FFT = 512
WIN_LENGTH = 400  # 25 ms Hann
HOP_LENGTH = 160  # 10 ms
N_MELS = 64
F_MIN = 0.0
F_MAX = 8_000.0
PREEMPHASIS_ALPHA = 0.97
EPS = 1e-5


def pre_emphasis(waveform: torch.Tensor, alpha: float = PREEMPHASIS_ALPHA) -> torch.Tensor:
    """y[n] = x[n] - alpha * x[n-1], with y[0] = x[0] (zero-pad convention)."""
    return torchaudio.functional.preemphasis(waveform, coeff=alpha)


def waveform_z_norm(waveform: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Amplitude-agnostic normalization over the window (last dim)."""
    mean = waveform.mean(dim=-1, keepdim=True)
    std = waveform.std(dim=-1, keepdim=True, unbiased=False)
    return (waveform - mean) / (std + eps)


def per_bin_z_norm(log_mel: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Z-score each of the 64 mel bins across its 21 time frames."""
    mean = log_mel.mean(dim=-1, keepdim=True)
    std = log_mel.std(dim=-1, keepdim=True, unbiased=False)
    return (log_mel - mean) / (std + eps)


class MelFrontend(torch.nn.Module):
    """Raw waveform (B, 3200) -> normalized log-Mel features (B, 64, 21)."""

    def __init__(self) -> None:
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            f_min=F_MIN,
            f_max=F_MAX,
            n_mels=N_MELS,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="htk",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[-1] != WINDOW_SAMPLES:
            raise ValueError(
                f"Expected exactly {WINDOW_SAMPLES} samples (200 ms @ 16 kHz), "
                f"got {waveform.shape[-1]}"
            )
        emphasized = pre_emphasis(waveform)
        normalized = waveform_z_norm(emphasized)
        mel = self.mel_spectrogram(normalized)  # (B, 64, 21), power spectrogram
        log_mel = torch.log(mel + EPS)
        return per_bin_z_norm(log_mel)
