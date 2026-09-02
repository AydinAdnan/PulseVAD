import torch
import pytest

from pulsevad.frontend import (
    MelFrontend,
    WINDOW_SAMPLES,
    per_bin_z_norm,
    pre_emphasis,
    waveform_z_norm,
)


def test_pre_emphasis():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = pre_emphasis(x)
    # y[n] = x[n] - 0.97 * x[n-1], y[0] = x[0]
    expected = torch.tensor([[1.0, 2.0 - 0.97, 3.0 - 0.97 * 2.0, 4.0 - 0.97 * 3.0]])
    assert torch.allclose(y, expected, atol=1e-6)


def test_waveform_z_norm_statistics():
    x = torch.randn(4, WINDOW_SAMPLES) * 50.0  # arbitrary scale
    z = waveform_z_norm(x)
    mean = z.mean(dim=-1)
    std = z.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-3)


def test_frontend_output_shape():
    frontend = MelFrontend()
    batch_size = 4
    audio_3200 = torch.randn(batch_size, WINDOW_SAMPLES)
    features = frontend(audio_3200)
    assert features.shape == (batch_size, 64, 21), f"Expected (4, 64, 21), got {features.shape}"


def test_frontend_frame_count_exactness():
    # frames = floor(3200 / 160) + 1 = 21 (rule 2 of the spec's engineering rules)
    frontend = MelFrontend()
    assert frontend(torch.randn(1, WINDOW_SAMPLES)).shape[-1] == 21


def test_frontend_zero_input():
    frontend = MelFrontend()
    silent_audio = torch.zeros(1, WINDOW_SAMPLES)
    features = frontend(silent_audio)
    assert not torch.isnan(features).any(), "NaN detected in silent input"
    assert not torch.isinf(features).any(), "Inf detected in silent input"


def test_frontend_statistics():
    frontend = MelFrontend()
    audio = torch.randn(2, WINDOW_SAMPLES)
    features = frontend(audio)
    mean = features.mean(dim=-1)
    std = features.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-4)


def test_frontend_rejects_wrong_window_length():
    frontend = MelFrontend()
    with pytest.raises(ValueError):
        frontend(torch.randn(1, 1600))


def test_frontend_deterministic():
    # Rule: no learned params -> same input must give bit-identical output
    frontend = MelFrontend().eval()
    audio = torch.randn(2, WINDOW_SAMPLES)
    assert torch.equal(frontend(audio), frontend(audio))


def test_per_bin_z_norm_order_matches_spec():
    # Normalization order (rule 3): log-mel -> per-bin z-norm last
    frontend = MelFrontend().eval()
    audio = torch.randn(1, WINDOW_SAMPLES)
    with torch.no_grad():
        emphasized = pre_emphasis(audio)
        normalized = waveform_z_norm(emphasized)
        mel = frontend.mel_spectrogram(normalized)
        manual = per_bin_z_norm(torch.log(mel + 1e-5))
    assert torch.allclose(frontend(audio), manual, atol=1e-6)
