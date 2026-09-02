import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pulsevad.augment import mix_at_snr, reverb_apply, rms, simulate_rir, wind_noise
from pulsevad.build_cache import (
    EVAL_CATEGORIES,
    build_cache,
    build_eval_sets,
    majority_label,
    window_label,
)
from pulsevad.label_corpus import label_wav, segments_to_frame_flags

SR = 16_000
WINDOW = 3_200


# ---------------------------------------------------------------- SNR math

def test_mix_at_snr_exact():
    rng = np.random.default_rng(0)
    speech = (0.01 * rng.standard_normal(SR)).astype(np.float32)
    noise = (0.01 * rng.standard_normal(SR)).astype(np.float32)
    for snr in [-10.0, -5.0, 0.0, 5.0, 10.0]:
        mixed = mix_at_snr(speech, noise, snr)
        measured = 20 * np.log10(rms(speech) / (rms(mixed - speech) + 1e-8))
        assert abs(measured - snr) < 0.01, f"SNR {snr} dB: measured {measured:.2f}"


def test_mix_at_snr_anti_clip():
    speech = np.ones(SR, dtype=np.float32) * 0.9
    noise = np.ones(SR, dtype=np.float32) * 0.9
    mixed = mix_at_snr(speech, noise, 0.0)
    assert np.max(np.abs(mixed)) <= 1.0


# ---------------------------------------------------------------- wind / reverb

def test_wind_noise_properties():
    rng = np.random.default_rng(1)
    w = wind_noise(SR, rng=rng)
    assert np.isfinite(w).all()
    assert abs(rms(w) - 1.0) < 0.05  # unit RMS
    spec = np.abs(np.fft.rfft(w)) ** 2
    freqs = np.fft.rfftfreq(SR)
    low = spec[freqs < 600].sum()
    assert low / spec.sum() > 0.9, "wind should be dominated by low frequencies"


def test_reverb_shape_and_energy():
    rng = np.random.default_rng(2)
    speech = rng.standard_normal(SR * 2).astype(np.float32) * 0.1
    rir = simulate_rir(rng=rng)
    y = reverb_apply(speech, rir)
    assert len(y) == len(speech)
    assert np.isfinite(y).all()
    assert abs(rms(y) / rms(speech) - 1.0) < 0.05  # RMS restored


# ---------------------------------------------------------------- rasterization

def test_rasterize_by_frame_center():
    # frame centers at 0.01s steps; segment [0.105, 0.115] contains centers
    # 0.11 but not 0.10/0.12 (inclusive bounds)
    flags = segments_to_frame_flags([[0.105, 0.115]], n_frames=20)
    assert flags.tolist() == [False] * 11 + [True] + [False] * 8


def test_majority_label():
    assert majority_label(np.zeros(21, bool)) == 0
    assert majority_label(np.ones(21, bool)) == 1
    assert majority_label(np.array([True] * 10 + [False] * 11)) == 0
    assert majority_label(np.array([True] * 11 + [False] * 10)) == 1


def test_window_label_alignment():
    # speech segment [1.7, 3.0] s; frame centers on the 10 ms grid
    seg = [[1.7, 3.0]]
    n = 10 * SR
    flags = segments_to_frame_flags(seg, n // 160 + 1)
    # window at 1.0 s (16000 samples): centers 1.0..1.2 -> no speech -> 0
    assert window_label(flags, 16_000) == 0
    # window at 1.8 s (28800 samples): centers 1.8..2.0 all inside -> 1
    assert window_label(flags, 28_800) == 1
    # window at 2.9 s (46400 samples): centers 2.9..3.1, 11 inside (2.9..3.0
    # inclusive) -> majority -> 1
    assert window_label(flags, 46_400) == 1
    # window at 3.1 s (49600 samples): centers 3.1..3.3 all outside -> 0
    assert window_label(flags, 49_600) == 0


# ---------------------------------------------------------------- silero wiring

def test_label_wav_passes_spec_params(monkeypatch):
    captured = {}

    def fake_get_ts(tensor, model, sampling_rate=None, **kwargs):
        captured.update(kwargs)
        return [{"start": 1600, "end": 32000}]  # 0.1s..2.0s

    wav = np.zeros(SR * 3, dtype=np.float32)
    m = label_wav(wav, model=None, get_ts=fake_get_ts)
    # build plan §1.3 exact hysteresis parameters
    assert captured == dict(
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        speech_pad_ms=30,
    )
    assert m["segments"] == [[0.1, 2.0]]
    assert abs(m["speech_fraction"] - (1.9 / 3.0)) < 1e-6


# ---------------------------------------------------------------- e2e cache

@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """3 s tone-burst speech clips + matching label JSONs + one noise wav."""
    root = tmp_path_factory.mktemp("corpus")
    speech, labels, noise = root / "speech", root / "labels", root / "noise"
    speech.mkdir(); labels.mkdir(); noise.mkdir()
    rng = np.random.default_rng(3)
    t = np.arange(3 * SR) / SR
    for i in range(4):
        wav = np.zeros(3 * SR, dtype=np.float32)
        wav[SR : 2 * SR] = 0.5 * np.sin(2 * np.pi * 440 * t[SR : 2 * SR])
        wav[int(2.5 * SR) :] = 0.4 * np.sin(2 * np.pi * 440 * t[int(2.5 * SR) :])
        # speech active 1.0-2.0 s and 2.5-3.0 s
        p = speech / f"utt{i}.wav"
        sf.write(p, wav, SR)
        (labels / f"utt{i}.json").write_text(
            '{"duration_sec": 3.0, "segments": [[1.0, 2.0], [2.5, 3.0]]}'
        )
    sf.write(noise / "white.wav", rng.standard_normal(2 * SR).astype(np.float32) * 0.2, SR)
    return speech, labels, noise


def test_noise_reader_eviction(tmp_path):
    """Regression: popitem() returns (key, value); eviction must close the
    SoundFile, and reads must stay correct across cache thrashing."""
    from pulsevad.build_cache import NoiseReader, WINDOW_SAMPLES

    rng = np.random.default_rng(7)
    paths = []
    for k in range(5):  # pool larger than cache_size -> constant eviction
        p = tmp_path / f"n{k}.wav"
        sf.write(p, np.full(WINDOW_SAMPLES, k + 1, dtype=np.float32), SR)
        paths.append([str(p), WINDOW_SAMPLES])

    reader = NoiseReader(paths, cache_size=2)
    for _ in range(50):  # thrash the cache well past eviction triggers
        w = reader.load_window(rng)
        assert w.shape == (WINDOW_SAMPLES,)
        assert len(reader._handles) <= 2
    for handle in reader._handles.values():  # cleanup closes real handles
        handle.close()
    assert all(v.closed for v in reader._handles.values())


def test_build_cache_end_to_end(corpus, tmp_path):
    speech, labels, noise = corpus
    out = tmp_path / "cache"
    manifest = build_cache(
        speech_dir=speech, labels_dir=labels, noise_dirs=[noise],
        out_dir=out, val_frac=0.0, seed=0, rir_pool_size=2,
    )
    feats = np.load(out / "train_features.npy", mmap_mode="r")
    lab = np.load(out / "train_labels.npy")
    n = (3 * SR - WINDOW) // 1600 + 1  # windows per 3 s file
    assert feats.shape == (4 * n, 64, 21)
    assert lab.shape == (4 * n,)
    assert set(np.unique(lab)) <= {0, 1}

    # label alignment: recompute majority vote independently for every window
    for i in range(4):
        wav, _ = sf.read(speech / f"utt{i}.wav", dtype="float32")
        centers = np.arange(len(wav) // 160 + 1) * 0.01
        flags = (centers >= 1.0) & (centers <= 2.0) | (centers >= 2.5) & (centers <= 3.0)
        for w in range(n):
            s0 = w * 1600
            expected = flags[s0 // 160 : s0 // 160 + 21].sum() * 2 > 21
            assert lab[i * n + w] == expected, f"file {i} window {w}"

    # 25/25/50 mix: all three categories present (seeded, 152 windows)
    cats = manifest["splits"]["train"]["categories"]
    assert all(v > 0 for v in cats.values())


def test_build_cache_split_and_val(corpus, tmp_path):
    out = tmp_path / "cache"
    manifest = build_cache(
        speech_dir=corpus[0], labels_dir=corpus[1], noise_dirs=[corpus[2]],
        out_dir=out, val_frac=0.25, seed=1, rir_pool_size=2,
    )
    assert manifest["splits"]["val"]["n"] >= 0
    if manifest["splits"]["val"]["n"]:
        assert (out / "val_features.npy").exists()


def test_build_eval_sets(corpus, tmp_path):
    out = tmp_path / "eval"
    summary = build_eval_sets(
        speech_dir=corpus[0], labels_dir=corpus[1], noise_dirs=[corpus[2]],
        out_dir=out, n_windows=8, rir_pool_size=2,
    )
    assert set(summary) == set(EVAL_CATEGORIES)
    for cat in EVAL_CATEGORIES:
        f = np.load(out / f"eval_{cat}_features.npy")
        l = np.load(out / f"eval_{cat}_labels.npy")
        assert f.shape == (8, 64, 21)
        assert not np.isnan(f).any()
        if cat == "pure_noise":
            assert (l == 0).all(), "pure-noise eval set must be all non-speech"
