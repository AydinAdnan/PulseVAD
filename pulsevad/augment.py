"""Acoustic augmentation (spec phase-03 §2.1, §3.3): SNR mixing math, synthetic
wind noise (Mirabilii-style proxy), and pyroomacoustics room reverb.
"""

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 16_000


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """x = s + g*w with g set so RMS(s)/RMS(g*w) = 10^(snr/20); anti-clip guard."""
    gain = rms(speech) / (rms(noise) + 1e-8) * 10.0 ** (-snr_db / 20.0)
    mixed = speech + gain * noise
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed = mixed / (peak + 1e-5)
    return mixed


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f power noise via FFT spectrum weighting (amplitude ~ 1/sqrt(f))."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]  # avoid div-by-zero on DC
    shaped = spec / np.sqrt(freqs)
    p = np.fft.irfft(shaped, n)
    return p / (rms(p) + 1e-8)


def wind_noise(n: int, sr: int = SR, rng: np.random.Generator | None = None) -> np.ndarray:
    """Synthetic airflow wind: pink noise, 500 Hz Butterworth low-pass, slow
    time-varying gust envelope. Mirabilii et al. 2022-style proxy."""
    rng = rng or np.random.default_rng()
    x = sosfilt(butter(4, 500, btype="low", fs=sr, output="sos"), pink_noise(n, rng))
    env = sosfilt(butter(2, 0.5, btype="low", fs=sr, output="sos"), rng.standard_normal(n))
    env = env / (rms(env) + 1e-8)
    x = x * (1.0 + 0.5 * env)
    return x / (rms(x) + 1e-8)


def simulate_rir(sr: int = SR, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random ShoeBox RIR: L in [3,8] m, W in [3,6], H in [2.5,4], T60 in [0.15,0.5]."""
    import pyroomacoustics as pra

    rng = rng or np.random.default_rng()
    room_dim = [rng.uniform(3, 8), rng.uniform(3, 6), rng.uniform(2.5, 4)]
    t60 = rng.uniform(0.15, 0.5)
    e_abs, max_order = pra.inverse_sabine(t60, room_dim)
    # max_order can be huge/invalid for small rooms + long T60 without ray tracing
    max_order = min(max(int(max_order), 1), 50)
    room = pra.ShoeBox(
        room_dim, fs=sr, materials=pra.Material(e_abs), max_order=max_order
    )
    for _ in range(5):  # place source/mic with >=0.5 m separation
        src = [rng.uniform(0.3, d - 0.3) for d in room_dim]
        mic = [rng.uniform(0.3, d - 0.3) for d in room_dim]
        if np.linalg.norm(np.array(src) - np.array(mic)) >= 0.5:
            break
    room.add_source(src)
    room.add_microphone(mic)
    room.compute_rir()  # RIR-only: simulate() needs source signals
    # ponytail: cap the image-source order — full tail for T60=0.5s small rooms
    # needs ray tracing; 50 images gives a correct-sounding tail in ~3ms less.
    rir = np.asarray(room.rir[0][0], dtype=np.float32)
    return rir / (np.max(np.abs(rir)) + 1e-8)


def reverb_apply(speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve with an RIR, keep the input length, restore input RMS."""
    y = fftconvolve(speech, rir)[: len(speech)]
    return y * (rms(speech) / (rms(y) + 1e-8))
