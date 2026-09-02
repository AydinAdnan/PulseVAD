"""Feature cache builder (spec phase-03 §3.4 + build plan §4.2).

Windows: 200 ms (3200 samples), hop 100 ms (50% overlap).
Per-window mix (paper-faithful, config libri_dns_full_no_pure_noise_v2):
  25% clean / 25% wind @ -5 dB / 50% DNS+MUSAN noise @ SNR in {-10,-5,0,+5,+10} dB,
  reverb on 50% of noisy windows (pre-generated RIR pool), 0% pure-noise files.
Labels: majority vote over the 21 frame centers (>=11 speech frames -> 1).
"""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from pulsevad.augment import (
    mix_at_snr,
    reverb_apply,
    simulate_rir,
    wind_noise,
)
from pulsevad.frontend import MelFrontend
from pulsevad.label_corpus import AUDIO_EXTS, read_mono, segments_to_frame_flags

WINDOW_SAMPLES = 3_200
HOP_SAMPLES = 1_600
FRAME_SEC = 0.01
FRAMES_PER_WINDOW = 21
STFT_FRAME_SEC = 0.01  # STFT center=True frame k sits at sample k*160

WIND_SNR_DB = -5.0
NOISE_SNRS_DB = [-10.0, -5.0, 0.0, 5.0, 10.0]


def majority_label(frame_flags: np.ndarray) -> int:
    """1 iff strictly more than half of the window's frames are speech."""
    return int(frame_flags.sum() * 2 > len(frame_flags))


def file_frame_flags(segments: list, n_samples: int, sr: int = 16_000) -> np.ndarray:
    n_frames = int(n_samples / (sr * FRAME_SEC)) + 1
    return segments_to_frame_flags(segments, n_frames, FRAME_SEC)


def window_label(flags: np.ndarray, start: int) -> int:
    """Window starting at sample `start` (multiple of hop): its 21 STFT frame
    centers sit at samples start + j*160 -> flag indices start//160 + j."""
    first = start // 160
    return majority_label(flags[first : first + FRAMES_PER_WINDOW])


def _count_windows(n_samples: int) -> int:
    return max(0, (n_samples - WINDOW_SAMPLES) // HOP_SAMPLES + 1)


def build_noise_pool(noise_dirs: list) -> list:
    """[(path, n_frames)] for 16 kHz wav files long enough to cover a window."""
    pool = []
    for d in noise_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.wav")):
            info = sf.info(p)
            if info.samplerate == 16_000 and info.frames >= WINDOW_SAMPLES:
                pool.append([str(p), info.frames])
    if not pool:
        raise FileNotFoundError(f"no usable 16 kHz noise wavs under {noise_dirs}")
    return pool


def load_noise_window(pool: list, rng: np.random.Generator) -> np.ndarray:
    """One-shot loader (tests / low-volume use). The cache builders use
    NoiseReader instead — a fresh handle per window is ruinous over NFS."""
    return NoiseReader(pool).load_window(rng)


class NoiseReader:
    """Random noise-window reader with an LRU cache of open file handles.
    ponytail: without handle reuse, cache building pays ~1.6M open/seek/close
    round-trips on the volume; with 64 cached handles the hit rate is near 100%.
    Raise cache size if the noise pool grows past a few hundred files.
    """

    def __init__(self, pool: list, cache_size: int = 64) -> None:
        self.pool = pool
        self.cache_size = cache_size
        self._handles: OrderedDict[str, sf.SoundFile] = OrderedDict()

    def _handle(self, path: str) -> sf.SoundFile:
        h = self._handles.get(path)
        if h is None:
            if len(self._handles) >= self.cache_size:
                _, old = self._handles.popitem(last=False)  # evict LRU
                old.close()
            h = sf.SoundFile(path)
            self._handles[path] = h
        else:
            self._handles.move_to_end(path)
        return h

    def load_window(self, rng: np.random.Generator) -> np.ndarray:
        path, n_frames = self.pool[rng.integers(len(self.pool))]
        start = int(rng.integers(0, n_frames - WINDOW_SAMPLES + 1))
        f = self._handle(path)
        f.seek(start)
        noise = f.read(WINDOW_SAMPLES, dtype="float32", always_2d=True)[:, 0]
        if len(noise) < WINDOW_SAMPLES:  # defensive: file truncated after scan
            reps = WINDOW_SAMPLES // len(noise) + 1
            noise = np.tile(noise, reps)[:WINDOW_SAMPLES]
        return noise


def draw_category(rng: np.random.Generator) -> str:
    r = rng.random()
    return "clean" if r < 0.25 else "wind" if r < 0.5 else "noise"


def augment_window(
    seg: np.ndarray,
    category: str,
    rng: np.random.Generator,
    noise_reader: NoiseReader,
    rir_pool: list,
    snr_db: float | None = None,
) -> np.ndarray:
    """Reverb 50% of noisy windows, then mix wind or corpus noise. `snr_db`
    overrides the uniform noise SNR draw (used by the eval-set builders)."""
    if category == "clean":
        return seg
    if category != "pure_noise" and rng.random() < 0.5:
        seg = reverb_apply(seg, rir_pool[rng.integers(len(rir_pool))])
    if category == "wind":
        return mix_at_snr(seg, wind_noise(WINDOW_SAMPLES, rng=rng), WIND_SNR_DB)
    noise = noise_reader.load_window(rng)
    if snr_db is None:
        snr_db = NOISE_SNRS_DB[rng.integers(len(NOISE_SNRS_DB))]
    return mix_at_snr(seg, noise, snr_db)


def _speech_files(speech_dir: Path, labels_dir: Path) -> list:
    files = []
    for p in sorted(Path(speech_dir).rglob("*")):
        if p.suffix.lower() in AUDIO_EXTS:
            lj = Path(labels_dir) / p.relative_to(speech_dir).with_suffix(".json")
            if lj.exists():
                files.append((str(p), str(lj)))
    if not files:
        n_audio = sum(
            1
            for p in Path(speech_dir).rglob("*")
            if p.suffix.lower() in AUDIO_EXTS
        )
        n_json = sum(1 for _ in Path(labels_dir).rglob("*.json"))
        raise FileNotFoundError(
            f"no labeled speech files: found {n_audio} audio files under "
            f"{speech_dir} but only {n_json} label manifests under {labels_dir}. "
            f"If {n_json} == 0, run ::label; if both counts are >0, check the "
            f"manifest sub-path layout mirrors the speech tree."
        )
    return files


def _frontend_batch(frontend: MelFrontend, windows: list[np.ndarray]) -> np.ndarray:
    x = torch.from_numpy(
        np.stack(windows).astype(np.float32, copy=False)
    )  # scipy reverb/mixing returns float64; the frontend is float32
    with torch.no_grad():
        return frontend(x).numpy().astype(np.float32)  # (W, 64, 21)


_W = {}  # per-process worker state (noise reader, RIR pool, frontend)


def _worker_init(noise_dirs, rir_pool_size, base_seed, noise_only_frac,
                 max_windows_per_file):
    """ProcessPoolExecutor initializer: build the heavy per-process state once."""
    _W["reader"] = NoiseReader(build_noise_pool(noise_dirs))
    rng = np.random.default_rng(base_seed)
    _W["rir_pool"] = [simulate_rir(rng=rng) for _ in range(rir_pool_size)]
    _W["frontend"] = MelFrontend()
    _W["noise_only_frac"] = noise_only_frac
    _W["max_windows_per_file"] = max_windows_per_file


def _process_file(item):
    """Featurize one file -> (offset, n, features, labels, cat_counts).

    Per-file RNG (base_seed + 7919*file_idx) keeps results deterministic
    regardless of process scheduling.
    """
    file_idx, path, lj, n_win, offset, base_seed = item
    rng = np.random.default_rng(base_seed + 7919 * file_idx)
    reader, rir_pool = _W["reader"], _W["rir_pool"]
    wav = read_mono(Path(path))
    seg_manifest = json.loads(Path(lj).read_text())
    flags = file_frame_flags(seg_manifest["segments"], len(wav))

    starts = list(range(0, len(wav) - WINDOW_SAMPLES + 1, HOP_SAMPLES))
    if not starts:
        return offset, 0, None, None, None
    mwpf = _W["max_windows_per_file"]
    if mwpf is not None and len(starts) > mwpf:
        step = len(starts) / mwpf
        starts = [starts[int(k * step)] for k in range(mwpf)]

    windows, win_labels = [], []
    cat_counts = {"clean": 0, "wind": 0, "noise": 0}
    for s0 in starts:
        if rng.random() < _W["noise_only_frac"]:
            # Pure-noise window labeled 0: without these, the model never sees
            # noise without speech (silence-window SNR mixing zeroes the noise
            # out) and scores pure noise at ~0.5 — failing the phase-07
            # pure-noise FPR < 5% gate.
            windows.append(reader.load_window(rng))
            win_labels.append(0)
            cat_counts["noise"] += 1
            continue
        category = draw_category(rng)
        seg = augment_window(
            wav[s0 : s0 + WINDOW_SAMPLES].astype(np.float32).copy(),
            category, rng, reader, rir_pool,
        )
        cat_counts[category] += 1
        windows.append(seg)
        win_labels.append(window_label(flags, s0))

    feats = _frontend_batch(_W["frontend"], windows)
    return offset, len(windows), feats, np.array(win_labels, dtype=np.uint8), cat_counts


def _max_starts(starts, max_windows_per_file=None):
    if max_windows_per_file is not None and len(starts) > max_windows_per_file:
        step = len(starts) / max_windows_per_file
        return [starts[int(k * step)] for k in range(max_windows_per_file)]
    return starts


def build_cache(
    speech_dir,
    labels_dir,
    noise_dirs: list,
    out_dir,
    val_frac: float = 0.05,
    seed: int = 0,
    rir_pool_size: int = 200,
    max_windows_per_file: int | None = None,
    noise_only_frac: float = 0.10,
    workers: int = 8,
) -> dict:
    """Two-pass build: count windows from label durations, pre-allocate memmaps,
    fill via a process pool (one task per file). Returns the manifest."""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    from multiprocessing import cpu_count

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _speech_files(speech_dir, labels_dir)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_frac)) if len(files) > 20 else 0
    splits = {"train": files[n_val:], "val": files[:n_val]}

    manifest = {
        "window_samples": WINDOW_SAMPLES,
        "hop_samples": HOP_SAMPLES,
        "seed": seed,
        "workers": workers,
        "splits": {},
    }

    for tag, split_files in splits.items():
        base_seed = seed + (1 if tag == "val" else 0)
        print(f"[build:{tag}] reading {len(split_files)} label manifests…", flush=True)
        with ThreadPoolExecutor(max_workers=32) as ex:
            durations = list(
                ex.map(
                    lambda lj: json.loads(Path(lj).read_text())["duration_sec"],
                    [lj for _, lj in split_files],
                )
            )
        counts = [_count_windows(int(d * 16_000)) for d in durations]
        if max_windows_per_file is not None:
            counts = [min(c, max_windows_per_file) for c in counts]
        n_total = sum(counts)

        feats = np.lib.format.open_memmap(
            out_dir / f"{tag}_features.npy", mode="w+",
            dtype=np.float32, shape=(n_total, 64, 21),
        )
        labels = np.lib.format.open_memmap(
            out_dir / f"{tag}_labels.npy", mode="w+", dtype=np.uint8, shape=(n_total,)
        )

        # work items with precomputed memmap offsets (prefix sums)
        items, offsets, done = [], [], 0
        for file_idx, ((path, lj), n_win) in enumerate(zip(split_files, counts)):
            offsets.append(done)
            done += n_win
            if n_win:
                items.append((file_idx, path, lj, n_win, offsets[-1], base_seed))

        n_workers = max(1, min(workers, cpu_count() or workers))
        print(f"[build:{tag}] {n_total} windows over {len(items)} files "
              f"on {n_workers} workers", flush=True)
        cat_counts = {"clean": 0, "wind": 0, "noise": 0}
        written = 0
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(noise_dirs, rir_pool_size, base_seed, noise_only_frac,
                      max_windows_per_file),
        ) as pool:
            futs = {pool.submit(_process_file, it): it[4] for it in items}
            done = 0
            for fut in as_completed(futs):
                offset, n, f, lab, cats = fut.result()
                if n:
                    feats[offset:offset + n] = f
                    labels[offset:offset + n] = lab
                    for k in cat_counts:
                        cat_counts[k] += cats[k]
                written += n
                done += 1
                if done % 500 == 0:
                    print(f"[build:{tag}] {done}/{len(items)} files, "
                          f"{written}/{n_total} windows "
                          f"({100 * written / max(n_total, 1):.1f}%)", flush=True)

        manifest["splits"][tag] = {
            "features": f"{tag}_features.npy",
            "labels": f"{tag}_labels.npy",
            "n": n_total,
            "hours": round(n_total * HOP_SAMPLES / 16_000 / 3600, 2),
            "categories": cat_counts,
        }
        print(f"[{tag}] {n_total} windows ({cat_counts})", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


EVAL_CATEGORIES = ["clean", "windy", "dns_synthetic", "speech_noise", "pure_noise"]


def build_eval_sets(
    speech_dir,
    labels_dir,
    noise_dirs: list,
    out_dir,
    n_windows: int = 2_000,
    seed: int = 123,
    rir_pool_size: int = 50,
    save_audio: bool = False,
) -> dict:
    """Five held-out categories (build plan §8.2), drawn from the full corpus.

    ponytail: 'speech_noise' is a proxy for the DNS Speech+Noise category —
    real multi-talker DNS recordings aren't downloaded, so we mix speech with
    DNS/MUSAN noise at close-range SNRs {-5,0,+5}. 'pure_noise' has all-zero
    labels and gates the false-positive check (<5%) in Phase 7.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _speech_files(speech_dir, labels_dir)
    rng.shuffle(files)
    noise_reader = NoiseReader(build_noise_pool(noise_dirs))
    print(f"[eval] noise pool ready; simulating {rir_pool_size} RIRs…", flush=True)
    rir_pool = [simulate_rir(rng=rng) for _ in range(rir_pool_size)]
    frontend = MelFrontend()

    summary = {}
    for cat in EVAL_CATEGORIES:
        feats, labels = [], []
        while len(feats) < n_windows:
            path, lj = files[rng.integers(len(files))]
            wav = read_mono(Path(path))
            seg_manifest = json.loads(Path(lj).read_text())
            flags = file_frame_flags(seg_manifest["segments"], len(wav))
            starts = list(range(0, len(wav) - WINDOW_SAMPLES + 1, HOP_SAMPLES))
            if not starts:
                continue
            s0 = starts[rng.integers(len(starts))]

            if cat == "pure_noise":
                seg = noise_reader.load_window(rng)
                label = 0
            else:
                seg = augment_window(
                    wav[s0 : s0 + WINDOW_SAMPLES].astype(np.float32).copy(),
                    {"windy": "wind", "dns_synthetic": "noise",
                     "speech_noise": "noise"}.get(cat, "clean"),
                    rng, noise_reader, rir_pool,
                    snr_db=NOISE_SNRS_DB[1:4][rng.integers(3)] if cat == "speech_noise" else None,
                )
                label = window_label(flags, s0)
            feats.append(seg)
            labels.append(label)

        f = _frontend_batch(frontend, feats)
        np.save(out_dir / f"eval_{cat}_features.npy", f)
        np.save(out_dir / f"eval_{cat}_labels.npy", np.array(labels, dtype=np.uint8))
        if save_audio:
            # raw mixed audio so competitor VADs (Silero, MarbleNet) can score
            # the SAME windows we score — the apples-to-apples phase-7 eval
            np.save(out_dir / f"eval_{cat}_audio.npy",
                    np.stack(feats).astype(np.float32))
        summary[cat] = {"n": n_windows, "speech_fraction": float(np.mean(labels))}
        print(f"[eval:{cat}] {n_windows} windows, speech fraction {summary[cat]['speech_fraction']:.3f}")

    (out_dir / "eval_manifest.json").write_text(json.dumps(summary, indent=2))
    return summary
