"""Dataset downloaders (spec phase-03 §3.1). Designed to run on Modal with the
data volume mounted; all targets are CC BY 4.0 / CC0 — see ATTRIBUTION.md.

STRICTLY EXCLUDED (do not add): LibriVAD, Silero released labeled dataset
(CC BY-NC-SA), kiloVAD checkpoints (CC BY-NC).
"""

import subprocess
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"  # 6.3 GB
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"  # ~11 GB
DNS_REPO_URL = "https://github.com/microsoft/DNS-Challenge.git"
DNS_BRANCH = "interspeech2020"


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already downloaded")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "pulsevad/0.1"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := resp.read(1 << 23):  # 8 MB
                fh.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)
    return dest


def _extract(tar_path: Path, dest: Path) -> None:
    marker = dest / (tar_path.stem.replace(".tar", "") + ".extracted")
    if marker.exists():
        print(f"[skip] {tar_path.name} already extracted")
        return
    print(f"[extract] {tar_path.name}")
    with tarfile.open(tar_path) as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:  # Python < 3.12
            tf.extractall(dest)
    marker.touch()


def download_librispeech(raw_dir: Path) -> Path:
    """LibriSpeech train-clean-100 (CC BY 4.0), the primary speech corpus."""
    tar = _download(LIBRISPEECH_URL, raw_dir / "train-clean-100.tar.gz")
    _extract(tar, raw_dir)
    return raw_dir / "LibriSpeech" / "train-clean-100"


def download_musan(raw_dir: Path) -> Path:
    """MUSAN (CC BY 4.0): we use only the noise subset for augmentation."""
    tar = _download(MUSAN_URL, raw_dir / "musan.tar.gz")
    _extract(tar, raw_dir)
    return raw_dir / "musan"


def download_dns_noise(raw_dir: Path) -> Path:
    """Sparse LFS checkout of DNS-Challenge Interspeech-2020 free_sound noise (CC0).

    ponytail: read_speech (AudioSet CC BY) and DEMAND (CC BY-SA) are skipped —
    free_sound alone provides diverse CC0 environmental noise; add
    'datasets/full/no_noise/read_speech' to the sparse set for the paper's
    full noise mix at the cost of tens of GB.
    """
    repo = raw_dir / "DNS-Challenge"
    if not repo.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             "-b", DNS_BRANCH, DNS_REPO_URL, str(repo)],
            check=True,
        )
    subprocess.run(["git", "-C", str(repo), "lfs", "install"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "sparse-checkout", "set",
         "datasets/full/no_noise/free_sound"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "lfs", "pull",
         "--include=datasets/full/no_noise/free_sound/*"],
        check=True,
    )
    return repo / "datasets" / "full" / "no_noise" / "free_sound"


def download_all(raw_dir: Path, with_dns: bool = False) -> dict:
    libri = download_librispeech(raw_dir)
    musan = download_musan(raw_dir)
    out = {"librispeech": str(libri), "musan": str(musan)}
    if with_dns:
        out["dns_noise"] = str(download_dns_noise(raw_dir))
    return out
