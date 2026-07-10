"""
One-off offline preprocessing for the Multimodal Digit-Sum (MDS) benchmark.

Downloads MNIST (visual digits) and FSDD (Free Spoken Digit Dataset, spoken digits),
then extracts small, deliberately-degraded feature vectors for each, using only
numpy/scipy (no torchvision/torchaudio/librosa). Caches everything into
knitwork/gens/data/mdsum_cache/{mnist,fsdd}_features.npz, which is all
knitwork/gens/multimodal_sum.py needs at runtime.

Usage:
    uv run knitwork/gens/data/prepare_mdsum_cache.py [--feat_dim 64] [--force]
"""
from __future__ import annotations

import argparse
import gzip
import io
import struct
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

CACHE_DIR = Path(__file__).parent / "mdsum_cache"

MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
FSDD_ZIP_URL = "https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/zip/refs/heads/master"


def _download(url: str) -> bytes:
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


#  MNIST

def _parse_idx(data: bytes) -> np.ndarray:
    magic, = struct.unpack(">I", data[:4])
    n_dims = magic & 0xFF
    dims = struct.unpack(f">{n_dims}I", data[4:4 + 4 * n_dims])
    arr = np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * n_dims)
    return arr.reshape(dims)


def _fetch_mnist_raw() -> dict:
    out = {}
    for key, fname in MNIST_FILES.items():
        raw = gzip.decompress(_download(f"{MNIST_BASE_URL}/{fname}"))
        out[key] = _parse_idx(raw)
    return out


def build_mnist_cache(feat_dim: int, variance_retained: float = 0.25) -> dict:
    raw = _fetch_mnist_raw()
    train_x = raw["train_images"].reshape(-1, 784).astype(np.float32) / 255.0
    train_y = raw["train_labels"].astype(np.int64)
    test_x = raw["test_images"].reshape(-1, 784).astype(np.float32) / 255.0
    test_y = raw["test_labels"].astype(np.int64)

    return _finalize_image_cache(train_x, train_y, test_x, test_y, feat_dim, variance_retained)


def _finalize_image_cache(train_x, train_y, test_x, test_y, feat_dim, variance_retained):
    mean = train_x.mean(axis=0, keepdims=True)
    centered = train_x - mean

    # PCA via SVD, fit on train only; keep enough components for the requested
    # retained-variance target, but hard-cap at feat_dim (deliberately lossy,
    # mirrors AV-MNIST's degrade-per-modality-on-purpose design).
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    explained = (s ** 2)
    explained_ratio = np.cumsum(explained) / explained.sum()
    n_variance = int(np.searchsorted(explained_ratio, variance_retained) + 1)
    n_components = min(feat_dim, n_variance, vt.shape[0])
    components = vt[:n_components]  # (n_components, 784)

    def project(x):
        z = (x - mean) @ components.T
        if n_components < feat_dim:
            pad = np.zeros((z.shape[0], feat_dim - n_components), dtype=np.float32)
            z = np.concatenate([z, pad], axis=1)
        return z.astype(np.float32)

    train_feat = project(train_x)
    test_feat = project(test_x)

    # normalize to unit-ish scale so it plays nicely with buffer noise std
    scale = np.abs(train_feat).mean() + 1e-8
    train_feat /= scale
    test_feat /= scale

    features = np.concatenate([train_feat, test_feat], axis=0)
    labels = np.concatenate([train_y, test_y], axis=0)
    n_train = len(train_y)
    split_train_idx = np.arange(n_train)
    split_test_idx = np.arange(n_train, len(labels))

    return dict(
        features=features.astype(np.float32),
        labels=labels.astype(np.int64),
        pca_components=components.astype(np.float32),
        pca_mean=mean.astype(np.float32),
        pca_scale=np.float32(scale),
        split_train_idx=split_train_idx,
        split_test_idx=split_test_idx,
    )


#  FSDD

def _fetch_fsdd_wavs() -> list[tuple[int, int, np.ndarray]]:
    """Returns list of (digit_label, speaker_hash, waveform)."""
    zip_bytes = _download(FSDD_ZIP_URL)
    items = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".wav") and "/recordings/" in n]
        for name in names:
            fname = name.rsplit("/", 1)[-1]
            digit_str, speaker, _idx = fname[:-4].split("_")
            digit = int(digit_str)
            speaker_hash = abs(hash(speaker)) % (2 ** 31)
            with zf.open(name) as f:
                _sr, wav = wavfile.read(io.BytesIO(f.read()))
            items.append((digit, speaker_hash, wav.astype(np.float32)))
    return items


def _wav_to_feature(wav: np.ndarray, feat_dim: int, sample_rate: int = 8000) -> np.ndarray:
    wav = wav / (np.abs(wav).max() + 1e-8)
    nperseg = min(256, max(32, len(wav) // 4))
    _f, _t, sxx = spectrogram(wav, fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2)
    log_sxx = np.log1p(sxx)
    # mean-pool over time -> fixed-size frequency-bin vector, then resize to feat_dim
    pooled = log_sxx.mean(axis=1)
    if len(pooled) >= feat_dim:
        feat = pooled[:feat_dim]
    else:
        feat = np.concatenate([pooled, np.zeros(feat_dim - len(pooled))])
    return feat.astype(np.float32)


def build_fsdd_cache(feat_dim: int, n_test_speakers: int = 1) -> dict:
    items = _fetch_fsdd_wavs()
    labels = np.array([d for d, _, _ in items], dtype=np.int64)
    speaker_ids = np.array([s for _, s, _ in items], dtype=np.int64)
    feats = np.stack([_wav_to_feature(w, feat_dim) for _, _, w in items])

    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-8
    feats = (feats - mean) / std

    unique_speakers = np.unique(speaker_ids)
    test_speakers = set(unique_speakers[:n_test_speakers].tolist())
    test_mask = np.array([s in test_speakers for s in speaker_ids])
    split_train_idx = np.flatnonzero(~test_mask)
    split_test_idx = np.flatnonzero(test_mask)

    return dict(
        features=feats.astype(np.float32),
        labels=labels,
        speaker_ids=speaker_ids,
        feat_mean=mean.astype(np.float32),
        feat_std=std.astype(np.float32),
        split_train_idx=split_train_idx,
        split_test_idx=split_test_idx,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dim", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mnist_path = CACHE_DIR / "mnist_features.npz"
    fsdd_path = CACHE_DIR / "fsdd_features.npz"

    if args.force or not mnist_path.exists():
        cache = build_mnist_cache(args.feat_dim)
        np.savez(mnist_path, **cache)
        print(f"Saved {mnist_path} ({cache['features'].shape[0]} samples, dim={cache['features'].shape[1]})")
    else:
        print(f"{mnist_path} exists, skipping (use --force to rebuild)")

    if args.force or not fsdd_path.exists():
        cache = build_fsdd_cache(args.feat_dim)
        np.savez(fsdd_path, **cache)
        print(f"Saved {fsdd_path} ({cache['features'].shape[0]} samples, dim={cache['features'].shape[1]})")
    else:
        print(f"{fsdd_path} exists, skipping (use --force to rebuild)")


if __name__ == "__main__":
    main()
