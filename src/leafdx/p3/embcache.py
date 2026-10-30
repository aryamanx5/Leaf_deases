"""Content-addressed embedding cache (P3-M01).

Key = (model name, weights hash, image hash):
    .cache/embeddings/<model>/<weights hash[:12]>/<image hash>[__<view>].npy
The image hash is the sha1 of the file bytes; for generated images (background-swap views)
it is the sha1 of the generation recipe (source image hash, view index, augmentation
config, seed), see ``recipe_hash``. Unchanged inputs are never re-embedded.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image

from leafdx.config import ROOT, load_config
from leafdx.p3.data import file_hash, load_rgb


def cache_root(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return ROOT / cfg["p3"]["embed_cache"]


def weights_file(name: str) -> Path:
    import torch
    from huggingface_hub import hf_hub_download

    if name == "dinov2_vits14":
        return Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vits14_pretrain.pth"
    repo = {"openclip_vitb32": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "bioclip2": "imageomics/bioclip-2"}[name]
    return Path(hf_hub_download(repo, "open_clip_model.safetensors"))


def _sha_file(f: Path, chunk: int = 8 << 20) -> str:
    """sha1 over size + first and last 8 MB (weights files are large and immutable)."""
    h = hashlib.sha1(str(f.stat().st_size).encode())
    with open(f, "rb") as fh:
        h.update(fh.read(chunk))
        fh.seek(max(0, f.stat().st_size - chunk))
        h.update(fh.read(chunk))
    return h.hexdigest()


@cache
def weights_hash(name: str) -> str:
    f = weights_file(name)
    idx = cache_root() / "weights_hashes.json"
    known = json.loads(idx.read_text()) if idx.exists() else {}
    k = f"{f}|{f.stat().st_size}|{f.stat().st_mtime_ns}"
    if k not in known:
        idx.parent.mkdir(parents=True, exist_ok=True)
        known[k] = _sha_file(f)
        idx.write_text(json.dumps(known, indent=1))
    return known[k]


def recipe_hash(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()


def _file(name: str, key: str) -> Path:
    return cache_root() / name / weights_hash(name)[:12] / f"{key}.npy"


def embed_keys(name: str, keys: list[str], make: Callable[[list[int]], list[Image.Image]],
               batch: int = 64, dev_str: str | None = None) -> np.ndarray:
    """Embeddings for cache keys; ``make(idx)`` builds the missing images on demand."""
    from leafdx.robust.backbones import embed

    files = [_file(name, k) for k in keys]
    missing = [i for i, f in enumerate(files) if not f.exists()]
    for s in range(0, len(missing), 512):
        part = missing[s:s + 512]
        e = embed(name, make(part), batch=batch, dev_str=dev_str)
        files[part[0]].parent.mkdir(parents=True, exist_ok=True)
        for i, v in zip(part, e, strict=True):
            np.save(files[i], v.astype(np.float32))
    return np.stack([np.load(f) for f in files]) if files else np.zeros((0, 1), np.float32)


def embed_paths(name: str, paths: list[str], dev_str: str | None = None) -> np.ndarray:
    """Embeddings of image files (full frame), keyed by the file's sha1."""
    keys = [file_hash(p) for p in paths]
    return embed_keys(name, keys, lambda idx: [load_rgb(paths[i]) for i in idx], dev_str=dev_str)


def n_cached(name: str) -> int:
    d = cache_root() / name / weights_hash(name)[:12]
    return len(list(d.glob("*.npy"))) if d.exists() else 0
