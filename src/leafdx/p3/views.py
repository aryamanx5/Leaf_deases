"""Derived images for Phase 3, cached as PNG under .cache/views/<kind>/<image hash>.png.

* ``leafonly``: Stage 1 leaf mask applied, background set to mid-grey (shortcut test, P3-M07).
* ``swap``: the image's own Stage 1 leaf cut-out composited once onto a TRAIN-bank or
  procedural background, no collage (VAL/TEST robustness proxy; never used for training).
* ``train_swap_views``: K background-swap views per TRAIN image (Phase 2 recipe) for the
  frozen-backbone heads, generated in memory with recipe-hash keys.
"""

from __future__ import annotations

import json
import random

import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as T
from joblib import Parallel, delayed
from PIL import Image

from leafdx.cnn.augment import BackgroundSwap, SwapDataset, _cutout, robust_transform
from leafdx.config import ROOT
from leafdx.ip.preprocess import load_bgr, preprocess
from leafdx.ip.segment import segment_leaf
from leafdx.p3.data import file_hash
from leafdx.p3.embcache import recipe_hash

GREY = 128


def view_path(kind: str, p: str, cfg: dict):
    return ROOT / ".cache" / "views" / kind / f"{file_hash(p)}.png"


def _leafonly(p: str, cfg: dict) -> str:
    out = view_path("leafonly", p, cfg)
    if not out.exists():
        pre = preprocess(load_bgr(ROOT / p), cfg)
        m = segment_leaf(pre, cfg)["leaf_mask"].astype(bool)
        rgb = cv2.cvtColor(pre["resized"], cv2.COLOR_BGR2RGB)
        rgb[~m] = GREY
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(out)
    return str(out)


def _swap(p: str, crop: str, banks_bg: np.ndarray, cfg: dict) -> str | None:
    out = view_path("swap", p, cfg)
    if not out.exists():
        c = _cutout(p, cfg)
        if c is None:
            return None
        aug = {**cfg["augment"], "p_collage": 0.0}
        sw = BackgroundSwap({"backgrounds": list(banks_bg), "by_crop": {crop: []}, "items": []}, {**cfg, "augment": aug},
                            seed=int(file_hash(p)[:8], 16))
        out.parent.mkdir(parents=True, exist_ok=True)
        sw(c[0], crop).save(out)
    return str(out)


def leafonly_paths(paths: list[str], cfg: dict) -> list[str]:
    return Parallel(n_jobs=-1)(delayed(_leafonly)(p, cfg) for p in paths)


def swap_paths(df: pd.DataFrame, banks: dict, cfg: dict) -> list[str | None]:
    """One swapped view per row (VAL/TEST); None when Stage 1 gives no usable cut-out."""
    bg = np.stack(banks["backgrounds"])  # one array: joblib memory-maps it instead of pickling per task
    return Parallel(n_jobs=-1)(delayed(_swap)(r.path, r.crop, bg, cfg) for r in df.itertuples())


def train_view_keys(train: pd.DataFrame, cfg: dict) -> list[str]:
    """Recipe keys, view-major (view k of every image, then k+1), matching train_views()."""
    k = cfg["p3"]["train_swap_views"]
    aug = json.dumps(cfg["augment"], sort_keys=True)
    run = recipe_hash(*train.path, aug, cfg["seed"], k)
    return [recipe_hash(run, file_hash(p), v) for v in range(k) for p in train.path]


def train_views(train: pd.DataFrame, banks: dict, cfg: dict) -> list[Image.Image]:
    random.seed(cfg["seed"])  # robust_transform draws from the global RNG
    ds = SwapDataset(train, banks, cfg, target_fn=lambda r: 0, seed=cfg["seed"])
    geo = T.Compose(robust_transform(cfg).transforms[:-3])
    return [geo(ds.sample_image(i)) for _ in range(cfg["p3"]["train_swap_views"]) for i in range(len(ds))]


def save_train_views(train: pd.DataFrame, banks: dict, cfg: dict) -> list[str]:
    """Write the K swap views per TRAIN image to .cache/views/trainswap/<recipe key>.png
    (generated once; embeddings and Stage 1 features then read the same pixels)."""
    keys = train_view_keys(train, cfg)
    files = [ROOT / ".cache" / "views" / "trainswap" / f"{k}.png" for k in keys]
    if not all(f.exists() for f in files):
        files[0].parent.mkdir(parents=True, exist_ok=True)
        for f, im in zip(files, train_views(train, banks, cfg), strict=True):
            im.save(f)
    return [str(f) for f in files]


def load_png(p: str) -> Image.Image:
    with Image.open(p) as im:
        return im.convert("RGB")


def mask_fraction(p: str) -> float:
    a = np.asarray(load_png(p))
    return float((a != GREY).any(-1).mean())
