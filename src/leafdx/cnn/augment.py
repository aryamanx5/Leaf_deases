"""Background-swap + robust augmentation (P2-B03), TRAIN split only.

The v1 CNN learned the crop from the photo style (turmeric always on white paper, black
gram mostly in the field). Here every training leaf is cut out with its Stage 1 mask and
composited, at load time, onto a background that carries no crop information:
  * backgrounds from BOTH crops' TRAIN images with the leaf removed + inpainted,
  * procedural soil / foliage textures and blurred versions of the above,
  * plain colours,
optionally as a multi-leaf collage (2-4 leaves of the SAME crop) to mimic field scenes.
Then RandomResizedCrop(0.25-1), rotation, colour jitter, Gaussian blur, JPEG
re-compression and random erasing. Nothing is written into the dataset; banks are cached.
"""

from __future__ import annotations

import hashlib
import io
import json
import random

import cv2
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from PIL import Image, ImageFilter
from torchvision import transforms as T

from leafdx.cnn.dataset import IMAGENET_MEAN, IMAGENET_STD
from leafdx.config import ROOT, path
from leafdx.ip.preprocess import load_bgr, preprocess
from leafdx.ip.segment import segment_leaf


class LeakageError(ValueError):
    """Raised when a non-TRAIN image would be used to build augmentation banks."""


def _bank_key(paths: list[str], cfg: dict) -> str:
    blob = yaml.safe_dump({k: cfg[k] for k in ("preprocess", "segment")}) + "|".join(sorted(paths)) + json.dumps(cfg["augment"])
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _cutout(p: str, cfg: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """RGBA leaf cut-out (cropped to its box) and an inpainted background, both RGB uint8."""
    a = cfg["augment"]
    pre = preprocess(load_bgr(ROOT / p), cfg)
    seg = segment_leaf(pre, cfg)
    img, m = pre["resized"], seg["leaf_mask"].astype(np.uint8)
    if not 0.03 <= m.mean() <= 0.97:
        return None
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    alpha = cv2.GaussianBlur((m * 255).astype(np.uint8), (5, 5), 0)  # feathered edge
    rgba = np.dstack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha])[y0:y1, x0:x1]
    s = a["leaf_max_side"] / max(rgba.shape[:2])
    if s < 1:
        rgba = cv2.resize(rgba, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    # background: remove the (dilated) leaf and inpaint at low resolution
    small = cv2.resize(img, (a["bg_side"], a["bg_side"]), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(m, (a["bg_side"], a["bg_side"]), interpolation=cv2.INTER_NEAREST)
    ms = cv2.dilate(ms, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bg = cv2.inpaint(small, ms * 255, 7, cv2.INPAINT_TELEA)
    return rgba, cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)


def build_banks(train_df: pd.DataFrame, cfg: dict) -> dict:
    """Leaf cut-outs + inpainted backgrounds from TRAIN rows only (cached on disk).

    Returns {"items": [{path, crop, label, rgba}], "by_crop": {crop: [item idx]},
             "backgrounds": [rgb], "index": {path: item idx}}.
    """
    if "split" in train_df and (train_df["split"] != "train").any():
        raise LeakageError("augmentation banks may only use TRAIN images")
    key = _bank_key(list(train_df.path), cfg)
    cache = path("cache", cfg) / f"augbank_{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        items, backgrounds = list(z["items"]), list(z["backgrounds"])
    else:
        out = Parallel(n_jobs=-1)(delayed(_cutout)(p, cfg) for p in train_df.path)
        items, backgrounds = [], []
        for r, o in zip(train_df.itertuples(), out, strict=True):
            if o is not None:
                items.append({"path": r.path, "crop": r.crop, "label": r.label, "rgba": o[0]})
                backgrounds.append(o[1])
        np.savez(cache, items=np.array(items, dtype=object), backgrounds=np.array(backgrounds, dtype=object))
    by_crop = {c: [i for i, it in enumerate(items) if it["crop"] == c] for c in ("turmeric", "blackgram")}
    return {"items": items, "by_crop": by_crop, "backgrounds": backgrounds,
            "index": {it["path"]: i for i, it in enumerate(items)}}


def procedural_background(rng: random.Random, side: int) -> np.ndarray:
    kind = rng.choice(["soil", "foliage", "plain"])
    nprng = np.random.default_rng(rng.randrange(1 << 30))
    if kind == "plain":
        return np.full((side, side, 3), [rng.randrange(256) for _ in range(3)], np.uint8)
    if kind == "soil":
        base = np.array([rng.randint(70, 140), rng.randint(50, 100), rng.randint(30, 70)], float)
        noise = cv2.GaussianBlur(nprng.normal(0, 25, (side, side)), (0, 0), rng.uniform(1, 4))
        return np.clip(base + noise[..., None], 0, 255).astype(np.uint8)
    img = np.full((side, side, 3), [rng.randint(20, 60), rng.randint(50, 110), rng.randint(20, 50)], np.uint8)
    for _ in range(rng.randint(20, 60)):
        c = (rng.randrange(side), rng.randrange(side))
        ax = (rng.randint(side // 20, side // 5), rng.randint(side // 40, side // 10))
        col = (rng.randint(30, 140), rng.randint(80, 200), rng.randint(20, 90))
        cv2.ellipse(img, c, ax, rng.uniform(0, 180), 0, 360, col, -1)
    return cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 3))


def paste(canvas: np.ndarray, rgba: np.ndarray, scale: float, angle: float, cx: float, cy: float) -> None:
    side = canvas.shape[0]
    target = max(8, int(scale * side))
    s = target / max(rgba.shape[:2])
    leaf = cv2.resize(rgba, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    h, w = leaf.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    leaf = cv2.warpAffine(leaf, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    x0, y0 = int(cx * side - nw / 2), int(cy * side - nh / 2)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + nw, side), min(y0 + nh, side)
    if xa >= xb or ya >= yb:
        return
    patch = leaf[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
    al = patch[..., 3:4] / 255.0
    region = canvas[ya:yb, xa:xb].astype(np.float32)
    canvas[ya:yb, xa:xb] = (al * patch[..., :3] + (1 - al) * region).astype(np.uint8)


class BackgroundSwap:
    """Callable PIL->PIL transform: composite the sample's own leaf onto a neutral background."""

    def __init__(self, banks: dict, cfg: dict, seed: int = 0):
        self.banks, self.a, self.rng = banks, cfg["augment"], random.Random(seed)

    def __call__(self, leaf_rgba: np.ndarray, crop: str) -> Image.Image:
        a, rng = self.a, self.rng
        side = a["canvas_side"]
        r = rng.random()
        if r < a["p_bank_background"] and self.banks["backgrounds"]:
            bg = cv2.resize(rng.choice(self.banks["backgrounds"]), (side, side), interpolation=cv2.INTER_LINEAR)
            if rng.random() < 0.3:
                bg = cv2.GaussianBlur(bg, (0, 0), rng.uniform(2, 6))
        else:
            bg = procedural_background(rng, side)
        canvas = bg.copy()
        if rng.random() < a["p_collage"] and len(self.banks["by_crop"][crop]) > 1:
            for _ in range(rng.randint(1, 3)):  # 1-3 extra leaves of the SAME crop, behind the target leaf
                other = self.banks["items"][rng.choice(self.banks["by_crop"][crop])]["rgba"]
                paste(canvas, other, rng.uniform(0.35, 0.8), rng.uniform(0, 360), rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9))
        paste(canvas, leaf_rgba, rng.uniform(*a["leaf_scale"]), rng.uniform(0, 360), rng.uniform(0.35, 0.65), rng.uniform(0.35, 0.65))
        return Image.fromarray(canvas)


class JpegRecompress:
    def __init__(self, qmin: int, qmax: int, p: float):
        self.qmin, self.qmax, self.p = qmin, qmax, p

    def __call__(self, im: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return im
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=random.randint(self.qmin, self.qmax))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


class RandomBlur:
    def __init__(self, p: float, max_radius: float):
        self.p, self.r = p, max_radius

    def __call__(self, im: Image.Image) -> Image.Image:
        return im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, self.r))) if random.random() < self.p else im


def robust_transform(cfg: dict) -> T.Compose:
    a, c = cfg["augment"], cfg["cnn"]
    return T.Compose([
        T.RandomResizedCrop(c["image_size"], scale=tuple(a["crop_scale"]), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(a["rotation"]),
        T.ColorJitter(a["brightness"], a["contrast"], a["saturation"], a["hue"]),
        RandomBlur(a["p_blur"], a["blur_max"]),
        JpegRecompress(a["jpeg_q"][0], a["jpeg_q"][1], a["p_jpeg"]),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=a["p_erase"], scale=(0.02, 0.12)),
    ])


class SwapDataset:
    """TRAIN dataset: with probability p_swap the sample's own leaf is composited onto a new
    background (optionally with same-crop companions), otherwise the original photo is
    used; then the robust transform. `target_fn(row)` returns the training target."""

    def __init__(self, train_df: pd.DataFrame, banks: dict, cfg: dict, target_fn, seed: int = 0):
        from leafdx.cnn.dataset import _load

        if "split" in train_df and (train_df["split"] != "train").any():
            raise LeakageError("SwapDataset is for TRAIN rows only")
        self.df = train_df.reset_index(drop=True)
        self.a, self.banks, self.target_fn = cfg["augment"], banks, target_fn
        self.swap = BackgroundSwap(banks, cfg, seed)
        self.tf = robust_transform(cfg)
        self.images = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_load)(p, cfg["cnn"]["cache_side"]) for p in self.df.path)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.df)

    def sample_image(self, i: int) -> Image.Image:
        r = self.df.iloc[i]
        j = self.banks["index"].get(r.path)
        if j is not None and self.rng.random() < self.a["p_swap"]:
            return self.swap(self.banks["items"][j]["rgba"], r.crop)
        return self.images[i]

    def __getitem__(self, i: int):
        return self.tf(self.sample_image(i)), self.target_fn(self.df.iloc[i])
