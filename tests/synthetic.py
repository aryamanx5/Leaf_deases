"""SYNTHETIC test images. For unit tests only; never used for any reported result.

make_leaf() draws a leaf-shaped green region with veins and mild noise on a chosen
background, optionally with painted lesions, and returns the image together with
the ground-truth leaf and lesion masks.
"""

from __future__ import annotations

import cv2
import numpy as np

BACKGROUNDS = ("paper", "grey", "cloth", "field")
LESION_COLOURS = {  # BGR
    "brown": (35, 75, 125),
    "yellow": (40, 205, 225),
    "dark": (25, 35, 45),
}


def _leaf_polygon(size: int, rng: np.random.Generator, scale: float = 0.36) -> np.ndarray:
    """Pointed-ellipse leaf outline, randomly rotated, centred."""
    t = np.linspace(0, 2 * np.pi, 200)
    a, b = scale * size, scale * size * rng.uniform(0.45, 0.6)
    x = a * np.cos(t)
    y = b * np.sin(t) * (1 - 0.35 * np.cos(t))  # slightly egg-shaped
    ang = rng.uniform(0, np.pi)
    xr, yr = x * np.cos(ang) - y * np.sin(ang), x * np.sin(ang) + y * np.cos(ang)
    return np.stack([xr + size / 2, yr + size / 2], 1).astype(np.int32)


def _background(kind: str, size: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "paper":
        img = np.full((size, size, 3), (238, 236, 228), np.uint8)
        for y in range(20, size, 40):  # ruled lines
            cv2.line(img, (0, y), (size, y), (215, 170, 150), 1)
    elif kind == "grey":
        img = np.full((size, size, 3), (170, 172, 175), np.uint8)
    elif kind == "cloth":
        img = np.full((size, size, 3), (215, 175, 150), np.uint8)
    elif kind == "field":  # darker, bluer foliage than the subject leaf + soil gaps
        img = np.full((size, size, 3), (40, 70, 60), np.uint8)
        for _ in range(60):
            c = tuple(int(v) for v in rng.integers(0, size, 2))
            axes = tuple(int(v) for v in rng.integers(10, 40, 2))
            col = tuple(int(v) for v in (rng.integers(30, 70), rng.integers(70, 110), rng.integers(30, 60)))
            cv2.ellipse(img, c, axes, float(rng.uniform(0, 180)), 0, 360, col, -1)
    else:
        raise ValueError(kind)
    noise = rng.normal(0, 3, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def make_leaf(background: str = "paper", size: int = 400, lesions: list[dict] | None = None,
              seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (BGR image, leaf mask, lesion mask). Lesion masks are clipped to the leaf.

    lesions: list of {"centre": (x, y) as fraction of the leaf bbox, "radius": px, "colour": name}
    """
    rng = np.random.default_rng(seed)
    img = _background(background, size, rng)
    poly = _leaf_polygon(size, rng)
    leaf = np.zeros((size, size), np.uint8)
    cv2.fillPoly(leaf, [poly], 1)

    green = np.array((55, 160, 75), np.float32) + rng.normal(0, 6, (size, size, 3))
    leaf_img = green.clip(0, 255).astype(np.uint8)
    # veins: slightly lighter lines along the main axis
    p0, p1 = poly[0], poly[100]
    cv2.line(leaf_img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), (95, 190, 120), 2)
    img[leaf == 1] = leaf_img[leaf == 1]

    lesion = np.zeros((size, size), np.uint8)
    x0, y0, w, h = cv2.boundingRect(poly)
    for les in lesions or []:
        cx, cy = x0 + les["centre"][0] * w, y0 + les["centre"][1] * h
        m = np.zeros_like(lesion)
        cv2.circle(m, (int(cx), int(cy)), int(les["radius"]), 1, -1)
        m &= leaf
        colour = np.array(LESION_COLOURS[les["colour"]], np.float32)
        patch = (colour + rng.normal(0, 4, (size, size, 3))).clip(0, 255).astype(np.uint8)
        img[m == 1] = patch[m == 1]
        lesion |= m
    return img, leaf, lesion


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0
