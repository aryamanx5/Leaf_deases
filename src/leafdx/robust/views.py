"""Test-time views: full image, 4 corner crops, horizontal flip; plus 3x3 plant tiles for
field scenes (many leaves / foliage background)."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageOps

from leafdx.ip.segment import border_ring, clean_mask, plant_colour_mask


def is_field_scene(rgb: np.ndarray, cfg: dict) -> bool:
    """Same rule as Stage 1: few bright non-plant pixels on the border -> field scene."""
    s = cfg["segment"]
    small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    colour = clean_mask(plant_colour_mask(hsv, s), s["open_ksize"], 0)
    ring = border_ring(colour.shape, s["ring_frac"])
    plain = (ring & (colour == 0) & (hsv[..., 2] >= s["bg_val_min"])).sum() / ring.sum()
    return bool(plain < s["plain_bg_min_frac"])


def plant_tiles(rgb: np.ndarray, cfg: dict, grid: int = 3, min_plant: float = 0.5) -> list[Image.Image]:
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    plant = plant_colour_mask(hsv, cfg["segment"])
    out = []
    for i in range(grid):
        for j in range(grid):
            ys, ye, xs, xe = i * h // grid, (i + 1) * h // grid, j * w // grid, (j + 1) * w // grid
            if plant[ys:ye, xs:xe].mean() >= min_plant:
                out.append(Image.fromarray(rgb[ys:ye, xs:xe]))
    return out


def tta_views(im: Image.Image, cfg: dict, tiles: bool = True) -> list[Image.Image]:
    rgb = np.array(im.convert("RGB"))
    h, w = rgb.shape[:2]
    ch, cw = int(0.8 * h), int(0.8 * w)
    views = [im, ImageOps.mirror(im)]
    for y, x in ((0, 0), (0, w - cw), (h - ch, 0), (h - ch, w - cw)):
        views.append(Image.fromarray(rgb[y:y + ch, x:x + cw]))
    if tiles and is_field_scene(rgb, cfg):
        views += plant_tiles(rgb, cfg)
    return views
