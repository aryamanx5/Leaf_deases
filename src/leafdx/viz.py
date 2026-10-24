"""Figure helpers: contact sheets and mask overlays."""

from __future__ import annotations

import cv2
import numpy as np


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = img.astype(np.float32)
        if img.max() <= 1:
            img = img * 255
        return cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return img


def tile(img: np.ndarray, size: int) -> np.ndarray:
    img = to_bgr(img)
    h, w = img.shape[:2]
    s = size / max(h, w)
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    y, x = (size - small.shape[0]) // 2, (size - small.shape[1]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    return canvas


def label(img: np.ndarray, text: str, scale: float = 0.45) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (255, 255, 255), -1)
    cv2.putText(out, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def contact_sheet(rows: list[list[np.ndarray]], titles: list[list[str]] | None = None,
                  size: int = 200, gap: int = 4) -> np.ndarray:
    """Grid of tiles; each inner list is one row."""
    out_rows = []
    for r, row in enumerate(rows):
        tiles = []
        for c, img in enumerate(row):
            t = tile(img, size)
            if titles:
                t = label(t, titles[r][c])
            tiles.append(t)
            tiles.append(np.full((size, gap, 3), 255, np.uint8))
        out_rows.append(np.hstack(tiles[:-1]))
        out_rows.append(np.full((gap, out_rows[-1].shape[1], 3), 255, np.uint8))
    return np.vstack(out_rows[:-1])


def overlay_mask(img: np.ndarray, mask: np.ndarray, colour=(0, 0, 255), alpha: float = 0.5,
                 outline: np.ndarray | None = None) -> np.ndarray:
    """Tint `mask` pixels; optionally draw the outline of another mask (e.g. the leaf) in yellow."""
    out = img.copy()
    m = mask.astype(bool)
    out[m] = (out[m] * (1 - alpha) + np.array(colour) * alpha).astype(np.uint8)
    if outline is not None:
        cnts, _ = cv2.findContours(outline.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 255), 2)
    return out
