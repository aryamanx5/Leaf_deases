"""Image quality gate (P3-M05): blur, exposure and leaf area."""

from __future__ import annotations

import cv2
import numpy as np


def quality_stats(bgr: np.ndarray, leaf_mask: np.ndarray | None = None, max_side: int = 512) -> dict:
    """Blur on the whole frame; dark / clipped-bright shares on the LEAF pixels (Stage 1 mask) when a mask is
    given -- a white studio background is not over-exposure."""
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = np.ones(g.shape, bool)
    if leaf_mask is not None and leaf_mask.any():
        m = cv2.resize(leaf_mask.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return {"blur_var_laplacian": float(cv2.Laplacian(g, cv2.CV_64F).var()),
            "dark_frac": float((g[m] < 25).mean()), "bright_frac": float((g[m] > 245).mean())}


def quality_verdict(st: dict, leaf_frac: float, q: dict) -> tuple[bool, str | None]:
    if st["blur_var_laplacian"] < q["blur_min"]:
        return False, "image too blurry"
    if st["dark_frac"] > q["dark_max"]:
        return False, "image too dark"
    if st["bright_frac"] > q["bright_max"]:
        return False, "image over-exposed"
    if leaf_frac < q["min_leaf_frac"]:
        return False, "too little leaf area found"
    return True, None
