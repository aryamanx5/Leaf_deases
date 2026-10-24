"""Stage 1 pre-processing: resize, noise reduction, contrast enhancement, colour spaces.

Every function returns plain arrays so each step can be shown on its own.
Images are BGR uint8 (OpenCV convention) unless stated otherwise.
"""

from __future__ import annotations

import cv2
import numpy as np


def load_bgr(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return img


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img.copy()
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def denoise(img: np.ndarray, median_ksize: int, gaussian_ksize: int) -> np.ndarray:
    """Median filter (removes salt/pepper and specular speckle) then a light Gaussian."""
    out = cv2.medianBlur(img, median_ksize) if median_ksize > 1 else img
    if gaussian_ksize > 1:
        out = cv2.GaussianBlur(out, (gaussian_ksize, gaussian_ksize), 0)
    return out


def clahe_l(img: np.ndarray, clip: float, tile: int) -> np.ndarray:
    """CLAHE on the L channel of Lab, leaving colour (a*, b*) untouched."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess(img: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """Run the pre-processing chain; returns every intermediate plus HSV/Lab views.

    `filtered` is used for colour measurements (CLAHE would distort them);
    `enhanced` (CLAHE) is used for texture and edge-sensitive steps.
    """
    p = cfg["preprocess"]
    resized = resize_max_side(img, p["max_side"])
    filtered = denoise(resized, p["median_ksize"], p["gaussian_ksize"])
    enhanced = clahe_l(filtered, p["clahe_clip"], p["clahe_tile"])
    return {
        "resized": resized,
        "filtered": filtered,
        "enhanced": enhanced,
        "hsv": cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV),
        "lab": cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB),
    }
