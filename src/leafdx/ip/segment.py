"""Stage 1 leaf segmentation: separate the photographed leaf from its background.

Steps (each returned for display):
  1. colour mask   - HSV thresholds for plant tissue (green, yellow, brown, pale);
                     flags grey cards, blue/white cloth, soil and sky as non-plant.
  2. GrabCut       - the photographed leaf is near the centre; plant pixels near the
                     centre seed the foreground model, the outer border seeds the
                     background, everything else is "probable" and left to GrabCut's
                     colour models + edges. Separates the leaf from other foliage.
  3. morphology    - opening + closing to clean the mask.
  4. largest blob  - keep the connected component closest to the centre, fill holes
                     (lesions inside the leaf belong to the leaf).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def otsu_threshold(values: np.ndarray) -> float:
    v = values.astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def plant_colour_mask(hsv: np.ndarray, s: dict) -> np.ndarray:
    """Binary mask of pixels whose colour is plausible for leaf tissue.

    Hue is the main discriminator: backgrounds in these datasets (blue/white cloth,
    grey card) can be MORE saturated than a dull, dark leaf, so an Otsu split on
    saturation picks the wrong side. Fixed HSV ranges (configurable) are used instead.
    """
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_ok = (h >= s["hue_min"]) & (h <= s["hue_max"])
    coloured = hue_ok & (sat >= s["sat_min"]) & (val >= s["val_min"])
    # pale tissue (powdery mildew, chlorosis) is low-saturation but bright
    pale = hue_ok & (sat >= s["pale_sat_min"]) & (val >= s["pale_val_min"])
    return (coloured | pale).astype(np.uint8)


def centre_prior(shape: tuple[int, int], frac: float) -> np.ndarray:
    """Elliptical mask covering the central `frac` of each dimension."""
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return (((yy - h / 2) / (frac * h / 2)) ** 2 + ((xx - w / 2) / (frac * w / 2)) ** 2) <= 1


def grabcut_leaf(img: np.ndarray, colour: np.ndarray, s: dict, seed: int = 0) -> np.ndarray:
    h, w = colour.shape
    work = s["grabcut_side"] / max(h, w)
    small = cv2.resize(img, (round(w * work), round(h * work)), interpolation=cv2.INTER_AREA)
    col = cv2.resize(colour, small.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    sh = col.shape
    gc = np.full(sh, cv2.GC_PR_BGD, np.uint8)
    inner = centre_prior(sh, s["centre_frac"])
    core = centre_prior(sh, s["core_frac"])
    gc[inner & col] = cv2.GC_PR_FGD
    gc[core & col] = cv2.GC_FGD
    b = max(1, round(s["border_frac"] * min(sh)))
    border = np.zeros(sh, bool)
    border[:b], border[-b:], border[:, :b], border[:, -b:] = True, True, True, True
    gc[border & ~inner] = cv2.GC_BGD
    if not (gc == cv2.GC_FGD).any() and not (gc == cv2.GC_PR_FGD).any():
        return np.zeros((h, w), np.uint8)
    cv2.setRNGSeed(seed)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(small, gc, None, bgd, fgd, s["grabcut_iters"], cv2.GC_INIT_WITH_MASK)
    fg = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)


def clean_mask(mask: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    m = mask.astype(np.uint8)
    if open_k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    if close_k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    return m


def central_component(mask: np.ndarray) -> np.ndarray:
    """Keep the component with the best size/centrality score, then fill its holes."""
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, np.uint8)
    h, w = mask.shape
    diag = np.hypot(h, w)
    best, best_score = 0, -1.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA] / (h * w)
        dist = np.hypot(cents[i][0] - w / 2, cents[i][1] - h / 2) / diag
        score = area * (1.0 - dist)
        if score > best_score:
            best, best_score = i, score
    return ndi.binary_fill_holes(lab == best).astype(np.uint8)


def segment_leaf(pre: dict[str, np.ndarray], cfg: dict) -> dict:
    """Leaf mask plus intermediates. `pre` is the output of preprocess()."""
    s = cfg["segment"]
    colour = plant_colour_mask(pre["hsv"], s)
    colour = clean_mask(colour, s["open_ksize"], 0)
    gc = grabcut_leaf(pre["filtered"], colour, s, seed=cfg["seed"])
    cleaned = clean_mask(gc, s["open_ksize"], s["close_ksize"])
    leaf = central_component(cleaned)
    method = "grabcut"
    # GrabCut occasionally collapses onto the definite-foreground core (e.g. a pale,
    # mildew-covered leaf on cloth). Fall back to the cleaned colour mask in that case.
    if leaf.mean() < s["fallback_min_coverage"]:
        alt = central_component(clean_mask(colour, s["open_ksize"], s["close_ksize"]))
        if alt.mean() > leaf.mean():
            leaf, method = alt, "colour_fallback"
    leaf_rgb = pre["filtered"].copy()
    leaf_rgb[leaf == 0] = 0
    return {
        "colour_mask": colour,
        "grabcut_mask": gc,
        "leaf_mask": leaf,
        "segmented": leaf_rgb,
        "coverage": float(leaf.mean()),
        "method": method,
        "n_components_before": int(cv2.connectedComponents(cleaned.astype(np.uint8))[0] - 1),
    }
