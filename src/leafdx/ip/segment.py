"""Stage 1 leaf segmentation: separate the photographed leaf from its background.

Two modes, chosen per image from the outer border ring:

* plain background (paper, grey card, cloth: most of the border is NOT plant-coloured)
    1. colour mask   - HSV thresholds for plant tissue -> which border pixels are background
    2. background model - k-means (k<=3) on the background border pixels in Lab
    3. threshold     - per-pixel colour distance to the nearest background centre
                       (L down-weighted so shadows stay background), Otsu threshold
* field background (other foliage around the leaf)
    1. colour mask   - as above
    2. GrabCut       - plant pixels near the centre seed the leaf model, the border
                       seeds the background, the rest is left to GrabCut
Both then:
    4. morphology    - opening + closing to clean the mask
    5. component     - keep the dominant component (size x centrality), fill holes
                       (lesions inside the leaf belong to the leaf)
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def otsu_threshold(values: np.ndarray) -> float:
    v = np.clip(values, 0, 255).astype(np.uint8).reshape(-1, 1)
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
    # pale tissue (powdery mildew, chlorosis): low saturation, bright, yellow-green hue
    pale = (h >= s["pale_hue_min"]) & (h <= s["pale_hue_max"]) & (sat >= s["pale_sat_min"]) & (val >= s["pale_val_min"])
    return (coloured | pale).astype(np.uint8)


def border_ring(shape: tuple[int, int], frac: float) -> np.ndarray:
    h, w = shape
    b = max(1, round(frac * min(h, w)))
    ring = np.zeros((h, w), bool)
    ring[:b], ring[-b:], ring[:, :b], ring[:, -b:] = True, True, True, True
    return ring


def centre_prior(shape: tuple[int, int], frac: float) -> np.ndarray:
    """Elliptical mask covering the central `frac` of each dimension."""
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return (((yy - h / 2) / (frac * h / 2)) ** 2 + ((xx - w / 2) / (frac * w / 2)) ** 2) <= 1


def background_distance(lab: np.ndarray, bg_pixels: np.ndarray, s: dict, seed: int) -> np.ndarray:
    """Distance of every pixel to the nearest of k background colour centres (Lab)."""
    w = np.array([s["l_weight"], 1.0, 1.0], np.float32)
    data = bg_pixels.astype(np.float32) * w
    rng = np.random.default_rng(seed)
    if len(data) > 4000:
        data = data[rng.choice(len(data), 4000, replace=False)]
    k = int(min(s["bg_clusters"], max(1, len(data) // 50)))
    cv2.setRNGSeed(seed)
    _, labels, centres = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
                                    3, cv2.KMEANS_PP_CENTERS)
    # drop tiny clusters: usually bits of leaf that reached the border, not background
    share = np.bincount(labels.ravel(), minlength=k) / len(labels)
    centres = centres[share >= s["bg_min_cluster_share"]]
    px = lab.reshape(-1, 3).astype(np.float32) * w
    d = np.sqrt(((px[:, None, :] - centres[None, :, :]) ** 2).sum(-1)).min(1)
    return d.reshape(lab.shape[:2])


def threshold_distance(dist: np.ndarray, s: dict) -> tuple[np.ndarray, float]:
    thr = float(np.clip(otsu_threshold(dist), s["dist_min"], s["dist_max"]))
    return (dist > thr).astype(np.uint8), thr


def grabcut_leaf(img: np.ndarray, colour: np.ndarray, s: dict, seed: int = 0,
                 core_seeding: bool = False) -> np.ndarray:
    """GrabCut seeded by a centre prior: plant-coloured pixels in the central ellipse
    start as probable leaf, the border as background. With `core_seeding`, only pixels
    whose Lab colour is close to the leaf core (Otsu split of the distance) start as
    probable leaf, so neighbouring foliage of a different shade starts as background."""
    h, w = colour.shape
    work = s["grabcut_side"] / max(h, w)
    small = cv2.resize(img, (round(w * work), round(h * work)), interpolation=cv2.INTER_AREA)
    col = cv2.resize(colour, small.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    sh = col.shape
    gc = np.full(sh, cv2.GC_PR_BGD, np.uint8)
    inner = centre_prior(sh, s["centre_frac"])
    core = centre_prior(sh, s["core_frac"])
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    wts = np.array([s["l_weight"], 1.0, 1.0], np.float32)
    seed_px = lab[core & col] if (core & col).any() else lab[core]
    d_core = np.sqrt((((lab - np.median(seed_px, axis=0)) * wts) ** 2).sum(-1))
    similar = d_core <= max(otsu_threshold(d_core[inner]), s["core_dist_min"])
    gc[inner & col & (similar if core_seeding else True)] = cv2.GC_PR_FGD
    gc[core & col] = cv2.GC_FGD
    gc[border_ring(sh, s["border_frac"]) & ~inner] = cv2.GC_BGD
    if not np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).any():
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


def dominant_component(mask: np.ndarray, centre_weight: float) -> np.ndarray:
    """Keep the component maximising area * (1 - centre_weight * distance-from-centre); fill holes."""
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, np.uint8)
    h, w = mask.shape
    diag = np.hypot(h, w) / 2
    area = stats[1:, cv2.CC_STAT_AREA] / (h * w)
    dist = np.hypot(cents[1:, 0] - w / 2, cents[1:, 1] - h / 2) / diag
    best = 1 + int(np.argmax(area * (1.0 - centre_weight * dist)))
    return ndi.binary_fill_holes(lab == best).astype(np.uint8)


def solidity(mask: np.ndarray) -> float:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    hull = cv2.convexHull(np.vstack(cnts))
    return float(mask.sum() / max(cv2.contourArea(hull), 1.0))


def _finish(raw: np.ndarray, open_k: int, s: dict, centre_weight: float) -> tuple[np.ndarray, np.ndarray]:
    cleaned = clean_mask(raw, open_k, s["close_ksize"])
    return cleaned, dominant_component(cleaned, centre_weight)


def segment_leaf(pre: dict[str, np.ndarray], cfg: dict) -> dict:
    """Leaf mask plus intermediates. `pre` is the output of preprocess()."""
    s = cfg["segment"]
    colour = clean_mask(plant_colour_mask(pre["hsv"], s), s["open_ksize"], 0)
    ring = border_ring(colour.shape, s["ring_frac"])
    # background candidates: non-plant colour AND reasonably bright (paper, card, cloth);
    # dark non-plant pixels at the border are usually necrotic leaf, not background
    bg_ring = ring & (colour == 0) & (pre["hsv"][..., 2] >= s["bg_val_min"])
    plain_frac = float(bg_ring.sum() / ring.sum())

    if plain_frac >= s["plain_bg_min_frac"]:
        method = "plain_bg_threshold"
        dist = background_distance(pre["lab"], pre["lab"][bg_ring], s, cfg["seed"])
        raw, thr = threshold_distance(dist, s)
        centre_weight = s["plain_centre_weight"]
        cleaned, leaf = _finish(raw, s["plain_open_ksize"], s, centre_weight)
    else:
        # Field shot: run GrabCut with loose and with core-colour seeding; keep the
        # more leaf-like (solid) result. Loose seeding suits two-tone diseased leaves,
        # core seeding rejects adjacent foliage of a different shade.
        thr = float("nan")
        centre_weight = s["field_centre_weight"]
        best = None
        for core_seeding in (False, True):
            r = grabcut_leaf(pre["filtered"], colour, s, seed=cfg["seed"], core_seeding=core_seeding)
            c, lf = _finish(r, s["open_ksize"], s, centre_weight)
            score = solidity(lf) if lf.mean() >= s["fallback_min_coverage"] else 0.0
            if best is None or score > best[0] + s["solidity_margin"]:
                best = (score, r, c, lf, "field_grabcut" + ("_core" if core_seeding else ""))
        _, raw, cleaned, leaf, method = best

    # GrabCut occasionally collapses onto the definite-foreground core. Fall back to
    # the cleaned colour mask in that case.
    if leaf.mean() < s["fallback_min_coverage"]:
        alt = dominant_component(clean_mask(colour, s["open_ksize"], s["close_ksize"]), centre_weight)
        if leaf.mean() < alt.mean() <= s["fallback_max_coverage"]:
            leaf, method = alt, method + "+colour_fallback"
    leaf_rgb = pre["filtered"].copy()
    leaf_rgb[leaf == 0] = 0
    n_comp = cv2.connectedComponents(cleaned)[0] - 1
    fg = int(cleaned.sum())
    dominant_share = float(min(1.0, (leaf & cleaned).sum() / fg)) if fg else 0.0
    return {
        "colour_mask": colour,
        "raw_mask": raw,
        "leaf_mask": leaf,
        "segmented": leaf_rgb,
        "coverage": float(leaf.mean()),
        "method": method,
        "plain_bg_frac": plain_frac,
        "dist_threshold": thr,
        "n_components_before": int(n_comp),
        "dominant_share": dominant_share,  # share of cleaned foreground in the kept component
    }


def is_single_dominant(seg: dict, cfg: dict) -> bool:
    """T03 quality criterion: one component holds most of the foreground and the leaf
    covers a plausible share of the frame."""
    q = cfg["segment"]["quality"]
    return (seg["dominant_share"] >= q["min_dominant_share"]
            and q["min_coverage"] <= seg["coverage"] <= q["max_coverage"])
