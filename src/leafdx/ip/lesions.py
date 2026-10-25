"""Stage 1 disease-region detection inside the segmented leaf.

1. Reference green: median HSV/Lab of the leaf's own green pixels (hue in the crop's
   green range); falls back to the crop default when too few green pixels exist.
2. Abnormal-colour classes (per-crop thresholds in config), relative to that reference:
     yellow  - hue below the green range (chlorosis, yellow mosaic, blotch halo)
     brown   - orange/brown/red-purple hue (necrosis, anthracnose spots, dry tissue)
     dark    - much darker than the leaf with a non-green hue (necrotic centres)
     pale    - low chroma and brighter than the leaf (powdery mildew)
2b. Small specks (P2-B06): black / white top-hat on L* at two scales (1x, 0.5x) finds
   small dark / bright spots that the colour rules alone miss; dark specks must be
   non-green, bright specks low-chroma. They join the brown/dark and pale classes.
3. Hysteresis: the class masks are strong seeds; a relaxed mask (Lab colour distance
   from the reference green above max(Otsu, floor)) is kept where it connects to a
   seed, so lesions extend to their soft edges without creating new regions.
4. Morphology: opening removes speckle, closing joins fragments of one lesion.
5. Connected components -> lesion list (area, perimeter, circularity, centroid),
   components below a minimum area are dropped.
"""

from __future__ import annotations

import cv2
import numpy as np

from leafdx.ip.segment import otsu_threshold

CLASSES = ("yellow", "brown", "dark", "pale")


def crop_params(cfg: dict, crop: str | None) -> dict:
    """Merge the default lesion parameters with per-crop overrides."""
    lc = cfg["lesions"]
    p = dict(lc["default"])
    if crop and crop in lc.get("per_crop", {}):
        p.update(lc["per_crop"][crop])
    return p


def reference_colour(hsv: np.ndarray, lab: np.ndarray, leaf: np.ndarray, p: dict) -> dict:
    m = leaf.astype(bool)
    h = hsv[..., 0]
    green = m & (h >= p["green_hue_min"]) & (h <= p["green_hue_max"]) & (hsv[..., 1] >= p["green_sat_min"])
    used_default = green.sum() < p["min_green_px"]
    if used_default:
        green = m
    if not green.any():
        return {"h": float(p["default_green_hue"]), "s": 120.0, "v": 130.0, "L": 130.0, "a": 108.0,
                "b": 150.0, "C": 40.0, "green_frac": 0.0, "used_default": True}
    ab = lab[..., 1:].astype(np.float32) - 128
    chroma = np.hypot(ab[..., 0], ab[..., 1])
    ref_h = float(np.median(h[green])) if not used_default else float(p["default_green_hue"])
    return {
        "h": ref_h,
        "s": float(np.median(hsv[..., 1][green])),
        "v": float(np.median(hsv[..., 2][green])),
        "L": float(np.median(lab[..., 0][green])),
        "a": float(np.median(lab[..., 1][green])),
        "b": float(np.median(lab[..., 2][green])),
        "C": float(np.median(chroma[green])),
        "green_frac": float(green.sum() / max(m.sum(), 1)),
        "used_default": bool(used_default),
    }


def abnormal_masks(hsv: np.ndarray, lab: np.ndarray, leaf: np.ndarray, ref: dict, p: dict) -> dict:
    """Raw (pre-morphology) masks of each abnormal-colour class, restricted to the leaf."""
    m = leaf.astype(bool)
    h = hsv[..., 0].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    L = lab[..., 0].astype(np.float32)
    ab = lab[..., 1:].astype(np.float32) - 128
    chroma = np.hypot(ab[..., 0], ab[..., 1])

    yellow_max = min(ref["h"] - p["yellow_hue_gap"], p["yellow_hue_max"])
    brown = m & (((h < p["brown_hue_max"]) | (h >= p["red_hue_min"])) & (s >= p["brown_sat_min"]))
    yellow = m & (h >= p["brown_hue_max"]) & (h < yellow_max) & (s >= p["yellow_sat_min"]) & (v >= p["yellow_val_min"])
    dark_thr = max(p["dark_val_ratio"] * ref["v"], p["dark_val_abs"])
    dark = m & (v < dark_thr) & ((h < yellow_max) | (h >= p["red_hue_min"]) | (s < p["dark_sat_max"]))
    pale = m & (chroma < p["pale_chroma_ratio"] * ref["C"]) & (L > ref["L"] + p["pale_l_gain"]) & (L < p["pale_l_max"])
    return {"yellow": yellow, "brown": brown & ~dark, "dark": dark, "pale": pale & ~(yellow | brown | dark)}


def speck_masks(pre: dict, inner: np.ndarray, ref: dict, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Dark and bright specks from multi-scale top-hats of L*, gated by colour. Uses a lightly
    smoothed copy of the resized image: the median filter of pre-processing erases specks
    only a few pixels wide."""
    light = cv2.GaussianBlur(pre["resized"], (3, 3), 0)
    lab, hsv = cv2.cvtColor(light, cv2.COLOR_BGR2LAB), cv2.cvtColor(light, cv2.COLOR_BGR2HSV)
    L = lab[..., 0]
    h, w = L.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["speck_ksize"], p["speck_ksize"]))
    dark = np.zeros((h, w), bool)
    bright = np.zeros((h, w), bool)
    for sc in p["speck_scales"]:
        Ls = L if sc == 1 else cv2.resize(L, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
        bh = cv2.morphologyEx(Ls, cv2.MORPH_BLACKHAT, k)
        th = cv2.morphologyEx(Ls, cv2.MORPH_TOPHAT, k)
        if sc != 1:
            bh = cv2.resize(bh, (w, h), interpolation=cv2.INTER_LINEAR)
            th = cv2.resize(th, (w, h), interpolation=cv2.INTER_LINEAR)
        dark |= bh > p["speck_dark_delta"]
        bright |= th > p["speck_bright_delta"]
    hue, sat = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32)
    ab = lab[..., 1:].astype(np.float32) - 128
    chroma = np.hypot(ab[..., 0], ab[..., 1])
    non_green = (hue < ref["h"] - p["yellow_hue_gap"]) | (hue >= p["red_hue_min"]) | (sat < p["dark_sat_max"])
    inside = inner.astype(bool)
    return dark & non_green & inside, bright & (chroma < p["pale_chroma_ratio"] * ref["C"]) & inside


def colour_distance(lab: np.ndarray, ref: dict, l_weight: float) -> np.ndarray:
    """Lab distance of every pixel from the leaf's reference green (L down-weighted)."""
    d = lab.astype(np.float32) - np.array([ref["L"], ref["a"], ref["b"]], np.float32)
    d[..., 0] *= l_weight
    return np.sqrt((d**2).sum(-1))


def hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Components of (weak | strong) that contain at least one strong pixel."""
    both = (weak | strong).astype(np.uint8)
    n, lab = cv2.connectedComponents(both, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong.astype(bool)])] = True
    keep[0] = False
    return keep[lab]


def clean_lesion_mask(mask: np.ndarray, leaf: np.ndarray, p: dict) -> np.ndarray:
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["open_ksize"], p["open_ksize"]))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["close_ksize"], p["close_ksize"]))
    out = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k_open)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close)
    return out & leaf.astype(np.uint8)


def connected_lesions(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, list[dict]]:
    """Filter components below `min_area`; return the filtered mask and a lesion list."""
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, np.uint8)
    lesions = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (lab == i).astype(np.uint8)
        keep |= comp
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perim = float(sum(cv2.arcLength(c, True) for c in cnts))
        lesions.append({
            "area": area,
            "perimeter": perim,
            "circularity": float(4 * np.pi * area / perim**2) if perim > 0 else 0.0,
            "centroid": (float(cents[i][0]), float(cents[i][1])),
        })
    return keep, lesions


def _assign_classes(raw: dict, mask: np.ndarray) -> dict:
    """Split the final lesion mask into classes; unlabelled (grown) pixels get the class
    of the nearest seed pixel."""
    seeds = np.zeros(mask.shape, np.int32)
    for i, name in enumerate(CLASSES, start=1):
        seeds[raw[name].astype(bool) & (seeds == 0)] = i
    seeds[mask == 0] = 0
    if not seeds.any():
        return {name: np.zeros_like(mask) for name in CLASSES}
    _, idx = _nearest(seeds == 0)
    filled = seeds[idx[0], idx[1]]
    return {name: ((filled == i) & (mask == 1)).astype(np.uint8) for i, name in enumerate(CLASSES, start=1)}


def _nearest(background: np.ndarray):
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(background, return_indices=True)


def detect_lesions(pre: dict, leaf: np.ndarray, cfg: dict, crop: str | None = None) -> dict:
    """Lesion mask, per-class masks, lesion list and summary for one segmented leaf."""
    p = crop_params(cfg, crop)
    leaf = leaf.astype(np.uint8)
    leaf_area = int(leaf.sum())
    ref = reference_colour(pre["hsv"], pre["lab"], leaf, p)
    raw = abnormal_masks(pre["hsv"], pre["lab"], leaf, ref, p)
    # erode the leaf a little so its outline (soft edge, shadow) is not read as lesion
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["edge_margin"] * 2 + 1,) * 2)
    inner = cv2.erode(leaf, k)
    sd, sb = speck_masks(pre, inner, ref, p)
    hue = pre["hsv"][..., 0]
    brownish = (hue < p["brown_hue_max"]) | (hue >= p["red_hue_min"])
    raw["brown"] = raw["brown"] | (sd & brownish & ~raw["dark"])
    raw["dark"] = raw["dark"] | (sd & ~brownish)
    raw["pale"] = raw["pale"] | (sb & ~(raw["yellow"] | raw["brown"] | raw["dark"]))
    union = np.zeros_like(leaf)
    for name in CLASSES:
        union |= raw[name].astype(np.uint8)
    dist = colour_distance(pre["lab"], ref, p["hyst_l_weight"])
    inside = inner.astype(bool)
    weak_thr = max(otsu_threshold(dist[inside]) if inside.any() else 0.0, p["hyst_dist_min"])
    grown = hysteresis(union.astype(bool) & inside, (dist > weak_thr) & inside)
    cleaned = clean_lesion_mask(grown, leaf, p)
    min_area = max(p["min_lesion_px"], int(p["min_lesion_frac"] * leaf_area))  # adaptive: max(8 px, 0.0002 x leaf)
    mask, lesions = connected_lesions(cleaned, min_area)
    # pixels added by hysteresis take the class of the nearest seed class pixel
    per_class = _assign_classes(raw, mask)
    affected = float(mask.sum() / leaf_area) if leaf_area else 0.0
    return {
        "lesion_mask": mask,
        "class_masks": per_class,
        "lesions": lesions,
        "reference": ref,
        "affected_pct": 100.0 * affected,
        "class_pct": {n: 100.0 * float(per_class[n].sum() / leaf_area) if leaf_area else 0.0 for n in CLASSES},
        "lesion_count": len(lesions),
        "leaf_area": leaf_area,
    }


def overlay_lesions(img: np.ndarray, leaf: np.ndarray, res: dict) -> np.ndarray:
    """Original image with lesion classes tinted and the leaf outline drawn."""
    colours = {"yellow": (0, 255, 255), "brown": (0, 80, 200), "dark": (255, 0, 255), "pale": (255, 160, 0)}
    out = img.copy()
    for name, col in colours.items():
        m = res["class_masks"][name].astype(bool)
        out[m] = (0.45 * out[m] + 0.55 * np.array(col)).astype(np.uint8)
    cnts, _ = cv2.findContours(res["lesion_mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 0, 255), 1)
    cnts, _ = cv2.findContours(leaf.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 255, 0), 2)
    return out
