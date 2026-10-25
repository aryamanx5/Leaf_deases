"""Stage 1 feature extraction: a fixed-order vector of colour, texture and shape features.

Degenerate cases (documented in results/feature_list.txt):
  * segmentation failed (leaf < `min_leaf_frac` of the frame): the whole frame is used as
    the leaf and `seg_failed` = 1.
  * no lesions: lesion colour means are set to the leaf means and lesion stds to 0
    ("lesions look like the leaf"); lesion shape features are 0.
  * constant texture: GLCM correlation is undefined -> 0.
Every value is finite by construction; a final nan_to_num guards against surprises.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from leafdx.ip.lesions import CLASSES, detect_lesions
from leafdx.ip.preprocess import preprocess
from leafdx.ip.segment import segment_leaf, solidity

CHANNELS = ("R", "G", "B", "H", "S", "V", "L", "a", "b")
GLCM_PROPS = ("contrast", "energy", "homogeneity", "correlation", "dissimilarity")

FEATURE_NAMES: list[str] = (
    [f"leaf_{c}_{s}" for c in CHANNELS for s in ("mean", "std")]
    + [f"lesion_{c}_{s}" for c in CHANNELS for s in ("mean", "std")]
    + [f"abn_{c}_pct" for c in CLASSES]
    + ["affected_pct", "green_frac", "ref_hue"]
    + [f"glcm_{p}" for p in GLCM_PROPS]
    + ["edge_density", "lesion_count", "lesion_total_area_pct", "lesion_total_perimeter_rel",
       "lesion_circularity_mean", "lesion_mean_size_pct", "lesion_max_size_pct",
       "leaf_solidity", "seg_failed"]
)


def _channels(pre: dict) -> np.ndarray:
    """H x W x 9 float array: R, G, B, H (red wrapped to negative), S, V, L, a*, b*."""
    bgr = pre["filtered"].astype(np.float32)
    hsv = pre["hsv"].astype(np.float32)
    lab = pre["lab"].astype(np.float32)
    hue = hsv[..., 0].copy()
    hue[hue >= 150] -= 180  # red/purple hues sit next to orange instead of at the far end
    return np.dstack([bgr[..., 2], bgr[..., 1], bgr[..., 0], hue, hsv[..., 1], hsv[..., 2],
                      lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128])


def _stats(ch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    px = ch[mask.astype(bool)]
    return np.column_stack([px.mean(0), px.std(0)]).ravel()


def glcm_features(gray: np.ndarray, mask: np.ndarray, f: dict) -> np.ndarray:
    """GLCM props over the masked region only: leaf pixels are quantised to levels
    1..L-1, background is level 0 and its row/column is dropped before normalising."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 16:
        return np.zeros(len(GLCM_PROPS))
    g = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(bool)
    s = f["glcm_max_side"] / max(g.shape)
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    levels = f["glcm_levels"]
    q = (g.astype(np.float32) / 256 * (levels - 1)).astype(np.uint8) + 1
    q[~m] = 0
    angles = [np.deg2rad(a) for a in f["glcm_angles"]]
    P = graycomatrix(q, f["glcm_distances"], angles, levels=levels + 1, symmetric=True, normed=False)
    P = P[1:, 1:].astype(np.float64)
    tot = P.sum(axis=(0, 1), keepdims=True)
    P = np.divide(P, tot, out=np.zeros_like(P), where=tot > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = [np.nan_to_num(graycoprops(P, prop), nan=0.0).mean() for prop in GLCM_PROPS]
    return np.array(vals)


def extract_features(pre: dict, seg: dict, les: dict, cfg: dict) -> dict[str, float]:
    f = cfg["features"]
    leaf = seg["leaf_mask"].astype(np.uint8)
    ch = _channels(pre)
    leaf_stats = _stats(ch, leaf)
    lesion = les["lesion_mask"].astype(np.uint8)
    if lesion.sum() > 0:
        lesion_stats = _stats(ch, lesion)
    else:
        lesion_stats = leaf_stats.copy()
        lesion_stats[1::2] = 0.0
    gray = cv2.cvtColor(pre["enhanced"], cv2.COLOR_BGR2GRAY)
    glcm = glcm_features(gray, leaf, f)
    edges = cv2.Canny(gray, f["canny_low"], f["canny_high"])
    inner = cv2.erode(leaf, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    edge_density = float((edges[inner == 1] > 0).mean()) if inner.any() else 0.0

    leaf_area = max(int(leaf.sum()), 1)
    L = les["lesions"]
    areas = np.array([x["area"] for x in L], float)
    perims = np.array([x["perimeter"] for x in L], float)
    shape = [
        float(len(L)),
        100.0 * areas.sum() / leaf_area if len(L) else 0.0,
        perims.sum() / np.sqrt(leaf_area) if len(L) else 0.0,  # scale-free total boundary
        float(np.mean([x["circularity"] for x in L])) if len(L) else 0.0,
        100.0 * areas.mean() / leaf_area if len(L) else 0.0,
        100.0 * areas.max() / leaf_area if len(L) else 0.0,
        solidity(leaf),
        float(seg.get("seg_failed", 0)),
    ]
    vec = np.concatenate([
        leaf_stats, lesion_stats,
        [les["class_pct"][c] for c in CLASSES],
        [les["affected_pct"], les["reference"]["green_frac"], les["reference"]["h"]],
        glcm, [edge_density], shape,
    ])
    vec = np.nan_to_num(vec.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    assert len(vec) == len(FEATURE_NAMES)
    return dict(zip(FEATURE_NAMES, vec.tolist(), strict=True))


def stage1(img: np.ndarray, cfg: dict, crop: str | None = None, roi: dict | None = None) -> dict:
    """Full Stage 1 on one BGR image: pre-processing, segmentation, lesions, features.
    roi: optional user click {"x", "y"} or {"box": (x, y, w, h)}, fractions of the image."""
    t0 = time.perf_counter()
    pre = preprocess(img, cfg)
    seg = segment_leaf(pre, cfg, roi=roi)
    if seg["coverage"] < cfg["features"]["min_leaf_frac"]:
        seg["leaf_mask"] = np.ones_like(seg["leaf_mask"])
        seg["seg_failed"] = 1
    les = detect_lesions(pre, seg["leaf_mask"], cfg, crop)
    feats = extract_features(pre, seg, les, cfg)
    return {"pre": pre, "seg": seg, "lesions": les, "features": feats,
            "ms": 1000.0 * (time.perf_counter() - t0)}
