"""P2-F01: export everything the website shows -> web/public/data/site.json + web/public/media/.

No number on the site is typed by hand: every value below is read from a results file or
computed here by the project's own code. Media are REAL pipeline outputs.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from leafdx.cnn.dataset import eval_transform  # noqa: E402
from leafdx.cnn.gradcam import gradcam, overlay_cam  # noqa: E402
from leafdx.config import ROOT, load_config, path  # noqa: E402
from leafdx.ip.features import stage1  # noqa: E402
from leafdx.ip.lesions import overlay_lesions  # noqa: E402
from leafdx.ip.segment import background_distance, border_ring, clean_mask, plant_colour_mask  # noqa: E402
from leafdx.robust.cnn_heads import crop_index  # noqa: E402
from leafdx.robust.infer import RobustModel  # noqa: E402

WEB = ROOT / "web/public"
MEDIA = WEB / "media"
HERO_TITLE = "File:Starr-190610-6608-Curcuma longa-leaves with morning dew-Hawea Pl Olinda-Maui (48301962546).jpg"
UA = {"User-Agent": "leafdx-research/0.1 (university research prototype)"}


def save_img(rgb_or_bgr: np.ndarray, stem: str, widths=(640, 1280), bgr: bool = True) -> dict:
    """WebP + AVIF at the given widths; returns {'webp': {w: url}, 'avif': {...}, 'w','h'}."""
    arr = cv2.cvtColor(rgb_or_bgr, cv2.COLOR_BGR2RGB) if (bgr and rgb_or_bgr.ndim == 3) else rgb_or_bgr
    im = Image.fromarray(arr)
    out = {"webp": {}, "avif": {}, "w": im.width, "h": im.height}
    f = MEDIA / stem
    f.parent.mkdir(parents=True, exist_ok=True)
    for w in widths:
        r = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
        for fmt, q in (("webp", 84), ("avif", 58)):
            name = f"{stem}-{w}.{fmt}"
            r.save(MEDIA / name, format=fmt.upper(), quality=q)
            out[fmt][w] = f"/media/{name}"
    return out


def colourise(gray: np.ndarray) -> np.ndarray:
    """Single-channel map -> dark-to-leaf-green ramp (BGR) for display."""
    g = np.clip(gray.astype(np.float32), 0, 1)[..., None]
    lo, hi = np.array([11, 14, 10], np.float32), np.array([107, 184, 141], np.float32)  # BGR of ink -> leaf
    return (lo + g * (hi - lo)).astype(np.uint8)


def pick_showcase(cfg: dict) -> dict:
    f = pd.read_csv(path("features", cfg))
    out = {}
    for crop, cls in (("turmeric", "leaf_blotch"), ("blackgram", "anthracnose")):
        c = f[(f.split == "test") & (f.crop == crop) & (f.orig_class == cls) & (f.bg_mode == "plain")
              & f.affected_pct.between(8, 40) & (f.lesion_count >= 4) & f.seg_coverage.between(0.2, 0.6)]
        out[crop] = c.sort_values("path").iloc[0].path
    return out


def stage1_steps(p: str, crop: str, cfg: dict, rm: RobustModel) -> dict:
    bgr = cv2.imread(str(ROOT / p))
    s1 = stage1(bgr, cfg, crop)
    pre, seg, les, feats = s1["pre"], s1["seg"], s1["lesions"], s1["features"]
    s = cfg["segment"]
    # intermediates of the plain-background route, recomputed exactly as segment_leaf does
    colour = clean_mask(plant_colour_mask(pre["hsv"], s), s["open_ksize"], 0)
    ring = border_ring(colour.shape, s["ring_frac"])
    bg_ring = ring & (colour == 0) & (pre["hsv"][..., 2] >= s["bg_val_min"])
    dist = background_distance(pre["lab"], pre["lab"][bg_ring] if bg_ring.any() else pre["lab"][ring], s, cfg["seed"])
    raw = seg["raw_mask"]
    cleaned = clean_mask(raw, s["plain_open_ksize"], s["close_ksize"])
    a_star = pre["lab"][..., 1].astype(np.float32)
    a_norm = 1 - (a_star - a_star.min()) / max(np.ptp(a_star), 1)  # green = bright
    key = f"showcase/{crop}"
    media = {
        "original": save_img(pre["resized"], f"{key}/original"),
        "denoised": save_img(pre["filtered"], f"{key}/denoised"),
        "colour_space": save_img(colourise(a_norm), f"{key}/lab_a"),
        "segmentation": save_img(colourise(np.clip(dist / max(np.percentile(dist, 99), 1), 0, 1)), f"{key}/distance"),
        "threshold": save_img(colourise(raw.astype(np.float32)), f"{key}/threshold"),
        "morphology": save_img(colourise(seg["leaf_mask"].astype(np.float32) * 0.85 + cleaned * 0.15), f"{key}/leaf_mask"),
        "lesions": save_img(overlay_lesions(pre["resized"], seg["leaf_mask"], les), f"{key}/lesions"),
        "segmented": save_img(seg["segmented"], f"{key}/segmented"),
    }
    params = {
        "denoise": {"median_ksize": cfg["preprocess"]["median_ksize"], "gaussian_ksize": cfg["preprocess"]["gaussian_ksize"],
                    "clahe_clip": cfg["preprocess"]["clahe_clip"]},
        "colour_space": {"a_star_min": float(a_star.min() - 128), "a_star_max": float(a_star.max() - 128)},
        "segmentation": {"mode": seg["method"], "border_background_share": seg["plain_bg_frac"]},
        "threshold": {"otsu_distance": None if np.isnan(seg["dist_threshold"]) else seg["dist_threshold"]},
        "morphology": {"open_ksize": s["plain_open_ksize"], "close_ksize": s["close_ksize"], "leaf_coverage": seg["coverage"]},
        "lesions": {"count": int(feats["lesion_count"]), "affected_pct": feats["affected_pct"],
                    "min_area_px": max(cfg["lesions"]["default"]["min_lesion_px"],
                                       int(cfg["lesions"]["default"]["min_lesion_frac"] * les["leaf_area"]))},
    }
    readouts = {k: feats[k] for k in ("affected_pct", "lesion_count", "glcm_contrast", "glcm_energy", "glcm_homogeneity",
                                      "lesion_circularity_mean")}
    readouts["abn_colour_pct"] = 100.0 * (1.0 - feats["green_frac"])
    readouts["class_pct"] = les["class_pct"]
    # Stage 2 on the same leaf: feature maps of the disease CNN, Grad-CAM, calibrated probabilities
    im = Image.open(ROOT / p).convert("RGB")
    pred = rm.predict(im, None)
    model = rm.disease_model
    x = eval_transform(cfg["cnn"])(im).unsqueeze(0)
    acts = {}
    hooks = [model.features[i].register_forward_hook(lambda m, inp, o, k=k: acts.__setitem__(k, o.detach()))
             for k, i in (("early", 3), ("late", 13))]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    tiles = {}
    for k, n in (("early", 16), ("late", 9)):
        a = acts[k][0].numpy()
        order = np.argsort(-a.reshape(a.shape[0], -1).var(1))[:n]
        tiles[k] = []
        for j, ch in enumerate(order):
            m = a[ch]
            m = (m - m.min()) / max(np.ptp(m), 1e-6)
            big = cv2.resize(m, (160, 160), interpolation=cv2.INTER_CUBIC)
            tiles[k].append(save_img(colourise(big), f"cnn/{crop}/{k}-{j:02d}", widths=(160,))["webp"][160])
        tiles[f"{k}_shape"] = list(acts[k].shape[1:])
    cam, _ = gradcam(model, x, target=lambda lg: lg[crop_index(pred["crop"]["label"])])
    media["gradcam"] = save_img(overlay_cam(np.array(im.resize((448, 448))), cv2.resize(cam, (448, 448))), f"{key}/gradcam",
                                widths=(448,))
    media["cnn_input"] = save_img(cv2.cvtColor(np.array(im.resize((448, 448))), cv2.COLOR_RGB2BGR), f"{key}/cnn_input",
                                  widths=(448,))
    return {"source": p, "crop": crop, "label": "diseased", "media": media, "params": params, "readouts": readouts,
            "cnn": {"tiles": tiles, "p_crop": pred["crop"]["probs"], "crop_status": pred["crop"]["status"],
                    "p_diseased_given_crop": pred["disease"]["prob"], "crop_pred": pred["crop"]["label"]}}


def hero(cfg: dict) -> dict:
    src_dir = ROOT / "web/assets-src/hero"
    src_dir.mkdir(parents=True, exist_ok=True)
    f = src_dir / "starr_6608.jpg"
    meta_f = src_dir / "starr_6608.json"
    if not f.exists():
        r = requests.get("https://commons.wikimedia.org/w/api.php", headers=UA, timeout=60, params={
            "action": "query", "format": "json", "titles": HERO_TITLE, "prop": "imageinfo",
            "iiprop": "url|extmetadata", "iiurlwidth": 2400}).json()
        ii = next(iter(r["query"]["pages"].values()))["imageinfo"][0]
        f.write_bytes(requests.get(ii["thumburl"], headers=UA, timeout=120).content)
        m = ii["extmetadata"]
        meta_f.write_text(json.dumps({"title": HERO_TITLE, "url": ii["descriptionurl"],
                                      "licence": m["LicenseShortName"]["value"],
                                      "author": "Forest and Kim Starr"}, indent=1))
    meta = json.loads(meta_f.read_text())
    bgr = cv2.imread(str(f))
    out = {"credit": meta}
    out["image"] = save_img(bgr, "hero/leaf", widths=(960, 1600, 2400))
    # Stage 1 on the hero photo: leaf outline + principal axis (midrib direction) for the callouts
    s1 = stage1(bgr, cfg, "turmeric")
    leaf = s1["seg"]["leaf_mask"].astype(np.uint8)
    h, w = leaf.shape
    cnts, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True)[:, 0, :]
    out["outline"] = [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in c]
    ys, xs = np.nonzero(leaf)
    cen = np.array([xs.mean(), ys.mean()])
    cov = np.cov(np.stack([xs, ys]))
    v = np.linalg.eigh(cov)[1][:, -1]
    perp = np.array([-v[1], v[0]])
    spread = np.sqrt(np.linalg.eigvalsh(cov))
    rel = lambda p: [round(float(p[0]) / w, 4), round(float(p[1]) / h, 4)]  # noqa: E731
    out["anchors"] = {"midrib": rel(cen), "venation": rel(cen + perp * spread[0] * 0.9),
                      "outline": rel(c[np.argmin(c[:, 0])])}
    out["stage1"] = {"coverage": s1["seg"]["coverage"], "lesion_count": int(s1["features"]["lesion_count"]),
                     "affected_pct": s1["features"]["affected_pct"], "method": s1["seg"]["method"]}
    return out


def crop_panels(cfg: dict) -> dict:
    """One healthy and one diseased TEST photo per crop (real dataset images) for the Crops section."""
    f = pd.read_csv(path("features", cfg))
    out = {}
    picks = {("turmeric", "healthy"): "healthy_leaf", ("turmeric", "diseased"): "leaf_blotch",
             ("blackgram", "healthy"): "healthy", ("blackgram", "diseased"): "yellow_mosaic"}
    for (crop, lab), cls in picks.items():
        c = f[(f.split == "test") & (f.crop == crop) & (f.orig_class == cls) & (f.bg_mode == "field" if crop == "blackgram"
                                                                                 else f.bg_mode == "plain")]
        if lab == "diseased":
            c = c[c.affected_pct.between(10, 45)]
        p = c.sort_values("path").iloc[0].path
        out.setdefault(crop, {})[lab] = {"img": save_img(cv2.imread(str(ROOT / p)), f"crops/{crop}-{lab}", widths=(480, 960)),
                                         "source": p, "orig_class": cls}
    return out


def metrics_block(m: dict) -> dict:
    return {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "macro_f1", "healthy_recall", "roc_auc") if k in m}


def main() -> int:
    cfg = load_config()
    res = path("results", cfg)
    comp = pd.read_csv(res / "comparison.csv")
    rob = json.loads((res / "robustness.json").read_text())
    cal = json.loads((res / "calibration.json").read_text())
    s = pd.read_csv(path("splits", cfg))
    sm = json.loads((path("processed", cfg) / "split_meta.json").read_text())

    def get(section, method, scope, metric):
        v = comp[(comp.section == section) & (comp.method == method) & (comp.scope == scope) & (comp.metric == metric)].value
        return float(v.iloc[0]) if len(v) else None

    methods = ("rule_based", "svm", "cnn")
    clean = {m: {k: get("metrics", m, "overall", k) for k in ("accuracy", "precision", "recall", "f1", "macro_f1",
                                                            "healthy_recall", "roc_auc", "n")} for m in methods}
    clean_ms = {"rule_based": get("inference", "rule_based", "overall", "ms_per_image"),
                "svm": get("inference", "svm", "overall", "ms_per_image"),
                "cnn": get("inference", "cnn_cpu", "overall", "ms_per_image")}
    wild = {m: metrics_block(rob["comparison_wild_test"][m]) | {"n": rob["comparison_wild_test"][m]["n"]} for m in methods}
    leak = {sc: {mm: {"accuracy": get("leakage_ablation", mm, sc, "accuracy"), "macro_f1": get("leakage_ablation", mm, sc, "macro_f1"),
                      "dup_share": get("leakage_ablation", mm, sc, "test_items_with_cluster_in_train")}
                 for mm in ("svm", "cnn_head")} for sc in ("clean", "naive_random", "augment_then_split")}
    rm = RobustModel.load()
    showcase = {crop: stage1_steps(p, crop, cfg, rm) for crop, p in pick_showcase(cfg).items()}
    heal = {v: json.loads((MEDIA / f"heal/heal_{v}.json").read_text()) for v in ("21x9", "9x16")}
    site = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "total": int(len(s)),
            "by_crop_label": {f"{c}_{lab}": int(n) for (c, lab), n in s.groupby(["crop", "label"]).size().items()},
            "by_class": {f"{c}/{o}": int(n) for (c, o), n in s.groupby(["crop", "orig_class"]).size().items()},
            "splits": {k: int(v) for k, v in s.split.value_counts().items()},
            "clusters": sm["n_clusters"], "multi_image_clusters": sm["n_multi_image_clusters"],
            "wild_test": rob["n"]["wild_test"], "wild_test_labelled": rob["n"]["wild_test_labelled"],
        },
        "comparison": {"clean": clean, "clean_ms": clean_ms, "wild": wild,
                       "mcnemar_p": get("mcnemar", "svm_vs_cnn", "overall", "p_value"),
                       "test_n": clean["cnn"]["n"]},
        "leakage": leak,
        "robustness": {"before": rob["before"], "after": rob["after"], "targets": rob["targets"]},
        "calibration": {"tau_crop": cal["tau_crop"], "ood_threshold": cal["ood"]["threshold"]},
        "showcase": showcase,
        "heal": heal,
        "crops": crop_panels(cfg),
        "hero": hero(cfg),
        "example_card": rob["regression"]["after"]["card"] if rob["regression"]["after"] else None,
    }
    (WEB / "data").mkdir(parents=True, exist_ok=True)
    (WEB / "data/site.json").write_text(json.dumps(site, indent=1, default=float))
    n_media = sum(1 for _ in MEDIA.rglob("*") if _.is_file())
    print(f"wrote web/public/data/site.json ({len(json.dumps(site, default=float)) // 1024} kB); media files: {n_media}")
    print("showcase:", {k: v["source"] for k, v in showcase.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
