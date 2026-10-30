"""Analysis service shared by the FastAPI app and Streamlit.

analyze_image(data, crop=None, roi=None) -> dict with hierarchical output:
  crop    {label, confidence, probs, status: confident|uncertain|user_selected, source}
  disease {label, confidence, by: {cnn, svm, rule}}
  measurements, images (RGB arrays), ood {is_plant, ood_score, flagged}, timings_ms
When the Phase 3 bundle exists (models/manifest.json) the decision comes from it and the
result also carries crop_set, disease_set, abstain, severity, per-arm predictions and a
plain-language explanation; otherwise the Phase 2 hierarchical model is used.
Raises InvalidImageError / NotALeafError for bad input.
"""

from __future__ import annotations

import hashlib
import io
import os
import time
import uuid
from functools import lru_cache

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from leafdx.cnn.dataset import eval_transform  # noqa: E402
from leafdx.cnn.gradcam import gradcam, overlay_cam  # noqa: E402
from leafdx.config import load_config, path  # noqa: E402
from leafdx.ip.decide import load_ip_models, predict_ip  # noqa: E402
from leafdx.ip.features import stage1  # noqa: E402
from leafdx.ip.lesions import overlay_lesions  # noqa: E402
from leafdx.ip.segment import clean_mask, plant_colour_mask  # noqa: E402
from leafdx.p3 import production as p3  # noqa: E402
from leafdx.pipeline import InvalidImageError, NotALeafError  # noqa: E402
from leafdx.robust.cnn_heads import crop_index  # noqa: E402
from leafdx.robust.infer import RobustModel, sigmoid  # noqa: E402

LABEL = {0: "healthy", 1: "diseased"}


def decode(data: bytes, cfg: dict) -> Image.Image:
    """Decode bytes -> RGB PIL with EXIF orientation applied; validate size."""
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        im = im.convert("RGB")
    except Exception as e:  # PIL raises many types for broken files
        raise InvalidImageError("could not decode the image (corrupt or unsupported format)") from e
    if min(im.size) < cfg["pipeline"]["min_side"]:
        raise InvalidImageError(f"image is too small ({im.size[0]}x{im.size[1]}); need at least "
                                f"{cfg['pipeline']['min_side']} px per side")
    return im


def plant_fraction(rgb: np.ndarray, cfg: dict) -> float:
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (128, max(1, round(128 * h / w))))
    hsv = cv2.cvtColor(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    return float(clean_mask(plant_colour_mask(hsv, cfg["segment"]), 3, 0).mean())


@lru_cache(maxsize=1)
def model_version() -> str:
    """Short hash of the model files, so responses say which models produced them."""
    cfg = load_config()
    h = hashlib.sha1()
    for f in ("crop_model.joblib", "disease_cnn.pt", "calibration.json", "ip_svm.joblib", "ip_rule.joblib", "manifest.json"):
        p = path("models", cfg) / f
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:10]


def disease_gradcam(model, im: Image.Image, crop: str, cfg: dict) -> np.ndarray:
    x = eval_transform(cfg["cnn"])(im).unsqueeze(0)
    k = crop_index(crop)
    cam, _ = gradcam(model, x, target=lambda lg: lg[k])
    size = cfg["cnn"]["image_size"]
    return cv2.cvtColor(overlay_cam(np.array(im.resize((size, size))), cam), cv2.COLOR_BGR2RGB)


def analyze_image(data: bytes, crop: str | None = None, roi: dict | None = None, with_gradcam: bool = True,
                  arms: str = "all") -> dict:
    cfg = load_config()
    t = {}
    t0 = time.perf_counter()
    im = decode(data, cfg)
    rgb = np.array(im)
    pf = plant_fraction(rgb, cfg)
    if pf < cfg["pipeline"]["min_plant_fraction"]:
        raise NotALeafError(f"only {pf:.0%} of the image has leaf-like colour; please upload a close-up photo of a "
                            "turmeric or black gram leaf")
    t["decode"] = 1000 * (time.perf_counter() - t0)
    if p3.available():
        return _analyze_p3(cfg, im, rgb, pf, crop, roi, with_gradcam, arms, t)

    t0 = time.perf_counter()
    rm = RobustModel.load()
    pred = rm.predict(im, crop if crop in ("turmeric", "blackgram") else None)
    t["cnn"] = 1000 * (time.perf_counter() - t0)
    ood = pred["ood"]
    is_plant = not (ood["flagged"] and pf < cfg["pipeline"]["ood_reject_plant_fraction"])
    if not is_plant:
        raise NotALeafError("this does not look like a leaf photo (unusual image and little leaf-coloured area)")

    c = pred["crop"]["label"]
    t0 = time.perf_counter()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    s1 = stage1(bgr, cfg, c, roi=roi)
    t["stage1"] = 1000 * (time.perf_counter() - t0)
    t0 = time.perf_counter()
    rule, svm = load_ip_models(path("models", cfg))
    ip = predict_ip(s1["features"], c, rule, svm)
    platt = svm.get("platt")
    svm_prob = float(sigmoid(platt[0] * ip["svm_score"] + platt[1])) if platt else float(sigmoid(ip["svm_score"]))
    t["ip_decision"] = 1000 * (time.perf_counter() - t0)

    t0 = time.perf_counter()
    cam = disease_gradcam(rm.disease_model, im, c, cfg) if with_gradcam else None
    t["gradcam"] = 1000 * (time.perf_counter() - t0)

    d = pred["disease"]
    f = s1["features"]
    les = s1["lesions"]
    areas = [x["area"] for x in les["lesions"]]
    pre, seg = s1["pre"], s1["seg"]
    out = {
        "request_id": uuid.uuid4().hex[:12],
        "model_version": model_version(),
        "crop": pred["crop"],
        "disease": {
            "label": d["label"], "confidence": d["confidence"],
            "by": {
                "cnn": {"label": d["label"], "prob": d["prob"]},
                "svm": {"label": LABEL[ip["svm_pred"]], "prob": svm_prob},
                "rule": {"label": LABEL[ip["rule_pred"]], "score": ip["rule_score"], "threshold": ip["rule_threshold"],
                         "components": ip["rule_components"]},
            },
        },
        "agreement": LABEL[ip["svm_pred"]] == d["label"],
        "measurements": {
            "affected_region_pct": f["affected_pct"], "lesion_count": int(f["lesion_count"]),
            "abn_colour_pct": 100.0 * (1.0 - f["green_frac"]), "glcm_contrast": f["glcm_contrast"],
            "glcm_energy": f["glcm_energy"], "glcm_homogeneity": f["glcm_homogeneity"],
            "mean_lesion_area_px": float(np.mean(areas)) if areas else 0.0,
            "segmentation_mode": seg["method"],
        },
        "abnormal_colour_pct": les["class_pct"],
        "ood": {"is_plant": is_plant, "ood_score": ood["ood_score"], "flagged": ood["flagged"]},
        "images": {
            "original": cv2.cvtColor(pre["resized"], cv2.COLOR_BGR2RGB),
            "segmented_leaf": cv2.cvtColor(seg["segmented"], cv2.COLOR_BGR2RGB),
            "abnormal_regions": cv2.cvtColor(overlay_lesions(pre["resized"], seg["leaf_mask"], les), cv2.COLOR_BGR2RGB),
            "gradcam": cam,
        },
        "features": f,
    }
    t["total"] = sum(t.values())
    out["timings_ms"] = t
    out["trace"] = run_trace(cfg, im, pf, crop, roi, s1, ip, svm_prob, pred, t)
    out["result_card"] = result_card(out)
    return out


def _analyze_p3(cfg: dict, im: Image.Image, rgb: np.ndarray, pf: float, crop: str | None, roi: dict | None,
                with_gradcam: bool, arms: str, t: dict) -> dict:
    """Phase 3 decision (models/manifest.json), returned in the v1 shape plus the v3 fields."""
    import pandas as pd

    from leafdx.p3.cnnpred import load_arm
    from leafdx.p3.explain import gradcam_pp

    m = p3.load()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    user = crop if crop in ("turmeric", "blackgram") else None
    t0 = time.perf_counter()
    r = m.analyze(im, bgr, lambda c: stage1(bgr, cfg, c, roi=roi), user, arms)
    t["model"] = 1000 * (time.perf_counter() - t0) - r["timing_ms"]["stage1"]
    t["stage1"] = r["timing_ms"]["stage1"]
    s1 = r.pop("stage1")
    if r["ood"]["flagged"] and pf < cfg["pipeline"]["ood_reject_plant_fraction"]:
        raise NotALeafError("this does not look like a leaf photo (unusual image and little leaf-coloured area)")
    c = r["crop"]
    a = r["arms"]
    rule = m.arms["A_rule"]["rule"]
    row = pd.DataFrame([{**s1["features"], "crop": c}])
    ip = {"rule_score": float(rule.score(row).iloc[0]), "rule_threshold": float(rule.threshold[c]),
          "rule_pred": int(a["A_rule"]["label"] == "diseased"),
          "rule_components": {k: float(v) for k, v in rule.components(row).iloc[0].items()},
          "svm_score": a["B_svm"]["z"], "svm_pred": int(a["B_svm"]["label"] == "diseased")}
    status = ("user_selected" if user else
              "uncertain" if (len(r["crop_set"]) != 1 or r["ood"]["flagged"]) else "confident")
    probs = r["crop_probs"]
    pred = {"crop": {"label": c, "confidence": probs[c], "probs": probs, "status": status,
                     "source": "user" if user else "model"},
            "disease": {"label": r["disease"], "prob": r["disease_prob"], "confidence": r["disease_confidence"]},
            "ood": {"ood_score": r["ood"]["distance"], "threshold": r["ood"]["threshold"], "flagged": r["ood"]["flagged"]}}
    t0 = time.perf_counter()
    cam = None
    if with_gradcam:
        cb = m.arms["C1_mobilenet_v2"]
        cam = gradcam_pp(load_arm(cb["checkpoint"], cb["arch"], "cpu"), im, c, cfg, s1)
    t["gradcam"] = 1000 * (time.perf_counter() - t0)
    t["ip_decision"] = a["A_rule"]["ms"] + a["B_svm"]["ms"]
    t["cnn"] = t["model"]
    f, les, pre, seg = s1["features"], s1["lesions"], s1["pre"], s1["seg"]
    areas = [x["area"] for x in les["lesions"]]
    out = {
        "request_id": uuid.uuid4().hex[:12],
        "model_version": model_version(),
        "crop": pred["crop"],
        "disease": {"label": r["disease"], "confidence": r["disease_confidence"],
                    "by": {"cnn": {"label": a["C1_mobilenet_v2"]["label"], "prob": a["C1_mobilenet_v2"]["prob"]},
                           "svm": {"label": a["B_svm"]["label"], "prob": a["B_svm"]["prob"]},
                           "rule": {"label": LABEL[ip["rule_pred"]], "score": ip["rule_score"],
                                    "threshold": ip["rule_threshold"], "components": ip["rule_components"]}}},
        "agreement": a["B_svm"]["label"] == r["disease"],
        "measurements": {
            "affected_region_pct": f["affected_pct"], "lesion_count": int(f["lesion_count"]),
            "abn_colour_pct": 100.0 * (1.0 - f["green_frac"]), "glcm_contrast": f["glcm_contrast"],
            "glcm_energy": f["glcm_energy"], "glcm_homogeneity": f["glcm_homogeneity"],
            "mean_lesion_area_px": float(np.mean(areas)) if areas else 0.0, "segmentation_mode": seg["method"]},
        "abnormal_colour_pct": les["class_pct"],
        "ood": {"is_plant": True, "ood_score": r["ood"]["distance"], "flagged": r["ood"]["flagged"]},
        "images": {
            "original": cv2.cvtColor(pre["resized"], cv2.COLOR_BGR2RGB),
            "segmented_leaf": cv2.cvtColor(seg["segmented"], cv2.COLOR_BGR2RGB),
            "abnormal_regions": cv2.cvtColor(overlay_lesions(pre["resized"], seg["leaf_mask"], les), cv2.COLOR_BGR2RGB),
            "gradcam": cam},
        "features": f,
        # v3 additions
        "crop_confidence": r["crop_confidence"], "crop_set": r["crop_set"],
        "disease_confidence": r["disease_confidence"], "disease_set": r["disease_set"], "abstain": r["abstain"],
        "affected_pct": r["affected_pct"], "severity": r["severity"], "severity_note": r["severity_note"],
        "arms": {k: {kk: v[kk] for kk in ("name", "label", "prob", "config", "ms")} for k, v in a.items()},
        "production_model": r["production_model"], "production_model_name": r["production_model_name"],
        "explanation": r["reason"], "segmenter": r["segmenter"], "quality": r["quality"],
    }
    t["total"] = sum(v for k, v in t.items() if k not in ("total", "cnn"))  # "cnn" aliases "model" for the v1 UI
    out["timings_ms"] = t
    out["trace"] = run_trace(cfg, im, pf, crop, roi, s1, ip, a["B_svm"]["prob"], pred, t)
    out["result_card"] = result_card(out)
    return out


def explain_image(data: bytes, crop: str | None = None) -> dict:
    """P3-M08 explanations for one photo (crop from the Phase 3 crop model unless given)."""
    from leafdx.p3.cnnpred import load_arm
    from leafdx.p3.explain import gradcam_pp, vit_patch_map

    cfg = load_config()
    im = decode(data, cfg)
    rgb = np.array(im)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    m = p3.load()
    r = m.analyze(im, bgr, lambda c: stage1(bgr, cfg, c), crop, arms="production")
    s1 = r["stage1"]
    arm = m.man["production"].get("explain_cnn", "C1_mobilenet_v2")
    cb = m.arms[arm]
    return {"crop": r["crop"], "cnn_arm": arm,
            "gradcam_pp": gradcam_pp(load_arm(cb["checkpoint"], cb["arch"], "cpu"), im, r["crop"], cfg, s1),
            "vit_patch_map": vit_patch_map(im, s1),
            "lesion_overlay": cv2.cvtColor(overlay_lesions(s1["pre"]["resized"], s1["seg"]["leaf_mask"], s1["lesions"]),
                                           cv2.COLOR_BGR2RGB),
            "explanation": r["reason"], "affected_pct": r["affected_pct"], "severity": r["severity"]}


def _num(x):
    """JSON-safe number: NaN (e.g. no threshold on a field photo) becomes None."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def run_trace(cfg: dict, im: Image.Image, pf: float, crop: str | None, roi: dict | None, s1: dict, ip: dict,
              svm_prob: float, pred: dict, t: dict) -> dict:
    """What each stage did on THIS photo: the values it used or produced and its time. Stage ids match
    GET /api/code, so the web app can show the code next to the numbers."""
    p, pre, seg, les, f, ms = cfg["preprocess"], s1["pre"], s1["seg"], s1["lesions"], s1["features"], s1["ms_steps"]
    h, w = pre["resized"].shape[:2]
    d, ood = pred["disease"], pred["ood"]
    return {
        "input": {"width": im.size[0], "height": im.size[1], "analysed_width": w, "analysed_height": h,
                  "crop_given": crop if crop in ("turmeric", "blackgram") else "auto",
                  "roi": None if roi is None else {k: (list(v) if isinstance(v, tuple) else v) for k, v in roi.items()},
                  "plant_colour_fraction": pf},
        "stages": [
            {"id": "denoise", "ms": ms["preprocess"],
             "values": {"max_side_px": p["max_side"], "median_ksize": p["median_ksize"], "gaussian_ksize": p["gaussian_ksize"]}},
            {"id": "colour", "ms": None,  # inside preprocess, timed with denoise
             "values": {"clahe_clip": p["clahe_clip"], "clahe_tile": p["clahe_tile"], "spaces": "HSV, Lab"}},
            {"id": "segment", "ms": ms["segment"],
             "values": {"method": seg["method"], "border_background_share": _num(seg["plain_bg_frac"]),
                        "distance_threshold": _num(seg["dist_threshold"]), "leaf_coverage_pct": 100.0 * seg["coverage"],
                        "fallback_whole_image": bool(seg.get("seg_failed", 0))}},
            {"id": "lesions", "ms": ms["lesions"],
             "values": {"leaf_area_px": les["leaf_area"], "min_lesion_px": les["min_area"], "lesion_count": les["lesion_count"],
                        "affected_pct": les["affected_pct"]}},
            {"id": "measure", "ms": ms["features"],
             "values": {"n_features": len(f), "abn_colour_pct": 100.0 * (1.0 - f["green_frac"]),
                        "glcm_contrast": f["glcm_contrast"], "glcm_homogeneity": f["glcm_homogeneity"]}},
            {"id": "ip_decision", "ms": t["ip_decision"],
             "values": {"rule_score": ip["rule_score"], "rule_threshold": ip["rule_threshold"], "rule_label": LABEL[ip["rule_pred"]],
                        "svm_margin": ip["svm_score"], "svm_prob_diseased": svm_prob, "svm_label": LABEL[ip["svm_pred"]]}},
            {"id": "cnn", "ms": t["cnn"],
             "values": {"p_turmeric": pred["crop"]["probs"]["turmeric"], "crop_status": pred["crop"]["status"],
                        "p_diseased": d["prob"], "cnn_label": d["label"], "ood_score": ood["ood_score"],
                        "ood_threshold": ood["threshold"]}},
            {"id": "compare", "ms": None,
             "values": {"svm_label": LABEL[ip["svm_pred"]], "cnn_label": d["label"], "agreement": LABEL[ip["svm_pred"]] == d["label"]}},
        ],
    }


def result_card(res: dict) -> list[str]:
    """Four-line card; an uncertain crop is shown as such, never as a confident answer."""
    c = res["crop"]
    names = {"turmeric": "Turmeric", "blackgram": "Black gram"}
    if c["status"] == "uncertain":
        crop_line = f"Crop: Unsure - Turmeric {c['probs']['turmeric']:.0%} / Black gram {c['probs']['blackgram']:.0%}"
    else:
        crop_line = f"Crop: {names[c['label']]} ({c['confidence']:.0%})" + (" - selected by user" if c["status"] == "user_selected" else "")
    if res.get("abstain", {}).get("flag"):
        return [crop_line, f"Prediction: Uncertain ({res['abstain']['reason']})",
                f"Affected Region: {res['measurements']['affected_region_pct']:.1f}%",
                f"Model leaning: {res['disease']['label']} ({res['disease']['confidence']:.0%})"]
    conf_name = "Model" if res.get("production_model") else "CNN"
    return [crop_line, f"Prediction: {res['disease']['label'].capitalize()}",
            f"Affected Region: {res['measurements']['affected_region_pct']:.1f}%",
            f"{conf_name} Confidence: {res['disease']['confidence']:.0%}"]
