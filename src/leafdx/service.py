"""Phase 2 analysis service shared by the FastAPI app and Streamlit.

analyze_image(data, crop=None, roi=None) -> dict with hierarchical output:
  crop    {label, confidence, probs, status: confident|uncertain|user_selected, source}
  disease {label, confidence, by: {cnn, svm, rule}}
  measurements, images (RGB arrays), ood {is_plant, ood_score, flagged}, timings_ms
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
    for f in ("crop_model.joblib", "disease_cnn.pt", "calibration.json", "ip_svm.joblib", "ip_rule.joblib"):
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


def analyze_image(data: bytes, crop: str | None = None, roi: dict | None = None, with_gradcam: bool = True) -> dict:
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
    return [crop_line, f"Prediction: {res['disease']['label'].capitalize()}",
            f"Affected Region: {res['measurements']['affected_region_pct']:.1f}%",
            f"CNN Confidence: {res['disease']['confidence']:.0%}"]
