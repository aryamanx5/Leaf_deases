"""Unified inference: one leaf image -> Stage 1 (image processing) + Stage 2 (CNN).

    analyze(image, crop=None) -> dict with
      crop, crop_source, ip_prediction {rule, svm, ...}, cnn_prediction, cnn_confidence,
      affected_region_pct, lesion_count, agreement, prediction (headline), images
      {original, segmented_leaf, abnormal_regions_overlay, gradcam} (RGB uint8)

Bad input raises InvalidImageError (unreadable / too small) or NotALeafError.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

from leafdx.cnn.dataset import CLASSES4, DISEASED_IDX  # noqa: E402
from leafdx.cnn.gradcam import gradcam_for_image  # noqa: E402
from leafdx.cnn.model import load_checkpoint  # noqa: E402
from leafdx.config import load_config, path  # noqa: E402
from leafdx.ip.decide import load_ip_models, predict_ip  # noqa: E402
from leafdx.ip.features import stage1  # noqa: E402
from leafdx.ip.lesions import overlay_lesions  # noqa: E402
from leafdx.ip.segment import clean_mask, plant_colour_mask  # noqa: E402

CROP_NAMES = {"turmeric": "Turmeric", "blackgram": "Black Gram"}
LABEL = {0: "Healthy", 1: "Diseased"}


class InvalidImageError(ValueError):
    """The input could not be decoded or is too small to analyse."""


class NotALeafError(ValueError):
    """The image does not look like a turmeric / black gram leaf photo."""


def to_bgr(image) -> np.ndarray:
    """Accept a file path, raw bytes, a file-like object, a PIL image or an array.
    NumPy arrays are taken as BGR (OpenCV order); PIL images as RGB."""
    try:
        if isinstance(image, (str, Path)):
            data = Path(image).read_bytes()
            image = data
        if isinstance(image, (bytes, bytearray)):
            arr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                raise InvalidImageError("could not decode the image (corrupt or unsupported format)")
            return arr
        if hasattr(image, "read"):
            return to_bgr(image.read())
        if isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            if image.ndim == 3 and image.shape[2] in (3, 4):
                return image[..., :3].astype(np.uint8)
    except (OSError, UnidentifiedImageError) as e:
        raise InvalidImageError(f"could not read the image: {e}") from e
    raise InvalidImageError(f"unsupported input type {type(image).__name__}")


def validate(bgr: np.ndarray, cfg: dict) -> None:
    p = cfg["pipeline"]
    h, w = bgr.shape[:2]
    if min(h, w) < p["min_side"]:
        raise InvalidImageError(f"image is too small ({w}x{h}); need at least {p['min_side']} px per side")
    small = cv2.resize(bgr, (128, round(128 * h / w)) if w >= h else (round(128 * w / h), 128))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    plant = clean_mask(plant_colour_mask(hsv, cfg["segment"]), 3, 0).mean()
    if plant < p["min_plant_fraction"]:
        raise NotALeafError(
            f"only {plant:.0%} of the image has leaf-like colour (need {p['min_plant_fraction']:.0%}); "
            "please upload a close-up photo of a single turmeric or black gram leaf")


@lru_cache(maxsize=1)
def _models():
    cfg = load_config()
    rule, svm = load_ip_models(path("models", cfg))
    cnn, meta = load_checkpoint(path("models", cfg) / "cnn_best.pt")
    return rule, svm, cnn, meta


@lru_cache(maxsize=1)
def tie_breaker() -> str:
    """Method with the higher measured TEST macro-F1 (from comparison.csv) settles disagreements."""
    f = path("results", load_config()) / "comparison.csv"
    if not f.exists():
        return "cnn"
    t = pd.read_csv(f)
    sel = t[(t.section == "metrics") & (t.scope == "overall") & (t.metric == "macro_f1") & t.method.isin(["svm", "cnn"])]
    return str(sel.sort_values("value").method.iloc[-1]) if len(sel) == 2 else "cnn"


def analyze(image, crop: str | None = None, with_gradcam: bool = True) -> dict:
    cfg = load_config()
    if crop not in (None, "auto", *cfg["crops"]):
        raise ValueError(f"crop must be one of {cfg['crops']} or None/'auto'")
    bgr = to_bgr(image)
    validate(bgr, cfg)
    rule, svm, cnn, _ = _models()

    # Stage 2 first: it also identifies the crop when none is given
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if with_gradcam:
        cam_bgr, probs = gradcam_for_image(cnn, pil, cfg)
    else:
        from leafdx.cnn.dataset import eval_transform  # noqa: PLC0415

        with torch.no_grad():
            probs = torch.softmax(cnn(eval_transform(cfg["cnn"])(pil).unsqueeze(0)), 1)[0].numpy()
        cam_bgr = None
    p_crop = {c: float(sum(probs[i] for i, k in enumerate(CLASSES4) if k.startswith(c))) for c in cfg["crops"]}
    if crop in (None, "auto"):
        crop, crop_source = max(p_crop, key=p_crop.get), "cnn"
    else:
        crop_source = "user"
    p_dis = float(probs[list(DISEASED_IDX)].sum())
    cnn_pred = int(p_dis >= 0.5)

    # Stage 1
    s1 = stage1(bgr, cfg, crop)
    if s1["features"]["seg_failed"]:
        raise NotALeafError("no leaf could be segmented from the image")
    ip = predict_ip(s1["features"], crop, rule, svm)
    agree = ip["svm_pred"] == cnn_pred
    final = cnn_pred if (agree or tie_breaker() == "cnn") else ip["svm_pred"]

    pre, seg, les = s1["pre"], s1["seg"], s1["lesions"]
    rgb = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2RGB)  # noqa: E731
    return {
        "crop": crop,
        "crop_name": CROP_NAMES[crop],
        "crop_source": crop_source,
        "crop_probabilities": p_crop,
        "prediction": LABEL[final],
        "prediction_basis": "both methods agree" if agree else f"methods disagree; {tie_breaker().upper()} (higher TEST macro-F1) decides",
        "ip_prediction": {
            "rule": LABEL[ip["rule_pred"]], "rule_score": ip["rule_score"], "rule_threshold": ip["rule_threshold"],
            "rule_components": ip["rule_components"],
            "svm": LABEL[ip["svm_pred"]], "svm_score": ip["svm_score"],
        },
        "cnn_prediction": LABEL[cnn_pred],
        "cnn_confidence": max(p_dis, 1 - p_dis),
        "cnn_p_diseased": p_dis,
        "cnn_class_probabilities": {k: float(v) for k, v in zip(CLASSES4, probs, strict=True)},
        "affected_region_pct": s1["features"]["affected_pct"],
        "lesion_count": int(s1["features"]["lesion_count"]),
        "abnormal_colour_pct": {k: les["class_pct"][k] for k in les["class_pct"]},
        "agreement": bool(agree),
        "segmentation_method": seg["method"],
        "features": s1["features"],
        "stage1_ms": s1["ms"],
        "images": {
            "original": rgb(pre["resized"]),
            "segmented_leaf": rgb(seg["segmented"]),
            "abnormal_regions_overlay": rgb(overlay_lesions(pre["resized"], seg["leaf_mask"], les)),
            "gradcam": rgb(cam_bgr) if cam_bgr is not None else None,
        },
    }


def result_card(res: dict) -> str:
    """The four-line result card."""
    return (f"Crop: {res['crop_name']}\nPrediction: {res['prediction']}\n"
            f"Affected Region: {res['affected_region_pct']:.1f}%\nCNN Confidence: {res['cnn_confidence']:.0%}")


def encode_png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()
