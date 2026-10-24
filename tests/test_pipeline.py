"""Unified pipeline: synthetic + real images, bad input handling, Stage 1 consistency."""

import cv2
import numpy as np
import pandas as pd
import pytest

from leafdx.config import ROOT, load_config, path
from leafdx.ip.features import stage1
from leafdx.pipeline import InvalidImageError, NotALeafError, analyze, result_card
from tests.synthetic import make_leaf

CFG = load_config()
pytestmark = pytest.mark.skipif(not (path("models", CFG) / "cnn_best.pt").exists(),
                                reason="models not trained yet")
KEYS = {"crop", "ip_prediction", "cnn_prediction", "cnn_confidence", "affected_region_pct",
        "lesion_count", "agreement", "images"}


def _check(res):
    assert KEYS <= set(res)
    assert res["cnn_prediction"] in ("Healthy", "Diseased")
    assert res["ip_prediction"]["svm"] in ("Healthy", "Diseased")
    assert 0.5 <= res["cnn_confidence"] <= 1.0
    for k in ("original", "segmented_leaf", "abnormal_regions_overlay", "gradcam"):
        assert res["images"][k].ndim == 3 and res["images"][k].dtype == np.uint8
    card = result_card(res)
    assert card.splitlines()[0].startswith("Crop: ") and "Affected Region: " in card and "CNN Confidence: " in card


def test_synthetic_leaf_with_given_crop():
    img, _, _ = make_leaf("paper", lesions=[{"centre": (0.5, 0.5), "radius": 25, "colour": "brown"}])
    res = analyze(img, crop="blackgram")
    _check(res)
    assert res["crop"] == "blackgram" and res["crop_source"] == "user"
    # the reported affected area IS the Stage 1 feature value
    assert res["affected_region_pct"] == pytest.approx(stage1(img, CFG, "blackgram")["features"]["affected_pct"])


def test_five_real_images_match_stage1_features():
    feats = pd.read_csv(path("features", CFG))
    sample = feats[feats.split == "test"].groupby("crop").sample(3, random_state=0).head(5)
    for r in sample.itertuples():
        res = analyze(ROOT / r.path, crop=r.crop)
        _check(res)
        assert res["affected_region_pct"] == pytest.approx(r.affected_pct, abs=1e-9)
    # auto crop: the CNN chooses one of the two crops
    res = analyze(ROOT / sample.path.iloc[0])
    assert res["crop"] in CFG["crops"] and res["crop_source"] == "cnn"


def test_bad_inputs_raise_clear_errors():
    with pytest.raises(InvalidImageError):
        analyze(b"definitely not an image")
    with pytest.raises(InvalidImageError):
        analyze(np.full((20, 20, 3), 100, np.uint8))
    with pytest.raises(NotALeafError):
        analyze(np.full((300, 300, 3), (200, 120, 40), np.uint8))   # plain blue (BGR) card, no leaf
    with pytest.raises(NotALeafError):
        analyze(cv2.cvtColor(np.full((300, 300), 128, np.uint8), cv2.COLOR_GRAY2BGR))
    with pytest.raises(ValueError):
        analyze(np.zeros((300, 300, 3), np.uint8), crop="rice")
