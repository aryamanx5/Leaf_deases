"""Crop decision status paths, kNN OOD distance, temperature fit (no model files needed)."""

import importlib.util

import numpy as np

from leafdx.config import ROOT
from leafdx.robust.infer import crop_decision, knn_distance, logit, sigmoid

CAL = {"tau_crop": 0.8, "ood_threshold": 0.3}


def test_confident_when_probable_and_in_distribution():
    d = crop_decision(0.93, 0.1, CAL)
    assert d["label"] == "turmeric" and d["status"] == "confident" and d["source"] == "model"
    assert abs(d["probs"]["turmeric"] + d["probs"]["blackgram"] - 1) < 1e-9


def test_uncertain_when_probability_low():
    d = crop_decision(0.35, 0.1, CAL)
    assert d["label"] == "blackgram" and d["confidence"] == 0.65 and d["status"] == "uncertain"


def test_uncertain_when_out_of_distribution():
    assert crop_decision(0.99, 0.9, CAL)["status"] == "uncertain"


def test_user_selected_crop_overrides_model():
    d = crop_decision(0.99, 0.9, CAL, user_crop="blackgram")
    assert d == {"label": "blackgram", "confidence": d["probs"]["blackgram"], "probs": d["probs"],
                 "status": "user_selected", "source": "user"}


def test_knn_distance_is_zero_for_reference_points():
    rng = np.random.default_rng(0)
    base = np.ones(8) / np.sqrt(8)
    ref = base + rng.normal(0, 0.05, size=(50, 8))
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    assert np.allclose(knn_distance(ref[:3], ref, 1), 0, atol=1e-6)
    assert knn_distance(-base[None], ref, 5)[0] > 1.5  # opposite of the cluster


def test_temperature_fit_shrinks_overconfident_logits():
    spec = importlib.util.spec_from_file_location("calibrate", ROOT / "scripts/calibrate.py")
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    x = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 1.0, 2000)  # true log-odds = 2x
    over = 4 * x                                               # 2x overconfident
    T = cal.fit_temperature(over, y)
    assert 1.6 < T < 2.5
    assert cal.ece(sigmoid(over / T), y) < cal.ece(sigmoid(over), y)
    assert np.allclose(sigmoid(logit(0.7)), 0.7)
