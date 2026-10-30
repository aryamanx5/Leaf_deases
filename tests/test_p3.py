"""Phase 3 unit tests: CLAHE flag, statistics, conformal sets, quality gate, cache keys."""

import numpy as np
import pytest

from leafdx.config import load_config
from leafdx.ip.preprocess import preprocess
from leafdx.p3.embcache import recipe_hash
from leafdx.p3.quality import quality_stats, quality_verdict
from leafdx.p3.stats import (
    abstention_curve,
    conformal_sets,
    conformal_threshold,
    ece,
    fast_metrics,
    fit_temperature,
    grouped_bootstrap,
    mcnemar,
    point_metrics,
    sigmoid,
)
from tests.synthetic import make_leaf


def _img():
    return make_leaf()[0]


def test_clahe_runs_on_l_channel_and_flag_disables_it():
    import cv2

    cfg = load_config()
    img = _img()
    on = preprocess(img, cfg)
    off = preprocess(img, {**cfg, "preprocess": {**cfg["preprocess"], "clahe_enabled": False}})
    lab_f = cv2.cvtColor(on["filtered"], cv2.COLOR_BGR2LAB).astype(int)
    lab_e = cv2.cvtColor(on["enhanced"], cv2.COLOR_BGR2LAB).astype(int)
    assert np.abs(lab_e[..., 0] - lab_f[..., 0]).mean() > np.abs(lab_e[..., 1:] - lab_f[..., 1:]).mean()
    assert (off["enhanced"] == off["filtered"]).all()


def test_fast_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.3 * y + 0.7 * rng.random(300), 0, 1)
    a, b = fast_metrics(y, p), point_metrics(y, p)
    for k in ("accuracy", "macro_f1", "roc_auc", "ece"):
        assert a[k] == pytest.approx(b[k], abs=1e-9)


def test_ece_zero_for_perfectly_calibrated_bins():
    y = np.array([1] * 9 + [0] + [0] * 9 + [1])
    p = np.array([0.9] * 10 + [0.1] * 10)
    assert ece(p, y) == pytest.approx(0.0, abs=1e-9)


def test_temperature_softens_overconfident_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    x = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 1, 2000)  # calibrated log-odds = 2x
    T = fit_temperature(8 * x, y)  # 4x overconfident -> T near 4
    assert 3 < T < 5
    assert ece(sigmoid(8 * x / T), y) < ece(sigmoid(8 * x), y)


def test_grouped_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.4 * y + 0.6 * rng.random(200), 0, 1)
    g = np.repeat(np.arange(100), 2)
    ci = grouped_bootstrap(y, p, g, n_boot=200, seed=0)
    acc = point_metrics(y, p)["accuracy"]
    assert ci["accuracy"][0] <= acc <= ci["accuracy"][1]


@pytest.mark.parametrize("method", ["lac", "aps"])
def test_conformal_marginal_coverage(method):
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)
    p1 = sigmoid(np.where(y == 1, 1.0, -1.0) * 1.5 + rng.normal(0, 1.5, n))
    probs = np.column_stack([1 - p1, p1])
    q = conformal_threshold(probs[:2000], y[:2000], 0.1, method)
    sets = conformal_sets(probs[2000:], q, method)
    cover = sets[np.arange(2000), y[2000:]].mean()
    assert cover >= 0.88


def test_lac_ambiguous_example_gets_two_labels():
    q = conformal_threshold(np.array([[0.9, 0.1]] * 50 + [[0.6, 0.4]] * 50), np.array([0] * 50 + [1] * 50), 0.1)
    s = conformal_sets(np.array([[0.5, 0.5], [0.99, 0.01]]), q)
    assert s[0].sum() == 2 and s[1].tolist() == [True, False]


def test_mcnemar_counts_discordant_pairs():
    y = np.array([1, 1, 1, 0, 0])
    r = mcnemar(y, np.array([1, 1, 0, 0, 0]), np.array([0, 1, 1, 0, 1]))
    assert (r["a_right_b_wrong"], r["a_wrong_b_right"]) == (2, 1)


def test_abstention_curve_is_monotone_on_sorted_confidence():
    conf = np.linspace(1, 0.5, 100)
    correct = np.r_[np.ones(80, bool), np.zeros(20, bool)]
    c = abstention_curve(conf, correct)
    assert c[0]["accuracy"] == 1.0 and c[-1]["accuracy"] <= c[0]["accuracy"]


def test_quality_gate_flags_blur_and_dark():
    import cv2

    rng = np.random.default_rng(0)
    sharp = (rng.random((200, 200, 3)) * 255).astype(np.uint8)
    q = load_config()["p3"]["quality"]
    assert quality_verdict(quality_stats(sharp), 0.5, q) == (True, None)
    blurred = cv2.GaussianBlur(np.full((200, 200, 3), 120, np.uint8), (31, 31), 10)
    assert quality_verdict(quality_stats(blurred), 0.5, q)[1] == "image too blurry"
    dark = np.zeros((200, 200, 3), np.uint8)
    dark[::4] = 255  # texture so it is not "blurry"
    assert quality_verdict(quality_stats(dark), 0.5, q)[1] == "image too dark"
    assert quality_verdict(quality_stats(sharp), 0.0, q)[1] == "too little leaf area found"


def test_recipe_hash_is_stable_and_input_sensitive():
    assert recipe_hash("a", 1) == recipe_hash("a", 1) != recipe_hash("a", 2)


def test_temperature_does_not_sharpen_on_separable_data():
    z = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    assert fit_temperature(z, (z > 0).astype(int)) >= 1.0


def test_white_background_is_not_over_exposure_when_leaf_mask_given():
    rng = np.random.default_rng(0)
    img = np.full((200, 200, 3), 255, np.uint8)
    leaf = np.zeros((200, 200), bool)
    leaf[50:150, 50:150] = True
    img[leaf] = (rng.random((leaf.sum(), 3)) * 120 + 40).astype(np.uint8)
    q = load_config()["p3"]["quality"]
    assert quality_verdict(quality_stats(img), 0.25, q)[1] == "image over-exposed"  # whole frame: mostly white
    assert quality_verdict(quality_stats(img, leaf), 0.25, q) == (True, None)
