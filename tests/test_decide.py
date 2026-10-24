import numpy as np
import pandas as pd
import pytest

from leafdx.config import load_config, path
from leafdx.ip.decide import RULE_INPUTS, RuleBasedScorer, load_ip_models, predict_ip
from leafdx.ip.features import stage1
from tests.synthetic import make_leaf

CFG = load_config()


def _toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    return pd.DataFrame({
        "crop": "blackgram", "label": np.where(y == 1, "diseased", "healthy"),
        "affected_pct": y * 20 + rng.uniform(0, 2, n), "lesion_count": y * 10 + rng.integers(0, 2, n),
        "green_frac": 1 - y * 0.3 - rng.uniform(0, 0.02, n), "glcm_contrast": 5 + y * 3 + rng.normal(0, 0.5, n),
    })


def test_rule_based_scorer_separates_toy_data():
    df = _toy()
    rule = RuleBasedScorer(dict.fromkeys(RULE_INPUTS, 1.0)).fit_scaling(df).fit_threshold(df, np.linspace(0, 1, 101))
    s = rule.score(df)
    assert ((s >= 0) & (s <= 1)).all()
    assert (rule.predict(df) == (df.label == "diseased")).mean() == 1.0


def test_predict_end_to_end_on_synthetic_image():
    if not (path("models", CFG) / "ip_svm.joblib").exists():
        pytest.skip("IP models not trained yet")
    rule, svm = load_ip_models(path("models", CFG))
    img, _, _ = make_leaf("paper", lesions=[{"centre": (0.5, 0.5), "radius": 25, "colour": "brown"}])
    out = predict_ip(stage1(img, CFG, "blackgram")["features"], "blackgram", rule, svm)
    assert out["rule_pred"] in (0, 1) and out["svm_pred"] in (0, 1)
    assert 0 <= out["rule_score"] <= 1 and np.isfinite(out["svm_score"])
