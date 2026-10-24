import numpy as np
import pandas as pd
import pytest

from leafdx.config import load_config, path
from leafdx.ip.features import FEATURE_NAMES, stage1
from tests.synthetic import make_leaf

CFG = load_config()


def test_feature_vector_is_fixed_and_finite():
    img, _, _ = make_leaf("field", lesions=[{"centre": (0.5, 0.5), "radius": 20, "colour": "yellow"}])
    f = stage1(img, CFG, "blackgram")["features"]
    assert list(f) == FEATURE_NAMES
    assert np.isfinite(list(f.values())).all()
    assert f["lesion_count"] == 1 and f["affected_pct"] > 0


def test_no_lesion_and_failed_segmentation_are_finite():
    img, _, _ = make_leaf("paper", seed=4)
    f = stage1(img, CFG, "turmeric")["features"]
    assert f["lesion_count"] == 0 and f["lesion_max_size_pct"] == 0
    assert f["lesion_L_std"] == 0 and f["lesion_L_mean"] == pytest.approx(f["leaf_L_mean"])
    blank = stage1(np.full((120, 120, 3), 200, np.uint8), CFG, None)["features"]
    assert blank["seg_failed"] == 1
    assert np.isfinite(list(blank.values())).all()


def test_diseased_leaf_has_higher_affected_pct_than_healthy():
    healthy, _, _ = make_leaf("grey", seed=5)
    sick, _, _ = make_leaf("grey", seed=5, lesions=[{"centre": (0.3, 0.5), "radius": 22, "colour": "brown"},
                                                     {"centre": (0.72, 0.5), "radius": 18, "colour": "yellow"}])
    fh = stage1(healthy, CFG, "blackgram")["features"]
    fs = stage1(sick, CFG, "blackgram")["features"]
    assert fs["affected_pct"] > fh["affected_pct"] + 5
    assert fs["lesion_count"] == 2


def test_real_features_file():
    f = path("features", CFG)
    if not f.exists():
        pytest.skip("features.csv not built yet")
    df = pd.read_csv(f)
    splits = pd.read_csv(path("splits", CFG))
    assert len(df) == len(splits) and df["path"].is_unique
    assert np.isfinite(df[FEATURE_NAMES].to_numpy()).all()
