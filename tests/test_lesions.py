import pytest

from leafdx.config import load_config
from leafdx.ip.lesions import connected_lesions, detect_lesions
from leafdx.ip.preprocess import preprocess
from leafdx.ip.segment import segment_leaf
from tests.synthetic import make_leaf

CFG = load_config()


def _run(img, crop="blackgram"):
    pre = preprocess(img, CFG)
    seg = segment_leaf(pre, CFG)
    return seg, detect_lesions(pre, seg["leaf_mask"], CFG, crop)


@pytest.mark.parametrize("colour", ["brown", "yellow", "dark"])
@pytest.mark.parametrize("background", ["paper", "field"])
def test_painted_lesions_area_and_count(colour, background):
    lesions = [{"centre": (0.35, 0.45), "radius": 14, "colour": colour},
               {"centre": (0.55, 0.55), "radius": 18, "colour": colour},
               {"centre": (0.70, 0.45), "radius": 10, "colour": colour}]
    img, leaf, truth = make_leaf(background, size=400, lesions=lesions, seed=7)
    _, res = _run(img)
    detected = int(res["lesion_mask"].sum())
    true_area = int(truth.sum())
    assert abs(detected - true_area) / true_area < 0.15, (detected, true_area, res["lesion_count"])
    assert res["lesion_count"] == 3
    # detected lesion pixels are where the paint is
    assert (res["lesion_mask"] & truth).sum() / detected > 0.85


def test_healthy_synthetic_leaf_has_no_lesions():
    img, _, _ = make_leaf("paper", size=400, seed=11)
    _, res = _run(img)
    assert res["affected_pct"] < 1.0
    assert res["lesion_count"] == 0


def test_affected_pct_matches_mask():
    img, _, _ = make_leaf("grey", lesions=[{"centre": (0.5, 0.5), "radius": 20, "colour": "brown"}], seed=2)
    seg, res = _run(img)
    expect = 100 * res["lesion_mask"].sum() / seg["leaf_mask"].sum()
    assert res["affected_pct"] == pytest.approx(expect)


def test_min_area_filter_and_shape_stats():
    import numpy as np

    m = np.zeros((100, 100), np.uint8)
    m[10:12, 10:12] = 1          # 4 px: dropped
    m[40:60, 40:60] = 1          # 400 px square
    keep, les = connected_lesions(m, min_area=25)
    assert len(les) == 1 and les[0]["area"] == 400
    assert 0.6 < les[0]["circularity"] < 0.9   # square ~0.785
