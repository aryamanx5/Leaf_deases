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


def test_sixty_small_speckles_area_recall_and_count():
    """P2-B06: 60 small brown speckles on a leaf (SYNTHETIC)."""
    import numpy as np

    rng = np.random.default_rng(5)
    pts = []
    while len(pts) < 60:
        c = rng.uniform(0.2, 0.8, 2)
        if all(np.hypot(*(c - q)) > 0.07 for q in pts):
            pts.append(c)
    lesions = [{"centre": tuple(c), "radius": float(rng.uniform(2.5, 4.0)), "colour": "brown"} for c in pts]
    img, leaf, truth = make_leaf("paper", size=400, lesions=lesions, seed=9)
    _, res = _run(img)
    recall = (res["lesion_mask"].astype(bool) & truth.astype(bool)).sum() / truth.sum()
    n_true = int(__import__("cv2").connectedComponents(truth)[0] - 1)
    assert recall >= 0.85, recall
    assert abs(res["lesion_count"] - n_true) <= 0.2 * n_true, (res["lesion_count"], n_true)
