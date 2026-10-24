import numpy as np
import pytest

from leafdx.config import load_config
from leafdx.ip.preprocess import clahe_l, denoise, preprocess, resize_max_side
from leafdx.ip.segment import segment_leaf
from tests.synthetic import BACKGROUNDS, iou, make_leaf

CFG = load_config()


def test_preprocess_returns_all_steps():
    img, _, _ = make_leaf("paper", size=700)
    pre = preprocess(img, CFG)
    assert max(pre["resized"].shape[:2]) == CFG["preprocess"]["max_side"]
    for k in ("filtered", "enhanced", "hsv", "lab"):
        assert pre[k].shape == pre["resized"].shape
    assert denoise(img, 5, 5).shape == img.shape
    assert clahe_l(img, 2.0, 8).dtype == np.uint8


def test_resize_never_upscales():
    img = np.zeros((100, 80, 3), np.uint8)
    assert resize_max_side(img, 512).shape == img.shape


@pytest.mark.parametrize("background", BACKGROUNDS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_segmentation_iou_on_synthetic_leaf(background, seed):
    img, leaf, _ = make_leaf(background, size=400, seed=seed)
    seg = segment_leaf(preprocess(img, CFG), CFG)
    assert iou(seg["leaf_mask"], leaf) > 0.9, seg["method"]


def test_lesions_inside_leaf_stay_in_leaf_mask():
    lesions = [{"centre": (0.5, 0.5), "radius": 25, "colour": "brown"},
               {"centre": (0.35, 0.45), "radius": 15, "colour": "dark"}]
    img, leaf, lesion = make_leaf("paper", lesions=lesions, seed=3)
    seg = segment_leaf(preprocess(img, CFG), CFG)
    assert seg["leaf_mask"][lesion == 1].mean() > 0.95
    assert iou(seg["leaf_mask"], leaf) > 0.9


def test_blank_image_does_not_crash():
    seg = segment_leaf(preprocess(np.full((200, 200, 3), 128, np.uint8), CFG), CFG)
    assert seg["leaf_mask"].shape == (200, 200)
