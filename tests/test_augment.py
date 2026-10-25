import numpy as np
import pandas as pd
import pytest

from leafdx.cnn.augment import BackgroundSwap, LeakageError, SwapDataset, build_banks, paste
from leafdx.config import load_config

CFG = load_config()


def _rgba(colour, size=60):
    a = np.zeros((size, size, 4), np.uint8)
    a[..., :3] = colour
    a[10:50, 10:50, 3] = 255
    return a


def test_banks_refuse_non_train_rows():
    df = pd.DataFrame({"path": ["a.jpg", "b.jpg"], "crop": ["turmeric"] * 2, "label": ["healthy"] * 2,
                       "split": ["train", "test"]})
    with pytest.raises(LeakageError):
        build_banks(df, CFG)
    with pytest.raises(LeakageError):
        SwapDataset(df.assign(split=["val", "train"]), {"items": [], "by_crop": {}, "backgrounds": [], "index": {}},
                    CFG, target_fn=lambda r: 0)


def test_real_banks_use_only_train_paths():
    from leafdx.config import path

    s = pd.read_csv(path("splits", CFG))
    train = s[s.split == "train"]
    banks = build_banks(train, CFG)  # cached after scripts/augment_preview.py
    assert set(banks["index"]) <= set(train.path)
    assert not set(banks["index"]) & set(s[s.split != "train"].path)


def test_background_swap_keeps_canvas_and_pastes_leaf():
    banks = {"items": [{"rgba": _rgba((0, 255, 0)), "crop": "turmeric"}, {"rgba": _rgba((0, 200, 0)), "crop": "turmeric"}],
             "by_crop": {"turmeric": [0, 1], "blackgram": []}, "backgrounds": [np.full((50, 50, 3), 128, np.uint8)]}
    im = np.array(BackgroundSwap(banks, CFG, seed=1)(_rgba((255, 0, 0)), "turmeric"))
    side = CFG["augment"]["canvas_side"]
    assert im.shape == (side, side, 3)
    red = (im[..., 0] > 200) & (im[..., 1] < 60)
    assert red.mean() > 0.05  # the target leaf is present


def test_paste_clips_at_borders():
    canvas = np.zeros((100, 100, 3), np.uint8)
    paste(canvas, _rgba((255, 255, 255)), 0.8, 30, 0.95, 0.95)
    assert canvas.any() and canvas.shape == (100, 100, 3)
