"""P2-B03: build the TRAIN-only augmentation banks and save a 32-sample contact sheet."""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pandas as pd
from torchvision import transforms as T

from leafdx.cnn.augment import SwapDataset, build_banks, robust_transform
from leafdx.config import load_config, path
from leafdx.viz import contact_sheet


def main() -> int:
    cfg = load_config()
    s = pd.read_csv(path("splits", cfg))
    train = s[s.split == "train"]
    banks = build_banks(train, cfg)
    print(f"banks: {len(banks['items'])} leaf cut-outs (turmeric {len(banks['by_crop']['turmeric'])}, "
          f"black gram {len(banks['by_crop']['blackgram'])}), {len(banks['backgrounds'])} backgrounds")
    ds = SwapDataset(train, banks, cfg, target_fn=lambda r: 0, seed=cfg["seed"])
    viz = T.Compose(robust_transform(cfg).transforms[:-3])  # everything before ToTensor/Normalize/Erasing
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.choice(len(ds), 32, replace=False)
    tiles, titles = [], []
    for i in idx:
        im = viz(ds.sample_image(int(i)))
        tiles.append(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
        titles.append(ds.df.stratum[i])
    rows = [tiles[k:k + 8] for k in range(0, 32, 8)]
    trows = [titles[k:k + 8] for k in range(0, 32, 8)]
    out = path("figures", cfg) / "augment_samples.png"
    cv2.imwrite(str(out), contact_sheet(rows, trows, size=170))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
