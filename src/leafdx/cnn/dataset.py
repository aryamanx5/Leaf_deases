"""Torch dataset over splits.csv for the 4-class (crop x state) CNN.

Images are decoded once, downsized to `cache_side`, and held in memory as PIL images;
augmentation is applied on the fly and ONLY to the training split.
"""

from __future__ import annotations

import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from leafdx.config import ROOT

CLASSES4 = ("turmeric_healthy", "turmeric_diseased", "blackgram_healthy", "blackgram_diseased")
DISEASED_IDX = tuple(i for i, c in enumerate(CLASSES4) if c.endswith("diseased"))
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _load(p: str, side: int) -> Image.Image:
    with Image.open(ROOT / p) as im:
        im = im.convert("RGB")
        im.thumbnail((side, side), Image.BILINEAR)
        return im.copy()


def train_transform(c: dict) -> T.Compose:
    a = c["augment"]
    return T.Compose([
        T.RandomResizedCrop(c["image_size"], scale=tuple(a["crop_scale"]), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(a["rotation"]),
        T.ColorJitter(a["brightness"], a["contrast"], a["saturation"], a["hue"]),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(c: dict) -> T.Compose:
    return T.Compose([
        T.Resize((c["image_size"], c["image_size"])),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class LeafDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, cache_side: int = 288, n_jobs: int = -1):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.targets = [CLASSES4.index(s) for s in self.df["stratum"]]
        self.images = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_load)(p, cache_side) for p in self.df["path"]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        return self.transform(self.images[i]), self.targets[i]

