"""Phase 3 image table: every image with its role.

split: train / val / test (grouped clean split), wild (all of data/wild, reporting only),
user (user-reported images, regression only). ``group`` is the near-duplicate cluster id,
used for grouped bootstrap resampling.
"""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path

import pandas as pd
from PIL import Image

from leafdx.config import ROOT, load_config, path

USER_IMAGES = [  # (path, crop, label); label "unknown": the user did not state the disease status
    ("data/wild/user_reported/turmeric_google_01.jpg", "turmeric", "unknown"),
]


def load_rgb(p: str | Path) -> Image.Image:
    with Image.open(ROOT / p) as im:
        return im.convert("RGB")


@cache
def file_hash(p: str) -> str:
    """sha1 of the file bytes (content-addressed cache key)."""
    return hashlib.sha1((ROOT / p).read_bytes()).hexdigest()


def image_table(cfg: dict | None = None, with_user: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    s = pd.read_csv(path("splits", cfg))
    clean = s.assign(group="c" + s.cluster_id.astype(str), source="clean", wild_half="")
    # Phase 2's wild splits.csv lists one Commons file twice (two titles saved to one filename): keep it once
    w = pd.read_csv(ROOT / cfg["wild"]["root"] / "splits.csv").drop_duplicates("path")
    wild = w.assign(split="wild", group="w" + w.cluster_id.astype(str), wild_half=w["split"])
    cols = ["path", "crop", "label", "split", "group", "source", "wild_half", "orig_class"]
    parts = [clean[cols], wild[cols]]
    if with_user:
        u = [{"path": p, "crop": c, "label": lab, "split": "user", "group": f"u{i}", "source": "user_reported",
              "wild_half": "", "orig_class": "user_reported"} for i, (p, c, lab) in enumerate(USER_IMAGES) if (ROOT / p).exists()]
        if u:
            parts.append(pd.DataFrame(u))
    return pd.concat(parts, ignore_index=True)
