"""Evaluation / training frames for Phase 3: one DataFrame per named set with labels,
groups, the image file actually shown to the model, its embedding cache key and the 57
Stage 1 features.

Sets: train, train_swap (K views per TRAIN image), val, val_swap, test, test_swap,
wild (disease-labelled wild images), wild_all (every wild image, for crop ID), user,
and *_leafonly variants of val/test/wild/wild_all/user (background masked, shortcut test).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from leafdx.cnn.augment import build_banks
from leafdx.config import ROOT, load_config, path, seed_everything
from leafdx.ip.features import FEATURE_NAMES
from leafdx.p3.data import file_hash, image_table
from leafdx.p3.embcache import embed_keys
from leafdx.p3.features import features_for
from leafdx.p3.views import load_png, save_train_views, train_view_keys


def _base(t: pd.DataFrame, setname: str) -> pd.DataFrame:
    d = t.copy().reset_index(drop=True)
    d["set"] = setname
    d["file"] = d.path
    d["key"] = [file_hash(p) for p in d.path]
    d["y"] = (d.label == "diseased").astype(int)
    return d


@lru_cache(maxsize=1)
def build_frames() -> dict[str, pd.DataFrame]:
    cfg = load_config()
    t = image_table(cfg)
    fc = pd.read_csv(path("features", cfg)).set_index("path")
    F: dict[str, pd.DataFrame] = {}
    for s in ("train", "val", "test"):
        d = _base(t[t.split == s], s)
        d[FEATURE_NAMES] = fc.loc[d.path, FEATURE_NAMES].to_numpy(float)
        F[s] = d
    w = t[t.split == "wild"]
    F["wild_all"] = _base(w, "wild_all")
    F["user"] = _base(t[t.split == "user"], "user")
    for s in ("wild_all", "user"):
        F[s][FEATURE_NAMES] = features_for(list(F[s].file), list(F[s].crop), cfg)[FEATURE_NAMES].to_numpy()
    F["wild"] = F["wild_all"][F["wild_all"].label != "unknown"].reset_index(drop=True).assign(set="wild")

    # background-swap views (VAL/TEST: one each; TRAIN: K each)
    sw = pd.read_csv(path("results", cfg) / "p3" / "views_swap.csv").dropna().set_index("path")
    for s in ("val", "test"):
        d = F[s][F[s].path.isin(sw.index)].copy().reset_index(drop=True)
        d["set"], d["file"] = f"{s}_swap", [sw.loc[p, "swap"] for p in d.path]
        d["key"] = [k + "__swap" for k in d.key]
        d[FEATURE_NAMES] = features_for(list(d.file), list(d.crop), cfg)[FEATURE_NAMES].to_numpy()
        F[f"{s}_swap"] = d
    train = F["train"]
    seed_everything(cfg["seed"])
    banks = build_banks(pd.read_csv(path("splits", cfg)).query("split == 'train'"), cfg)
    files = save_train_views(train[["path", "crop", "label", "split"]], banks, cfg)
    k = cfg["p3"]["train_swap_views"]
    d = pd.concat([train] * k, ignore_index=True)
    d["set"], d["file"], d["key"] = "train_swap", files, train_view_keys(train, cfg)
    d[FEATURE_NAMES] = features_for(files, list(d.crop), cfg)[FEATURE_NAMES].to_numpy()
    F["train_swap"] = d

    lo = pd.read_csv(path("results", cfg) / "p3" / "views_leafonly.csv").set_index("path")
    for s in ("val", "test", "wild", "wild_all", "user"):
        d = F[s].copy()
        d["set"], d["file"] = f"{s}_leafonly", [lo.loc[p, "leafonly"] for p in d.path]
        d["key"] = [k + "__leafonly" for k in d.key]
        d[FEATURE_NAMES] = features_for(list(d.file), list(d.crop), cfg)[FEATURE_NAMES].to_numpy()
        F[f"{s}_leafonly"] = d
    return F


_MEMO: dict = {}


def embeddings(name: str, d: pd.DataFrame) -> np.ndarray:
    """Cached embeddings for a frame (computed on demand from ``file`` if missing); memoised per process."""
    mk = (name, tuple(d.key))
    if mk in _MEMO:
        return _MEMO[mk]
    files = list(d.file)

    def make(idx):
        from leafdx.p3.data import load_rgb

        return [load_png(files[i]) if str(files[i]).endswith(".png") and str(ROOT / ".cache") in str(files[i])
                else load_rgb(files[i]) for i in idx]

    _MEMO[mk] = embed_keys(name, list(d.key), make)
    return _MEMO[mk]


def crop_ind(d: pd.DataFrame) -> np.ndarray:
    return (d.crop == "turmeric").to_numpy(float)[:, None]
