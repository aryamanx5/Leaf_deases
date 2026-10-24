"""Split integrity. Synthetic images (tests only) plus a check of the real splits.csv if present."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from leafdx.config import load_config, path
from leafdx.split import (
    cluster_images,
    grouped_stratified_split,
    leakage_report,
    phash_bits,
    stem_key,
)


def _blob_image(rng: np.random.Generator, size: int = 128) -> np.ndarray:
    """SYNTHETIC: random smooth colour field, distinct per seed."""
    small = rng.integers(0, 255, (6, 6, 3), dtype=np.uint8)
    return np.array(Image.fromarray(small).resize((size, size), Image.BICUBIC))


@pytest.fixture()
def synthetic_set(tmp_path):
    """12 base images per stratum, each with a flipped and a noisy near-duplicate."""
    rng = np.random.default_rng(0)
    rows = []
    for crop in ("turmeric", "blackgram"):
        for label in ("healthy", "diseased"):
            for i in range(12):
                base = _blob_image(rng)
                variants = {
                    f"{label}_{i}.png": base,
                    f"{label}_{i}_flip.png": base[:, ::-1],
                    f"{label}_{i}_x.png": np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8),
                }
                for name, arr in variants.items():
                    p = tmp_path / crop / label / name
                    p.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(arr).save(p)
                    rows.append({"path": str(p), "crop": crop, "label": label, "orig_class": label,
                                 "base": f"{crop}/{label}/{i}"})
    return pd.DataFrame(rows)


def _split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    hashes = np.stack([phash_bits(p) for p in df.path])
    df["cluster_id"] = cluster_images(df, hashes, max_distance=6)
    df["stratum"] = df["crop"] + "_" + df["label"]
    df["split"] = grouped_stratified_split(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed)
    return df


def test_stem_key_strips_augmentation_suffix():
    assert stem_key("a/leaf_12_flip.jpg") == stem_key("a/leaf_12.jpg") == "leaf_12"
    assert stem_key("leaf_12 - Copy.jpg") == "leaf_12"
    assert stem_key("leaf_12.jpg") != stem_key("leaf_13.jpg")


def test_near_duplicates_share_a_cluster(synthetic_set):
    df = _split(synthetic_set)
    # every base image and its flipped/noisy copies end up in one cluster
    assert (df.groupby("base")["cluster_id"].nunique() == 1).all()
    # distinct base images are not merged wholesale
    assert df["cluster_id"].nunique() >= 0.8 * df["base"].nunique()


def test_no_cluster_spans_two_splits(synthetic_set):
    df = _split(synthetic_set)
    assert leakage_report(df)["clusters_spanning_splits"] == 0


def test_every_stratum_in_every_split(synthetic_set):
    df = _split(synthetic_set)
    table = df.groupby(["stratum", "split"]).size().unstack(fill_value=0)
    assert set(table.columns) == {"train", "val", "test"}
    assert (table > 0).all().all()


def test_reproducible_with_same_seed(synthetic_set):
    a, b = _split(synthetic_set, 42), _split(synthetic_set, 42)
    assert (a["split"] == b["split"]).all()


def test_real_splits_file_is_leakage_free():
    f = path("splits", load_config())
    if not f.exists():
        pytest.skip("splits.csv not built yet")
    df = pd.read_csv(f)
    assert df.groupby("cluster_id")["split"].nunique().max() == 1
    table = df.groupby(["stratum", "split"]).size().unstack(fill_value=0)
    assert set(table.columns) == {"train", "val", "test"}
    assert (table > 0).all().all()
    # no file path appears twice
    assert df["path"].is_unique
