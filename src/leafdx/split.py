"""Leakage-free splitting: cluster near-duplicates, then split by cluster.

1. Perceptual hash (pHash) per image and per flip/rot90 variant; images whose
   closest variants are within `phash_max_distance` bits (same crop) are joined
   into one cluster (union-find). This also catches flipped/rotated copies.
2. Filename-stem grouping: files whose names differ only by an augmentation /
   copy suffix (e.g. "_flip", "_aug3", " - Copy") are joined too.
3. Clusters are assigned whole to train/val/test, stratified by crop+label,
   greedily filling the split with the largest deficit (seeded shuffle).
"""

from __future__ import annotations

import re
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

AUG_SUFFIX = re.compile(
    r"([ _\-.]*(aug(mented)?|flip(ped)?|rot(ate[d]?)?\d*|bright(ness)?|noise|zoom|shift(ed)?|shear|copy)\d*)+$",
    re.IGNORECASE,
)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def phash_bits(path: str | Path, hash_size: int = 8) -> np.ndarray:
    """pHash bits of the 8 dihedral variants (4 rotations x optional mirror): shape (8, bits)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((256, 256))
    variants = []
    for mirror in (False, True):
        base = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if mirror else im
        for k in range(4):
            v = base.rotate(90 * k, expand=True)
            variants.append(imagehash.phash(v, hash_size=hash_size).hash.flatten())
    return np.stack(variants).astype(np.uint8)


def stem_key(path: str) -> str:
    stem = Path(path).stem
    return AUG_SUFFIX.sub("", stem).strip().lower() or stem.lower()


def cluster_images(df: pd.DataFrame, hashes: np.ndarray, max_distance: int,
                   use_stems: bool = True) -> np.ndarray:
    """Cluster id per row. `hashes` is (n, variants, bits); rows are only joined within a crop."""
    n = len(df)
    uf = UnionFind(n)
    crops = df["crop"].to_numpy()
    for crop in np.unique(crops):
        idx = np.flatnonzero(crops == crop)
        h = hashes[idx].astype(np.int16)
        for k, i in enumerate(idx[:-1]):
            # Hamming distance from i's canonical hash to the closest variant of each later image
            d = np.abs(h[k + 1:] - h[k, 0]).sum(axis=2).min(axis=1)
            for j in idx[k + 1:][d <= max_distance]:
                uf.union(i, j)
    if use_stems:
        keys = df["crop"] + "|" + df["orig_class"] + "|" + df["path"].map(stem_key)
        first: dict[str, int] = {}
        for i, k in enumerate(keys):
            if k in first:
                uf.union(first[k], i)
            else:
                first[k] = i
    roots = np.array([uf.find(i) for i in range(n)])
    _, cluster_ids = np.unique(roots, return_inverse=True)
    return cluster_ids


def grouped_stratified_split(df: pd.DataFrame, ratios: dict[str, float], seed: int) -> pd.Series:
    """Assign whole clusters to splits, stratified by crop+label of the cluster's majority."""
    rng = np.random.default_rng(seed)
    names = list(ratios)
    split = pd.Series(index=df.index, dtype=object)
    strata = df.groupby("cluster_id").agg(
        stratum=("stratum", lambda s: s.value_counts().index[0]), size=("stratum", "size")
    )
    for _, block in strata.groupby("stratum", sort=True):
        clusters = block.index.to_numpy()
        clusters = clusters[rng.permutation(len(clusters))]
        sizes = block.loc[clusters, "size"].to_numpy()
        total = sizes.sum()
        filled = dict.fromkeys(names, 0)
        # Seed each split with one cluster so every stratum appears everywhere.
        order = list(range(len(clusters)))
        assign = {}
        for name, ci in zip(sorted(names, key=lambda n: ratios[n]), order[: len(names)], strict=False):
            assign[ci] = name
            filled[name] += sizes[ci]
        for ci in order[len(names):]:
            deficit = {n: ratios[n] * total - filled[n] for n in names}
            name = max(names, key=lambda n: (deficit[n], ratios[n]))
            assign[ci] = name
            filled[name] += sizes[ci]
        for ci, name in assign.items():
            split[df["cluster_id"] == clusters[ci]] = name
    return split


def class_weights(train: pd.DataFrame, column: str) -> dict[str, float]:
    """Inverse-frequency weights normalised to mean 1 (sklearn 'balanced')."""
    counts = train[column].value_counts()
    w = len(train) / (len(counts) * counts)
    return {k: float(v) for k, v in w.sort_index().items()}


def leakage_report(df: pd.DataFrame) -> dict:
    spans = df.groupby("cluster_id")["split"].nunique()
    mixed = df.groupby("cluster_id")["stratum"].nunique()
    return {
        "clusters_spanning_splits": int((spans > 1).sum()),
        "clusters_with_mixed_labels": int((mixed > 1).sum()),
        "n_clusters": int(df["cluster_id"].nunique()),
        "n_multi_image_clusters": int((df.groupby("cluster_id").size() > 1).sum()),
    }
