"""Cluster near-duplicates and write data/processed/splits.csv (grouped, stratified 70/15/15)."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from leafdx.config import ROOT, load_config, path
from leafdx.split import (
    class_weights,
    cluster_images,
    grouped_stratified_split,
    leakage_report,
    phash_pair,
)


def main() -> int:
    cfg = load_config()
    sc = cfg["split"]
    inv = pd.read_csv(path("inventory", cfg))
    df = inv[inv.readable].reset_index(drop=True)
    print(f"{len(df)} readable images ({len(inv) - len(df)} unreadable excluded)")

    pairs = Parallel(n_jobs=-1)(delayed(phash_pair)(ROOT / p) for p in df.path)
    h_small, h_large = (np.stack(h) for h in zip(*pairs, strict=True))
    df["cluster_id"] = cluster_images(df, h_small, h_large, sc["dedup"], sc["stem_grouping"])
    df["stratum"] = df["crop"] + "_" + df["label"]
    df["split"] = grouped_stratified_split(df, sc["ratios"], cfg["seed"])

    out = path("splits", cfg)
    df[["path", "crop", "label", "orig_class", "cluster_id", "stratum", "split"]].to_csv(out, index=False)
    rep = leakage_report(df)
    train = df[df.split == "train"]
    meta = {
        "seed": cfg["seed"],
        "dedup_rules": sc["dedup"],
        **rep,
        "counts": df.groupby(["split", "stratum"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "class_weights_binary": class_weights(train, "label"),
        "class_weights_4class": class_weights(train, "stratum"),
    }
    (path("processed", cfg) / "split_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {out.relative_to(ROOT)}")
    print(json.dumps(rep, indent=2))
    print(df.groupby(["stratum", "split"]).size().unstack(fill_value=0).to_string())
    sizes = df.groupby("cluster_id").size()
    print("cluster size distribution:", sizes.value_counts().sort_index().to_dict())
    return 1 if rep["clusters_spanning_splits"] else 0


if __name__ == "__main__":
    sys.exit(main())
