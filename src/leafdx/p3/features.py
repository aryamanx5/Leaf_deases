"""Stage 1 handcrafted features (57) for any image file, cached by content hash.

Cache: data/processed/cache/features_p3_<stage1 hash>/<file sha1>_<crop>.json. Clean
TRAIN/VAL/TEST rows are read from data/processed/features.csv (same Stage 1 code); every
other image (wild, user, derived views) goes through ``stage1`` here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from leafdx.config import ROOT, path
from leafdx.ip.features import FEATURE_NAMES, stage1
from leafdx.ip.preprocess import load_bgr

sys.path.insert(0, str(ROOT / "scripts"))
from extract_features import stage1_hash  # noqa: E402

KEEP = [*FEATURE_NAMES, "seg_method", "seg_coverage", "stage1_ms"]


def _one(f: str, crop: str, cache_dir: Path) -> dict:
    import hashlib

    key = hashlib.sha1(Path(f).read_bytes()).hexdigest()
    cf = cache_dir / f"{key}_{crop}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    import cv2

    from leafdx.config import load_config

    cv2.setNumThreads(1)  # one OpenCV thread per joblib worker (avoids oversubscription)

    out = stage1(load_bgr(f), load_config(), crop)
    rec = {**out["features"], "seg_method": out["seg"]["method"], "seg_coverage": out["seg"]["coverage"],
           "stage1_ms": out["ms"]}
    cf.write_text(json.dumps(rec))
    return rec


def features_for(files: list[str], crops: list[str], cfg: dict) -> pd.DataFrame:
    """One row per file (absolute or repo-relative), columns KEEP."""
    cache_dir = path("cache", cfg) / f"features_p3_{stage1_hash(cfg)}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    absf = [str(f) if Path(f).is_absolute() else str(ROOT / f) for f in files]
    recs = Parallel(n_jobs=-1, batch_size=8)(delayed(_one)(f, c, cache_dir) for f, c in zip(absf, crops, strict=True))
    df = pd.DataFrame(recs)[KEEP]
    df[FEATURE_NAMES] = np.nan_to_num(df[FEATURE_NAMES].to_numpy(float))
    return df


def clean_features(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(path("features", cfg))
