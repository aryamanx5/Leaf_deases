"""Cached logits of the fine-tuned disease CNN arms (C1-C3) for any frame.

Logit = mean over the 6 TTA views of the true crop's "diseased" unit (as in Phase 2).
Cache: .cache/cnnpreds/<checkpoint sha1[:12]>/<image key>.npy
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from leafdx.cnn.model import build_model
from leafdx.config import ROOT, load_config
from leafdx.p3.data import load_rgb
from leafdx.robust.cnn_heads import predict_disease_logit


def ckpt_hash(ck: Path) -> str:
    return hashlib.sha1(Path(ck).read_bytes()).hexdigest()


@lru_cache(maxsize=8)
def load_arm(ck: str, arch: str, dev: str = "cpu"):
    m, _ = build_model(arch, 2, pretrained=False)
    m.load_state_dict(torch.load(ROOT / ck, map_location="cpu"))
    return m.eval().to(dev)


def logits(ck: str, arch: str, d: pd.DataFrame, dev: str | None = None) -> np.ndarray:
    from leafdx.cnn.model import device

    dev = dev or str(device())
    cfg = load_config()
    out_dir = ROOT / ".cache" / "cnnpreds" / ckpt_hash(ROOT / ck)[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [out_dir / f"{k}.npy" for k in d.key]
    miss = [i for i, f in enumerate(files) if not f.exists()]
    if miss:
        m = load_arm(ck, arch, dev)
        for s in range(0, len(miss), 64):
            part = miss[s:s + 64]
            ims = [load_rgb(d.file.iloc[i]) for i in part]
            z = predict_disease_logit(m, ims, list(d.crop.iloc[part]), cfg, dev=torch.device(dev))
            for i, v in zip(part, z, strict=True):
                np.save(files[i], np.float32(v))
    return np.array([float(np.load(f)) for f in files])
