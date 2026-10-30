"""Score any saved Phase 3 disease configuration on any image frame (no fitting).

A frame needs columns path, file, key (content hash) and crop; `feats` are its 57 Stage 1 features computed with
that crop. `b` is a saved bundle: a head bundle from .cache/p3models / models/p3/arms (kind, backbone, mode, heads,
temperature), the rule bundle, or {"checkpoint", "arch"} for a fine-tuned CNN arm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from leafdx.ip.features import FEATURE_NAMES
from leafdx.p3 import heads
from leafdx.p3.frames import embeddings


def config_z(b: dict, d: pd.DataFrame, feats: np.ndarray) -> np.ndarray:
    """Raw score z (log-odds or margin) of the 'diseased' class for every row, given the row's crop."""
    ci = (d.crop == "turmeric").to_numpy(float)[:, None]
    if "checkpoint" in b:  # CNN arm
        from leafdx.p3.cnnpred import logits

        return logits(b["checkpoint"], b["arch"], d)
    if b["kind"] == "rule":
        row = pd.DataFrame(feats, columns=FEATURE_NAMES).assign(crop=d.crop.values)
        return 10 * (b["rule"].score(row).to_numpy() - d.crop.map(b["rule"].threshold).to_numpy())
    parts = []
    if b["kind"] in ("emb", "fusion"):
        parts.append(embeddings(b["backbone"], d))
    if b["kind"] in ("feat", "fusion"):
        parts.append(feats)
    X = np.column_stack([*parts, ci]).astype(np.float32)
    if b["mode"] == "pooled":
        return heads.raw_score(b["heads"]["all"], X)
    z = np.zeros(len(d))
    for c in ("turmeric", "blackgram"):
        m = (d.crop == c).to_numpy()
        if m.any():
            z[m] = heads.raw_score(b["heads"][c], X[m])
    return z
