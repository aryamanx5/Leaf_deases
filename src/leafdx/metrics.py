"""Shared binary-classification metrics (positive class = diseased)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

LABELS = ("healthy", "diseased")


def binary_metrics(y_true, y_pred, score=None) -> dict:
    """y_* are 0/1 (1 = diseased). Confusion matrix rows = true [healthy, diseased]."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "healthy_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, score))
    return out


def with_per_crop(df, pred_col: str, score_col: str | None = None) -> dict:
    """Overall metrics plus a per-crop breakdown. `df` needs y (0/1), crop and pred_col."""
    res = {"overall": binary_metrics(df.y, df[pred_col], df[score_col] if score_col else None)}
    res["per_crop"] = {
        crop: binary_metrics(g.y, g[pred_col], g[score_col] if score_col else None)
        for crop, g in df.groupby("crop")
    }
    return res
