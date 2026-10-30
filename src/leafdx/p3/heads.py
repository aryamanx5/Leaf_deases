"""Small heads for the classical / frozen-backbone / fusion arms (P3-M03).

Every candidate is fitted on TRAIN rows only and scored on clean VAL and VAL-swap; the
selection score is the mean of the two macro-F1 values (tie-break: VAL log-loss). The wild
set never enters here. Heads return a raw score z (log-odds or SVM margin) per row.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def pin_lightgbm_openmp() -> bool:
    """Set LightGBM's own OpenMP runtime to one thread. On macOS LightGBM links Homebrew's libomp while PyTorch
    ships another; a multi-threaded LightGBM pool next to PyTorch's segfaults or deadlocks, including inside
    model unpickling (which ignores n_jobs). The setting is per thread: call it in the thread that loads or
    predicts (e.g. inside a web request). Returns True if a libomp was pinned."""
    import ctypes
    import os

    for lib in ("/opt/homebrew/opt/libomp/lib/libomp.dylib", "/usr/local/opt/libomp/lib/libomp.dylib",
                "/opt/local/lib/libomp/libomp.dylib"):
        if os.path.exists(lib):
            try:
                ctypes.CDLL(lib).omp_set_num_threads(1)
                return True
            except OSError:
                pass
    return False


pin_lightgbm_openmp()


def _lgbm(num_leaves: int, seed: int):
    try:
        from lightgbm import LGBMClassifier

        # n_jobs=1: a multi-threaded LightGBM OpenMP pool segfaults or deadlocks next to PyTorch's own libomp
        # on macOS (tested); single-threaded with torch imported first is stable
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=num_leaves, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, class_weight="balanced", random_state=seed,
                              verbose=-1, n_jobs=1)
    except ImportError:  # spec fallback
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_leaf_nodes=num_leaves, class_weight="balanced", random_state=seed)


def candidates(kind: str, seed: int, C_grid=(0.01, 0.1, 1.0, 10.0)) -> list[tuple[str, object]]:
    if kind == "svm":
        return [(f"svm C={C} g={g}", Pipeline([("s", StandardScaler()), ("m", SVC(C=C, gamma=g, class_weight="balanced",
                                                                                  random_state=seed))]))
                for C in (0.3, 1, 3, 10, 30) for g in ("scale", 0.003, 0.01, 0.03)]
    if kind == "lgbm":
        return [(f"lgbm leaves={n}", _lgbm(n, seed)) for n in (7, 15, 31)]
    if kind == "lr":
        return [(f"logreg C={C}", Pipeline([("s", StandardScaler()), ("m", LogisticRegression(
            C=C, max_iter=3000, class_weight="balanced"))])) for C in C_grid]
    if kind == "mlp":
        return [(f"mlp(256) alpha={a}", Pipeline([("s", StandardScaler()), ("m", MLPClassifier(
            (256,), alpha=a, max_iter=200, random_state=seed))])) for a in (1e-3, 1e-1)]
    raise ValueError(kind)


def raw_score(model, X) -> np.ndarray:
    if hasattr(model, "decision_function") and not hasattr(model, "predict_proba"):
        return model.decision_function(X)
    m = model[-1] if isinstance(model, Pipeline) else model
    if isinstance(m, SVC):
        return model.decision_function(X)
    p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class Fitted:
    name: str
    model: object
    sel_score: float
    val_f1: float
    val_swap_f1: float
    val_logloss: float
    tried: list = field(default_factory=list)


def select(cands, Xtr, ytr, Xval, yval, Xvs, yvs, n_jobs: int = -1) -> Fitted:
    """Fit every candidate on TRAIN, pick by mean(VAL F1, VAL-swap F1), tie-break VAL log-loss."""
    from joblib import Parallel, delayed

    def fit_one(name, m):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xtr, ytr)
        zv, zs = raw_score(m, Xval), raw_score(m, Xvs)
        f_v = f1_score(yval, (zv >= 0).astype(int), average="macro")
        f_s = f1_score(yvs, (zs >= 0).astype(int), average="macro") if len(Xvs) else f_v
        ll = log_loss(yval, 1 / (1 + np.exp(-np.clip(zv, -30, 30))), labels=[0, 1])
        return name, m, (f_v + f_s) / 2, f_v, f_s, ll

    res = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(fit_one)(n, m) for n, m in cands)
    best = sorted(res, key=lambda r: (-round(r[2], 4), r[5]))[0]
    tried = [{"candidate": r[0], "sel": r[2], "val_f1": r[3], "val_swap_f1": r[4], "val_logloss": r[5]} for r in res]
    return Fitted(best[0], best[1], best[2], best[3], best[4], best[5], tried)
