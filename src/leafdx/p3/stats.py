"""Phase 3 evaluation statistics: metrics with grouped bootstrap CIs, ECE, McNemar,
temperature scaling, split-conformal prediction sets, abstention curves."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def ece(p_pos, y, bins: int = 10) -> float:
    """Binary expected calibration error over confidence = max(p, 1-p), equal-width bins."""
    p_pos, y = np.asarray(p_pos, float), np.asarray(y, int)
    conf = np.maximum(p_pos, 1 - p_pos)
    correct = ((p_pos >= 0.5).astype(int) == y)
    edges = np.linspace(0.5, 1.0, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < bins - 1 else (conf <= edges[i + 1]))
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def point_metrics(y, p_pos, thr: float = 0.5) -> dict:
    y, p_pos = np.asarray(y, int), np.asarray(p_pos, float)
    pred = (p_pos >= thr).astype(int)
    out = {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
           "ece": ece(p_pos, y)}
    out["roc_auc"] = float(roc_auc_score(y, p_pos)) if len(np.unique(y)) == 2 else None
    return out


def fast_metrics(y: np.ndarray, p_pos: np.ndarray, thr: float = 0.5) -> dict:
    """numpy-only accuracy / macro-F1 / ROC-AUC / ECE (same definitions as point_metrics)."""
    from scipy.stats import rankdata

    pred = p_pos >= thr
    yb = y.astype(bool)
    tp, tn = float((pred & yb).sum()), float((~pred & ~yb).sum())
    fp, fn = float((pred & ~yb).sum()), float((~pred & yb).sum())
    f1p = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    f1n = 2 * tn / (2 * tn + fn + fp) if tn + fn + fp else 0.0
    npos, nneg = yb.sum(), (~yb).sum()
    auc = None
    if npos and nneg:
        r = rankdata(p_pos)
        auc = float((r[yb].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return {"accuracy": (tp + tn) / len(y), "macro_f1": (f1p + f1n) / 2, "roc_auc": auc, "ece": ece(p_pos, y)}


def grouped_bootstrap(y, p_pos, groups, n_boot: int = 1000, seed: int = 0, thr: float = 0.5,
                      keys=("accuracy", "macro_f1", "roc_auc", "ece")) -> dict:
    """95% percentile CIs, resampling whole near-duplicate groups with replacement."""
    y, p_pos, groups = np.asarray(y, int), np.asarray(p_pos, float), np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(ug))]
    rng = np.random.default_rng(seed)
    vals: dict = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(ug), len(ug))])
        m = fast_metrics(y[idx], p_pos[idx], thr)
        for k in keys:
            if m.get(k) is not None:
                vals[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 10 else None
            for k, v in vals.items()}


def evaluate(y, p_pos, groups, n_boot: int = 1000, seed: int = 0) -> dict:
    out = point_metrics(y, p_pos)
    out["ci95"] = grouped_bootstrap(y, p_pos, groups, n_boot, seed)
    return out


def paired_bootstrap_diff(y, pa, pb, groups, n_boot: int = 1000, seed: int = 0, metric: str = "macro_f1") -> dict:
    """CI of metric(a) - metric(b) on the same resampled groups."""
    y, pa, pb, groups = np.asarray(y, int), np.asarray(pa, float), np.asarray(pb, float), np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(ug))]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(ug), len(ug))])
        d.append(fast_metrics(y[idx], pa[idx])[metric] - fast_metrics(y[idx], pb[idx])[metric])
    diff = point_metrics(y, pa)[metric] - point_metrics(y, pb)[metric]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"metric": metric, "diff": float(diff), "ci95": [float(lo), float(hi)], "ci_contains_0": bool(lo <= 0 <= hi)}


def mcnemar(y, pred_a, pred_b) -> dict:
    """Exact (binomial) McNemar test on discordant pairs."""
    y, a, b = (np.asarray(v, int) for v in (y, pred_a, pred_b))
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": p}


# -- calibration -------------------------------------------------------------------------

def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def fit_temperature(z, y) -> float:
    """T minimising binary NLL of sigmoid(z / T); searched over log T in [-3, 3].

    If the calibration set is perfectly separated (every prediction right), NLL keeps falling as
    T -> 0, so the fit only says "no errors seen", not how sharp to be: then T is not allowed
    below 1 (no sharpening beyond the raw model)."""
    z, y = np.asarray(z, float), np.asarray(y, int)
    separable = bool(((z >= 0).astype(int) == y).all())

    def nll(lt):
        p = np.clip(sigmoid(z / np.exp(lt)), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    T = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
    return max(T, 1.0) if separable else T


def fit_temperature_multi(logits, y) -> float:
    logits, y = np.asarray(logits, float), np.asarray(y, int)

    def nll(lt):
        z = logits / np.exp(lt)
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean()

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def softmax(z):
    z = np.asarray(z, float)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# -- conformal prediction --------------------------------------------------------------

def conformal_threshold(probs_cal, y_cal, alpha: float, method: str = "lac") -> float:
    """Split-conformal score threshold q-hat for (1 - alpha) marginal coverage.

    LAC: score = 1 - p(true class); set = {k : 1 - p_k <= q}.
    APS: score = cumulative mass of classes ranked above and including the true class.
    """
    probs_cal, y_cal = np.asarray(probs_cal, float), np.asarray(y_cal, int)
    n = len(y_cal)
    if method == "lac":
        s = 1 - probs_cal[np.arange(n), y_cal]
    else:
        order = np.argsort(-probs_cal, 1)
        cum = np.cumsum(np.take_along_axis(probs_cal, order, 1), 1)
        rank = np.argmax(order == y_cal[:, None], 1)
        s = cum[np.arange(n), rank]
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


def conformal_sets(probs, q: float, method: str = "lac") -> np.ndarray:
    """Boolean (n, K) membership matrix."""
    probs = np.asarray(probs, float)
    if method == "lac":
        return (1 - probs) <= q
    order = np.argsort(-probs, 1)
    cum = np.cumsum(np.take_along_axis(probs, order, 1), 1)
    keep_sorted = np.concatenate([np.ones((len(probs), 1), bool), cum[:, :-1] < q], 1)
    out = np.zeros_like(probs, bool)
    np.put_along_axis(out, order, keep_sorted, 1)
    return out


def abstention_curve(conf, correct, points: int = 21) -> list[dict]:
    """Accuracy on the most-confident fraction `coverage` of samples."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    order = np.argsort(-conf, kind="stable")
    out = []
    for cov in np.linspace(0.05, 1.0, points - 1).tolist() + [1.0]:
        k = max(1, int(round(cov * len(conf))))
        out.append({"coverage": round(float(k / len(conf)), 4), "accuracy": float(correct[order[:k]].mean()),
                    "min_conf": float(conf[order[k - 1]])})
    return out[:-1]
