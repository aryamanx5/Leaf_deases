"""Phase 5 helpers (docs/PLAN_phase5_confidence.md): the mean-of-logits ensemble and risk-targeted refusal
thresholds. Shared by scripts/p5_select.py, scripts/p5_eval.py, the production loader and the tests."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta

from leafdx.p3.stats import logit, sigmoid


class MeanLogit:
    """Combines member models by the plain mean of their log-odds: no learned weights.

    It stands where Phase 3's stacking meta-model stood in the production bundle, so it speaks the same interface:
    `predict_proba(X)` takes one column of member logits per member and returns [[P(healthy), P(diseased)], ...].
    """

    def predict_proba(self, X) -> np.ndarray:
        p = sigmoid(np.asarray(X, float).mean(1))
        return np.column_stack([1 - p, p])

    def __repr__(self) -> str:
        return "MeanLogit()"


def ensemble_logit(member_probs: list[np.ndarray]) -> np.ndarray:
    """Mean of the members' log-odds, from their temperature-scaled probabilities (as production computes it)."""
    return np.mean([logit(p) for p in member_probs], axis=0)


def nll(p_pos, y) -> float:
    """Mean negative log-likelihood (nats) of binary labels under P(positive) = p_pos."""
    p = np.clip(np.asarray(p_pos, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, int)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def error_upper_bound(errors: int, n: int, level: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper confidence bound on an error rate, given `errors` wrong out of `n`."""
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    return float(beta.ppf(level, errors + 1, n - errors))


def selective_threshold(conf, correct, lo: float, hi: float, max_error: float, level: float = 0.95,
                        extra_ok=None) -> tuple[float, dict]:
    """Lowest confidence threshold tau in [lo, hi] such that, among rows with conf >= tau, the Clopper-Pearson upper
    bound on the error rate is at most `max_error` (and `extra_ok(tau)` holds, if given). Falls back to `hi`.

    Candidate values are `lo`, `hi` and every observed confidence in between: the answered set only changes there.
    Returns (tau, record of the chosen point).
    """
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    grid = sorted({lo, hi, *[float(c) for c in conf if lo <= c <= hi]})
    chosen = None
    for tau in grid:
        m = conf >= tau
        n, e = int(m.sum()), int((~correct[m]).sum())
        ub = error_upper_bound(e, n, level)
        if ub <= max_error and (extra_ok is None or extra_ok(tau)):
            chosen = (tau, {"tau": tau, "answered": n, "errors": e, "error_upper_bound": ub, "from_rule": True})
            break
    if chosen is None:
        m = conf >= hi
        n, e = int(m.sum()), int((~correct[m]).sum())
        chosen = (hi, {"tau": hi, "answered": n, "errors": e, "error_upper_bound": error_upper_bound(e, n, level),
                       "from_rule": False})
    chosen[1]["n"] = int(len(conf))
    chosen[1]["answered_share"] = chosen[1]["answered"] / max(1, len(conf))
    return chosen


def confidence_summary(p_pos, y) -> dict:
    """How much probability sits on the answer given: mean confidence overall, on right and on wrong answers."""
    p_pos, y = np.asarray(p_pos, float), np.asarray(y, int)
    conf = np.maximum(p_pos, 1 - p_pos)
    right = (p_pos >= 0.5).astype(int) == y
    return {"mean_confidence": float(conf.mean()),
            "mean_confidence_right": float(conf[right].mean()) if right.any() else None,
            "mean_confidence_wrong": float(conf[~right].mean()) if (~right).any() else None,
            "share_conf_ge_0.9": float((conf >= 0.9).mean()), "share_conf_ge_0.99": float((conf >= 0.99).mean()),
            "nll": nll(p_pos, y)}
