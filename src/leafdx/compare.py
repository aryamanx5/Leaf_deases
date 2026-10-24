"""Rule-based vs SVM vs CNN on the identical TEST split, plus a leakage ablation.

All reported numbers are collected in one long table (section, method, scope, metric,
value) -> results/comparison.csv; the markdown files are rendered from that table so
every number they show is present in comparison.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, f1_score

from leafdx.metrics import binary_metrics

METHODS = {  # method -> (prediction column, score column)
    "rule_based": ("rule_pred", "rule_score"),
    "svm": ("svm_pred", "svm_score"),
    "cnn": ("cnn_pred", "cnn_p_diseased"),
}
METRICS = ("accuracy", "precision", "recall", "f1", "macro_f1", "healthy_recall", "roc_auc", "pr_auc")


def merge_test_predictions(ip_pred: pd.DataFrame, cnn_pred: pd.DataFrame) -> pd.DataFrame:
    ip = ip_pred[ip_pred.split == "test"]
    cnn = cnn_pred[cnn_pred.split == "test"][["path", "cnn_pred", "cnn_p_diseased", "cnn_conf", "cnn_pred4", "cnn_crop_pred"]]
    df = ip.merge(cnn, on="path", how="inner", validate="one_to_one")
    assert len(df) == len(ip) == len(cnn), "IP and CNN must be scored on the identical TEST split"
    return df


def bootstrap_ci(y, pred, n_boot: int, seed: int, fn=None) -> tuple[float, float]:
    fn = fn or (lambda a, b: f1_score(a, b, average="macro", zero_division=0))
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    vals = [fn(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(y, pred_a, pred_b) -> dict:
    """Exact McNemar test on the discordant pairs."""
    a_ok, b_ok = np.asarray(pred_a) == np.asarray(y), np.asarray(pred_b) == np.asarray(y)
    b01, b10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    n = b01 + b10
    p = 1.0 if n == 0 else float(binomtest(b01, n, 0.5).pvalue)
    return {"a_right_b_wrong": b01, "a_wrong_b_right": b10, "p_value": p}


def comparison_table(df: pd.DataFrame, timing_ms: dict, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []

    def add(section, method, scope, metric, value):
        rows.append({"section": section, "method": method, "scope": scope, "metric": metric, "value": float(value)})

    for scope, part in [("overall", df), *[(c, g) for c, g in df.groupby("crop")]]:
        for m, (pc, sc) in METHODS.items():
            met = binary_metrics(part.y, part[pc], part[sc])
            met["pr_auc"] = average_precision_score(part.y, part[sc])
            add("metrics", m, scope, "n", met["n"])
            for k in METRICS:
                add("metrics", m, scope, k, met[k])
            lo, hi = bootstrap_ci(part.y, part[pc], n_boot, seed)
            add("metrics", m, scope, "macro_f1_ci_low", lo)
            add("metrics", m, scope, "macro_f1_ci_high", hi)
            tn, fp, fn, tp = np.ravel(met["confusion_matrix"])
            for k, v in (("tn", tn), ("fp", fp), ("fn", fn), ("tp", tp)):
                add("confusion", m, scope, k, v)
    for m, ms in timing_ms.items():
        add("inference", m, "overall", "ms_per_image", ms)
    for cls, g in df.groupby("orig_class"):
        for m, (pc, _) in METHODS.items():
            add("per_class", m, cls, "accuracy", (g[pc] == g.y).mean())
            add("per_class", m, cls, "n", len(g))
    for a, b in (("svm", "cnn"), ("rule_based", "cnn"), ("rule_based", "svm")):
        r = mcnemar(df.y, df[METHODS[a][0]], df[METHODS[b][0]])
        for k, v in r.items():
            add("mcnemar", f"{a}_vs_{b}", "overall", k, v)
    agree = df["svm_pred"] == df["cnn_pred"]
    add("agreement", "svm_vs_cnn", "overall", "agreement_rate", agree.mean())
    add("agreement", "svm_vs_cnn", "overall", "accuracy_when_agree", (df.loc[agree, "cnn_pred"] == df.loc[agree, "y"]).mean())
    add("agreement", "svm_vs_cnn", "overall", "n_disagree", (~agree).sum())
    if "cnn_crop_pred" in df:
        add("metrics", "cnn", "overall", "crop_accuracy", (df["cnn_crop_pred"] == df["crop"]).mean())
    return pd.DataFrame(rows)


def get(tab: pd.DataFrame, section: str, method: str, scope: str, metric: str) -> float:
    v = tab[(tab.section == section) & (tab.method == method) & (tab.scope == scope) & (tab.metric == metric)].value
    if len(v) != 1:
        raise KeyError((section, method, scope, metric))
    return float(v.iloc[0])


def fmt(x: float, metric: str) -> str:
    """Canonical number formatting shared by comparison.csv consumers and the markdown."""
    if metric in ("n", "tn", "fp", "fn", "tp", "a_right_b_wrong", "a_wrong_b_right", "n_disagree", "n_images"):
        return f"{int(round(x))}"
    if metric == "p_value":
        return f"{x:.4f}"
    if metric.startswith("ms"):
        return f"{x:.1f}"
    return f"{x:.3f}"
