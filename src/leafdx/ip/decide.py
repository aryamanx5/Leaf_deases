"""Stage 1 decisions on the feature vector.

(a) Rule-based disease score: a transparent, fixed-weight combination of four
    measurements, each min-max scaled per crop by its TRAIN 1st-99th percentile:
        score = sum_i w_i * clip((x_i - p1_i) / (p99_i - p1_i), 0, 1)
    with x = affected-%, log(1 + lesion count), abnormal-colour % (leaf pixels outside
    the green hue range) and GLCM contrast. Weights come from config (equal by default,
    not fitted). The only learned quantity is the per-crop decision threshold, chosen on
    VAL by a grid search maximising macro-F1.
(b) SVM: StandardScaler + RBF SVC (class_weight="balanced") on all features plus a crop
    indicator; C and gamma by grouped (cluster-aware) stratified CV on TRAIN only. The
    SVM score is its decision function (signed distance; > 0 means diseased).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from leafdx.ip.features import FEATURE_NAMES

RULE_INPUTS = ("affected_pct", "log_lesion_count", "abn_colour_pct", "glcm_contrast")


def rule_inputs(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "affected_pct": df["affected_pct"],
        "log_lesion_count": np.log1p(df["lesion_count"]),
        "abn_colour_pct": 100.0 * (1.0 - df["green_frac"]),
        "glcm_contrast": df["glcm_contrast"],
    }, index=df.index)


class RuleBasedScorer:
    """Fixed-weight disease score with per-crop scaling and per-crop VAL threshold."""

    def __init__(self, weights: dict[str, float]):
        self.weights = {k: float(weights[k]) for k in RULE_INPUTS}
        self.scale: dict[str, dict] = {}
        self.threshold: dict[str, float] = {}

    def fit_scaling(self, train: pd.DataFrame) -> RuleBasedScorer:
        X = rule_inputs(train)
        for crop, idx in train.groupby("crop").groups.items():
            lo, hi = X.loc[idx].quantile(0.01), X.loc[idx].quantile(0.99)
            self.scale[crop] = {k: (float(lo[k]), float(max(hi[k], lo[k] + 1e-9))) for k in RULE_INPUTS}
        return self

    def components(self, df: pd.DataFrame) -> pd.DataFrame:
        X = rule_inputs(df)
        out = pd.DataFrame(index=df.index, columns=list(RULE_INPUTS), dtype=float)
        for crop, idx in df.groupby("crop").groups.items():
            sc = self.scale[crop]
            for k in RULE_INPUTS:
                lo, hi = sc[k]
                out.loc[idx, k] = ((X.loc[idx, k] - lo) / (hi - lo)).clip(0, 1)
        return out

    def score(self, df: pd.DataFrame) -> pd.Series:
        c = self.components(df)
        total = sum(self.weights.values())
        return sum(self.weights[k] * c[k] for k in RULE_INPUTS) / total

    def fit_threshold(self, val: pd.DataFrame, grid: np.ndarray) -> RuleBasedScorer:
        s = self.score(val)
        y = (val["label"] == "diseased").astype(int)
        for crop, idx in val.groupby("crop").groups.items():
            f1s = [f1_score(y[idx], (s[idx] >= t).astype(int), average="macro", zero_division=0) for t in grid]
            self.threshold[crop] = float(grid[int(np.argmax(f1s))])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        s = self.score(df)
        t = df["crop"].map(self.threshold)
        return (s >= t).astype(int).to_numpy()


def svm_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([df[FEATURE_NAMES].to_numpy(float), (df["crop"] == "turmeric").to_numpy(float)])


def fit_svm(train: pd.DataFrame, cfg: dict) -> GridSearchCV:
    d = cfg["decide"]["svm"]
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight="balanced", random_state=cfg["seed"])),
    ])
    grid = {"svc__C": d["C"], "svc__gamma": d["gamma"]}
    X, y = svm_matrix(train), (train["label"] == "diseased").astype(int).to_numpy()
    # folds keep near-duplicate clusters together and are stratified by crop+label
    sgkf = StratifiedGroupKFold(n_splits=d["cv_folds"], shuffle=True, random_state=cfg["seed"])
    folds = list(sgkf.split(X, train["stratum"], groups=train["cluster_id"]))
    search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=folds, n_jobs=-1, refit=True)
    search.fit(X, y)
    return search


def load_ip_models(model_dir) -> tuple[RuleBasedScorer, dict]:
    import joblib

    return joblib.load(model_dir / "ip_rule.joblib"), joblib.load(model_dir / "ip_svm.joblib")


def predict_ip(features: dict, crop: str, rule: RuleBasedScorer, svm_bundle: dict) -> dict:
    """Rule-based and SVM decisions for one feature dict (from features.stage1)."""
    row = pd.DataFrame([{**features, "crop": crop}])
    X = svm_matrix(row)
    svm = svm_bundle["model"]
    return {
        "rule_score": float(rule.score(row).iloc[0]),
        "rule_threshold": float(rule.threshold[crop]),
        "rule_pred": int(rule.predict(row)[0]),
        "rule_components": {k: float(v) for k, v in rule.components(row).iloc[0].items()},
        "svm_score": float(svm.decision_function(X)[0]),
        "svm_pred": int(svm.predict(X)[0]),
    }
