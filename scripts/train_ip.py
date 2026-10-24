"""Fit the Stage 1 decisions (rule-based score + SVM) and report VAL and TEST metrics.

Rule: scaling from TRAIN, threshold from VAL. SVM: hyperparameters by grouped CV on
TRAIN, fitted on TRAIN. TEST is only scored, never used for any choice.
Outputs: models/ip_rule.joblib, models/ip_svm.joblib, results/ip_metrics.json,
results/ip_predictions.csv.
"""

from __future__ import annotations

import json
import sys
import time

import joblib
import numpy as np
import pandas as pd

from leafdx.config import ROOT, load_config, path, seed_everything
from leafdx.ip.decide import RuleBasedScorer, fit_svm, svm_matrix
from leafdx.ip.features import FEATURE_NAMES, stage1
from leafdx.ip.preprocess import load_bgr
from leafdx.metrics import with_per_crop


def time_inference(sample: pd.DataFrame, rule: RuleBasedScorer, svm, cfg: dict) -> dict:
    """Sequential single-process timing of the full Stage 1 path per image."""
    stage_ms, decide_ms = [], []
    for r in sample.itertuples():
        img = load_bgr(ROOT / r.path)
        t0 = time.perf_counter()
        out = stage1(img, cfg, r.crop)
        t1 = time.perf_counter()
        row = pd.DataFrame([{**out["features"], "crop": r.crop}])
        rule.predict(row)
        svm.predict(svm_matrix(row))
        t2 = time.perf_counter()
        stage_ms.append(1000 * (t1 - t0))
        decide_ms.append(1000 * (t2 - t1))
    return {
        "n_images": len(sample),
        "stage1_ms_median": float(np.median(stage_ms)),
        "stage1_ms_mean": float(np.mean(stage_ms)),
        "decision_ms_median": float(np.median(decide_ms)),
        "total_ms_median": float(np.median(np.add(stage_ms, decide_ms))),
        "note": "single process, image already decoded; includes pre-processing, segmentation, lesions, features",
    }


def main() -> int:
    cfg = load_config()
    seed_everything(cfg["seed"])
    df = pd.read_csv(path("features", cfg))
    df["y"] = (df["label"] == "diseased").astype(int)
    train, val, test = (df[df.split == s].copy() for s in ("train", "val", "test"))
    print(f"train {len(train)}  val {len(val)}  test {len(test)}")

    # (a) rule-based score
    rc = cfg["decide"]["rule"]
    grid = np.linspace(rc["threshold_grid"]["start"], rc["threshold_grid"]["stop"], rc["threshold_grid"]["num"])
    rule = RuleBasedScorer(rc["weights"]).fit_scaling(train).fit_threshold(val, grid)
    # (b) SVM
    t0 = time.time()
    search = fit_svm(train, cfg)
    svm = search.best_estimator_
    print(f"SVM grid search {time.time() - t0:.0f}s best {search.best_params_} CV macro-F1 {search.best_score_:.3f}")

    preds = []
    for name, part in (("val", val), ("test", test)):
        part = part.assign(rule_score=rule.score(part).to_numpy(), rule_pred=rule.predict(part),
                           svm_score=svm.decision_function(svm_matrix(part)),
                           svm_pred=svm.predict(svm_matrix(part)))
        preds.append(part)
    pred = pd.concat(preds)
    keep = ["path", "split", "crop", "label", "orig_class", "y", "rule_score", "rule_pred", "svm_score", "svm_pred"]
    pred[keep].to_csv(path("results", cfg) / "ip_predictions.csv", index=False)

    timing = time_inference(val.sample(cfg["decide"]["timing_images"], random_state=cfg["seed"]), rule, svm, cfg)
    cvres = pd.DataFrame(search.cv_results_)
    metrics = {
        "rule_based": {
            "weights": rule.weights,
            "thresholds_from_val": rule.threshold,
            "val": with_per_crop(pred[pred.split == "val"], "rule_pred", "rule_score"),
            "test": with_per_crop(pred[pred.split == "test"], "rule_pred", "rule_score"),
        },
        "svm": {
            "best_params": {k: (v if isinstance(v, str) else float(v)) for k, v in search.best_params_.items()},
            "cv_macro_f1_train": float(search.best_score_),
            "cv_grid": cvres[["params", "mean_test_score", "std_test_score"]].astype({"params": str}).to_dict(orient="records"),
            "n_features": len(FEATURE_NAMES) + 1,
            "val": with_per_crop(pred[pred.split == "val"], "svm_pred", "svm_score"),
            "test": with_per_crop(pred[pred.split == "test"], "svm_pred", "svm_score"),
        },
        "inference_time": timing,
    }
    (path("results", cfg) / "ip_metrics.json").write_text(json.dumps(metrics, indent=2))
    mdir = path("models", cfg)
    joblib.dump(rule, mdir / "ip_rule.joblib")
    joblib.dump({"model": svm, "features": FEATURE_NAMES + ["is_turmeric"], "best_params": search.best_params_},
                mdir / "ip_svm.joblib")

    for m in ("rule_based", "svm"):
        for s in ("val", "test"):
            o = metrics[m][s]["overall"]
            pc = {c: round(v["macro_f1"], 3) for c, v in metrics[m][s]["per_crop"].items()}
            print(f"{m:10s} {s:4s} acc {o['accuracy']:.3f} P {o['precision']:.3f} R {o['recall']:.3f} "
                  f"F1 {o['f1']:.3f} macroF1 {o['macro_f1']:.3f} AUC {o.get('roc_auc', float('nan')):.3f} "
                  f"CM {o['confusion_matrix']} per-crop macroF1 {pc}")
    print(f"rule thresholds (VAL): {rule.threshold}")
    print(f"inference: stage1 median {timing['stage1_ms_median']:.0f} ms + decision {timing['decision_ms_median']:.1f} ms")
    print("saved models/ip_rule.joblib, models/ip_svm.joblib, results/ip_metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
