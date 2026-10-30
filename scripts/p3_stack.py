"""P3-M04: stacking ensemble.

Meta-learner: logistic regression over the calibrated log-odds of each arm's selected
configuration (A-F). It is fitted only on VAL + VAL-swap predictions; every base model was
trained on TRAIN only, so none of them saw those images. (VAL was also used to select the
base configurations and temperatures; that optimism is shared by every arm and is noted.)

Comparison with the best single arm (highest selection score) on TEST and on the wild set:
exact McNemar test and a grouped paired-bootstrap CI of the macro-F1 difference. If the CI
contains 0 the result is "no significant difference" and the simpler model is preferred.
Writes results/stacking.json and adds STACK rows to results/p3/preds_disease.csv.gz.
"""

from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from leafdx.config import load_config, path
from leafdx.p3.stats import evaluate, logit, mcnemar, paired_bootstrap_diff, point_metrics

SETS = ["val", "val_swap", "test", "test_swap", "wild", "user", "val_leafonly", "test_leafonly", "wild_leafonly",
        "user_leafonly"]


def main() -> int:
    cfg = load_config()
    out = path("results", cfg)
    P = pd.read_csv(out / "p3" / "preds_disease.csv.gz", keep_default_na=False)
    P = P[P.arm != "STACK"]
    P["wild_half"] = P.wild_half.astype(str)
    D = json.loads((out / "disease_models.json").read_text())
    members = {a: v["selected_config"] for a, v in D["arms"].items() if "selected_config" in v}
    cols = list(members)

    def matrix(s):
        parts = []
        for a, cid in members.items():
            d = P[(P.config == cid) & (P.set == s)].reset_index(drop=True)
            parts.append(d.assign(col=a))
        base = parts[0][["path", "crop", "label", "y", "group", "wild_half"]]
        X = np.column_stack([logit(p.p_cal.to_numpy()) for p in parts])
        return base, X

    Xs = {s: matrix(s) for s in SETS}
    Xfit = np.r_[Xs["val"][1], Xs["val_swap"][1]]
    yfit = np.r_[Xs["val"][0].y, Xs["val_swap"][0].y]
    meta = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xfit, yfit)
    rows = []
    for s in SETS:
        b, X = Xs[s]
        p = meta.predict_proba(X)[:, 1]
        rows.append(b.assign(arm="STACK", config="STACK", set=s, z=logit(p), p_raw=p, p_cal=p))
    S = pd.concat(rows, ignore_index=True)
    pd.concat([P, S[P.columns]], ignore_index=True).to_csv(out / "p3" / "preds_disease.csv.gz", index=False)
    joblib.dump({"members": members, "meta": meta}, path("models", cfg) / "p3" / "stacker.joblib")

    best = max(members, key=lambda a: (round(D["arms"][a]["selection_f1"], 4), -D["arms"][a]["selection_ece"]))
    res = {"members": members, "weights": dict(zip(cols, meta.coef_[0].tolist(), strict=True)),
           "intercept": float(meta.intercept_[0]), "fit_on": "VAL + VAL-swap predictions",
           "best_single_arm": best, "best_single_config": members[best],
           "selection_f1": {"STACK": float(np.mean([point_metrics(Xs[s][0].y, meta.predict_proba(Xs[s][1])[:, 1])["macro_f1"]
                                                    for s in ("val", "val_swap")])),
                            best: D["arms"][best]["selection_f1"]}, "comparison": {}}
    nb, seed = cfg["p3"]["bootstrap"], cfg["seed"]
    for s in ("test", "wild", "wild_test"):
        ss = "wild" if s == "wild_test" else s
        b, X = Xs[ss]
        m = (b.wild_half == "wild_test").to_numpy() if s == "wild_test" else np.ones(len(b), bool)
        ps = meta.predict_proba(X)[:, 1][m]
        pb = P[(P.config == members[best]) & (P.set == ss)].reset_index(drop=True).p_cal.to_numpy()[m]
        y, g = b.y.to_numpy()[m], b.group.to_numpy()[m]
        diff = paired_bootstrap_diff(y, ps, pb, g, nb, seed)
        res["comparison"][s] = {"n": int(m.sum()), "STACK": evaluate(y, ps, g, nb, seed), best: evaluate(y, pb, g, nb, seed),
                                "mcnemar_stack_vs_best": mcnemar(y, (ps >= 0.5).astype(int), (pb >= 0.5).astype(int)),
                                "macro_f1_diff_stack_minus_best": diff,
                                "verdict": "no significant difference" if diff["ci_contains_0"] else
                                ("stack better" if diff["diff"] > 0 else "best single arm better")}
        print(s, res["comparison"][s]["verdict"], f"diff {diff['diff']:+.3f} CI {diff['ci95']}",
              f"McNemar p={res['comparison'][s]['mcnemar_stack_vs_best']['p_value']:.3f}", flush=True)
    sig_better = all(res["comparison"][s]["verdict"] == "stack better" for s in ("test", "wild"))
    res["production_choice"] = "STACK" if sig_better else best
    res["production_reason"] = ("stack significantly better on TEST and wild" if sig_better else
                                "stack not significantly better on both TEST and wild -> prefer the simpler single arm")
    (out / "stacking.json").write_text(json.dumps(res, indent=1))
    print("production:", res["production_choice"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
