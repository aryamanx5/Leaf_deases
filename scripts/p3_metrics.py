"""P3-M03 metrics: results/disease_models.json from results/p3/preds_disease.csv.gz.

Per arm: the configuration (regime pooled/percrop, data orig/swap, head) with the best
selection score = mean macro-F1 on clean VAL and VAL-swap (tie-break: VAL+VAL-swap ECE);
F_fusion picks its backbone the same way. Metrics (accuracy, precision, recall, macro-F1,
ROC-AUC, ECE) for TEST, wild (all disease-labelled wild images) and wild_test (the half of
the wild set Phase 2 never used for selection), pooled and per crop, with grouped bootstrap
95% CIs (1000 resamples by near-duplicate group). Inference ms/image from results/p3/timing.json.
The wild set is reporting-only: nothing here feeds back into training or selection.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from leafdx.config import load_config, path
from leafdx.p3.stats import evaluate, point_metrics

CI_SETS = ("test", "wild", "wild_test")
OTHER_SETS = ("val", "val_swap", "test_swap", "test_leafonly", "wild_leafonly")
SCOPES = ("pooled", "turmeric", "blackgram")
ARM_ORDER = ["A_rule", "B_svm", "B2_lgbm", "C1_mobilenet_v2", "C2_efficientnet_b0", "C3_convnext_tiny",
             "D_dinov2", "E_openclip", "E_bioclip2", "F_fusion"]


def subset(P: pd.DataFrame, cid: str, s: str) -> pd.DataFrame:
    if s == "wild_test":
        return P[(P.config == cid) & (P.set == "wild") & (P.wild_half == "wild_test")]
    return P[(P.config == cid) & (P.set == s)]


def selection_score(P, cid) -> tuple[float, float]:
    v = pd.concat([subset(P, cid, "val"), subset(P, cid, "val_swap")])
    f = np.mean([point_metrics(subset(P, cid, s).y, subset(P, cid, s).p_cal)["macro_f1"] for s in ("val", "val_swap")])
    return float(f), float(point_metrics(v.y, v.p_cal)["ece"])


def arm_of(cid: str) -> str:
    return "F_fusion" if cid.startswith("F_fusion") else cid.split("[")[0]


def main() -> int:
    cfg = load_config()
    P = pd.read_csv(path("results", cfg) / "p3" / "preds_disease.csv.gz", keep_default_na=False)
    P["wild_half"] = P.wild_half.astype(str)
    log = json.loads((path("results", cfg) / "p3" / "disease_configs.json").read_text())
    timing = json.loads((path("results", cfg) / "p3" / "timing.json").read_text()) \
        if (path("results", cfg) / "p3" / "timing.json").exists() else {}
    nb, seed = cfg["p3"]["bootstrap"], cfg["seed"]

    sel = {cid: selection_score(P, cid) for cid in P.config.unique()}
    all_cfg = {}
    for cid in sel:
        all_cfg[cid] = {"arm": arm_of(cid), "selection_f1": sel[cid][0], "selection_ece": sel[cid][1],
                        "temperature": log.get(cid, {}).get("temperature"),
                        **{s: point_metrics(subset(P, cid, s).y, subset(P, cid, s).p_cal) for s in (*CI_SETS, *OTHER_SETS)
                           if len(subset(P, cid, s))}}
    arms = {}
    for arm in ARM_ORDER:
        cids = [c for c in sel if arm_of(c) == arm]
        skipped = {c: v.get("skipped") for c, v in log.items() if v.get("arm", "").startswith(arm) and "skipped" in v}
        if not cids:
            arms[arm] = {"skipped": skipped or "no configuration"}
            continue
        best = sorted(cids, key=lambda c: (-round(sel[c][0], 4), sel[c][1]))[0]
        out = {"selected_config": best, "selection_f1": sel[best][0], "selection_ece": sel[best][1],
               "temperature": log[best].get("temperature"), "configs_considered": len(cids), "skipped_configs": skipped,
               "inference_ms_per_image": timing.get("arms", {}).get(arm, {}).get("ms_per_image")}
        for s in CI_SETS:
            d = subset(P, best, s)
            out[s] = {}
            for sc in SCOPES:
                g = d if sc == "pooled" else d[d.crop == sc]
                if len(g) and g.y.nunique() == 2:
                    m = evaluate(g.y, g.p_cal, g.group, nb, seed)
                    m["ece_before_temperature"] = point_metrics(g.y, g.p_raw)["ece"]
                    out[s][sc] = m
        arms[arm] = out
        print(f"{arm:20s} {best:40s} sel {sel[best][0]:.3f} | test F1 {out['test']['pooled']['macro_f1']:.3f} "
              f"| wild F1 {out['wild']['pooled']['macro_f1']:.3f} AUC {out['wild']['pooled']['roc_auc']:.3f}", flush=True)
    res = {"selection_rule": "per arm: max mean macro-F1 over clean VAL and VAL-swap (tie-break lower ECE); wild "
                             "reporting only. Probabilities temperature-scaled on VAL + VAL-swap.",
           "sets": {"test": "clean grouped TEST", "wild": "all disease-labelled wild images (Mendeley CC BY sets)",
                    "wild_test": "wild_test half only (never used by Phase 2 either)"},
           "n": {s: int(len(subset(P, next(iter(sel)), s))) for s in CI_SETS},
           "bootstrap": {"resamples": nb, "unit": "near-duplicate group", "ci": "95% percentile"},
           "small_sample_warning": "wild CIs rest on 125 labelled images (55 turmeric / 70 black gram; wild_test about "
                                   "half of that) from two Bangladeshi datasets: treat them as unreliable",
           "arms": arms, "all_configs": all_cfg}
    (path("results", cfg) / "disease_models.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
