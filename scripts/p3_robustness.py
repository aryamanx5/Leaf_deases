"""P3-M07: augmentation effect and shortcut test.

* Background-swap + strong colour/blur/JPEG augmentation: each fine-tuned CNN arm trained
  with it (swap) vs the v1 augmentation (plain); frozen/fusion arms trained on TRAIN photos
  (orig) vs photos + swap views (swap), in the arm's selected regime. Effect on wild and TEST
  macro-F1 with a grouped paired-bootstrap CI.
* Shortcut test: every selected arm (and the crop model) on leaf-only images (Stage 1 leaf
  mask, background set to grey) vs full images. A large drop (> 0.10 macro-F1) means the
  model leans on the background.
* Few-shot adaptation with wild images: not run (the wild set stays evaluation-only).
Writes results/robustness_p3.json.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from leafdx.config import load_config, path
from leafdx.p3.stats import paired_bootstrap_diff, point_metrics

LARGE_DROP = 0.10


def main() -> int:
    cfg = load_config()
    res = path("results", cfg)
    P = pd.read_csv(res / "p3" / "preds_disease.csv.gz", keep_default_na=False)
    D = json.loads((res / "disease_models.json").read_text())
    nb, seed = cfg["p3"]["bootstrap"], cfg["seed"]
    out = {"large_drop_threshold": LARGE_DROP, "augmentation": {}, "shortcut_disease": {}, "shortcut_crop": {},
           "few_shot_adaptation": "not run: the wild set is reserved for evaluation (no adaptation pool carved out)"}

    def pair(c_a, c_b, s):
        a = P[(P.config == c_a) & (P.set == s)].reset_index(drop=True)
        b = P[(P.config == c_b) & (P.set == s)].set_index("path").loc[a.path].reset_index()
        return a, b

    for arm, v in D["arms"].items():
        if "selected_config" not in v:
            continue
        cid = v["selected_config"]
        if arm.startswith("C"):
            base = arm
            sw, pl = f"{base}[swap]", f"{base}[plain]"
        else:
            if arm == "A_rule":
                continue
            mode = cid.split("[")[1].split(",")[0]
            stem = cid.split("[")[0]
            sw, pl = f"{stem}[{mode},swap]", f"{stem}[{mode},orig]"
        if not ((P.config == sw).any() and (P.config == pl).any()):
            continue
        r = {"with_swap": sw, "without_swap": pl}
        for s in ("test", "wild"):
            a, b = pair(sw, pl, s)
            r[s] = {"macro_f1_with": point_metrics(a.y, a.p_cal)["macro_f1"],
                    "macro_f1_without": point_metrics(b.y, b.p_cal)["macro_f1"],
                    "diff_with_minus_without": paired_bootstrap_diff(a.y, a.p_cal, b.p_cal, a.group, nb, seed)}
        out["augmentation"][arm] = r
        print("aug", arm, f"wild {r['wild']['macro_f1_without']:.3f} -> {r['wild']['macro_f1_with']:.3f}", flush=True)

    for arm, v in list(D["arms"].items()) + [("STACK", {"selected_config": "STACK"})]:
        if "selected_config" not in v or not (P.config == v["selected_config"]).any():
            continue
        cid = v["selected_config"]
        r = {"config": cid}
        for s in ("test", "wild"):
            full, lo = pair(cid, cid, s)[0], P[(P.config == cid) & (P.set == f"{s}_leafonly")]
            lo = lo.set_index("path").loc[full.path].reset_index()
            mf, ml = point_metrics(full.y, full.p_cal), point_metrics(lo.y, lo.p_cal)
            r[s] = {"macro_f1_full": mf["macro_f1"], "macro_f1_leafonly": ml["macro_f1"],
                    "accuracy_full": mf["accuracy"], "accuracy_leafonly": ml["accuracy"],
                    "drop_macro_f1": mf["macro_f1"] - ml["macro_f1"],
                    "large_drop": bool(mf["macro_f1"] - ml["macro_f1"] > LARGE_DROP)}
        out["shortcut_disease"][arm] = r
        print("shortcut", arm, {s: round(r[s]["drop_macro_f1"], 3) for s in ("test", "wild")}, flush=True)

    C = json.loads((res / "crop_models.json").read_text())
    PC = pd.read_csv(res / "p3" / "preds_crop.csv.gz", keep_default_na=False)
    for n in C["candidates"]:
        x = PC[PC.config == n]
        r = {}
        for s, slo in (("test", "test_leafonly"), ("wild_all", "wild_all_leafonly")):
            f, lo = x[x.set == s], x[x.set == slo].set_index("path").loc[x[x.set == s].path].reset_index()
            yf = (f.crop == "turmeric").astype(int).to_numpy()
            mf, ml = point_metrics(yf, f.p_turmeric), point_metrics(yf, lo.p_turmeric)
            r[s] = {"accuracy_full": mf["accuracy"], "accuracy_leafonly": ml["accuracy"],
                    "macro_f1_full": mf["macro_f1"], "macro_f1_leafonly": ml["macro_f1"],
                    "drop_macro_f1": mf["macro_f1"] - ml["macro_f1"],
                    "large_drop": bool(mf["macro_f1"] - ml["macro_f1"] > LARGE_DROP)}
        out["shortcut_crop"][n] = r
    out["chosen_crop_model"] = C["chosen"]
    (res / "robustness_p3.json").write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
