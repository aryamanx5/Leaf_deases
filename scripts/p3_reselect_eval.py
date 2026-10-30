"""Field-validated re-selection, stage 2 (docs/PLAN_field_validation.md): evaluate ONCE on the frozen test sets.

Requires results/reselection_selection.json and models/manifest_field.json from scripts/p3_reselect.py. Checks the
test files against results/test_manifest.json first. New (field-validated) vs old (Phase 3) production on wild_ext and
wild: macro-F1 with grouped bootstrap CIs, paired-bootstrap difference, exact McNemar. Switch rule (pre-registered):
switch iff the paired macro-F1 CI upper bound on wild_ext is >= 0; "better" only if its lower bound is > 0.
On a switch, models/manifest.json is replaced (the old one kept as models/manifest_phase3.json).
The candidate bundle's trust layer is scored by `p3_wild_ext.py --manifest models/manifest_field.json --out wild_ext_field`.
Writes results/reselection.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from leafdx.config import ROOT, load_config, path  # noqa: E402
from leafdx.ip.features import FEATURE_NAMES  # noqa: E402
from leafdx.p3 import heads  # noqa: E402
from leafdx.p3.data import file_hash  # noqa: E402
from leafdx.p3.features import features_for  # noqa: E402
from leafdx.p3.frames import build_frames  # noqa: E402
from leafdx.p3.score import config_z  # noqa: E402
from leafdx.p3.stats import evaluate, logit, mcnemar, paired_bootstrap_diff, sigmoid  # noqa: E402


def prod_prob(man: dict, d: pd.DataFrame, feats: np.ndarray) -> np.ndarray:
    prod = man["production"]["disease"]
    if prod == "STACK":
        st = joblib.load(ROOT / man["stacker"]["file"])
        X = []
        for arm, cid in st["members"].items():
            b = joblib.load(ROOT / man["arms"][arm]["file"]) if man["arms"][arm]["config"] == cid else \
                joblib.load(ROOT / ".cache" / "p3models" / f"{cid}.joblib")
            X.append(logit(sigmoid(config_z(b, d, feats) / b["temperature"])))
        p = st["meta"].predict_proba(np.column_stack(X))[:, 1]
        return sigmoid(logit(p) / st.get("temperature", 1.0))
    b = joblib.load(ROOT / man["arms"][prod]["file"])
    return sigmoid(config_z(b, d, feats) / b["temperature"])


def main() -> int:
    cfg = load_config()
    heads.pin_lightgbm_openmp()
    nb, seed = cfg["p3"]["bootstrap"], cfg["seed"]
    res_dir, mdir = path("results", cfg), path("models", cfg)
    tm = json.loads((res_dir / "test_manifest.json").read_text())
    changed = [f["path"] for f in tm["files"] if hashlib.sha1((ROOT / f["path"]).read_bytes()).hexdigest() != f["sha1"]]
    if changed:
        raise SystemExit(f"test files changed since they were frozen: {changed[:5]}")
    old = json.loads((mdir / "manifest.json").read_text())
    if (mdir / "manifest_phase3.json").exists():  # already switched once: compare against the Phase 3 bundle
        old = json.loads((mdir / "manifest_phase3.json").read_text())
    new = json.loads((mdir / "manifest_field.json").read_text())
    sel = json.loads((res_dir / "reselection_selection.json").read_text())

    sets = {}
    lab = pd.read_csv(ROOT / cfg["wild_ext"]["root"] / "labels.csv")
    d = lab[lab.label.isin(["healthy", "diseased"])].reset_index(drop=True).copy()
    d["file"], d["key"], d["y"] = d.path, [file_hash(p) for p in d.path], (d.label == "diseased").astype(int)
    sets["wild_ext"] = (d, features_for(list(d.path), list(d.crop), cfg)[FEATURE_NAMES].to_numpy(float))
    w = build_frames()["wild"]
    sets["wild"] = (w, w[FEATURE_NAMES].to_numpy(float))

    out = {"selection_file": "results/reselection_selection.json", "new_production": new["production"],
           "old_production": old["production"], "test_files_verified": len(tm["files"]), "sets": {}}
    for name, (d, F) in sets.items():
        po, pn = prod_prob(old, d, F), prod_prob(new, d, F)
        y, g = d.y.to_numpy(), d.group.to_numpy()
        diff = paired_bootstrap_diff(y, pn, po, g, nb, seed)
        out["sets"][name] = {"n": len(d), "old": evaluate(y, po, g, nb, seed), "new": evaluate(y, pn, g, nb, seed),
                             "new_minus_old_macro_f1": diff,
                             "mcnemar_new_vs_old": mcnemar(y, (pn >= 0.5).astype(int), (po >= 0.5).astype(int))}
        print(name, f"old {out['sets'][name]['old']['macro_f1']:.3f} new {out['sets'][name]['new']['macro_f1']:.3f} "
                    f"diff {diff['diff']:+.3f} {np.round(diff['ci95'], 3).tolist()}", flush=True)
    lo, hi = out["sets"]["wild_ext"]["new_minus_old_macro_f1"]["ci95"]
    same = new["production"]["disease_config"] == old["production"].get("disease_config")
    out["switch"] = bool(hi >= 0 and not same)
    out["verdict"] = ("same model re-selected" if same else "new model better on wild_ext" if lo > 0 else
                      "no significant difference on wild_ext" if hi >= 0 else "new model worse on wild_ext: no switch")
    out["loso_stable"] = sel["loso_stable"]
    subprocess.run([sys.executable, str(ROOT / "scripts" / "p3_wild_ext.py"), "--manifest", "models/manifest_field.json",
                    "--out", "wild_ext_field"], check=True, cwd=ROOT)
    out["trust_wild_ext_field"] = "results/wild_ext_field.json"
    if out["switch"]:
        if not (mdir / "manifest_phase3.json").exists():
            shutil.copyfile(mdir / "manifest.json", mdir / "manifest_phase3.json")
        shutil.copyfile(mdir / "manifest_field.json", mdir / "manifest.json")
    (res_dir / "reselection.json").write_text(json.dumps(out, indent=1, default=float))
    print("verdict:", out["verdict"], "| switch:", out["switch"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
