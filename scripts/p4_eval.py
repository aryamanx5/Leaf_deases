"""Phase 4 once-only evaluation and switch rule (docs/PLAN_phase4_more_data.md, pre-registered).

Scores the current production bundle (models/manifest.json, "old") and the Phase 4 bundle (models/manifest_p4.json,
"new") on clean TEST, the held-out p4_test, data/wild and data/wild_ext. Nothing is fitted here.
Disease is scored with the true crop. Also reported: crop ID and the trust gates (OOD, conformal crop, conformal
disease; the quality gate is not re-run here) on p4_test, and the refusal rate on p4_test other-species leaves.

Switch rule: new becomes production iff the paired macro-F1 difference (new - old) on p4_test has a 95% CI lower bound
> 0 AND clean TEST macro-F1 drops by no more than 0.02. Writes results/p4/eval.json; on a switch, models/manifest.json
= manifest_p4.json (the previous bundle stays as models/manifest_field.json).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
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
from leafdx.p3.frames import build_frames, embeddings  # noqa: E402
from leafdx.p3.score import config_z  # noqa: E402
from leafdx.p3.stats import conformal_sets, evaluate, mcnemar, paired_bootstrap_diff, sigmoid  # noqa: E402
from leafdx.robust.infer import knn_distance  # noqa: E402

CROPS = ("turmeric", "blackgram")


def frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df.reset_index(drop=True).copy()
    d["file"] = d.path
    d["key"] = [file_hash(p) for p in d.path]
    d["y"] = (d.label == "diseased").astype(int)
    return d


class Bundle:
    def __init__(self, man_file: str):
        self.file = man_file
        self.man = json.loads((ROOT / man_file).read_text())
        prod = self.man["production"]["disease"]
        self.disease = joblib.load(ROOT / self.man["arms"][prod]["file"])
        self.crop = joblib.load(ROOT / self.man["crop"]["file"])
        self.ref = np.load(ROOT / self.man["trust"]["ood_reference"]).astype(np.float32)

    def p_disease(self, d: pd.DataFrame, feats: np.ndarray) -> np.ndarray:
        return sigmoid(config_z(self.disease, d, feats) / self.disease["temperature"])

    def p_turmeric(self, d: pd.DataFrame) -> np.ndarray:
        info = self.crop["info"]
        bbs = info.get("backbones") or [info["backbone"]]
        X = np.column_stack([embeddings(bb, d) for bb in bbs])
        return sigmoid(self.crop["model"].decision_function(X) / self.crop["temperature"])

    def ood(self, d: pd.DataFrame) -> np.ndarray:
        tr = self.man["trust"]
        return knn_distance(embeddings(tr["ood_backbone"], d), self.ref, tr["ood_k"]) > tr["ood_threshold"]


def per_crop(y, p, g, crop, nb, seed):
    out = {"pooled": evaluate(y, p, g, nb, seed)}
    for c in CROPS:
        m = crop == c
        if m.sum() and len(np.unique(y[m])) == 2:
            out[c] = evaluate(y[m], p[m], g[m], nb, seed)
    return out


def main() -> int:
    cfg = load_config()
    heads.pin_lightgbm_openmp()
    nb, seed = cfg["p3"]["bootstrap"], cfg["seed"]
    old, new = Bundle("models/manifest.json"), Bundle("models/manifest_p4.json")
    if old.man["production"] == new.man["production"] and old.man["crop"] == new.man["crop"]:
        raise SystemExit("manifest.json already is the Phase 4 bundle: evaluation runs once, before a switch")

    F = build_frames()
    lab = pd.read_csv(ROOT / cfg["p4"]["root"] / "labels.csv")
    p4t = lab[lab.p4_split == "p4_test"]
    we = pd.read_csv(ROOT / cfg["wild_ext"]["root"] / "labels.csv")
    S = {"test": F["test"], "wild": F["wild"],
         "p4_test": frame(p4t[p4t.label.isin(["healthy", "diseased"])]),
         "wild_ext": frame(we[we.crop.isin(CROPS) & we.label.isin(["healthy", "diseased"])])}
    for s in ("p4_test", "wild_ext"):
        S[s][FEATURE_NAMES] = features_for(list(S[s].path), list(S[s].crop), cfg)[FEATURE_NAMES].to_numpy(float)

    res = {"protocol": "once-only evaluation; nothing fitted on these sets", "old": old.man["production"],
           "new": new.man["production"], "sets": {}, "test_files_sha1": {}}
    for s, d in S.items():
        feats = d[FEATURE_NAMES].to_numpy(float)
        y, g, crop = d.y.to_numpy(), d.group.astype(str).to_numpy(), d.crop.to_numpy()
        po, pn = old.p_disease(d, feats), new.p_disease(d, feats)
        res["sets"][s] = {"n": len(d), "old": per_crop(y, po, g, crop, nb, seed), "new": per_crop(y, pn, g, crop, nb, seed),
                          "new_minus_old_macro_f1": paired_bootstrap_diff(y, pn, po, g, nb, seed),
                          "mcnemar_new_vs_old": mcnemar(y, pn >= 0.5, po >= 0.5)}
        res["test_files_sha1"][s] = hashlib.sha1("".join(sorted(d.key)).encode()).hexdigest()
        r = res["sets"][s]
        print(f"{s:9s} n={len(d):4d}  old {r['old']['pooled']['macro_f1']:.3f}  new {r['new']['pooled']['macro_f1']:.3f}  "
              f"diff {r['new_minus_old_macro_f1']['diff']:+.3f} {np.round(r['new_minus_old_macro_f1']['ci95'], 3)}", flush=True)

    # ---- crop ID + trust gates on p4_test (incl. crop-only turmeric) and other-species refusal ---------------------
    tgt = frame(p4t[p4t.crop.isin(CROPS)])
    oth = frame(p4t[p4t.crop == "other"])
    yc = (tgt.crop == "turmeric").to_numpy()
    dlab = tgt.label.isin(["healthy", "diseased"]).to_numpy()
    tfe = features_for(list(tgt.path), list(tgt.crop), cfg)[FEATURE_NAMES].to_numpy(float)
    gates = {}
    for name, b in (("old", old), ("new", new)):
        tr = b.man["trust"]
        pt = b.p_turmeric(tgt)
        cset = conformal_sets(np.column_stack([pt, 1 - pt]), tr["crop_q_lac"]).sum(1) == 1
        oflag = b.ood(tgt)
        pd_ = b.p_disease(tgt, tfe)
        dset = conformal_sets(np.column_stack([pd_, 1 - pd_]), tr["disease_q_lac"]).sum(1) == 1
        ans = cset & ~oflag & dset
        crop_ok = (pt >= 0.5) == yc
        dis_ok = (pd_ >= 0.5) == tgt.y.to_numpy().astype(bool)
        pto = b.p_turmeric(oth) if len(oth) else np.array([])
        refused = (b.ood(oth) | (conformal_sets(np.column_stack([pto, 1 - pto]), tr["crop_q_lac"]).sum(1) != 1)) \
            if len(oth) else np.array([])
        gates[name] = {"crop_accuracy_all_targets": float(crop_ok.mean()), "n_targets": int(len(tgt)),
                       "answered_share_labelled": float(ans[dlab].mean()),
                       "end_to_end_accuracy_answered": float((crop_ok & dis_ok)[dlab & ans].mean()),
                       "end_to_end_accuracy_all": float((crop_ok & dis_ok)[dlab].mean()),
                       "ood_flag_rate_targets": float(oflag.mean()),
                       "other_species_refused": float(refused.mean()) if len(oth) else None, "n_other": int(len(oth)),
                       "note": "quality gate not re-run here; gates = OOD + conformal crop + conformal disease"}
    res["trust_p4_test"] = gates

    # ---- switch rule -----------------------------------------------------------------------------------------------
    d4 = res["sets"]["p4_test"]["new_minus_old_macro_f1"]
    t_old = res["sets"]["test"]["old"]["pooled"]["macro_f1"]
    t_new = res["sets"]["test"]["new"]["pooled"]["macro_f1"]
    better = d4["ci95"][0] > 0
    clean_ok = t_new >= t_old - 0.02
    res["switch"] = {"p4_test_ci_low": d4["ci95"][0], "p4_test_better": better, "clean_test_old": t_old,
                     "clean_test_new": t_new, "clean_test_within_0.02": clean_ok, "switched": bool(better and clean_ok)}
    out = path("results", cfg) / "p4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(res, indent=1, default=float))
    mdir = path("models", cfg)
    if better and clean_ok:
        if not (mdir / "manifest_field.json").exists():
            shutil.copy(mdir / "manifest.json", mdir / "manifest_field.json")
        shutil.copy(mdir / "manifest_p4.json", mdir / "manifest.json")
    print("switch:", json.dumps(res["switch"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
