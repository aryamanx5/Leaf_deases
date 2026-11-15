"""Phase 5 evaluation and switch rule (docs/PLAN_phase5_confidence.md, pre-registered).

Scores the deployed bundle (models/manifest.json, "old") and the Phase 5 bundle (models/manifest_p5.json, "new") on
clean TEST, p4_test, data/wild and data/wild_ext. Nothing is fitted here. Disease is scored with the true crop.
Also reported on p4_test: the trust gates (OOD, crop threshold, disease threshold; the quality gate is not re-run
here, as in Phase 4), other-species refusal, and how much probability each bundle puts on its answers.

CAVEAT recorded in the output: p4_test was read once before, by scripts/p4_eval.py, for Phase 4's own switch. This
is its second read, for a different pre-registered decision; nothing is selected on it.

Switch rule (all must hold): p4_test macro-F1 difference (new - old) point >= -0.005 and 95% CI lower bound > -0.02;
answered share on p4_test higher; end-to-end accuracy on answered >= 0.97; clean TEST macro-F1 drop <= 0.02;
other-species refusal on p4_test not lower by more than one leaf. Writes results/p5/eval.json; on a switch,
models/manifest.json = manifest_p5.json (the Phase 4 bundle stays as models/manifest_p4.json).
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
from leafdx.p3.stats import conformal_sets, evaluate, logit, mcnemar, paired_bootstrap_diff, sigmoid  # noqa: E402
from leafdx.p5 import confidence_summary  # noqa: E402
from leafdx.robust.infer import knn_distance  # noqa: E402

CROPS = ("turmeric", "blackgram")


def frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df.reset_index(drop=True).copy()
    d["file"] = d.path
    d["key"] = [file_hash(p) for p in d.path]
    d["y"] = (d.label == "diseased").astype(int)
    return d


class Bundle:
    """A production bundle scored offline exactly as leafdx.p3.production scores one image."""

    def __init__(self, man_file: str):
        self.file = man_file
        self.man = json.loads((ROOT / man_file).read_text())
        self.prod = self.man["production"]["disease"]
        self.stacker = joblib.load(ROOT / self.man["stacker"]["file"]) if self.man.get("stacker") else None
        names = self.stacker["members"] if self.prod == "STACK" else [self.prod]
        self.arms = {a: joblib.load(ROOT / self.man["arms"][a]["file"]) for a in names}
        self.crop = joblib.load(ROOT / self.man["crop"]["file"])
        self.ref = np.load(ROOT / self.man["trust"]["ood_reference"]).astype(np.float32)

    def p_disease(self, d: pd.DataFrame, feats: np.ndarray) -> np.ndarray:
        p = {a: sigmoid(config_z(b, d, feats) / b["temperature"]) for a, b in self.arms.items()}
        if self.prod != "STACK":
            return p[self.prod]
        X = np.column_stack([logit(p[a]) for a in self.stacker["members"]])
        return sigmoid(logit(self.stacker["meta"].predict_proba(X)[:, 1]) / self.stacker.get("temperature", 1.0))

    def p_turmeric(self, d: pd.DataFrame) -> np.ndarray:
        info = self.crop["info"]
        bbs = info.get("backbones") or [info["backbone"]]
        X = np.column_stack([embeddings(bb, d) for bb in bbs])
        return sigmoid(self.crop["model"].decision_function(X) / self.crop["temperature"])

    def ood(self, d: pd.DataFrame) -> np.ndarray:
        tr = self.man["trust"]
        return knn_distance(embeddings(tr["ood_backbone"], d), self.ref, tr["ood_k"]) > tr["ood_threshold"]


def per_crop(y, p, g, crop, nb, seed):
    out = {"pooled": {**evaluate(y, p, g, nb, seed), "confidence": confidence_summary(p, y)}}
    for c in CROPS:
        m = crop == c
        if m.sum() and len(np.unique(y[m])) == 2:
            out[c] = evaluate(y[m], p[m], g[m], nb, seed)
    return out


def main() -> int:
    cfg = load_config()
    heads.pin_lightgbm_openmp()
    nb, seed = cfg["p3"]["bootstrap"], cfg["seed"]
    old, new = Bundle("models/manifest.json"), Bundle("models/manifest_p5.json")
    if old.man["production"] == new.man["production"] and old.man["trust"] == new.man["trust"]:
        raise SystemExit("manifest.json already is the Phase 5 bundle: evaluation runs before a switch")

    F = build_frames()
    lab = pd.read_csv(ROOT / cfg["p4"]["root"] / "labels.csv")
    p4t = lab[lab.p4_split == "p4_test"]
    we = pd.read_csv(ROOT / cfg["wild_ext"]["root"] / "labels.csv")
    S = {"test": F["test"], "wild": F["wild"],
         "p4_test": frame(p4t[p4t.label.isin(["healthy", "diseased"])]),
         "wild_ext": frame(we[we.crop.isin(CROPS) & we.label.isin(["healthy", "diseased"])])}
    for s in ("p4_test", "wild_ext"):
        S[s][FEATURE_NAMES] = features_for(list(S[s].path), list(S[s].crop), cfg)[FEATURE_NAMES].to_numpy(float)

    res = {"protocol": "nothing fitted on these sets; p4_test was read once before by scripts/p4_eval.py (Phase 4), "
                       "so this is its second read", "plan": "docs/PLAN_phase5_confidence.md",
           "old": old.man["production"], "new": new.man["production"], "sets": {}, "test_files_sha1": {}}
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
              f"diff {r['new_minus_old_macro_f1']['diff']:+.3f} {np.round(r['new_minus_old_macro_f1']['ci95'], 3)}  "
              f"conf(right) {r['old']['pooled']['confidence']['mean_confidence_right']:.3f} -> "
              f"{r['new']['pooled']['confidence']['mean_confidence_right']:.3f}", flush=True)

    # ---- trust gates on p4_test (incl. crop-only turmeric) and other-species refusal -------------------------------
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
                       "n_labelled": int(dlab.sum()),
                       "answered_share_labelled": float(ans[dlab].mean()),
                       "answered_labelled": int(ans[dlab].sum()),
                       "errors_answered": int((~(crop_ok & dis_ok))[dlab & ans].sum()),
                       "end_to_end_accuracy_answered": float((crop_ok & dis_ok)[dlab & ans].mean()),
                       "end_to_end_accuracy_all": float((crop_ok & dis_ok)[dlab].mean()),
                       "refused_by": {"crop_gate": float((~cset)[dlab].mean()), "ood_gate": float(oflag[dlab].mean()),
                                      "disease_gate": float((~dset)[dlab].mean())},
                       "ood_flag_rate_targets": float(oflag.mean()),
                       "other_species_refused": float(refused.mean()) if len(oth) else None,
                       "other_species_refused_n": int(refused.sum()) if len(oth) else None, "n_other": int(len(oth)),
                       "disease_min_confidence": 1 - tr["disease_q_lac"], "crop_min_confidence": 1 - tr["crop_q_lac"],
                       "note": "quality gate not re-run here; gates = OOD + crop threshold + disease threshold"}
    res["trust_p4_test"] = gates
    print("gates p4_test:", json.dumps({k: {x: v[x] for x in ("answered_share_labelled", "end_to_end_accuracy_answered",
                                                               "errors_answered", "other_species_refused")}
                                        for k, v in gates.items()}), flush=True)

    # ---- switch rule -----------------------------------------------------------------------------------------------
    d4 = res["sets"]["p4_test"]["new_minus_old_macro_f1"]
    t_old = res["sets"]["test"]["old"]["pooled"]["macro_f1"]
    t_new = res["sets"]["test"]["new"]["pooled"]["macro_f1"]
    go, gn = gates["old"], gates["new"]
    rule = {
        "f1_no_loss": bool(d4["diff"] >= -0.005 and d4["ci95"][0] > -0.02),
        "answers_more": bool(gn["answered_share_labelled"] > go["answered_share_labelled"]),
        "accuracy_answered_ok": bool(gn["end_to_end_accuracy_answered"] >= 0.97),
        "clean_test_within_0.02": bool(t_new >= t_old - 0.02),
        "other_species_ok": bool(gn["other_species_refused_n"] is None
                                 or gn["other_species_refused_n"] >= go["other_species_refused_n"] - 1),
    }
    switched = all(rule.values())
    res["switch"] = {**rule, "p4_test_diff": d4["diff"], "p4_test_ci95": d4["ci95"], "clean_test_old": t_old,
                     "clean_test_new": t_new, "switched": switched}
    out = path("results", cfg) / "p5"
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(res, indent=1, default=float))
    mdir = path("models", cfg)
    if switched:
        if not (mdir / "manifest_p4.json").exists():
            shutil.copy(mdir / "manifest.json", mdir / "manifest_p4.json")
        shutil.copy(mdir / "manifest_p5.json", mdir / "manifest.json")
    print("switch:", json.dumps(res["switch"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
