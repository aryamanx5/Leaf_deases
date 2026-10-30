"""Field-validated re-selection (docs/PLAN_field_validation.md, pre-registered).

Stage 1 -- SELECT and CALIBRATE on data/field_val only (no test image is read):
  * candidates: every Phase 3 configuration (39) + a stack over the Phase 3 member arms whose meta-learner is
    cross-fitted on field_val (grouped 5-fold); base models are not retrained;
  * score = field_val pooled macro-F1; tie rule: among candidates whose point estimate reaches the lower 95% CI bound
    of the best, take the lowest CPU ms/image;
  * calibrate the winner on field_val: temperature, LAC disease threshold (alpha 0.10); crop head kept if its
    field_val accuracy >= 0.98, its temperature and LAC threshold refitted on field_val; kNN OOD threshold = 95th
    percentile of field_val target distances (refusal rate on field_val other species reported); quality gate unchanged;
  * grouped 5-fold stability of T and q; leave-one-source-out winner check.
  -> results/reselection_selection.json and models/manifest_field.json (candidate bundle), written BEFORE stage 2.
Stage 2 (scripts/p3_reselect_eval.py, run once afterwards) evaluates on the frozen test sets.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from leafdx.config import ROOT, load_config, path  # noqa: E402
from leafdx.ip.features import FEATURE_NAMES  # noqa: E402
from leafdx.p3 import heads  # noqa: E402
from leafdx.p3.data import file_hash  # noqa: E402
from leafdx.p3.features import features_for  # noqa: E402
from leafdx.p3.frames import embeddings  # noqa: E402
from leafdx.p3.score import config_z  # noqa: E402
from leafdx.p3.stats import (  # noqa: E402
    conformal_threshold,
    fit_temperature,
    grouped_bootstrap,
    logit,
    point_metrics,
    sigmoid,
)
from leafdx.robust.infer import knn_distance  # noqa: E402

TIE_KEY = "macro_f1"


def frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df.reset_index(drop=True).copy()
    d["file"] = d.path
    d["key"] = [file_hash(p) for p in d.path]
    d["y"] = (d.label == "diseased").astype(int)
    return d


def bundles(cfg) -> dict[str, dict]:
    """config id -> scoring bundle, for every Phase 3 configuration that can be scored here."""
    log = json.loads((path("results", cfg) / "p3" / "disease_configs.json").read_text())
    tr = json.loads((path("results", cfg) / "p3_cnn_train.json").read_text())
    arch = {a["name"]: a["arch"] for a in cfg["p3"]["cnn_arms"]["arms"]}
    out = {}
    for cid, info in log.items():
        if "skipped" in info:
            continue
        f = ROOT / ".cache" / "p3models" / f"{cid}.joblib"
        if f.exists():
            out[cid] = joblib.load(f)
        elif cid.startswith("C"):
            arm, variant = cid.split("[")[0], cid.split("[")[1].rstrip("]")
            ck = tr.get(f"{arm}_{variant}", {}).get("checkpoint")
            if ck and (ROOT / ck).exists():
                out[cid] = {"checkpoint": ck, "arch": arch[arm], "temperature": info["temperature"]}
    return out


def arm_of(cid: str) -> str:
    return "F_fusion" if cid.startswith("F_fusion") else cid.split("[")[0]


def score_all(B, d, feats) -> dict[str, np.ndarray]:
    z = {}
    for cid, b in B.items():
        t0 = time.time()
        z[cid] = config_z(b, d, feats)
        print(f"  {cid:42s} {time.time() - t0:5.1f}s", flush=True)
    return z


def stack_inputs(members: dict, z: dict, B: dict) -> np.ndarray:
    """Phase 3 stack inputs: calibrated (Phase 3 temperatures) log-odds of each member configuration."""
    return np.column_stack([logit(sigmoid(z[c] / B[c]["temperature"])) for c in members.values()])


def oof_stack(X, y, groups, seed) -> np.ndarray:
    p = np.zeros(len(y))
    for tr_i, te_i in GroupKFold(n_splits=5).split(X, y, groups):
        m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X[tr_i], y[tr_i])
        p[te_i] = m.predict_proba(X[te_i])[:, 1]
    return p


def choose(scores: dict, ms: dict, y, groups, nb, seed) -> tuple[str, dict]:
    """Best pooled macro-F1; ties (point >= best's lower CI bound) go to the lowest CPU ms/image."""
    f1 = {c: point_metrics(y, s)["macro_f1"] for c, s in scores.items()}
    best = max(f1, key=f1.get)
    lo = grouped_bootstrap(y, scores[best], groups, nb, seed, keys=(TIE_KEY,))[TIE_KEY][0]
    tied = [c for c in f1 if f1[c] >= lo]
    win = min(tied, key=lambda c: (ms.get(c, np.inf), -f1[c]))
    return win, {"best_point": best, "best_f1": f1[best], "best_ci_low": lo, "tied": sorted(tied, key=lambda c: -f1[c]),
                 "winner": win, "winner_f1": f1[win], "ms": {c: ms.get(c) for c in tied}}


def main() -> int:
    cfg = load_config()
    heads.pin_lightgbm_openmp()
    nb, seed, alpha = cfg["p3"]["bootstrap"], cfg["seed"], cfg["p3"]["conformal_alpha"]
    res_dir, mdir = path("results", cfg), path("models", cfg)
    p3_man = mdir / "manifest_phase3.json"  # present after a switch: always start from the Phase 3 bundle
    man = json.loads((p3_man if p3_man.exists() else mdir / "manifest.json").read_text())
    timing = json.loads((res_dir / "p3" / "timing.json").read_text())

    # ================= stage 1: field_val only ============================================
    lab = pd.read_csv(ROOT / cfg["field_val"]["root"] / "labels.csv")
    fv = frame(lab[lab.label.isin(["healthy", "diseased"])])
    fo = frame(lab[lab.crop == "other"])
    feats = features_for(list(fv.path), list(fv.crop), cfg)[FEATURE_NAMES].to_numpy(float)
    B = bundles(cfg)
    print(f"field_val: {len(fv)} labelled leaves ({fv.group.nunique()} groups), {len(fo)} other; {len(B)} configs", flush=True)
    z = score_all(B, fv, feats)
    y, g = fv.y.to_numpy(), fv.group.to_numpy()
    scores = {c: sigmoid(v) for c, v in z.items()}  # raw (threshold at 0.5 == z at 0); calibration comes later
    members = {a: v["config"] for a, v in man["arms"].items()}
    members = {a: c for a, c in members.items() if c in z}
    Xs = stack_inputs(members, z, B)
    scores["STACK_fv"] = oof_stack(Xs, y, g, seed)
    comp = timing["components_ms"]
    arm_ms = {a: v["ms_per_image"] for a, v in timing["arms"].items()}
    ms = {c: arm_ms.get(arm_of(c)) for c in z}
    for c in z:  # F configs differ by backbone: stage1 + that backbone's embedding
        if c.startswith("F_fusion_"):
            bb = B[c]["backbone"]
            ms[c] = comp["stage1"] + comp[f"embed_{bb}"]
    needed = {"stage1"} | {f"embed_{B[c]['backbone']}" for c in members.values() if B[c].get("backbone")} | \
        {f"cnn_{arm_of(c)}" for c in members.values() if c.startswith("C")}
    ms["STACK_fv"] = float(sum(comp[k] for k in needed if k in comp))
    win, sel = choose(scores, ms, y, g, nb, seed)
    print("winner:", win, json.dumps({k: sel[k] for k in ("best_point", "best_f1", "best_ci_low", "winner_f1")}), flush=True)

    # leave-one-source-out
    loso = {}
    for s in sorted(fv.source.unique()):
        m = (fv.source != s).to_numpy()
        sc = {c: v[m] for c, v in scores.items()}
        loso[s] = choose(sc, ms, y[m], g[m], 200, seed)[0]

    # calibration of the winner on field_val (stack: out-of-fold scores)
    if win == "STACK_fv":
        z_w = logit(scores["STACK_fv"])
        meta = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xs, y)
    else:
        z_w = z[win]
    T = fit_temperature(z_w, y)
    p_w = sigmoid(z_w / T)
    q_dis = conformal_threshold(np.column_stack([p_w, 1 - p_w]), 1 - y, alpha, "lac")
    stab = []
    for tr_i, te_i in GroupKFold(n_splits=5).split(z_w, y, g):
        Tk = fit_temperature(z_w[tr_i], y[tr_i])
        pk = sigmoid(z_w / Tk)
        qk = conformal_threshold(np.column_stack([pk[tr_i], 1 - pk[tr_i]]), 1 - y[tr_i], alpha, "lac")
        sets = np.column_stack([1 - pk[te_i], pk[te_i]]) <= qk  # LAC: 1 - p_k <= q
        stab.append({"T": Tk, "q": qk, "held_out_coverage": float(sets[np.arange(len(te_i)), 1 - y[te_i]].mean())})

    # crop head + OOD on field_val
    cb = joblib.load(ROOT / man["crop"]["file"])
    bbs = cb["info"].get("backbones") or [cb["info"]["backbone"]]
    allfv = pd.concat([fv, fo], ignore_index=True)
    zc = cb["model"].decision_function(np.column_stack([embeddings(bb, allfv) for bb in bbs]))
    tgt = (allfv.crop != "other").to_numpy()
    yc = (allfv.crop[tgt] == "turmeric").astype(int).to_numpy()
    crop_acc = float(((zc[tgt] >= 0).astype(int) == yc).mean())
    Tc = fit_temperature(zc[tgt], yc) if crop_acc >= 0.98 else None
    pc = sigmoid(zc[tgt] / Tc) if Tc else None
    q_crop = conformal_threshold(np.column_stack([pc, 1 - pc]), 1 - yc, alpha, "lac") if Tc else None
    ref = np.load(ROOT / man["trust"]["ood_reference"]).astype(np.float32)
    dist = knn_distance(embeddings(man["trust"]["ood_backbone"], allfv), ref, man["trust"]["ood_k"])
    ood_thr = float(np.quantile(dist[tgt], cfg["p3"]["ood_quantile"]))
    other_refused = float((dist[~tgt] > ood_thr).mean()) if (~tgt).any() else None

    selection = {
        "plan": "docs/PLAN_field_validation.md", "field_val_sha1": hashlib.sha1(
            (ROOT / cfg["field_val"]["root"] / "labels.csv").read_bytes()).hexdigest(),
        "n": {"labelled": len(fv), "groups": int(fv.group.nunique()), "other": len(fo),
              "by_source": fv.source.value_counts().to_dict(), "by_crop_label": {f"{a}/{b}": int(v) for (a, b), v in
                                                                                  fv.groupby(["crop", "label"]).size().items()}},
        "candidates": {c: {"macro_f1": point_metrics(y, s)["macro_f1"], "ms": ms.get(c),
                           "turmeric_f1": point_metrics(y[fv.crop == "turmeric"], s[(fv.crop == "turmeric").to_numpy()])["macro_f1"],
                           "blackgram_f1": point_metrics(y[fv.crop == "blackgram"], s[(fv.crop == "blackgram").to_numpy()])["macro_f1"]}
                       for c, s in scores.items()},
        "selection": sel, "leave_one_source_out_winner": loso, "loso_stable": all(v == win for v in loso.values()),
        "calibration": {"temperature": T, "disease_q_lac": q_dis, "fold_stability": stab,
                        "crop_accuracy_field_val": crop_acc, "crop_kept": crop_acc >= 0.98, "crop_temperature": Tc,
                        "crop_q_lac": q_crop, "ood_threshold": ood_thr, "ood_other_species_refused": other_refused,
                        "quality": "unchanged (config p3.quality)"},
        "old_production": man["production"],
    }
    (res_dir / "reselection_selection.json").write_text(json.dumps(selection, indent=1, default=float))

    # candidate bundle
    adir = mdir / "p3" / "arms_field"
    adir.mkdir(parents=True, exist_ok=True)
    new = json.loads(json.dumps(man))
    if win == "STACK_fv":
        sf = mdir / "p3" / "stacker_field.joblib"
        joblib.dump({"members": members, "meta": meta, "temperature": T}, sf)
        new["stacker"] = {"file": str(sf.relative_to(ROOT)), "members": members, "fit_on": "field_val"}
        new["production"] = {**man["production"], "disease": "STACK", "disease_config": "STACK_fv"}
    else:
        arm = arm_of(win)
        b = dict(B[win])
        b["temperature"] = T
        f = adir / f"{arm}.joblib"
        joblib.dump(b, f)
        new["arms"][arm] = {**man["arms"].get(arm, {}), "config": win, "file": str(f.relative_to(ROOT)),
                            "kind": "cnn" if "checkpoint" in b else b["kind"]}
        if "checkpoint" in b:
            new["arms"][arm]["checkpoint"] = b["checkpoint"]
        if new.get("stacker") and new["stacker"]["members"].get(arm, win) != win:
            new["stacker"] = None  # the Phase 3 stacker was fitted on the arm configuration replaced here
        new["production"] = {**man["production"], "disease": arm, "disease_config": win}
    new["production"]["reason"] = "field-validated re-selection (docs/PLAN_field_validation.md)"
    if Tc:
        cf = mdir / "p3" / "crop_head_field.joblib"
        joblib.dump({**cb, "temperature": Tc}, cf)
        new["crop"] = {**man["crop"], "file": str(cf.relative_to(ROOT))}
    new["trust"] = {**man["trust"], "disease_q_lac": q_dis, "ood_threshold": ood_thr,
                    **({"crop_q_lac": q_crop} if q_crop is not None else {}), "calibrated_on": "field_val"}
    (mdir / "manifest_field.json").write_text(json.dumps(new, indent=1))
    print("stage 1 written: results/reselection_selection.json, models/manifest_field.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
