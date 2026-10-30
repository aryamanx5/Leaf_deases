"""P3-M05: trust layer.

* Temperature scaling (fitted per model on VAL + VAL-swap in p3_crop / p3_disease): ECE
  before/after collected here.
* Split conformal prediction (LAC, and APS for comparison), target coverage 1 - alpha.
  Calibration set = VAL + VAL-swap. Crop set size != 1 -> abstain "crop uncertain";
  disease set size != 1 -> "disease status uncertain".
* kNN OOD on DINOv2 embeddings: mean cosine distance to the k nearest TRAIN embeddings
  (TRAIN photos + their background-swap views); threshold = 95th percentile of VAL + VAL-swap
  distances. Above -> "this does not look like a turmeric or black gram leaf".
* Quality gate: blur (variance of Laplacian), dark / over-exposed share, Stage 1 leaf area.
* Abstention curves (accuracy vs coverage) on TEST and wild.

The wild set is evaluated only. Writes results/trust.json, models/p3/trust.json,
models/p3/ood_reference.npy.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from leafdx.config import ROOT, load_config, path
from leafdx.ip.preprocess import load_bgr
from leafdx.p3.frames import build_frames, embeddings
from leafdx.p3.quality import quality_stats, quality_verdict
from leafdx.p3.stats import abstention_curve, conformal_sets, conformal_threshold, point_metrics
from leafdx.robust.infer import knn_distance

OOD_BB = "dinov2_vits14"


def _q(p: str, leafonly: str) -> dict:
    import cv2

    from leafdx.p3.views import GREY

    lo = cv2.imread(leafonly)
    return quality_stats(load_bgr(ROOT / p), (lo != GREY).any(-1))


def two(p):
    p = np.asarray(p, float)
    return np.column_stack([p, 1 - p])


def main() -> int:
    cfg = load_config()
    c3 = cfg["p3"]
    out = path("results", cfg)
    F = build_frames()
    alpha = c3["conformal_alpha"]

    # ---- OOD ------------------------------------------------------------------------------
    ref = np.r_[embeddings(OOD_BB, F["train"]), embeddings(OOD_BB, F["train_swap"])].astype(np.float32)
    dist = {s: knn_distance(embeddings(OOD_BB, F[s]), ref, c3["ood_k"])
            for s in ("val", "val_swap", "test", "wild_all", "user")}
    ood_thr = float(np.quantile(np.r_[dist["val"], dist["val_swap"]], c3["ood_quantile"]))
    np.save(path("models", cfg) / "p3" / "ood_reference.npy", ref.astype(np.float16))

    # ---- quality ---------------------------------------------------------------------------
    fc = pd.read_csv(path("features", cfg)).set_index("path")
    from leafdx.p3.features import features_for

    qs = {}
    for s in ("val", "test", "wild_all", "user"):
        d = F[s]
        lo = pd.read_csv(out / "p3" / "views_leafonly.csv").set_index("path")  # Stage 1 leaf mask = non-grey pixels
        st = Parallel(n_jobs=-1)(delayed(_q)(p, lo.loc[p, "leafonly"]) for p in d.path)
        cov = (fc.loc[d.path, "seg_coverage"].to_numpy() if s in ("val", "test")
               else features_for(list(d.path), list(d.crop), cfg).seg_coverage.to_numpy())
        qs[s] = [dict(**a, leaf_frac=float(b)) for a, b in zip(st, cov, strict=True)]
    q = c3["quality"]
    qv = {s: [quality_verdict(x, x["leaf_frac"], q) for x in qs[s]] for s in qs}

    # ---- crop conformal --------------------------------------------------------------------
    cm = json.loads((out / "crop_models.json").read_text())
    crop_cfg = cm["chosen"]
    PC = pd.read_csv(out / "p3" / "preds_crop.csv.gz", keep_default_na=False)
    PC = PC[PC.config == crop_cfg]

    def crop_p(s):
        d = PC[PC.set == s].set_index("path").loc[F[s].path]
        return d.p_turmeric.to_numpy(float), (F[s].crop == "turmeric").to_numpy(int)

    pv = np.r_[crop_p("val")[0], crop_p("val_swap")[0]]
    yv = np.r_[crop_p("val")[1], crop_p("val_swap")[1]]
    # class index 0 = turmeric
    qc = {m: conformal_threshold(two(pv), 1 - yv, alpha, m) for m in ("lac", "aps")}

    # ---- disease conformal (production arm) -------------------------------------------------
    stk = json.loads((out / "stacking.json").read_text())
    dis_arm = stk["production_choice"]
    dis_cfg = "STACK" if dis_arm == "STACK" else stk["members"][dis_arm]
    PD = pd.read_csv(out / "p3" / "preds_disease.csv.gz", keep_default_na=False)
    PD = PD[PD.config == dis_cfg]

    def dis_p(s):
        d = PD[PD.set == s].set_index("path").loc[F[s].path]
        return d.p_cal.to_numpy(float), F[s].y.to_numpy(int)

    dv = np.r_[dis_p("val")[0], dis_p("val_swap")[0]]
    dy = np.r_[dis_p("val")[1], dis_p("val_swap")[1]]
    qd = {m: conformal_threshold(two(dv), 1 - dy, alpha, m) for m in ("lac", "aps")}

    res = {"alpha": alpha, "target_coverage": 1 - alpha, "crop_model": crop_cfg, "disease_model": dis_cfg,
           "ood": {"backbone": OOD_BB, "k": c3["ood_k"], "quantile": c3["ood_quantile"], "threshold": ood_thr,
                   "reference": "TRAIN photos + background-swap views", "n_reference": int(len(ref)),
                   "flag_rate": {s: float((dist[s] > ood_thr).mean()) for s in dist}},
           "quality": {"thresholds": q, "pass_rate": {s: float(np.mean([v[0] for v in qv[s]])) for s in qv},
                       "fail_reasons": {s: pd.Series([v[1] for v in qv[s] if not v[0]]).value_counts().to_dict()
                                        for s in qv}},
           "conformal": {"crop_q": qc, "disease_q": qd}, "sets": {}}

    # ---- per-set evaluation ----------------------------------------------------------------
    for s in ("test", "wild_all", "user"):
        d = F[s].reset_index(drop=True)
        pt, yc = crop_p(s)
        cs = {m: conformal_sets(two(pt), qc[m], m) for m in qc}
        crop_ok = (np.where(pt >= 0.5, 1, 0) == yc)
        r = {"n": len(d), "crop": {"accuracy": float(crop_ok.mean())}}
        for m in qc:
            size = cs[m].sum(1)
            cover = cs[m][np.arange(len(d)), 1 - yc]
            r["crop"][m] = {"coverage": float(cover.mean()), "mean_set_size": float(size.mean()),
                            "share_singleton": float((size == 1).mean()), "share_empty": float((size == 0).mean())}
        lab = (d.label != "unknown").to_numpy()
        dp = np.full(len(d), np.nan)
        if s != "user":
            ss = "wild" if s == "wild_all" else s
            dd = PD[PD.set == ss].set_index("path")
            dp[lab] = dd.loc[d.path[lab], "p_cal"].to_numpy(float)
        ds = {m: conformal_sets(two(np.nan_to_num(dp, nan=0.5)), qd[m], m) for m in qd}
        ok_q = np.array([v[0] for v in qv[s]])
        ood = dist[s] > ood_thr
        crop_unc = cs["lac"].sum(1) != 1
        dis_unc = ds["lac"].sum(1) != 1
        reason = np.where(~ok_q, "quality", np.where(ood, "ood", np.where(crop_unc, "crop uncertain",
                          np.where(dis_unc & lab, "disease status uncertain", ""))))
        accepted = reason == ""
        r["abstain_reasons"] = pd.Series(reason[~accepted]).value_counts().to_dict()
        r["coverage_after_gates"] = float(accepted.mean())
        r["crop_accuracy_accepted"] = float(crop_ok[accepted].mean()) if accepted.any() else None
        r["crop_confidently_wrong_accepted"] = int((~crop_ok & accepted).sum())
        if lab.any():
            dy_ = d.y.to_numpy()
            dis_ok = ((dp >= 0.5).astype(int) == dy_)
            e2e = crop_ok & dis_ok
            r["disease"] = {"n_labelled": int(lab.sum()), "accuracy_all": float(dis_ok[lab].mean()),
                            "macro_f1_all": point_metrics(dy_[lab], dp[lab])["macro_f1"]}
            for m in qd:
                size = ds[m][lab].sum(1)
                r["disease"][m] = {"coverage": float(ds[m][lab][np.arange(lab.sum()), 1 - dy_[lab]].mean()),
                                   "mean_set_size": float(size.mean()), "share_singleton": float((size == 1).mean())}
            al = accepted & lab
            r["disease"]["accepted_n"] = int(al.sum())
            r["disease"]["accepted_share"] = float(al.sum() / lab.sum())
            r["disease"]["accuracy_accepted"] = float(dis_ok[al].mean()) if al.any() else None
            r["disease"]["end_to_end_accuracy_accepted"] = float(e2e[al].mean()) if al.any() else None
            r["disease"]["end_to_end_accuracy_all"] = float(e2e[lab].mean())
            conf = np.maximum(dp[lab], 1 - dp[lab])
            r["disease"]["abstention_curve"] = abstention_curve(conf, dis_ok[lab])
        r["crop"]["abstention_curve"] = abstention_curve(np.maximum(pt, 1 - pt), crop_ok)
        if s == "user":
            r["images"] = [{"path": p, "p_turmeric": float(a), "crop_set": [c for c, k in zip(("turmeric", "blackgram"),
                                                                                              cs["lac"][i], strict=True) if k],
                            "ood_distance": float(dist[s][i]), "ood_flagged": bool(ood[i]), "quality_ok": bool(ok_q[i]),
                            "abstain_reason": reason[i] or None,
                            "outcome": ("abstain" if reason[i] else ("turmeric" if a >= 0.5 else "blackgram"))}
                           for i, (p, a) in enumerate(zip(d.path, pt, strict=True))]
        res["sets"][s] = r
        print(s, {k: v for k, v in r.items() if k in ("coverage_after_gates", "abstain_reasons", "crop_accuracy_accepted")},
              flush=True)

    # ---- ECE before / after temperature ---------------------------------------------------
    D = json.loads((out / "disease_models.json").read_text())
    res["ece_before_after"] = {
        "disease": {a: {s: {"before": v[s]["pooled"]["ece_before_temperature"], "after": v[s]["pooled"]["ece"]}
                        for s in ("test", "wild") if "pooled" in v.get(s, {})}
                    for a, v in D["arms"].items() if "selected_config" in v}}
    res["ece_before_after"]["crop"] = {}
    sub = pd.read_csv(out / "p3" / "preds_crop.csv.gz", keep_default_na=False)
    for n in cm["candidates"]:
        x = sub[sub.config == n]
        e = {}
        for s in ("test", "wild_all"):
            g = x[x.set == s]
            y = (g.crop == "turmeric").astype(int)
            e[s] = {"before": point_metrics(y, 1 / (1 + np.exp(-g.z)))["ece"], "after": point_metrics(y, g.p_turmeric)["ece"]}
        res["ece_before_after"]["crop"][n] = e
    (out / "trust.json").write_text(json.dumps(res, indent=1, default=float))
    (path("models", cfg) / "p3" / "trust.json").write_text(json.dumps(
        {"alpha": alpha, "crop_model": crop_cfg, "disease_model": dis_cfg, "crop_q_lac": qc["lac"],
         "disease_q_lac": qd["lac"], "ood_threshold": ood_thr, "ood_k": c3["ood_k"], "ood_backbone": OOD_BB,
         "quality": q}, indent=1))
    print(json.dumps({k: res[k] for k in ("ood", "quality", "conformal")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
