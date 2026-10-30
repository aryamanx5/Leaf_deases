"""Evaluate the frozen Phase 3 bundle on the extended field test set (data/wild_ext, evaluation ONLY).

Nothing is fitted: every head, temperature, conformal threshold, OOD threshold and quality threshold comes from
models/manifest.json / models/p3/ (fitted on TRAIN / VAL only). Reports, with grouped bootstrap 95% CIs:
  * disease: every selected arm + STACK on the labelled target leaves (true crop), pooled / per crop / per source;
  * crop ID: accuracy on every target image incl. whole field scenes, per source;
  * other-plant leaves (five species): share refused by the gates vs share answered as a target crop;
  * end-to-end with every gate (quality, OOD, conformal crop, conformal disease): coverage, accuracy on answered,
    abstention reasons and curves.
Writes results/wild_ext.json and results/p3/preds_wild_ext.csv.gz.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from leafdx.config import ROOT, load_config, path  # noqa: E402
from leafdx.ip.features import FEATURE_NAMES  # noqa: E402
from leafdx.ip.preprocess import load_bgr  # noqa: E402
from leafdx.p3 import heads  # noqa: E402
from leafdx.p3.data import file_hash  # noqa: E402
from leafdx.p3.features import features_for  # noqa: E402
from leafdx.p3.frames import embeddings  # noqa: E402
from leafdx.p3.quality import quality_stats, quality_verdict  # noqa: E402
from leafdx.p3.stats import (  # noqa: E402
    abstention_curve,
    conformal_sets,
    evaluate,
    logit,
    sigmoid,
)
from leafdx.p3.views import GREY, leafonly_paths  # noqa: E402
from leafdx.robust.infer import knn_distance  # noqa: E402


def frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df.reset_index(drop=True).copy()
    d["file"] = d.path
    d["key"] = [file_hash(p) for p in d.path]
    d["y"] = (d.label == "diseased").astype(int)
    return d


def arm_z(arm: str, b: dict, d: pd.DataFrame, feats: np.ndarray) -> np.ndarray:
    ci = (d.crop == "turmeric").to_numpy(float)[:, None]
    if "checkpoint" in b:  # CNN arm
        from leafdx.p3.cnnpred import logits

        return logits(b["checkpoint"], b["arch"], d)
    if b["kind"] == "rule":
        row = pd.DataFrame(feats, columns=FEATURE_NAMES).assign(crop=d.crop.values)
        return 10 * (b["rule"].score(row).to_numpy() - d.crop.map(b["rule"].threshold).to_numpy())
    parts = []
    if b["kind"] in ("emb", "fusion"):
        parts.append(embeddings(b["backbone"], d))
    if b["kind"] in ("feat", "fusion"):
        parts.append(feats)
    X = np.column_stack([*parts, ci]).astype(np.float32)
    if b["mode"] == "pooled":
        return heads.raw_score(b["heads"]["all"], X)
    z = np.zeros(len(d))
    for c in ("turmeric", "blackgram"):
        m = (d.crop == c).to_numpy()
        if m.any():
            z[m] = heads.raw_score(b["heads"][c], X[m])
    return z


def _q(p: str, leafonly: str) -> dict:
    lo = cv2.imread(leafonly)
    return quality_stats(load_bgr(ROOT / p), (lo != GREY).any(-1))


def main() -> int:
    cfg = load_config()
    heads.pin_lightgbm_openmp()
    man = json.loads((ROOT / "models" / "manifest.json").read_text())
    tr = man["trust"]
    nb, seed = cfg["p3"]["bootstrap"], cfg["seed"]
    lab = pd.read_csv(ROOT / cfg["wild_ext"]["root"] / "labels.csv")
    target = frame(lab[lab.crop.isin(["turmeric", "blackgram"])])
    other = frame(lab[lab.crop == "other"])
    alld = pd.concat([target, other], ignore_index=True)

    # ---- crop ID (production crop head), OOD, quality -----------------------------------
    cb = joblib.load(ROOT / man["crop"]["file"])
    info = cb["info"]
    if not info["kind"].startswith("logistic"):
        raise SystemExit(f"crop model kind {info['kind']} not supported here")
    bbs = info.get("backbones") or [info["backbone"]]
    Xc = np.column_stack([embeddings(bb, alld) for bb in bbs])
    p_tu = sigmoid(cb["model"].decision_function(Xc) / cb["temperature"])
    ref = np.load(ROOT / tr["ood_reference"]).astype(np.float32)
    ood_d = knn_distance(embeddings(tr["ood_backbone"], alld), ref, tr["ood_k"])
    lo = leafonly_paths(list(alld.path), cfg)
    qs = Parallel(n_jobs=-1)(delayed(_q)(p, q) for p, q in zip(alld.path, lo, strict=True))
    crop_for_feats = np.where(alld.crop.isin(["turmeric", "blackgram"]), alld.crop, np.where(p_tu >= 0.5, "turmeric", "blackgram"))
    F = features_for(list(alld.path), list(crop_for_feats), cfg)
    qv = [quality_verdict(q, float(c), tr["quality"]) for q, c in zip(qs, F.seg_coverage, strict=True)]
    alld["p_turmeric"], alld["ood_distance"] = p_tu, ood_d
    alld["ood_flag"] = ood_d > tr["ood_threshold"]
    alld["quality_ok"] = [v[0] for v in qv]
    alld["quality_reason"] = [v[1] for v in qv]
    cs = conformal_sets(np.column_stack([p_tu, 1 - p_tu]), tr["crop_q_lac"])
    alld["crop_set_size"] = cs.sum(1)
    alld["crop_pred"] = np.where(p_tu >= 0.5, "turmeric", "blackgram")

    # ---- disease arms on labelled target leaves (true crop) -----------------------------
    labd = alld[alld.label.isin(["healthy", "diseased"])].reset_index(drop=True)
    Fl = features_for(list(labd.path), list(labd.crop), cfg)[FEATURE_NAMES].to_numpy(float)
    P = {}
    for arm, v in man["arms"].items():
        b = joblib.load(ROOT / v["file"])
        if "checkpoint" in b and not (ROOT / b["checkpoint"]).exists():
            continue
        P[arm] = sigmoid(arm_z(arm, b, labd, Fl) / b["temperature"])
        print(arm, f"acc {np.mean((P[arm] >= 0.5) == labd.y):.3f}", flush=True)
    stk = joblib.load(ROOT / man["stacker"]["file"])
    if all(a in P for a in stk["members"]):
        P["STACK"] = stk["meta"].predict_proba(np.column_stack([logit(P[a]) for a in stk["members"]]))[:, 1]
    prod = man["production"]["disease"]

    res = {"protocol": "frozen Phase 3 bundle; nothing fitted on this set", "n": {}, "disease": {}, "crop": {},
           "other_plants": {}, "end_to_end": {}}
    res["n"] = {"target_images": len(target), "labelled_leaves": len(labd), "other_plants": len(other),
                "groups_labelled": int(labd.group.nunique()),
                "by_source_label": {f"{s} / {c} / {lb}": int(n) for (s, c, lb), n in
                                    alld.groupby(["source", "crop", "label"]).size().items()}}
    scopes = {"pooled": np.ones(len(labd), bool), "turmeric": (labd.crop == "turmeric").to_numpy(),
              "blackgram": (labd.crop == "blackgram").to_numpy()}
    for s in sorted(labd.source.unique()):
        scopes[s] = (labd.source == s).to_numpy()
    for arm, p in P.items():
        res["disease"][arm] = {}
        for sc, m in scopes.items():
            if m.any() and labd.y[m].nunique() == 2:
                res["disease"][arm][sc] = evaluate(labd.y[m], p[m], labd.group[m], nb, seed)
    tgt = alld[alld.crop.isin(["turmeric", "blackgram"])]
    yc = (tgt.crop == "turmeric").astype(int)
    res["crop"]["pooled"] = evaluate(yc, tgt.p_turmeric, tgt.group, nb, seed)
    for s, g in tgt.groupby(np.where(tgt.label == "unknown", "field scenes", tgt.source)):
        res["crop"][s] = {"n": len(g), "accuracy": float((g.crop_pred == g.crop).mean())}

    # ---- other plants -------------------------------------------------------------------
    o = alld[alld.crop == "other"]
    refused = (~o.quality_ok) | o.ood_flag | (o.crop_set_size != 1)
    res["other_plants"] = {"n": len(o), "refused_share": float(refused.mean()),
                           "refused_by": {"quality": int((~o.quality_ok).sum()),
                                          "ood": int((o.quality_ok & o.ood_flag).sum()),
                                          "crop uncertain": int((o.quality_ok & ~o.ood_flag & (o.crop_set_size != 1)).sum())},
                           "answered_as_target_crop": int((~refused).sum()),
                           "ood_flag_rate": float(o.ood_flag.mean()),
                           "per_species_folder": {k: {"n": len(g), "refused": int(((~g.quality_ok) | g.ood_flag |
                                                                                   (g.crop_set_size != 1)).sum())}
                                                  for k, g in o.groupby("orig_class")}}

    # ---- end-to-end with every gate on labelled target leaves ---------------------------
    e = alld.set_index("path").loc[labd.path].reset_index()
    pd_ = P[prod]
    ds = conformal_sets(np.column_stack([pd_, 1 - pd_]), tr["disease_q_lac"])
    reason = np.where(~e.quality_ok, "quality", np.where(e.ood_flag, "ood", np.where(e.crop_set_size != 1, "crop uncertain",
                      np.where(ds.sum(1) != 1, "disease status uncertain", ""))))
    ans = reason == ""
    crop_ok = (e.crop_pred == e.crop).to_numpy()
    dis_ok = ((pd_ >= 0.5).astype(int) == labd.y.to_numpy())
    res["end_to_end"] = {"production_model": prod, "coverage": float(ans.mean()),
                         "abstain_reasons": pd.Series(reason[~ans]).value_counts().to_dict(),
                         "crop_accuracy_answered": float(crop_ok[ans].mean()) if ans.any() else None,
                         "disease_accuracy_answered": float(dis_ok[ans].mean()) if ans.any() else None,
                         "end_to_end_accuracy_answered": float((crop_ok & dis_ok)[ans].mean()) if ans.any() else None,
                         "end_to_end_accuracy_all": float((crop_ok & dis_ok).mean()),
                         "disease_conformal_coverage": float(ds[np.arange(len(ds)), 1 - labd.y.to_numpy()].mean()),
                         "quality_pass_rate_targets": float(alld[alld.crop != "other"].quality_ok.mean()),
                         "ood_flag_rate_targets": float(alld[alld.crop != "other"].ood_flag.mean()),
                         "abstention_curve": abstention_curve(np.maximum(pd_, 1 - pd_), dis_ok),
                         "per_source": {s: {"n": int(m.sum()), "coverage": float(ans[m].mean()),
                                            "accuracy_answered": float((crop_ok & dis_ok)[m & ans].mean()) if (m & ans).any() else None}
                                        for s, m in ((s, (labd.source == s).to_numpy()) for s in sorted(labd.source.unique()))}}
    res["small_print"] = ("labels come from the source datasets' authors and were not re-checked by an agronomist; "
                          "all sources are from Bangladesh; pest classes (aphids, insect, flea beetle) are excluded")
    out = path("results", cfg)
    (out / "wild_ext.json").write_text(json.dumps(res, indent=1, default=float))
    preds = labd[["path", "crop", "label", "y", "group", "source"]].copy()
    for a, p in P.items():
        preds[a] = p
    preds.to_csv(out / "p3" / "preds_wild_ext.csv.gz", index=False)
    print(json.dumps({k: res[k] for k in ("n", "end_to_end", "other_plants")}, indent=1, default=float)[:3000])
    print({a: round(v["pooled"]["macro_f1"], 3) for a, v in res["disease"].items() if "pooled" in v})
    return 0


if __name__ == "__main__":
    sys.exit(main())
