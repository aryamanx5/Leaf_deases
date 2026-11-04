"""Phase 4 training, selection and calibration (docs/PLAN_phase4_more_data.md, pre-registered).

Reads clean TRAIN / VAL (+ TRAIN background-swap views) and data/p4 p4_train / p4_val. Never reads p4_test, clean
TEST, data/wild or data/wild_ext (those are scored once by scripts/p4_eval.py).

  1. every frozen-backbone / feature configuration of Phase 3 is re-fitted on clean TRAIN (+ swap views for `swap`)
     PLUS p4_train; head choice = mean(p4_val macro-F1, clean VAL macro-F1);
  2. configuration choice = best p4_val pooled macro-F1; tie (point >= best's grouped-bootstrap lower 95% bound)
     -> lowest CPU ms/image;
  3. calibration on p4_val: temperature, LAC disease threshold; crop head (BioCLIP 2 logistic) re-fitted with
     p4_train target leaves (incl. crop-only PhilMedic turmeric), temperature + LAC on p4_val; kNN OOD reference
     = Phase 3 reference + p4_train target leaves, threshold keeps 95% of p4_val target leaves.
Writes results/p4/selection.json, models/p4/*, models/manifest_p4.json.
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
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from leafdx.config import ROOT, load_config, path  # noqa: E402
from leafdx.ip.features import FEATURE_NAMES  # noqa: E402
from leafdx.p3 import heads  # noqa: E402
from leafdx.p3.data import file_hash  # noqa: E402
from leafdx.p3.features import features_for  # noqa: E402
from leafdx.p3.frames import build_frames, embeddings  # noqa: E402
from leafdx.p3.stats import (  # noqa: E402
    conformal_threshold,
    fit_temperature,
    grouped_bootstrap,
    point_metrics,
    sigmoid,
)
from leafdx.robust.infer import knn_distance  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from p3_disease import EMB_ARMS, X_of  # noqa: E402

CROPS = ("turmeric", "blackgram")
COLS = ["path", "file", "key", "crop", "label", "y", "group", *FEATURE_NAMES]
TIE_KEY = "macro_f1"


def frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df.reset_index(drop=True).copy()
    d["file"] = d.path
    d["key"] = [file_hash(p) for p in d.path]
    d["y"] = (d.label == "diseased").astype(int)
    return d


def specs() -> list[tuple[str, str, str | None, list[str]]]:
    s = [("B_svm", "feat", None, ["svm"]), ("B2_lgbm", "feat", None, ["lgbm"])]
    s += [(a, "emb", bb, ["lr", "mlp", "lgbm"]) for a, bb in EMB_ARMS.items()]
    s += [(f"F_fusion_{a.split('_')[1]}", "fusion", bb, ["lr", "mlp", "lgbm"]) for a, bb in EMB_ARMS.items()]
    return s


def arm_of(cid: str) -> str:
    return "F_fusion" if cid.startswith("F_fusion") else cid.split("[")[0]


def fit_config(kind, bb, head_kinds, mode, variant, F, P, cfg) -> tuple[dict, dict, dict]:
    base = F["train"] if variant == "orig" else pd.concat([F["train"], F["train_swap"]], ignore_index=True)
    tr = pd.concat([base[COLS], P["p4_train"][COLS]], ignore_index=True)
    Xtr, ytr = X_of(kind, tr, bb), tr.y.to_numpy()
    Xs = {s: X_of(kind, d, bb) for s, d in (("p4_val", P["p4_val"]), ("val", F["val"]))}
    ys = {"p4_val": P["p4_val"].y.to_numpy(), "val": F["val"].y.to_numpy()}
    crop_of = {"p4_val": P["p4_val"].crop.to_numpy(), "val": F["val"].crop.to_numpy()}
    z = {s: np.zeros(len(ys[s])) for s in Xs}
    info, fitted = {"mode": mode, "variant": variant, "n_train": len(tr), "n_train_p4": len(P["p4_train"])}, {}
    for g in ([None] if mode == "pooled" else list(CROPS)):
        mt = np.ones(len(tr), bool) if g is None else (tr.crop == g).to_numpy()
        mv = {s: np.ones(len(ys[s]), bool) if g is None else crop_of[s] == g for s in Xs}
        cands = [c for k in head_kinds for c in heads.candidates(k, cfg["seed"], cfg["p3"]["head_C"])]
        f = heads.select(cands, Xtr[mt], ytr[mt], Xs["p4_val"][mv["p4_val"]], ys["p4_val"][mv["p4_val"]],
                         Xs["val"][mv["val"]], ys["val"][mv["val"]])
        fitted[g or "all"] = f.model
        info[g or "all"] = {"head": f.name, "sel": f.sel_score, "p4_val_f1": f.val_f1, "val_f1": f.val_swap_f1}
        for s in Xs:
            if mv[s].any():
                z[s][mv[s]] = heads.raw_score(f.model, Xs[s][mv[s]])
    return info, z, {"kind": kind, "backbone": bb, "mode": mode, "heads": fitted}


def cached(cid, keys, cfg, fit):
    key = hashlib.sha1(json.dumps([cid, keys, cfg["p3"]["head_C"], cfg["seed"]], default=str).encode()).hexdigest()[:16]
    f = ROOT / ".cache" / "p4configs" / f"{cid}_{key}.joblib"
    if f.exists():
        return joblib.load(f)
    t0 = time.time()
    res = fit()
    res[0]["fit_s"] = time.time() - t0
    f.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(res, f)
    return res


def choose(scores, ms, y, groups, nb, seed):
    f1 = {c: point_metrics(y, s)["macro_f1"] for c, s in scores.items()}
    best = max(f1, key=f1.get)
    lo = grouped_bootstrap(y, scores[best], groups, nb, seed, keys=(TIE_KEY,))[TIE_KEY][0]
    tied = [c for c in f1 if f1[c] >= lo]
    win = min(tied, key=lambda c: (ms.get(c, np.inf), -f1[c]))
    return win, {"best_point": best, "best_f1": f1[best], "best_ci_low": lo, "tied": sorted(tied, key=lambda c: -f1[c]),
                 "winner": win, "winner_f1": f1[win]}


def main() -> int:
    cfg = load_config()
    heads.pin_lightgbm_openmp()
    nb, seed, alpha = cfg["p3"]["bootstrap"], cfg["seed"], cfg["p3"]["conformal_alpha"]
    mdir, out = path("models", cfg), path("results", cfg) / "p4"
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((mdir / "manifest.json").read_text())
    timing = json.loads((path("results", cfg) / "p3" / "timing.json").read_text())["components_ms"]
    t0 = time.time()

    lab = pd.read_csv(ROOT / cfg["p4"]["root"] / "labels.csv")
    lab = lab[lab.p4_split.isin(["p4_train", "p4_val"])]  # p4_test is not read here
    dis = lab[lab.label.isin(["healthy", "diseased"])]
    P = {s: frame(dis[dis.p4_split == s]) for s in ("p4_train", "p4_val")}
    for d in P.values():
        d[FEATURE_NAMES] = features_for(list(d.path), list(d.crop), cfg)[FEATURE_NAMES].to_numpy(float)
    F = build_frames()
    print("p4:", {s: len(d) for s, d in P.items()}, "clean:", {s: len(F[s]) for s in ("train", "train_swap", "val")},
          f"{time.time() - t0:.0f}s", flush=True)

    # ---- 1. re-fit every configuration ----------------------------------------------------------------------------
    keys = {s: list(P[s].key) for s in P} | {s: list(F[s].key) for s in ("train", "train_swap", "val")}
    results, bundles = {}, {}
    for arm, kind, bb, hk in specs():
        for mode in ("pooled", "percrop"):
            for variant in ("orig", "swap"):
                cid = f"{arm}[{mode},{variant}]"
                info, z, b = cached(cid, keys, cfg, lambda kind=kind, bb=bb, hk=hk, mode=mode, variant=variant:
                                    fit_config(kind, bb, hk, mode, variant, F, P, cfg))
                results[cid], bundles[cid] = (info, z), b
                print(f"{cid:38s} p4_val F1 {point_metrics(P['p4_val'].y, sigmoid(z['p4_val']))['macro_f1']:.3f}  "
                      f"val F1 {point_metrics(F['val'].y, sigmoid(z['val']))['macro_f1']:.3f}  {info['fit_s']:.0f}s",
                      flush=True)

    # ---- 2. choose -------------------------------------------------------------------------------------------------
    y, g = P["p4_val"].y.to_numpy(), P["p4_val"].group.to_numpy()
    scores = {c: sigmoid(z["p4_val"]) for c, (_i, z) in results.items()}
    ms = {}
    for c, b in bundles.items():
        ms[c] = (timing["stage1"] if b["kind"] in ("feat", "fusion") else 0.0) + \
            (timing[f"embed_{b['backbone']}"] if b["kind"] in ("emb", "fusion") else 0.0)
    win, sel = choose(scores, ms, y, g, nb, seed)
    print("winner:", win, json.dumps({k: sel[k] for k in ("best_point", "best_f1", "best_ci_low", "winner_f1")}), flush=True)

    # ---- 3. calibrate on p4_val ------------------------------------------------------------------------------------
    z_w = results[win][1]["p4_val"]
    T = fit_temperature(z_w, y)
    p_w = sigmoid(z_w / T)
    q_dis = conformal_threshold(np.column_stack([p_w, 1 - p_w]), 1 - y, alpha, "lac")

    # crop head: BioCLIP 2 logistic on clean TRAIN (+ swap views) + p4_train target leaves (incl. crop-only)
    tgt = lab[lab.crop.isin(CROPS)]
    C4 = {s: frame(tgt[tgt.p4_split == s]) for s in ("p4_train", "p4_val")}
    bb_c = "bioclip2"
    ctr = pd.concat([F["train"], F["train_swap"], C4["p4_train"]], ignore_index=True)
    Xc, yc = embeddings(bb_c, ctr), (ctr.crop == "turmeric").astype(int).to_numpy()
    Xcv = {"p4_val": embeddings(bb_c, C4["p4_val"]), "val": embeddings(bb_c, F["val"])}
    ycv = {"p4_val": (C4["p4_val"].crop == "turmeric").astype(int).to_numpy(),
           "val": (F["val"].crop == "turmeric").astype(int).to_numpy()}
    best = None
    for Cc in cfg["p3"]["head_C"]:
        mc = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=Cc, max_iter=3000, class_weight="balanced"))])
        mc.fit(Xc, yc)
        selc = np.mean([point_metrics(ycv[s], sigmoid(mc.decision_function(Xcv[s])))["macro_f1"] for s in Xcv])
        if best is None or selc > best[0] + 1e-9:
            best = (selc, Cc, mc)
    zc = best[2].decision_function(Xcv["p4_val"])
    Tc = fit_temperature(zc, ycv["p4_val"])
    pc = sigmoid(zc / Tc)
    q_crop = conformal_threshold(np.column_stack([pc, 1 - pc]), 1 - ycv["p4_val"], alpha, "lac")
    crop_acc = float(((zc >= 0).astype(int) == ycv["p4_val"]).mean())

    # OOD: Phase 3 reference + p4_train target leaves; threshold keeps 95% of p4_val target leaves
    ood_bb, k = man["trust"]["ood_backbone"], man["trust"]["ood_k"]
    ref = np.r_[np.load(ROOT / man["trust"]["ood_reference"]).astype(np.float32),
                embeddings(ood_bb, C4["p4_train"]).astype(np.float32)]
    d_tgt = knn_distance(embeddings(ood_bb, C4["p4_val"]), ref, k)
    ood_thr = float(np.quantile(d_tgt, cfg["p3"]["ood_quantile"]))
    oth = frame(lab[(lab.crop == "other") & (lab.p4_split == "p4_val")])
    other_refused = float((knn_distance(embeddings(ood_bb, oth), ref, k) > ood_thr).mean()) if len(oth) else None

    # ---- write the candidate bundle ----------------------------------------------------------------------------------
    pdir = mdir / "p4"
    (pdir / "arms").mkdir(parents=True, exist_ok=True)
    arm = arm_of(win)
    bw = {**bundles[win], "temperature": T, "config": win}
    joblib.dump(bw, pdir / "arms" / f"{arm}.joblib")
    joblib.dump({"name": "bioclip2_head", "model": best[2], "temperature": Tc,
                 "info": {"kind": "logistic head", "backbone": bb_c, "C": best[1], "trained_on": "clean train + swap + p4_train"}},
                pdir / "crop_head.joblib")
    np.save(pdir / "ood_reference.npy", ref.astype(np.float16))
    new = json.loads(json.dumps(man))
    new["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    new["arms"][arm] = {"config": win, "kind": bw["kind"], "file": f"models/p4/arms/{arm}.joblib",
                        **({"backbone": bw["backbone"]} if bw.get("backbone") else {}),
                        "trained_on": "clean train (+ swap views) + p4_train"}
    new["production"] = {"disease": arm, "disease_config": win, "crop": "bioclip2_head",
                         "reason": "Phase 4: trained with field data (docs/PLAN_phase4_more_data.md)"}
    new["crop"] = {**man["crop"], "file": "models/p4/crop_head.joblib"}
    new["trust"] = {**man["trust"], "disease_model": win, "disease_q_lac": q_dis, "crop_q_lac": q_crop,
                    "ood_threshold": ood_thr, "ood_reference": "models/p4/ood_reference.npy", "calibrated_on": "p4_val"}
    new["stacker"] = None
    new["p4"] = {"labels_sha1": hashlib.sha1((ROOT / cfg["p4"]["root"] / "labels.csv").read_bytes()).hexdigest(),
                 "n_p4_train": len(P["p4_train"]), "n_p4_val": len(P["p4_val"])}
    (mdir / "manifest_p4.json").write_text(json.dumps(new, indent=1))

    selection = {
        "plan": "docs/PLAN_phase4_more_data.md", "labels_sha1": new["p4"]["labels_sha1"],
        "n": {"p4_train": len(P["p4_train"]), "p4_val": len(P["p4_val"]), "p4_val_groups": int(P["p4_val"].group.nunique()),
              "p4_train_by_crop_label": {f"{a}/{b}": int(v) for (a, b), v in P["p4_train"].groupby(["crop", "label"]).size().items()},
              "p4_val_by_crop_label": {f"{a}/{b}": int(v) for (a, b), v in P["p4_val"].groupby(["crop", "label"]).size().items()},
              "crop_rows_p4_train": len(C4["p4_train"]), "other_p4_val": len(oth)},
        "candidates": {c: {"p4_val_macro_f1": point_metrics(y, scores[c])["macro_f1"],
                           "val_macro_f1": point_metrics(F["val"].y, sigmoid(results[c][1]["val"]))["macro_f1"],
                           "turmeric_p4_val_f1": point_metrics(y[P["p4_val"].crop == "turmeric"],
                                                               scores[c][(P["p4_val"].crop == "turmeric").to_numpy()])["macro_f1"],
                           "blackgram_p4_val_f1": point_metrics(y[P["p4_val"].crop == "blackgram"],
                                                                scores[c][(P["p4_val"].crop == "blackgram").to_numpy()])["macro_f1"],
                           "ms": ms[c], "heads": {kk: v["head"] for kk, v in results[c][0].items() if isinstance(v, dict)}}
                       for c in results},
        "selection": sel,
        "calibration": {"temperature": T, "disease_q_lac": q_dis, "crop_C": best[1], "crop_temperature": Tc,
                        "crop_q_lac": q_crop, "crop_accuracy_p4_val": crop_acc, "ood_threshold": ood_thr,
                        "ood_reference_rows": int(len(ref)), "ood_other_species_refused_p4_val": other_refused},
        "seconds": time.time() - t0,
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=1, default=float))
    print("written: results/p4/selection.json, models/manifest_p4.json, models/p4/", f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
