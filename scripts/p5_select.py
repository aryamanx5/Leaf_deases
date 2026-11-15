"""Phase 5 selection and calibration (docs/PLAN_phase5_confidence.md, pre-registered).

Reads clean VAL and data/p4 p4_val only (the selection / calibration sets of Phase 4). Never reads p4_test, clean
TEST, data/wild or data/wild_ext (those are scored by scripts/p5_eval.py). Nothing is trained: every member is a
Phase 4 fit loaded from .cache/p4configs (this script fails if one is missing rather than re-fitting it).

  1. candidates M0..M3 (the deployed head and three mean-of-logits ensembles of Phase 4 heads); admissible if p4_val
     macro-F1 >= M0's and clean VAL macro-F1 >= M0's - 0.01; lowest p4_val NLL wins, ties (0.005 nats) -> fewest members;
  2. refusal thresholds by selective prediction with a risk target (Clopper-Pearson 95% upper bound on the error
     among answered p4_val leaves): disease <= 2%, never below confidence 0.70; crop <= 1% and other-species refusal
     on p4_val not below its current value.
Writes results/p5/selection.json, models/p5/*, models/manifest_p5.json.
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

from leafdx.config import ROOT, load_config, path  # noqa: E402
from leafdx.p3 import heads  # noqa: E402
from leafdx.p3.frames import build_frames, embeddings  # noqa: E402
from leafdx.p3.stats import conformal_sets, fit_temperature, point_metrics, sigmoid  # noqa: E402
from leafdx.p5 import MeanLogit, confidence_summary, ensemble_logit, selective_threshold  # noqa: E402
from leafdx.robust.infer import knn_distance  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from p4_train import CROPS, frame  # noqa: E402

VARIANT = "[pooled,orig]"
MEMBERS = {  # arm name in the bundle -> Phase 4 configuration
    "D_dinov2": "D_dinov2" + VARIANT, "E_openclip": "E_openclip" + VARIANT, "E_bioclip2": "E_bioclip2" + VARIANT,
    "F_fusion_dinov2": "F_fusion_dinov2" + VARIANT, "F_fusion_openclip": "F_fusion_openclip" + VARIANT,
    "F_fusion_bioclip2": "F_fusion_bioclip2" + VARIANT,
}
CANDIDATES = {
    "M0": ["D_dinov2"],
    "M1": ["D_dinov2", "E_openclip", "E_bioclip2"],
    "M2": ["D_dinov2", "E_openclip", "E_bioclip2", "F_fusion_dinov2", "F_fusion_openclip", "F_fusion_bioclip2"],
    "M3": ["D_dinov2", "E_bioclip2"],
}
NLL_TIE = 0.005
DISEASE_FLOOR, DISEASE_MAX_ERROR = 0.70, 0.02
CROP_FLOOR, CROP_MAX_ERROR = 0.50, 0.01
LABELS = {"F_fusion_dinov2": "Fusion: DINOv2 + 57 features", "F_fusion_openclip": "Fusion: OpenCLIP + 57 features",
          "F_fusion_bioclip2": "Fusion: BioCLIP 2 + 57 features"}


def load_fit(cid: str, keys: dict, cfg: dict):
    """The Phase 4 fit of one configuration: (info, z on p4_val / val, bundle). Same cache key as p4_train.cached."""
    key = hashlib.sha1(json.dumps([cid, keys, cfg["p3"]["head_C"], cfg["seed"]], default=str).encode()).hexdigest()[:16]
    f = ROOT / ".cache" / "p4configs" / f"{cid}_{key}.joblib"
    if not f.exists():
        raise SystemExit(f"missing Phase 4 fit {f.name}: run scripts/p4_train.py first (Phase 5 never re-fits)")
    return joblib.load(f)


def main() -> int:
    cfg = load_config()
    heads.pin_lightgbm_openmp()
    mdir, out = path("models", cfg), path("results", cfg) / "p5"
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((mdir / "manifest.json").read_text())
    if man["production"]["disease"] != "D_dinov2" or man["production"].get("disease_config") != MEMBERS["D_dinov2"]:
        raise SystemExit("models/manifest.json is not the Phase 4 bundle this plan starts from")
    t0 = time.time()

    lab = pd.read_csv(ROOT / cfg["p4"]["root"] / "labels.csv")
    lab = lab[lab.p4_split.isin(["p4_train", "p4_val"])]  # p4_test is not read here
    dis = lab[lab.label.isin(["healthy", "diseased"])]
    P = {s: frame(dis[dis.p4_split == s]) for s in ("p4_train", "p4_val")}
    F = build_frames()
    keys = {s: list(P[s].key) for s in P} | {s: list(F[s].key) for s in ("train", "train_swap", "val")}
    y = {"p4_val": P["p4_val"].y.to_numpy(), "val": F["val"].y.to_numpy()}

    # ---- members: Phase 4 fits, each with its own temperature on p4_val --------------------------------------------
    member, bundles = {}, {}
    for arm, cid in MEMBERS.items():
        info, z, b = load_fit(cid, keys, cfg)
        T = fit_temperature(z["p4_val"], y["p4_val"])
        member[arm] = {s: sigmoid(z[s] / T) for s in z}
        bundles[arm] = {**b, "temperature": T, "config": cid}
        print(f"{cid:34s} T {T:.3f}  p4_val F1 {point_metrics(y['p4_val'], member[arm]['p4_val'])['macro_f1']:.3f}  "
              f"val F1 {point_metrics(y['val'], member[arm]['val'])['macro_f1']:.3f}", flush=True)
    T0 = joblib.load(ROOT / man["arms"]["D_dinov2"]["file"])["temperature"]
    if abs(T0 - bundles["D_dinov2"]["temperature"]) > 1e-6:
        raise SystemExit("the cached D_dinov2 fit does not reproduce the deployed temperature")

    # ---- 1. candidates ---------------------------------------------------------------------------------------------
    cand, probs = {}, {}
    for cid, arms in CANDIDATES.items():
        if len(arms) == 1:
            T, p = 1.0, member[arms[0]]
        else:
            zz = {s: ensemble_logit([member[a][s] for a in arms]) for s in ("p4_val", "val")}
            T = fit_temperature(zz["p4_val"], y["p4_val"])
            p = {s: sigmoid(zz[s] / T) for s in zz}
        probs[cid] = p
        cand[cid] = {"members": arms, "ensemble_temperature": T,
                     "p4_val": {**point_metrics(y["p4_val"], p["p4_val"]), **confidence_summary(p["p4_val"], y["p4_val"])},
                     "val": {**point_metrics(y["val"], p["val"]), **confidence_summary(p["val"], y["val"])}}
    base = cand["M0"]
    for cid, c in cand.items():
        c["admissible"] = bool(c["p4_val"]["macro_f1"] >= base["p4_val"]["macro_f1"] - 1e-12
                               and c["val"]["macro_f1"] >= base["val"]["macro_f1"] - 0.01)
        print(f"{cid} {len(c['members'])} members  p4_val F1 {c['p4_val']['macro_f1']:.4f} NLL {c['p4_val']['nll']:.4f} "
              f"ECE {c['p4_val']['ece']:.3f} conf(right) {c['p4_val']['mean_confidence_right']:.3f}  "
              f"val F1 {c['val']['macro_f1']:.4f}  admissible {c['admissible']}", flush=True)
    adm = [c for c in cand if cand[c]["admissible"]]
    best_nll = min(cand[c]["p4_val"]["nll"] for c in adm)
    tied = [c for c in adm if cand[c]["p4_val"]["nll"] <= best_nll + NLL_TIE]
    win = min(tied, key=lambda c: (len(cand[c]["members"]), cand[c]["p4_val"]["nll"]))
    print("winner:", win, "tied:", tied, flush=True)

    # ---- 2. refusal thresholds -------------------------------------------------------------------------------------
    tr = man["trust"]
    p_w = probs[win]["p4_val"]
    conf = np.maximum(p_w, 1 - p_w)
    right = (p_w >= 0.5).astype(int) == y["p4_val"]
    tau_old = 1 - tr["disease_q_lac"]
    tau_d, rec_d = selective_threshold(conf, right, DISEASE_FLOOR, tau_old, DISEASE_MAX_ERROR)
    p0 = probs["M0"]["p4_val"]
    old_ans = np.maximum(p0, 1 - p0) >= tau_old
    rec_d["old"] = {"tau": tau_old, "answered_share": float(old_ans.mean()),
                    "errors": int((((p0 >= 0.5).astype(int) != y["p4_val"]) & old_ans).sum())}

    # crop gate: the deployed crop head on p4_val target leaves; other species must stay refused as often as now
    full = pd.read_csv(ROOT / cfg["p4"]["root"] / "labels.csv")
    v = full[full.p4_split == "p4_val"]
    tgt, oth = frame(v[v.crop.isin(CROPS)]), frame(v[v.crop == "other"])
    crop = joblib.load(ROOT / man["crop"]["file"])
    bbs = crop["info"].get("backbones") or [crop["info"]["backbone"]]

    def p_tu(d):
        return sigmoid(crop["model"].decision_function(np.column_stack([embeddings(bb, d) for bb in bbs])) / crop["temperature"])

    ref = np.load(ROOT / tr["ood_reference"]).astype(np.float32)

    def ood(d):
        return knn_distance(embeddings(tr["ood_backbone"], d), ref, tr["ood_k"]) > tr["ood_threshold"]

    pt, po = p_tu(tgt), p_tu(oth)
    cconf, cright = np.maximum(pt, 1 - pt), (pt >= 0.5) == (tgt.crop == "turmeric").to_numpy()
    oconf, oood = np.maximum(po, 1 - po), ood(oth)
    tau_c_old = 1 - tr["crop_q_lac"]

    def other_refused(tau):
        return float((oood | (oconf < tau)).mean()) if len(oth) else 1.0

    refused_now = other_refused(tau_c_old)
    tau_c, rec_c = selective_threshold(cconf, cright, CROP_FLOOR, tau_c_old, CROP_MAX_ERROR,
                                       extra_ok=lambda tau: other_refused(tau) >= refused_now - 1e-12)
    rec_c["other_species_refused"] = other_refused(tau_c)
    rec_c["old"] = {"tau": tau_c_old, "answered_share": float((cconf >= tau_c_old).mean()), "other_species_refused": refused_now,
                    "n_other": int(len(oth))}
    # all gates together on p4_val disease-labelled leaves (quality gate not run here, as in Phase 4's evaluation)
    dl = tgt.label.isin(["healthy", "diseased"]).to_numpy()
    assert list(tgt[dl].key) == list(P["p4_val"].key), "target frame and disease frame disagree on row order"
    t_ood = ood(tgt)
    gates = {}
    for name, (pp, td, tc) in {"old": (p0, tau_old, tau_c_old), "new": (p_w, tau_d, tau_c)}.items():
        ans = (cconf[dl] >= tc) & ~t_ood[dl] & (np.maximum(pp, 1 - pp) >= td)
        ok = cright[dl] & ((pp >= 0.5).astype(int) == y["p4_val"])
        gates[name] = {"answered_share": float(ans.mean()), "accuracy_answered": float(ok[ans].mean()) if ans.any() else None,
                       "errors_answered": int((~ok[ans]).sum()), "n": int(len(ans))}
    print("disease tau", round(tau_d, 4), rec_d, "\ncrop tau", round(tau_c, 4), rec_c, "\ngates p4_val", gates, flush=True)
    # the thresholds really are equivalent to the manifest's LAC fields (set = one label iff confidence >= tau)
    chk = conformal_sets(np.column_stack([p_w, 1 - p_w]), 1 - tau_d).sum(1) == 1
    assert (chk == (conf >= tau_d - 1e-12)).all()

    # ---- write the candidate bundle --------------------------------------------------------------------------------
    pdir = mdir / "p5"
    (pdir / "arms").mkdir(parents=True, exist_ok=True)
    new = json.loads(json.dumps(man))
    new["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    arms = cand[win]["members"]
    if win != "M0":
        for a in arms:
            if a == "D_dinov2":
                continue  # the deployed Phase 4 head, already in the bundle with this temperature
            joblib.dump(bundles[a], pdir / "arms" / f"{a}.joblib")
            new["arms"][a] = {"config": bundles[a]["config"], "kind": bundles[a]["kind"], "file": f"models/p5/arms/{a}.joblib",
                              "backbone": bundles[a]["backbone"], "trained_on": "clean train + p4_train (Phase 4 fit)",
                              **({"label": LABELS[a]} if a in LABELS else {})}
        name = f"Average of {len(arms)} models"
        joblib.dump({"members": arms, "meta": MeanLogit(), "temperature": cand[win]["ensemble_temperature"],
                     "name": name, "kind": "mean of member log-odds"}, pdir / "stacker.joblib")
        new["stacker"] = {"file": "models/p5/stacker.joblib", "members": arms, "kind": "mean_logit"}
        new["production"] = {"disease": "STACK", "disease_config": f"{win}: mean logit of " + " + ".join(MEMBERS[a] for a in arms),
                             "crop": man["production"]["crop"],
                             "reason": "Phase 5: ensemble of tied Phase 4 heads (docs/PLAN_phase5_confidence.md)"}
    else:
        new["production"] = {**man["production"], "reason": "Phase 5: thresholds re-derived (docs/PLAN_phase5_confidence.md)"}
    new["trust"] = {**tr, "disease_model": new["production"]["disease_config"], "disease_q_lac": 1 - tau_d,
                    "crop_q_lac": 1 - tau_c, "calibrated_on": "p4_val",
                    "threshold_rule": {"kind": "selective prediction, Clopper-Pearson 95% upper bound on error among answered",
                                       "disease_min_confidence": tau_d, "disease_max_error": DISEASE_MAX_ERROR,
                                       "crop_min_confidence": tau_c, "crop_max_error": CROP_MAX_ERROR}}
    new["p5"] = {"plan": "docs/PLAN_phase5_confidence.md", "candidate": win, "members": arms}
    (mdir / "manifest_p5.json").write_text(json.dumps(new, indent=1))

    selection = {"plan": "docs/PLAN_phase5_confidence.md", "n": {"p4_val": int(len(y["p4_val"])), "val": int(len(y["val"])),
                                                                "p4_val_targets": int(len(tgt)), "p4_val_other": int(len(oth))},
                 "member_temperatures": {a: bundles[a]["temperature"] for a in MEMBERS},
                 "candidates": cand, "selection": {"admissible": adm, "best_nll": best_nll, "tied": tied, "winner": win},
                 "thresholds": {"disease": rec_d, "crop": rec_c}, "gates_p4_val": gates, "seconds": time.time() - t0}
    (out / "selection.json").write_text(json.dumps(selection, indent=1, default=float))
    print("written: results/p5/selection.json, models/manifest_p5.json, models/p5/", f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
