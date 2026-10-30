"""P3-M03: disease arms on the identical grouped split, per crop and pooled.

Arms: A rule score; B 57 features -> SVM; B2 57 features -> LightGBM; C1-C3 fine-tuned
CNNs (checkpoints from p3_train_cnns.py); D DINOv2 frozen + head; E OpenCLIP / BioCLIP 2
frozen + head; F fusion [embedding, 57 features] + head.

Training regimes: ``pooled`` (one model, crop indicator as an input) and ``percrop`` (one
model per crop); data variants ``orig`` (TRAIN photos) and ``swap`` (TRAIN photos + K
background-swap views). Heads and regimes are selected on clean VAL + VAL-swap only
(p3.heads.select). Every configuration is temperature-scaled on VAL + VAL-swap (P3-M05) and
evaluated on TEST and on the wild set (reporting only).

Outputs: results/p3/preds_disease.csv.gz (every configuration x set x image),
results/p3/disease_configs.json (selection log), results/p3/timing.json.
Metrics/CIs are computed by scripts/p3_metrics.py.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from leafdx.config import ROOT, load_config, path  # noqa: E402
from leafdx.ip.decide import RuleBasedScorer  # noqa: E402
from leafdx.ip.features import FEATURE_NAMES  # noqa: E402
from leafdx.p3 import heads  # noqa: E402
from leafdx.p3.frames import build_frames, crop_ind, embeddings  # noqa: E402
from leafdx.p3.stats import fit_temperature, sigmoid  # noqa: E402

EVAL_SETS = ["val", "val_swap", "test", "test_swap", "wild", "user",
             "val_leafonly", "test_leafonly", "wild_leafonly", "user_leafonly"]
EMB_ARMS = {"D_dinov2": "dinov2_vits14", "E_openclip": "openclip_vitb32", "E_bioclip2": "bioclip2"}
CROPS = ("turmeric", "blackgram")


def X_of(kind: str, d: pd.DataFrame, bb: str | None) -> np.ndarray:
    parts = []
    if kind in ("emb", "fusion"):
        parts.append(embeddings(bb, d))
    if kind in ("feat", "fusion"):
        parts.append(d[FEATURE_NAMES].to_numpy(float))
    parts.append(crop_ind(d))
    return np.column_stack(parts).astype(np.float32)


def fit_config(kind, bb, head_kinds, mode, variant, F, cfg) -> tuple[dict, dict]:
    """Returns (info, z per eval set)."""
    tr = F["train"] if variant == "orig" else pd.concat([F["train"], F["train_swap"]], ignore_index=True)
    cands = lambda: [c for k in head_kinds for c in heads.candidates(k, cfg["seed"], cfg["p3"]["head_C"])]  # noqa: E731
    Xs = {s: X_of(kind, F[s], bb) for s in EVAL_SETS}
    Xtr = X_of(kind, tr, bb)
    z = {s: np.zeros(len(F[s])) for s in EVAL_SETS}
    info = {"mode": mode, "variant": variant, "n_train": len(tr)}
    fitted = {}
    groups = [None] if mode == "pooled" else list(CROPS)
    for g in groups:
        mt = np.ones(len(tr), bool) if g is None else (tr.crop == g).to_numpy()
        mv = np.ones(len(F["val"]), bool) if g is None else (F["val"].crop == g).to_numpy()
        ms = np.ones(len(F["val_swap"]), bool) if g is None else (F["val_swap"].crop == g).to_numpy()
        f = heads.select(cands(), Xtr[mt], tr.y.to_numpy()[mt], Xs["val"][mv], F["val"].y.to_numpy()[mv],
                         Xs["val_swap"][ms], F["val_swap"].y.to_numpy()[ms])
        fitted[g or "all"] = f.model
        info[g or "all"] = {"head": f.name, "sel": f.sel_score, "val_f1": f.val_f1, "val_swap_f1": f.val_swap_f1,
                            "tried": f.tried}
        for s in EVAL_SETS:
            m = np.ones(len(F[s]), bool) if g is None else (F[s].crop == g).to_numpy()
            if m.any():
                z[s][m] = heads.raw_score(f.model, Xs[s][m])
    info["_models"] = {"kind": kind, "backbone": bb, "mode": mode, "heads": fitted}
    return info, z


def rule_arm(F, cfg) -> tuple[dict, dict]:
    r = RuleBasedScorer(cfg["decide"]["rule"]["weights"]).fit_scaling(F["train"])
    g = cfg["decide"]["rule"]["threshold_grid"]
    r.fit_threshold(F["val"], np.linspace(g["start"], g["stop"], g["num"]))
    z = {s: 10 * (r.score(F[s]).to_numpy() - F[s].crop.map(r.threshold).to_numpy()) for s in EVAL_SETS}
    return {"mode": "pooled (per-crop scaling and threshold)", "variant": "orig", "thresholds": r.threshold,
            "_models": {"kind": "rule", "rule": r}}, z


def cnn_arms(F, cfg) -> list[tuple[str, dict, dict]]:
    from leafdx.p3.cnnpred import logits

    tr = json.loads((path("results", cfg) / "p3_cnn_train.json").read_text())
    out = []
    for arm in cfg["p3"]["cnn_arms"]["arms"]:
        for v in ("swap", "plain"):
            rec = tr.get(f"{arm['name']}_{v}", {"failed": "not trained"})
            ck = rec.get("checkpoint")
            if "failed" in rec or not ck or not (ROOT / ck).exists():
                out.append((f"{arm['name']}[{v}]", {"skipped": rec.get("failed", "missing checkpoint")}, None))
                continue
            t0 = time.time()
            z = {s: logits(ck, arm["arch"], F[s]) for s in EVAL_SETS}
            out.append((f"{arm['name']}[{v}]", {"mode": "shared backbone, one head per crop",
                                                "variant": v, "train": rec, "predict_s": time.time() - t0}, z))
            print(out[-1][0], f"{time.time() - t0:.0f}s", flush=True)
    return out


def cached(cid: str, F, cfg, fit):
    """(info, z) for one configuration, cached in .cache/p3configs keyed by the config id, the
    frames' image keys and the p3 config, so a crash later in the run does not lose finished fits."""
    import hashlib

    key = hashlib.sha1(json.dumps([cid, {s: list(F[s].key) for s in ("train", "train_swap", *EVAL_SETS)},
                                   cfg["p3"], cfg["decide"]], sort_keys=True, default=str).encode()).hexdigest()[:16]
    f = ROOT / ".cache" / "p3configs" / f"{cid}_{key}.joblib"
    if f.exists():
        return joblib.load(f)
    t1 = time.time()
    info, z = fit()
    info["fit_s"] = time.time() - t1
    f.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((info, z), f)
    return info, z


def main() -> int:
    cfg = load_config()
    t0 = time.time()
    F = build_frames()
    print("frames:", {k: len(v) for k, v in F.items()}, f"{time.time() - t0:.0f}s", flush=True)
    configs: list[tuple[str, str, dict, dict]] = []  # (arm, config id, info, z)

    info, z = cached("A_rule", F, cfg, lambda: rule_arm(F, cfg))
    configs.append(("A_rule", "A_rule", info, z))
    specs = [("B_svm", "feat", None, ["svm"]), ("B2_lgbm", "feat", None, ["lgbm"])]
    specs += [(a, "emb", bb, ["lr", "mlp", "lgbm"]) for a, bb in EMB_ARMS.items()]
    specs += [(f"F_fusion_{a.split('_')[1]}", "fusion", bb, ["lr", "mlp", "lgbm"]) for a, bb in EMB_ARMS.items()]
    man = json.loads((path("results", cfg) / "p3" / "embed_manifest.json").read_text())
    for arm, kind, bb, hk in specs:
        if bb and man["backbones"].get(bb, {}).get("status") != "ok":
            configs.append((arm, f"{arm}", {"skipped": man["backbones"].get(bb, {}).get("reason", "no embeddings")}, None))
            continue
        for mode in ("pooled", "percrop"):
            for variant in ("orig", "swap"):
                cid = f"{arm}[{mode},{variant}]"
                info, z = cached(cid, F, cfg, lambda kind=kind, bb=bb, hk=hk, mode=mode, variant=variant:
                                 fit_config(kind, bb, hk, mode, variant, F, cfg))
                configs.append((arm, cid, info, z))
                print(configs[-1][1], {k: info[k]["head"] for k in ("all", *CROPS) if k in info},
                      f"{info['fit_s']:.0f}s", flush=True)
    for cid, info, z in cnn_arms(F, cfg):
        configs.append((cid.split("[")[0], cid, info, z))

    # temperature scaling, then the long prediction table
    rows, log = [], {}
    for arm, cid, info, z in configs:
        if z is None:
            log[cid] = {"arm": arm, **info}
            continue
        # VAL + VAL-swap: both derive from VAL images only; clean VAL alone is near-separable for
        # some arms, which drives T -> 0 (over-sharpening)
        T = fit_temperature(np.r_[z["val"], z["val_swap"]], np.r_[F["val"].y, F["val_swap"].y])
        info["temperature"] = T
        info = dict(info)
        if "_models" in info:  # fitted heads for the API (selected ones are copied to models/p3/arms)
            mdir = ROOT / ".cache" / "p3models"
            mdir.mkdir(parents=True, exist_ok=True)
            joblib.dump({**info.pop("_models"), "temperature": T, "config": cid}, mdir / f"{cid}.joblib")
        log[cid] = {"arm": arm, **info}
        for s in EVAL_SETS:
            d = F[s]
            rows.append(pd.DataFrame({"arm": arm, "config": cid, "set": s, "path": d.path, "crop": d.crop,
                                      "label": d.label, "y": d.y, "group": d.group, "wild_half": d.wild_half,
                                      "z": z[s], "p_raw": sigmoid(z[s]), "p_cal": sigmoid(z[s] / T)}))
    P = pd.concat(rows, ignore_index=True)
    out = path("results", cfg) / "p3"
    P.to_csv(out / "preds_disease.csv.gz", index=False)
    (out / "disease_configs.json").write_text(json.dumps(log, indent=1, default=str))
    print(f"done in {time.time() - t0:.0f}s; {P.config.nunique()} configurations", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
