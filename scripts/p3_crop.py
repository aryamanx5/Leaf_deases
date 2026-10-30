"""P3-M02: crop identification v3.

Candidates (all frozen backbones, cached embeddings):
  (a) zero-shot with text prompts (turmeric / black gram / "other") -- BioCLIP 2 and OpenCLIP;
  (b) DINOv2 + logistic head; (c) OpenCLIP + logistic head; (c2) BioCLIP 2 + logistic head;
  (d) fusions: concatenated-embedding head, and product of the best head with BioCLIP 2
      zero-shot.
Heads train on TRAIN photos + background-swap views. Selection: mean macro-F1 over clean VAL
and VAL-swap, tie-break by VAL+VAL-swap ECE (the add-on asks to select on wild macro-F1; that
conflicts with its own hard rule "do not pick models using the wild set", so the wild set is
reporting-only here). Temperature scaling on VAL + VAL-swap.

No negative ("other") training images are available under a usable licence in this repo, so
the third class is handled by the zero-shot "other" prompt and the kNN OOD gate (P3-M05).

Outputs: results/crop_models.json, results/crop_errors.md, results/p3/preds_crop.csv.gz.
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from leafdx.config import load_config, path  # noqa: E402
from leafdx.p3.frames import build_frames, embeddings  # noqa: E402
from leafdx.p3.stats import evaluate, fit_temperature, logit, point_metrics, sigmoid  # noqa: E402

SETS = ["val", "val_swap", "test", "test_swap", "wild_all", "user", "test_leafonly", "wild_all_leafonly"]
BB = {"dinov2": "dinov2_vits14", "openclip": "openclip_vitb32", "bioclip2": "bioclip2"}
ZS = {"openclip": "openclip_vitb32", "bioclip2": "bioclip2"}


def y_crop(d: pd.DataFrame) -> np.ndarray:
    return (d.crop == "turmeric").to_numpy(int)


def zero_shot(name: str, X: np.ndarray, prompts: dict) -> np.ndarray:
    """(n, 3) softmax over [turmeric, blackgram, other]; each class = mean prompt similarity."""
    from leafdx.robust.backbones import clip_zero_shot_text

    cls = ["turmeric", "blackgram", "other"]
    txt = clip_zero_shot_text([p for c in cls for p in prompts[c]], name)
    sims, k, out = 100 * X @ txt.T, 0, []
    for c in cls:
        out.append(sims[:, k:k + len(prompts[c])].mean(1))
        k += len(prompts[c])
    z = np.stack(out, 1)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main() -> int:
    cfg = load_config()
    F = build_frames()
    tr = pd.concat([F["train"], F["train_swap"]], ignore_index=True)
    man = json.loads((path("results", cfg) / "p3" / "embed_manifest.json").read_text())
    ok = {k: v for k, v in BB.items() if man["backbones"].get(v, {}).get("status") == "ok"}
    E = {k: {s: embeddings(bb, F[s]) for s in [*SETS, "train_swap", "train"]} for k, bb in ok.items()}
    Etr = {k: np.r_[E[k]["train"], E[k]["train_swap"]] for k in ok}
    ytr = y_crop(tr)
    cands: dict[str, dict] = {}  # name -> {"z": {set: logit P(turmeric)}, "p_other": ..., "info": ...}
    models = {}

    def lr_head(X, Xs, name, info):
        best = None
        for C in cfg["p3"]["head_C"]:
            m = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])
            m.fit(X, ytr)
            z = {s: m.decision_function(Xs[s]) for s in SETS}
            sel = np.mean([point_metrics(y_crop(F[s]), sigmoid(z[s]))["macro_f1"] for s in ("val", "val_swap")])
            if best is None or sel > best[0] + 1e-9:
                best = (sel, C, m, z)
        cands[name] = {"z": best[3], "info": {**info, "C": best[1]}}
        models[name] = best[2]

    t0 = time.time()
    for k in ok:
        lr_head(Etr[k], E[k], f"{k}_head", {"kind": "logistic head", "backbone": BB[k]})
    if len(ok) >= 2:
        ks = [k for k in ("dinov2", "bioclip2", "openclip") if k in ok][:2]
        lr_head(np.column_stack([Etr[k] for k in ks]), {s: np.column_stack([E[k][s] for k in ks]) for s in SETS},
                f"concat_{'+'.join(ks)}_head", {"kind": "logistic head on concatenated embeddings", "backbones": ks})
    for k in ZS:
        if k in ok:
            pz = {s: zero_shot(ZS[k], E[k][s], cfg["p3"]["crop_prompts"]) for s in SETS}
            cands[f"{k}_zeroshot"] = {"z": {s: logit(pz[s][:, 0] / (pz[s][:, 0] + pz[s][:, 1])) for s in SETS},
                                      "p_other": {s: pz[s][:, 2] for s in SETS},
                                      "info": {"kind": "zero-shot", "backbone": ZS[k], "prompts": cfg["p3"]["crop_prompts"]}}

    def sel_of(z):
        return np.mean([point_metrics(y_crop(F[s]), sigmoid(z[s]))["macro_f1"] for s in ("val", "val_swap")])

    heads = [n for n in cands if n.endswith("_head")]
    best_head = max(heads, key=lambda n: sel_of(cands[n]["z"]))
    if "bioclip2_zeroshot" in cands:
        zs = cands["bioclip2_zeroshot"]
        cands[f"{best_head}+bioclip2_zeroshot"] = {
            "z": {s: cands[best_head]["z"][s] + zs["z"][s] for s in SETS}, "p_other": zs["p_other"],
            "info": {"kind": "product of experts (sum of log-odds)", "members": [best_head, "bioclip2_zeroshot"]}}

    # temperature on VAL + VAL-swap, metrics
    res, rows = {}, []
    for n, c in cands.items():
        z = c["z"]
        zv, yv = np.r_[z["val"], z["val_swap"]], np.r_[y_crop(F["val"]), y_crop(F["val_swap"])]
        T = fit_temperature(zv, yv)
        p = {s: sigmoid(z[s] / T) for s in SETS}
        sel = sel_of(z)
        ece_v = point_metrics(yv, sigmoid(zv / T))["ece"]
        r = {"info": c["info"], "temperature": T, "selection": {"val_valswap_macro_f1": sel, "val_valswap_ece": ece_v}}
        for s in SETS:
            d = F[s]
            g = d.group
            if s == "user":
                r[s] = [{"path": pp, "p_turmeric": float(pt), "p_other": float(c["p_other"][s][i]) if "p_other" in c else None}
                        for i, (pp, pt) in enumerate(zip(d.path, p[s], strict=True))]
            else:
                r[s] = evaluate(y_crop(d), p[s], g, cfg["p3"]["bootstrap"] if s in ("test", "wild_all") else 0, cfg["seed"]) \
                    if s in ("test", "wild_all") else point_metrics(y_crop(d), p[s])
            rows.append(pd.DataFrame({"config": n, "set": s, "path": d.path, "crop": d.crop, "group": g,
                                      "wild_half": d.wild_half, "z": z[s], "p_turmeric": p[s],
                                      "p_other": c["p_other"][s] if "p_other" in c else np.nan}))
        res[n] = r
        print(n, f"sel {sel:.4f} ece {ece_v:.4f} T {T:.2f} test acc {res[n]['test']['accuracy']:.3f} "
                 f"wild acc {res[n]['wild_all']['accuracy']:.3f} f1 {res[n]['wild_all']['macro_f1']:.3f}", flush=True)
    chosen = sorted(res, key=lambda n: (-round(res[n]["selection"]["val_valswap_macro_f1"], 4),
                                        res[n]["selection"]["val_valswap_ece"]))[0]
    P = pd.concat(rows, ignore_index=True)
    out = path("results", cfg)
    P.to_csv(out / "p3" / "preds_crop.csv.gz", index=False)
    out_json = {"selection_rule": "max mean macro-F1 on clean VAL and VAL-swap, tie-break lower ECE on the same; "
                                  "wild set reporting only", "chosen": chosen, "candidates": res,
                "negatives": "no openly licensed 'other' training images in the repository; 'other' handled by the "
                             "zero-shot prompt and the kNN OOD gate", "seconds": time.time() - t0}
    (out / "crop_models.json").write_text(json.dumps(out_json, indent=1, default=float))
    if chosen in models:
        joblib.dump({"name": chosen, "model": models[chosen], "temperature": res[chosen]["temperature"],
                     "info": cands[chosen]["info"]}, path("models", cfg) / "p3" / "crop_head.joblib")
    else:
        mem = cands[chosen]["info"].get("members", [])
        joblib.dump({"name": chosen, "model": models.get(mem[0]) if mem else None, "temperature": res[chosen]["temperature"],
                     "info": cands[chosen]["info"], "prompts": cfg["p3"]["crop_prompts"]},
                    path("models", cfg) / "p3" / "crop_head.joblib")

    # crop errors on the wild set for the chosen model (confident = calibrated conf >= 0.9)
    w = P[(P.config == chosen) & (P.set == "wild_all")].copy()
    w["pred"] = np.where(w.p_turmeric >= 0.5, "turmeric", "blackgram")
    w["conf"] = np.maximum(w.p_turmeric, 1 - w.p_turmeric)
    err = w[w.pred != w.crop].sort_values("conf", ascending=False)
    src = pd.read_csv(path("results", cfg).parent / "data/wild/labels.csv").set_index("path")
    lines = [f"# Crop-ID errors on the wild set ({chosen})", "",
             "Generated by scripts/p3_crop.py. Calibrated confidence; 'confident' = >= 0.90 before any abstention gate.",
             "Abstention (conformal set, OOD, quality) is applied in P3-M05; see results/trust.json for which of "
             "these the app abstains on.", "",
             f"Wild images: {len(w)}; wrong crop: {len(err)}; confidently wrong (>= 0.90): {int((err.conf >= 0.9).sum())}.", "",
             "| image | true | predicted | confidence | source | near-duplicate group | likely reason |",
             "|---|---|---|---|---|---|---|"]
    for r in err.itertuples():
        reason = ("whole-plant field scene, many overlapping leaves (not a single leaf)" if "unknown" in r.path
                  else "cross-dataset leaf photo")
        lines.append(f"| {r.path.split('/')[-1]} | {r.crop} | {r.pred} | {r.conf:.2f} | {src.loc[r.path, 'source']} | "
                     f"{r.group} | {reason} |")
    u = P[(P.config == chosen) & (P.set == "user")]
    lines += ["", "## User-reported Google turmeric image", ""]
    for r in u.itertuples():
        lines.append(f"- {r.path}: P(turmeric) = {r.p_turmeric:.3f}"
                     + (f", P(other prompt) = {r.p_other:.3f}" if r.p_other == r.p_other else "")
                     + f" -> {'turmeric' if r.p_turmeric >= 0.5 else 'black gram'}")
    (out / "crop_errors.md").write_text("\n".join(lines) + "\n")
    print("chosen:", chosen, "| user:", u[["p_turmeric", "p_other"]].to_dict("records"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
