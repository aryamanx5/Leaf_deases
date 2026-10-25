"""P2-B05: temperature scaling, crop abstention threshold and OOD score.

* Temperature scaling (one T for the crop model, one for the disease head) fitted by NLL on
  clean VAL + wild_val; ECE before/after on every evaluation set.
* OOD score: mean cosine distance of the DINOv2 embedding to its 10 nearest TRAIN images;
  threshold = 99th percentile of clean-VAL scores.
* tau_crop: on wild_val, the smallest threshold whose accepted predictions (confidence >=
  tau and not OOD-flagged) reach >= 95% precision -> maximal coverage at that precision.
Writes results/calibration.json (report) and models/calibration.json + ood_reference.npy.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from leafdx.cnn.model import build_model  # noqa: E402
from leafdx.config import ROOT, load_config, path  # noqa: E402
from leafdx.robust.backbones import embed  # noqa: E402
from leafdx.robust.cnn_heads import predict_disease_logit  # noqa: E402
from leafdx.robust.infer import knn_distance, logit, sigmoid  # noqa: E402

PRECISION_TARGET = 0.95
OOD_K = 10
OOD_PCT = 99


def ece(p_pos: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """ECE of the predicted class confidence (binary)."""
    conf = np.maximum(p_pos, 1 - p_pos)
    correct = (p_pos >= 0.5).astype(int) == y
    edges = np.linspace(0.5, 1.0, bins + 1)
    out = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        m = (conf >= lo) & ((conf <= hi) if i == bins - 1 else (conf < hi))
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    def nll(logT):
        p = sigmoid(z / np.exp(logT))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def choose_tau(conf: np.ndarray, correct: np.ndarray, ok_ood: np.ndarray) -> tuple[float, float, float]:
    """Smallest tau with precision >= target on accepted predictions; (tau, coverage, precision)."""
    for tau in np.round(np.arange(0.5, 1.0001, 0.005), 3):
        acc_m = (conf >= tau) & ok_ood
        if acc_m.sum() and correct[acc_m].mean() >= PRECISION_TARGET:
            return float(tau), float(acc_m.mean()), float(correct[acc_m].mean())
    return 1.0, 0.0, float("nan")


def main() -> int:
    cfg = load_config()
    res, mdir = path("results", cfg), path("models", cfg)
    pr = pd.read_csv(res / "robust_preds.csv")
    bundle = joblib.load(mdir / "crop_model.joblib")
    col = f"p_turmeric__{bundle['key']}"
    y_crop = (pr.crop == "turmeric").astype(int).to_numpy()
    z_crop = logit(pr[col].to_numpy())
    fit_m = pr.set.isin(["clean_val", "wild_val"]).to_numpy()
    T_crop = fit_temperature(z_crop[fit_m], y_crop[fit_m])

    # disease: mean TTA logit of the crop's head, for labelled images
    dm, _ = build_model(cfg["robust"]["disease_cnn"]["arch"], 2, pretrained=False)
    dm.load_state_dict(torch.load(mdir / "disease_cnn.pt", map_location="cpu"))
    lab = (pr.label != "unknown").to_numpy()
    ims = [Image.open(ROOT / p).convert("RGB") for p in pr.path[lab]]
    z_dis = np.full(len(pr), np.nan)
    z_dis[lab] = predict_disease_logit(dm, ims, list(pr.crop[lab]), cfg)
    y_dis = (pr.label == "diseased").astype(int).to_numpy()
    fit_d = fit_m & lab
    T_dis = fit_temperature(z_dis[fit_d], y_dis[fit_d])

    # OOD reference: TRAIN originals (full image), DINOv2
    s = pd.read_csv(path("splits", cfg))
    train_paths = s[s.split == "train"].path
    ref_f = mdir / "ood_reference.npy"
    ref = embed("dinov2_vits14", [Image.open(ROOT / p).convert("RGB") for p in train_paths])
    np.save(ref_f, ref)
    q = embed("dinov2_vits14", [Image.open(ROOT / p).convert("RGB") for p in pr.path])
    ood = knn_distance(q, ref, OOD_K)
    ood_thr = float(np.percentile(ood[(pr.set == "clean_val").to_numpy()], OOD_PCT))

    p_cal = sigmoid(z_crop / T_crop)
    conf = np.maximum(p_cal, 1 - p_cal)
    correct = ((p_cal >= 0.5).astype(int) == y_crop)
    ok = ood <= ood_thr
    wv = (pr.set == "wild_val").to_numpy()
    tau, cov, prec = choose_tau(conf[wv], correct[wv], ok[wv])

    report = {"T_crop": T_crop, "T_disease": T_dis, "tau_crop": tau, "precision_target": PRECISION_TARGET,
              "ood": {"backbone": "dinov2_vits14", "k": OOD_K, "percentile": OOD_PCT, "threshold": ood_thr},
              "crop_model": bundle["candidate"], "sets": {}}
    for st in ("clean_val", "clean_test", "wild_val", "wild_test", "regression"):
        m = (pr.set == st).to_numpy()
        if not m.any():
            continue
        acc_m = m & (conf >= tau) & ok
        d = {"n": int(m.sum()),
             "crop_ece_before": ece(sigmoid(z_crop[m]), y_crop[m]), "crop_ece_after": ece(p_cal[m], y_crop[m]),
             "crop_accuracy": float(correct[m].mean()),
             "coverage": float(acc_m.sum() / m.sum()),
             "accepted_precision": float(correct[acc_m].mean()) if acc_m.any() else None,
             "abstention_rate": float(1 - acc_m.sum() / m.sum()),
             "ood_flag_rate": float((~ok[m]).mean())}
        ml = m & lab
        if ml.any():
            d["disease_ece_before"] = ece(sigmoid(z_dis[ml]), y_dis[ml])
            d["disease_ece_after"] = ece(sigmoid(z_dis[ml] / T_dis), y_dis[ml])
        if st == "regression":
            d["p_turmeric_calibrated"] = float(p_cal[m][0])
            d["ood_score"] = float(ood[m][0])
            d["status"] = "confident" if acc_m[m][0] else "uncertain"
        report["sets"][st] = d
    report["wild_val_tau_choice"] = {"coverage": cov, "accepted_precision": prec}
    (res / "calibration.json").write_text(json.dumps(report, indent=2))
    (mdir / "calibration.json").write_text(json.dumps({"T_crop": T_crop, "T_disease": T_dis, "tau_crop": tau,
                                                       "ood_threshold": ood_thr, "ood_k": OOD_K,
                                                       "crop_key": bundle["key"]}, indent=2))
    pr.assign(p_turmeric_cal=p_cal, ood_score=ood, z_disease=z_dis).to_csv(res / "robust_preds_calibrated.csv", index=False)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
