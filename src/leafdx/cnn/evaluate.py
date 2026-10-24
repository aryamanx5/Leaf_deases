"""Evaluate the trained CNN on VAL and TEST: derived binary healthy/diseased metrics (same
set as Stage 1), 4-class crop x state metrics, crop accuracy, confidence statistics and
per-image inference time. Writes results/cnn_metrics.json and results/cnn_predictions.csv.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from leafdx.cnn.dataset import CLASSES4, DISEASED_IDX, LeafDataset, eval_transform
from leafdx.cnn.model import device, load_checkpoint
from leafdx.cnn.train import predict_probs
from leafdx.config import ROOT, path
from leafdx.metrics import with_per_crop


def predictions_frame(df: pd.DataFrame, probs4: np.ndarray) -> pd.DataFrame:
    p_dis = probs4[:, list(DISEASED_IDX)].sum(1)
    pred4 = probs4.argmax(1)
    out = df[["path", "split", "crop", "label", "orig_class", "stratum"]].copy()
    out["y"] = (out["label"] == "diseased").astype(int)
    out["cnn_p_diseased"] = p_dis
    out["cnn_pred"] = (p_dis >= 0.5).astype(int)
    out["cnn_conf"] = np.maximum(p_dis, 1 - p_dis)  # confidence of the binary decision
    out["cnn_pred4"] = [CLASSES4[i] for i in pred4]
    out["cnn_conf4"] = probs4.max(1)
    out["cnn_crop_pred"] = [CLASSES4[i].split("_")[0] for i in pred4]
    for i, c in enumerate(CLASSES4):
        out[f"p_{c}"] = probs4[:, i]
    return out


def four_class_metrics(pred: pd.DataFrame) -> dict:
    y = pred["stratum"].map(CLASSES4.index).to_numpy()
    p = pred["cnn_pred4"].map(CLASSES4.index).to_numpy()
    return {
        "accuracy": float((y == p).mean()),
        "macro_f1": float(f1_score(y, p, average="macro", labels=range(4), zero_division=0)),
        "confusion_matrix": confusion_matrix(y, p, labels=range(4)).tolist(),
        "classes": list(CLASSES4),
        "crop_accuracy": float((pred["cnn_crop_pred"] == pred["crop"]).mean()),
    }


def confidence_stats(pred: pd.DataFrame) -> dict:
    ok = pred["cnn_pred"] == pred["y"]
    return {
        "mean_conf_correct": float(pred.loc[ok, "cnn_conf"].mean()) if ok.any() else None,
        "mean_conf_wrong": float(pred.loc[~ok, "cnn_conf"].mean()) if (~ok).any() else None,
        "median_conf": float(pred["cnn_conf"].median()),
        "share_conf_ge_0.9": float((pred["cnn_conf"] >= 0.9).mean()),
    }


@torch.no_grad()
def time_inference(model, df: pd.DataFrame, cfg: dict, dev) -> dict:
    """Per-image latency at batch size 1 (decode + resize + forward)."""
    tf = eval_transform(cfg["cnn"])
    model = model.to(dev).eval()
    ms = []
    for i, p in enumerate(df["path"]):
        t0 = time.perf_counter()
        with Image.open(ROOT / p) as im:
            x = tf(im.convert("RGB")).unsqueeze(0).to(dev)
        y = model(x)
        if dev.type == "mps":
            torch.mps.synchronize()
        _ = y.cpu()
        if i > 0:  # first call includes warm-up
            ms.append(1000 * (time.perf_counter() - t0))
    return {"device": str(dev), "n_images": len(ms), "ms_median": float(np.median(ms)), "ms_mean": float(np.mean(ms))}


def evaluate_and_report(cfg: dict, splits: pd.DataFrame, ckpt) -> dict:
    dev = device()
    model, meta = load_checkpoint(ckpt)
    model.to(dev)
    frames = []
    for s in ("val", "test"):
        part = splits[splits.split == s]
        dl = DataLoader(LeafDataset(part, eval_transform(cfg["cnn"]), cfg["cnn"]["cache_side"]),
                        batch_size=cfg["cnn"]["batch_size"], shuffle=False)
        probs, _ = predict_probs(model, dl, dev)
        frames.append(predictions_frame(part, probs))
    pred = pd.concat(frames)
    pred.to_csv(path("results", cfg) / "cnn_predictions.csv", index=False)

    timing_df = splits[splits.split == "val"].sample(cfg["decide"]["timing_images"] + 1, random_state=cfg["seed"])
    metrics = {"checkpoint": {k: meta[k] for k in ("arch", "epoch", "image_size", "pretrained") if k in meta}}
    for s in ("val", "test"):
        part = pred[pred.split == s]
        metrics[s] = {
            "binary": with_per_crop(part, "cnn_pred", "cnn_p_diseased"),
            "four_class": four_class_metrics(part),
            "confidence": confidence_stats(part),
        }
    metrics["inference_time"] = {
        "accelerator": time_inference(model, timing_df, cfg, dev),
        "cpu": time_inference(model, timing_df, cfg, torch.device("cpu")),
    }
    (path("results", cfg) / "cnn_metrics.json").write_text(json.dumps(metrics, indent=2))
    for s in ("val", "test"):
        o, f = metrics[s]["binary"]["overall"], metrics[s]["four_class"]
        pc = {c: round(v["macro_f1"], 3) for c, v in metrics[s]["binary"]["per_crop"].items()}
        print(f"cnn {s:4s} binary acc {o['accuracy']:.3f} P {o['precision']:.3f} R {o['recall']:.3f} "
              f"F1 {o['f1']:.3f} macroF1 {o['macro_f1']:.3f} AUC {o['roc_auc']:.3f} CM {o['confusion_matrix']} "
              f"per-crop macroF1 {pc} | 4-class acc {f['accuracy']:.3f} macroF1 {f['macro_f1']:.3f} "
              f"crop acc {f['crop_accuracy']:.3f}")
    t = metrics["inference_time"]
    print(f"inference ms/image: {t['accelerator']['device']} {t['accelerator']['ms_median']:.1f}, cpu {t['cpu']['ms_median']:.1f}")
    print("confidence (test):", metrics["test"]["confidence"])
    return metrics
