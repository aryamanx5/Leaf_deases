"""Inspection figures + checks for Stage 1 (image processing).

  --stage segment : results/figures/segmentation_samples.png (24 random images:
                    original | filtered | HSV mask | final leaf mask) and
                    results/segmentation_check.json (single-dominant-mask rate on 100
                    random images).
  --stage lesions : results/figures/lesion_samples_{diseased,healthy}.png and
                    results/lesion_check.json (mean affected-% healthy vs diseased).
Images are sampled from TRAIN+VAL only (test images stay untouched until reporting).
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from leafdx.config import ROOT, load_config, path
from leafdx.ip.lesions import detect_lesions, overlay_lesions
from leafdx.ip.preprocess import load_bgr, preprocess
from leafdx.ip.segment import is_single_dominant, segment_leaf
from leafdx.viz import contact_sheet, overlay_mask


def _segment(p: str, cfg: dict) -> dict:
    pre = preprocess(load_bgr(ROOT / p), cfg)
    return {"pre": pre, "seg": segment_leaf(pre, cfg)}


def _seg_stats(p: str, cfg: dict) -> dict:
    seg = _segment(p, cfg)["seg"]
    return {"path": p, "coverage": seg["coverage"], "dominant_share": seg["dominant_share"],
            "method": seg["method"], "ok": is_single_dominant(seg, cfg)}


def stage_segment(df: pd.DataFrame, cfg: dict) -> int:
    seed = cfg["seed"]
    sample = df.groupby("stratum").sample(6, random_state=seed)
    rows, titles = [], []
    for r in sample.itertuples():
        out = _segment(r.path, cfg)
        pre, seg = out["pre"], out["seg"]
        final = overlay_mask(pre["resized"], seg["leaf_mask"] == 0, (0, 0, 0), 0.65, seg["leaf_mask"])
        rows.append([pre["resized"], pre["filtered"], seg["colour_mask"], final])
        titles.append([f"{r.stratum}", "filtered", "HSV plant mask", f"{seg['method'][:18]} {seg['coverage']:.0%}"])
    sheet = contact_sheet([rows[i] + rows[i + 1] for i in range(0, len(rows), 2)],
                          [titles[i] + titles[i + 1] for i in range(0, len(rows), 2)], size=170)
    fig = path("figures", cfg) / "segmentation_samples.png"
    fig.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(fig), sheet)

    check = df.sample(100, random_state=seed)
    stats = pd.DataFrame(Parallel(n_jobs=-1)(delayed(_seg_stats)(p, cfg) for p in check.path))
    stats = stats.merge(check[["path", "stratum"]], on="path")
    result = {
        "n_images": len(stats),
        "single_dominant_rate": float(stats.ok.mean()),
        "criterion": cfg["segment"]["quality"],
        "rate_by_stratum": stats.groupby("stratum").ok.mean().round(3).to_dict(),
        "method_counts": stats.method.value_counts().to_dict(),
        "coverage_median": float(stats.coverage.median()),
        "failures": stats.loc[~stats.ok, ["path", "coverage", "dominant_share", "method"]].to_dict(orient="records"),
    }
    (path("results", cfg) / "segmentation_check.json").write_text(json.dumps(result, indent=2))
    print(f"saved {fig.relative_to(ROOT)}")
    print(f"single dominant leaf mask (coverage 5-95%): {stats.ok.sum()}/{len(stats)} = {stats.ok.mean():.1%}")
    print("by stratum:", result["rate_by_stratum"], "| methods:", result["method_counts"])
    return 0 if result["single_dominant_rate"] >= 0.9 else 1


def _lesion_stats(r, cfg: dict) -> dict:
    out = _segment(r.path, cfg)
    res = detect_lesions(out["pre"], out["seg"]["leaf_mask"], cfg, r.crop)
    return {"path": r.path, "crop": r.crop, "label": r.label, "orig_class": r.orig_class,
            "affected_pct": res["affected_pct"], "lesion_count": res["lesion_count"], **res["class_pct"]}


def stage_lesions(df: pd.DataFrame, cfg: dict) -> int:
    seed = cfg["seed"]
    for label in ("diseased", "healthy"):
        sub = df[df.label == label]
        sample = sub.groupby("crop").sample(12, random_state=seed)
        rows, titles = [], []
        for r in sample.itertuples():
            out = _segment(r.path, cfg)
            pre, seg = out["pre"], out["seg"]
            res = detect_lesions(pre, seg["leaf_mask"], cfg, r.crop)
            rows.append([pre["resized"], seg["segmented"], res["lesion_mask"],
                         overlay_lesions(pre["resized"], seg["leaf_mask"], res)])
            titles.append([f"{r.crop} {r.orig_class}", "segmented leaf", "abnormal mask",
                           f"{res['affected_pct']:.1f}% n={res['lesion_count']}"])
        sheet = contact_sheet([rows[i] + rows[i + 1] for i in range(0, len(rows), 2)],
                              [titles[i] + titles[i + 1] for i in range(0, len(rows), 2)], size=170)
        fig = path("figures", cfg) / f"lesion_samples_{label}.png"
        cv2.imwrite(str(fig), sheet)
        print(f"saved {fig.relative_to(ROOT)}")

    check = df.groupby("stratum").sample(60, random_state=seed)
    stats = pd.DataFrame(Parallel(n_jobs=-1)(delayed(_lesion_stats)(r, cfg) for r in check.itertuples()))
    by = stats.groupby(["crop", "label"]).affected_pct.agg(["mean", "median", "count"]).round(2)
    by_class = stats.groupby("orig_class")[["affected_pct", "lesion_count", "yellow", "brown", "dark", "pale"]].mean().round(2)
    means = stats.groupby("label").affected_pct.mean()
    result = {
        "n_images": len(stats),
        "mean_affected_pct": {"healthy": float(means["healthy"]), "diseased": float(means["diseased"])},
        "by_crop_label": {f"{c}_{lab}": row.to_dict() for (c, lab), row in by.iterrows()},
        "by_orig_class": by_class.to_dict(orient="index"),
    }
    (path("results", cfg) / "lesion_check.json").write_text(json.dumps(result, indent=2))
    print(by.to_string())
    print(by_class.to_string())
    print(f"mean affected-%: healthy {means['healthy']:.2f} vs diseased {means['diseased']:.2f}")
    ok = all(by.loc[(c, "healthy"), "mean"] < 0.5 * by.loc[(c, "diseased"), "mean"] for c in cfg["crops"])
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["segment", "lesions"], required=True)
    args = ap.parse_args()
    cfg = load_config()
    df = pd.read_csv(path("splits", cfg))
    df = df[df.split != "test"].reset_index(drop=True)
    np.random.seed(cfg["seed"])
    if args.stage == "segment":
        return stage_segment(df, cfg)
    return stage_lesions(df, cfg)


if __name__ == "__main__":
    sys.exit(main())
