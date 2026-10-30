"""P3-M03 / P3-M07: fine-tuned disease CNN arms C1-C3, each trained twice on the same
grouped TRAIN split:

* ``swap``  -- background-swap + strong colour/blur/JPEG augmentation (Phase 2 recipe);
* ``plain`` -- the v1 augmentation only (crop, flip, rotation, mild colour jitter).

Same crop-conditioned 2-logit head as the Phase 2 disease CNN; early stopping on clean VAL
macro-F1; hard time budget per arm (config p3.cnn_arms). The wild set is not touched here.
C1 ``swap`` reuses the Phase 2 checkpoint (models/disease_cnn.pt), which was trained with
exactly this recipe. Writes models/p3/<arm>_<variant>.pt and results/p3_cnn_train.json.

    python scripts/p3_train_cnns.py [--only C2_efficientnet_b0] [--variant swap]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd  # noqa: E402
from torch import nn  # noqa: E402

from leafdx.cnn.augment import SwapDataset, build_banks  # noqa: E402
from leafdx.cnn.dataset import train_transform  # noqa: E402
from leafdx.cnn.model import build_model  # noqa: E402
from leafdx.config import ROOT, load_config, path, seed_everything  # noqa: E402
from leafdx.p3.data import load_rgb  # noqa: E402
from leafdx.robust.cnn_heads import _fit, crop_index, disease_weights, predict_disease  # noqa: E402


class PlainDataset(SwapDataset):
    """Same rows/targets as SwapDataset but never swaps and uses the v1 transform."""

    def __init__(self, train_df, banks, cfg, target_fn, seed=0):
        super().__init__(train_df, banks, cfg, target_fn, seed)
        self.tf = train_transform(cfg["cnn"])

    def sample_image(self, i):
        return self.images[i]


def train_arm(arm: dict, variant: str, train, val_df, val_ims, banks, cfg, log=print) -> dict:
    out = path("models", cfg) / "p3" / f"{arm['name']}_{variant}.pt"
    if variant == "swap" and arm.get("reuse_swap") and (ROOT / arm["reuse_swap"]).exists():
        shutil.copyfile(ROOT / arm["reuse_swap"], out)
        return {"arm": arm["name"], "variant": variant, "reused": arm["reuse_swap"], "checkpoint": str(out.relative_to(ROOT))}
    seed_everything(cfg["seed"])
    w = disease_weights(train)
    model, loaded = build_model(arm["arch"], 2, pretrained=True)
    target = lambda r: (crop_index(r.crop), int(r.label == "diseased"), w[(r.crop, r.label)])  # noqa: E731
    ds = (SwapDataset if variant == "swap" else PlainDataset)(train, banks, cfg, target, seed=cfg["seed"])
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def step(m, x, tgt, dev):
        c, y, wt = tgt[0].to(dev), tgt[1].to(dev), tgt[2].float().to(dev)
        z = m(x).gather(1, c.view(-1, 1)).squeeze(1)
        return (bce(z, y.float()) * wt).mean()

    def val(m):
        from sklearn.metrics import f1_score

        p = predict_disease(m, val_ims, list(val_df.crop), cfg, tta=False)
        return f1_score((val_df.label == "diseased").astype(int), (p >= 0.5).astype(int), average="macro")

    c = cfg["p3"]["cnn_arms"]
    s = {"stages": arm["stages"], "batch_size": c["batch_size"], "patience": c["patience"],
         "time_budget_min": arm["time_budget_min"], "seed": cfg["seed"]}
    t0 = time.time()
    info = _fit(model, ds, s, step, val, out, log)
    hit_budget = info["minutes"] > arm["time_budget_min"]
    return {"arm": arm["name"], "variant": variant, "arch": arm["arch"], "pretrained": loaded,
            "checkpoint": str(out.relative_to(ROOT)), "best_val_macro_f1": info["score"], "best_epoch": info["epoch"],
            "epochs_run": info["epochs_run"], "minutes": (time.time() - t0) / 60, "device": info["device"],
            "time_budget_min": arm["time_budget_min"], "stopped_by_time_budget": bool(hit_budget)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--variant", nargs="*", default=["swap", "plain"])
    a = ap.parse_args()
    cfg = load_config()
    (path("models", cfg) / "p3").mkdir(exist_ok=True)
    s = pd.read_csv(path("splits", cfg))
    train = s[s.split == "train"].reset_index(drop=True)
    val_df = s[s.split == "val"].reset_index(drop=True)
    val_ims = [load_rgb(p) for p in val_df.path]
    banks = build_banks(train, cfg)
    res_f = path("results", cfg) / "p3_cnn_train.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    for arm in cfg["p3"]["cnn_arms"]["arms"]:
        if a.only and arm["name"] not in a.only:
            continue
        for v in a.variant:
            print(f"== {arm['name']} [{v}]", flush=True)
            try:
                res[f"{arm['name']}_{v}"] = train_arm(arm, v, train, val_df, val_ims, banks, cfg,
                                                      log=lambda m: print(m, flush=True))
            except Exception as e:  # failed experiments are logged, not hidden
                res[f"{arm['name']}_{v}"] = {"arm": arm["name"], "variant": v, "failed": repr(e)}
            print(res[f"{arm['name']}_{v}"], flush=True)
            res_f.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
