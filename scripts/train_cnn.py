"""Train the 4-class CNN (MobileNetV2 by default).

  --smoke : 1 epoch on 64 TRAIN images, save models/cnn_smoke.pt, reload, predict.
  (default): full two-stage training -> models/cnn_best.pt, results/cnn_train_log.csv,
             results/figures/cnn_training.png, then evaluation + Grad-CAM (T08).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from leafdx.cnn.dataset import CLASSES4, LeafDataset, eval_transform  # noqa: E402
from leafdx.cnn.model import load_checkpoint  # noqa: E402
from leafdx.cnn.train import predict_probs, train_cnn  # noqa: E402
from leafdx.config import load_config, path, seed_everything  # noqa: E402


def smoke(cfg: dict, splits: pd.DataFrame) -> int:
    n = cfg["cnn"]["smoke"]["n_images"]
    tr = splits[splits.split == "train"].groupby("stratum").sample(n // 4, random_state=cfg["seed"])
    va = splits[splits.split == "val"].groupby("stratum").sample(4, random_state=cfg["seed"])
    ckpt = path("models", cfg) / "cnn_smoke.pt"
    stages = [{"name": "head", "epochs": cfg["cnn"]["smoke"]["epochs"], "lr": 1e-3, "unfreeze_last_blocks": 0}]
    hist, info = train_cnn(cfg, tr, va, ckpt, stages=stages)
    model, meta = load_checkpoint(ckpt)
    dl = DataLoader(LeafDataset(va.head(4), eval_transform(cfg["cnn"]), cfg["cnn"]["cache_side"]), batch_size=4)
    probs, y = predict_probs(model, dl, torch.device("cpu"))
    print(f"smoke: trained {len(tr)} images, {info['epochs_run']} epoch on {info['device']}, "
          f"pretrained={info['pretrained_weights']}; checkpoint {ckpt.name} reloaded (epoch {meta['epoch']})")
    for p, t in zip(probs, y, strict=True):
        print(f"  true {CLASSES4[t]:20s} pred {CLASSES4[p.argmax()]:20s} conf {p.max():.2f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg["seed"])
    splits = pd.read_csv(path("splits", cfg))
    if args.smoke:
        return smoke(cfg, splits)

    from leafdx.cnn.evaluate import evaluate_and_report  # noqa: PLC0415
    from leafdx.cnn.gradcam import gradcam_figure  # noqa: PLC0415

    from leafdx.viz import plot_training  # noqa: PLC0415

    ckpt = path("models", cfg) / "cnn_best.pt"
    hist, info = train_cnn(cfg, splits[splits.split == "train"], splits[splits.split == "val"], ckpt)
    hist.to_csv(path("results", cfg) / "cnn_train_log.csv", index=False)
    (path("results", cfg) / "cnn_train_info.json").write_text(json.dumps(info, indent=2))
    plot_training(hist, path("figures", cfg) / "cnn_training.png")
    print(f"best epoch {info['best']['epoch']} VAL binary macro-F1 {info['best']['macro_f1_bin']:.3f} "
          f"({info['epochs_run']} epochs, {info['minutes']:.1f} min on {info['device']})")
    evaluate_and_report(cfg, splits, ckpt)
    gradcam_figure(cfg, splits, ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
