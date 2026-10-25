"""Fine-tuned CNNs for the hierarchical model (P2-B04).

* Disease CNN: MobileNetV2 with a 2-unit head; unit k is the "diseased" logit for crop k
  (0 = turmeric, 1 = black gram), i.e. one binary head per crop on a shared backbone.
  P(diseased | crop) = sigmoid(logit[crop]).
* Crop CNN baseline: EfficientNet-B0 with a 2-way softmax over crops.
Both are trained on the background-swap dataset (TRAIN only) and evaluated with TTA.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from leafdx.cnn.augment import SwapDataset
from leafdx.cnn.dataset import eval_transform
from leafdx.cnn.model import build_model, device, set_trainable
from leafdx.robust.views import tta_views

CROPS = ("turmeric", "blackgram")


def crop_index(c: str) -> int:
    return CROPS.index(c)


def disease_weights(train: pd.DataFrame) -> dict:
    """Per (crop, label) sample weight so each crop's healthy/diseased classes count equally."""
    w = {}
    for c, g in train.groupby("crop"):
        for lab, n in g.label.value_counts().items():
            w[(c, lab)] = len(g) / (2 * n)
    return w


def _fit(model, ds, cfg_s: dict, step_fn, val_fn, ckpt, log=print) -> dict:
    dev = device()
    model.to(dev)
    dl = DataLoader(ds, batch_size=cfg_s["batch_size"], shuffle=True, num_workers=0,
                    generator=torch.Generator().manual_seed(cfg_s["seed"]))
    best, t0, ep = {"score": -1.0}, time.time(), 0
    patience = cfg_s["patience"]
    for st in cfg_s["stages"]:
        set_trainable(model, st["unfreeze_last_blocks"])
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=st["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(st["epochs"], 1))
        for _ in range(st["epochs"]):
            ep += 1
            model.train()
            tot, n, te = 0.0, 0, time.time()
            for x, tgt in dl:
                opt.zero_grad()
                loss = step_fn(model, x.to(dev), tgt, dev)
                loss.backward()
                opt.step()
                tot += loss.item() * len(x)
                n += len(x)
            sched.step()
            score = val_fn(model)
            improved = score > best["score"] + 1e-4
            if improved:
                best = {"score": score, "epoch": ep}
                torch.save(model.state_dict(), ckpt)
                patience = cfg_s["patience"]
            elif st is not cfg_s["stages"][0]:
                patience -= 1
            log(f"  epoch {ep} loss {tot / n:.3f} val {score:.4f} ({time.time() - te:.0f}s){' *' if improved else ''}")
            if patience <= 0 or (time.time() - t0) / 60 > cfg_s["time_budget_min"]:
                break
        if patience <= 0 or (time.time() - t0) / 60 > cfg_s["time_budget_min"]:
            break
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return {**best, "epochs_run": ep, "minutes": (time.time() - t0) / 60, "device": str(dev)}


@torch.no_grad()
def _logits(model, images: list[Image.Image], cfg: dict, dev=None) -> np.ndarray:
    dev = dev or device()
    model.to(dev).eval()
    tf = eval_transform(cfg["cnn"])
    out = []
    for i in range(0, len(images), 64):
        out.append(model(torch.stack([tf(im.convert("RGB")) for im in images[i:i + 64]]).to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def predict_disease(model, images: list[Image.Image], crops: list[str], cfg: dict, tta: bool = True, dev=None) -> np.ndarray:
    """P(diseased | crop) per image, mean over TTA views (no tiles: lesions may sit in one tile)."""
    views, owner = [], []
    for k, im in enumerate(images):
        v = tta_views(im, cfg, tiles=False) if tta else [im]
        views += v
        owner += [k] * len(v)
    lg = _logits(model, views, cfg, dev)
    owner = np.array(owner)
    cidx = np.array([crop_index(crops[o]) for o in owner])
    p = 1 / (1 + np.exp(-lg[np.arange(len(lg)), cidx]))
    return np.array([p[owner == k].mean() for k in range(len(images))])


def predict_disease_logit(model, images: list[Image.Image], crops: list[str], cfg: dict, dev=None) -> np.ndarray:
    """Mean TTA logit of the given crop's head (for temperature scaling)."""
    views, owner = [], []
    for k, im in enumerate(images):
        v = tta_views(im, cfg, tiles=False)
        views += v
        owner += [k] * len(v)
    lg = _logits(model, views, cfg, dev)
    owner = np.array(owner)
    z = lg[np.arange(len(lg)), [crop_index(crops[o]) for o in owner]]
    return np.array([z[owner == k].mean() for k in range(len(images))])


def train_disease_cnn(train, val_images, val_df, banks, cfg, ckpt, log=print):
    r = cfg["robust"]["disease_cnn"]
    weights = disease_weights(train)
    model, _ = build_model(r["arch"], 2, pretrained=True)
    ds = SwapDataset(train, banks, cfg, target_fn=lambda row: (crop_index(row.crop), int(row.label == "diseased"),
                                                               weights[(row.crop, row.label)]), seed=cfg["seed"])
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def step(m, x, tgt, dev):
        c, y, w = (t.to(dev) for t in tgt)
        z = m(x).gather(1, c.view(-1, 1)).squeeze(1)
        return (bce(z, y.float()) * w.float()).mean()

    def val(m):
        p = predict_disease(m, val_images, list(val_df.crop), cfg, tta=False)
        return f1_score((val_df.label == "diseased").astype(int), (p >= 0.5).astype(int), average="macro")

    info = _fit(model, ds, {**r, "seed": cfg["seed"]}, step, val, ckpt, log)
    return model, info


def train_crop_cnn(train, val_images, val_df, banks, cfg, ckpt, log=print):
    r = cfg["robust"]["crop_cnn"]
    model, _ = build_model(r["arch"], 2, pretrained=True)
    ds = SwapDataset(train, banks, cfg, target_fn=lambda row: crop_index(row.crop), seed=cfg["seed"])
    ce = nn.CrossEntropyLoss()

    def step(m, x, tgt, dev):
        return ce(m(x), tgt.to(dev))

    def val(m):
        p = torch.softmax(torch.tensor(_logits(m, val_images, cfg)), 1).numpy()
        return float((p.argmax(1) == np.array([crop_index(c) for c in val_df.crop])).mean()) - \
            float(np.mean(-np.log(p[np.arange(len(p)), [crop_index(c) for c in val_df.crop]] + 1e-9))) * 1e-3

    info = _fit(model, ds, {**r, "seed": cfg["seed"]}, step, val, ckpt, log)
    return model, info


def crop_cnn_probs(model, images: list[Image.Image], cfg: dict, dev=None) -> np.ndarray:
    """P(crop) per image: mean softmax over TTA views (+ plant tiles for field scenes)."""
    views, owner = [], []
    for k, im in enumerate(images):
        v = tta_views(im, cfg, tiles=True)
        views += v
        owner += [k] * len(v)
    p = torch.softmax(torch.tensor(_logits(model, views, cfg, dev)), 1).numpy()
    owner = np.array(owner)
    return np.stack([p[owner == k].mean(0) for k in range(len(images))])
