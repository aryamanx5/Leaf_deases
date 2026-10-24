"""Two-stage transfer learning with early stopping on VAL binary macro-F1.

Stage 1 trains only the new head; stage 2 unfreezes the last backbone blocks at a lower
learning rate. Loss: class-weighted cross-entropy over the 4 crop x state classes.
Optimiser: AdamW with cosine annealing inside each stage.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from leafdx.cnn.dataset import CLASSES4, DISEASED_IDX, LeafDataset, eval_transform, train_transform
from leafdx.cnn.model import build_model, device, save_checkpoint, set_trainable


def class_weights4(train_df: pd.DataFrame) -> torch.Tensor:
    counts = train_df["stratum"].value_counts().reindex(CLASSES4).fillna(0).to_numpy(float)
    w = len(train_df) / (len(CLASSES4) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, dev) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(dev)), 1).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def scores(probs4: np.ndarray, y4: np.ndarray) -> dict:
    yb = np.isin(y4, DISEASED_IDX).astype(int)
    pb = (probs4[:, list(DISEASED_IDX)].sum(1) >= 0.5).astype(int)
    return {
        "acc4": float((probs4.argmax(1) == y4).mean()),
        "macro_f1_4": float(f1_score(y4, probs4.argmax(1), average="macro", labels=range(4), zero_division=0)),
        "acc_bin": float((pb == yb).mean()),
        "macro_f1_bin": float(f1_score(yb, pb, average="macro", zero_division=0)),
    }


def train_cnn(cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, ckpt_path, *,
              stages: list[dict] | None = None, log=print) -> tuple[pd.DataFrame, dict]:
    c = cfg["cnn"]
    dev = device()
    torch.manual_seed(cfg["seed"])
    model, pretrained_ok = build_model(c["arch"], len(CLASSES4), c["pretrained"])
    model.to(dev)
    gen = torch.Generator().manual_seed(cfg["seed"])
    train_ds = LeafDataset(train_df, train_transform(c), c["cache_side"])
    val_ds = LeafDataset(val_df, eval_transform(c), c["cache_side"])
    train_dl = DataLoader(train_ds, batch_size=c["batch_size"], shuffle=True, generator=gen, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=c["batch_size"], shuffle=False, num_workers=0)
    crit = nn.CrossEntropyLoss(weight=class_weights4(train_df).to(dev), label_smoothing=c["label_smoothing"])

    stages = stages or [{"name": "head", **c["stage1"]}, {"name": "finetune", **c["stage2"]}]
    t_start, history = time.time(), []
    best = {"macro_f1_bin": -1.0, "epoch": -1}
    patience_left, stop = c["early_stopping_patience"], False
    epoch = 0
    for st in stages:
        set_trainable(model, st["unfreeze_last_blocks"])
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=st["lr"], weight_decay=c["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(st["epochs"], 1))
        for _ in range(st["epochs"]):
            epoch += 1
            model.train()
            t0, tot, correct, n = time.time(), 0.0, 0, 0
            for x, y in train_dl:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad()
                out = model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
                tot += loss.item() * len(y)
                correct += (out.argmax(1) == y).sum().item()
                n += len(y)
            sched.step()
            probs, y4 = predict_probs(model, val_dl, dev)
            val_loss = float(nn.functional.nll_loss(torch.log(torch.tensor(probs) + 1e-9), torch.tensor(y4),
                                                    weight=class_weights4(train_df)))
            sc = scores(probs, y4)
            row = {"epoch": epoch, "stage": st["name"], "lr": opt.param_groups[0]["lr"],
                   "train_loss": tot / n, "train_acc4": correct / n, "val_loss": val_loss, **{f"val_{k}": v for k, v in sc.items()},
                   "epoch_s": time.time() - t0}
            history.append(row)
            improved = sc["macro_f1_bin"] > best["macro_f1_bin"] + 1e-4 or (
                abs(sc["macro_f1_bin"] - best["macro_f1_bin"]) <= 1e-4 and sc["macro_f1_4"] > best.get("macro_f1_4", -1))
            if improved:
                best = {"epoch": epoch, **sc}
                save_checkpoint(model, ckpt_path, {"arch": c["arch"], "classes": list(CLASSES4),
                                                   "image_size": c["image_size"], "epoch": epoch,
                                                   "val": sc, "pretrained": pretrained_ok})
                patience_left = c["early_stopping_patience"]
            elif st["name"] != "head":
                patience_left -= 1
            log(f"epoch {epoch:2d} [{st['name']}] loss {row['train_loss']:.3f} val_loss {val_loss:.3f} "
                f"val acc4 {sc['acc4']:.3f} binF1 {sc['macro_f1_bin']:.3f} ({row['epoch_s']:.0f}s)"
                + (" *" if improved else ""))
            elapsed_min = (time.time() - t_start) / 60
            if patience_left <= 0:
                log(f"early stopping at epoch {epoch}")
                stop = True
            elif elapsed_min > c["time_budget_min"]:
                log(f"time budget {c['time_budget_min']} min reached at epoch {epoch}")
                stop = True
            if stop:
                break
        if stop:
            break
    info = {"best": best, "epochs_run": epoch, "minutes": (time.time() - t_start) / 60,
            "device": str(dev), "pretrained_weights": pretrained_ok}
    return pd.DataFrame(history), info
