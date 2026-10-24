"""Leakage ablation: how much does apparent accuracy inflate under naive splitting?

Scenarios (each trains on its own TRAIN part and reports on its own TEST part):
  clean               - the leakage-free cluster-grouped split used everywhere else
  naive_random        - random image-level stratified split, near-duplicates may straddle
  augment_then_split  - originals + deterministic augmented copies (flip, rot90, brightness)
                        pooled BEFORE a random split, as in pre-augmented public datasets
Models (kept quick): the Stage 1 SVM with the T06 hyperparameters, and a "CNN head":
logistic regression on frozen ImageNet MobileNetV2 embeddings. The fine-tuned CNN is
NOT reused because it has seen clean-TRAIN images that a naive split puts in its test.
The augmented copies exist only in memory / the cache, never in data/raw.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from leafdx.cnn.dataset import eval_transform
from leafdx.cnn.model import build_model, device
from leafdx.config import ROOT
from leafdx.ip.decide import svm_matrix

AUGMENTS = ("orig", "hflip", "rot90", "bright")


def augment(img_bgr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "orig":
        return img_bgr
    if kind == "hflip":
        return cv2.flip(img_bgr, 1)
    if kind == "rot90":
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    if kind == "bright":
        return cv2.convertScaleAbs(img_bgr, alpha=1.15, beta=12)
    raise ValueError(kind)


@torch.no_grad()
def imagenet_embeddings(items: pd.DataFrame, cfg: dict, batch: int = 64) -> np.ndarray:
    """Global-average-pooled MobileNetV2 (ImageNet weights, frozen) features per item."""
    model, _ = build_model("mobilenet_v2", 4, pretrained=True)
    model.classifier = torch.nn.Identity()
    dev = device()
    model.to(dev).eval()
    tf = eval_transform(cfg["cnn"])
    out = []
    for i in range(0, len(items), batch):
        xs = []
        for r in items.iloc[i:i + batch].itertuples():
            bgr = augment(cv2.imread(str(ROOT / r.path)), r.aug)
            xs.append(tf(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
        out.append(model(torch.stack(xs).to(dev)).cpu().numpy())
    return np.concatenate(out)


def scenario_splits(items: pd.DataFrame, clean_split: pd.Series, seed: int, ratios: dict) -> dict[str, pd.Series]:
    """Train/test assignment per scenario (val is not needed: hyperparameters are fixed)."""
    out = {}
    orig = items.aug == "orig"
    s = pd.Series("unused", index=items.index)
    s[orig] = clean_split[orig].replace({"val": "unused"})
    out["clean"] = s

    def random_split(idx):
        tr, rest = train_test_split(idx, train_size=ratios["train"], stratify=items.loc[idx, "stratum"], random_state=seed)
        val_share = ratios["val"] / (ratios["val"] + ratios["test"])
        _, te = train_test_split(rest, train_size=val_share, stratify=items.loc[rest, "stratum"], random_state=seed)
        r = pd.Series("unused", index=items.index)
        r[tr], r[te] = "train", "test"
        return r

    out["naive_random"] = random_split(items.index[orig])
    out["augment_then_split"] = random_split(items.index)
    return out


def leakage_share(items: pd.DataFrame, split: pd.Series) -> float:
    """Share of TEST items whose near-duplicate cluster also has an item in TRAIN."""
    train_clusters = set(items.loc[split == "train", "cluster_id"])
    test = items.loc[split == "test", "cluster_id"]
    return float(test.isin(train_clusters).mean())


def run_ablation(items: pd.DataFrame, X_feat: np.ndarray, X_emb: np.ndarray, splits: dict,
                 svm_params: dict, seed: int) -> pd.DataFrame:
    y = (items.label == "diseased").astype(int).to_numpy()
    rows = []
    for scen, s in splits.items():
        tr, te = (s == "train").to_numpy(), (s == "test").to_numpy()
        models = {
            "svm": (make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced",
                                                        C=svm_params["svc__C"], gamma=svm_params["svc__gamma"],
                                                        random_state=seed)), X_feat),
            "cnn_head": (make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced",
                                                                            max_iter=3000)), X_emb),
        }
        for name, (model, X) in models.items():
            model.fit(X[tr], y[tr])
            p = model.predict(X[te])
            rows.append({"scenario": scen, "method": name, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                         "accuracy": accuracy_score(y[te], p),
                         "macro_f1": f1_score(y[te], p, average="macro"),
                         "test_items_with_cluster_in_train": leakage_share(items, s)})
    return pd.DataFrame(rows)


def stage1_matrix(feats: pd.DataFrame) -> np.ndarray:
    return svm_matrix(feats)
