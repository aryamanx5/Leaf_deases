"""Hierarchical inference (P2-B04/B05): P(crop) and P(diseased | crop), calibrated, with
abstention ("uncertain") and an out-of-distribution score.

    m = RobustModel.load()
    m.predict(pil_image, crop=None) -> {"crop": {...}, "disease": {...}, "ood": {...}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import torch
from PIL import Image

from leafdx.cnn.model import build_model
from leafdx.config import load_config, path
from leafdx.robust.backbones import clip_zero_shot_text, embed
from leafdx.robust.cnn_heads import CROPS, crop_cnn_probs, predict_disease_logit
from leafdx.robust.views import tta_views

OOD_BACKBONE = "dinov2_vits14"


def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def crop_decision(p_turmeric: float, ood_score: float, cal: dict, user_crop: str | None = None) -> dict:
    """Pure decision rule. p_turmeric is already calibrated."""
    probs = {"turmeric": float(p_turmeric), "blackgram": float(1 - p_turmeric)}
    if user_crop is not None:
        return {"label": user_crop, "confidence": probs[user_crop], "probs": probs, "status": "user_selected",
                "source": "user"}
    label = max(probs, key=probs.get)
    conf = probs[label]
    confident = conf >= cal["tau_crop"] and ood_score <= cal["ood_threshold"]
    return {"label": label, "confidence": conf, "probs": probs, "status": "confident" if confident else "uncertain",
            "source": "model"}


def knn_distance(q: np.ndarray, ref: np.ndarray, k: int) -> np.ndarray:
    """Mean cosine distance to the k nearest reference embeddings (all L2-normalised)."""
    sims = q @ ref.T
    k = min(k, ref.shape[0])
    top = np.sort(sims, axis=1)[:, -k:]
    return 1 - top.mean(1)


@dataclass
class RobustModel:
    crop_bundle: dict
    disease_model: torch.nn.Module
    cal: dict
    ood_ref: np.ndarray
    cfg: dict
    effnet: torch.nn.Module | None = None

    @classmethod
    def load(cls) -> RobustModel:
        return _load_cached()

    # -- crop -------------------------------------------------------------------------
    def crop_prob_raw(self, im: Image.Image, dev_str: str | None = None) -> float:
        """Uncalibrated P(turmeric) from the selected crop model (mean over TTA views + tiles)."""
        b = self.crop_bundle
        if b["backbone"] == "efficientnet_b0":
            return float(crop_cnn_probs(self.effnet, [im], self.cfg, dev=torch.device("cpu"))[0, 0])
        e = embed(b["backbone"], tta_views(im, self.cfg, tiles=True), dev_str=dev_str)
        p = b["clf"].predict_proba(e) if b["key"] != "openclip_zeroshot" else None
        if "zeroshot" in b["key"]:
            txt = _text_embeddings(tuple(b["prompts"]["turmeric"] + b["prompts"]["blackgram"]))
            nt = len(b["prompts"]["turmeric"])
            sims = 100 * e @ txt.T
            zl = np.stack([sims[:, :nt].mean(1), sims[:, nt:].mean(1)], 1)
            zp = np.exp(zl - zl.max(1, keepdims=True))
            zp /= zp.sum(1, keepdims=True)
            p = zp if p is None else p * zp / (p * zp).sum(1, keepdims=True)
        return float(p[:, 0].mean())

    def ood_score(self, im: Image.Image, dev_str: str | None = None) -> float:
        e = embed(OOD_BACKBONE, [im], dev_str=dev_str)
        return float(knn_distance(e, self.ood_ref, self.cal["ood_k"])[0])

    # -- full prediction -----------------------------------------------------------------
    def predict(self, im: Image.Image, crop: str | None = None, dev_str: str | None = None) -> dict:
        c = self.cal
        p_tu = float(sigmoid(logit(self.crop_prob_raw(im, dev_str)) / c["T_crop"]))
        ood = self.ood_score(im, dev_str)
        cd = crop_decision(p_tu, ood, c, crop if crop in CROPS else None)
        z = predict_disease_logit(self.disease_model, [im], [cd["label"]], self.cfg, dev=torch.device("cpu"))[0]
        p_dis = float(sigmoid(z / c["T_disease"]))
        return {
            "crop": cd,
            "disease": {"label": "diseased" if p_dis >= 0.5 else "healthy", "prob": p_dis,
                        "confidence": max(p_dis, 1 - p_dis), "under_crop": cd["label"]},
            "ood": {"ood_score": ood, "threshold": c["ood_threshold"], "flagged": ood > c["ood_threshold"]},
        }


@lru_cache(maxsize=4)
def _text_embeddings(prompts: tuple[str, ...]) -> np.ndarray:
    return clip_zero_shot_text(list(prompts))


@lru_cache(maxsize=1)
def _load_cached() -> RobustModel:
    cfg = load_config()
    mdir = path("models", cfg)
    bundle = joblib.load(mdir / "crop_model.joblib")
    dm, _ = build_model(cfg["robust"]["disease_cnn"]["arch"], 2, pretrained=False)
    dm.load_state_dict(torch.load(mdir / "disease_cnn.pt", map_location="cpu"))
    dm.eval()
    eff = None
    if bundle["backbone"] == "efficientnet_b0":
        eff, _ = build_model("efficientnet_b0", 2, pretrained=False)
        eff.load_state_dict(torch.load(mdir / "crop_cnn_effb0.pt", map_location="cpu"))
        eff.eval()
    cal = json.loads((mdir / "calibration.json").read_text())
    return RobustModel(bundle, dm, cal, np.load(mdir / "ood_reference.npy"), cfg, eff)
