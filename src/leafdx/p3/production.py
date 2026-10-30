"""Phase 3 production inference (P3-M09): crop ID with conformal set, kNN OOD, quality gate,
every disease arm, the production disease decision with its conformal set, abstention,
severity and a plain-language reason. Everything selected/fitted is read from
models/manifest.json (written by scripts/p3_finalize.py); nothing here is fitted.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image

from leafdx.config import ROOT, load_config
from leafdx.ip.features import FEATURE_NAMES
from leafdx.p3 import heads
from leafdx.p3.quality import quality_stats, quality_verdict
from leafdx.p3.stats import conformal_sets, logit, sigmoid
from leafdx.robust.backbones import clip_zero_shot_text, embed
from leafdx.robust.infer import knn_distance

MANIFEST = ROOT / "models" / "manifest.json"
CROPS = ("turmeric", "blackgram")
ARM_LABELS = {"A_rule": "Rule score (Stage 1)", "B_svm": "57 features + SVM", "B2_lgbm": "57 features + LightGBM",
              "C1_mobilenet_v2": "MobileNetV2 (fine-tuned)", "C2_efficientnet_b0": "EfficientNet-B0 (fine-tuned)",
              "C3_convnext_tiny": "ConvNeXt-Tiny (fine-tuned)", "D_dinov2": "DINOv2 frozen + head",
              "E_openclip": "OpenCLIP frozen + head", "E_bioclip2": "BioCLIP 2 frozen + head",
              "F_fusion": "Fusion: embedding + 57 features", "STACK": "Stacking ensemble"}


def severity(affected_pct: float, cuts: dict) -> str:
    """Heuristic grade from affected leaf area (cut-offs in config p3.severity_cuts)."""
    if affected_pct < cuts["mild"]:
        return "mild"
    return "moderate" if affected_pct <= cuts["moderate"] else "severe"


def reason_string(f: dict, model_label: str, conf: float) -> str:
    yb = f.get("abn_yellow_pct", 0.0) + f.get("abn_brown_pct", 0.0)
    return (f"affected area {f['affected_pct']:.1f}%, {int(f['lesion_count'])} lesion"
            f"{'' if int(f['lesion_count']) == 1 else 's'}, yellow-brown pixels {yb:.1f}%; "
            f"{model_label.lower()} confidence {100 * conf:.0f}%")


def available() -> bool:
    return MANIFEST.exists()


class P3Model:
    def __init__(self, man: dict):
        heads.pin_lightgbm_openmp()  # before unpickling any LightGBM head, in this thread
        self.man, self.cfg = man, load_config()
        self.trust = man["trust"]
        # CNN arms whose checkpoint is absent (large ConvNeXt files are not committed) are skipped, not fatal
        self.arms = {a: joblib.load(ROOT / v["file"]) for a, v in man["arms"].items()
                     if v["kind"] != "cnn" or (ROOT / v["checkpoint"]).exists()}
        self.crop = joblib.load(ROOT / man["crop"]["file"])
        self.stacker = joblib.load(ROOT / man["stacker"]["file"]) if man.get("stacker") else None
        self.ood_ref = np.load(ROOT / man["trust"]["ood_reference"]).astype(np.float32)

    # -- embeddings ---------------------------------------------------------------------
    def _emb(self, im: Image.Image, cache: dict, bb: str) -> np.ndarray:
        if bb not in cache:
            cache[bb] = embed(bb, [im])
        return cache[bb]

    # -- crop ---------------------------------------------------------------------------
    def crop_probs(self, im: Image.Image, cache: dict) -> tuple[float, float | None]:
        c = self.crop
        info = c["info"]
        z = 0.0
        p_other = None
        kind = info["kind"]
        if kind.startswith("logistic") or kind.startswith("product"):
            head_info = info if kind.startswith("logistic") else c.get("head_info", {})
            bbs = head_info.get("backbones") or [head_info.get("backbone")]
            X = np.column_stack([self._emb(im, cache, bb) for bb in bbs])
            z += float(c["model"].decision_function(X)[0])
        if kind == "zero-shot" or kind.startswith("product"):
            bb = "bioclip2" if kind.startswith("product") else info["backbone"]
            pr = c.get("prompts") or info["prompts"]
            cls = ["turmeric", "blackgram", "other"]
            txt = _text(tuple(p for k in cls for p in pr[k]), bb)
            sims, k, s = 100 * self._emb(im, cache, bb) @ txt.T, 0, []
            for cl in cls:
                s.append(sims[:, k:k + len(pr[cl])].mean(1))
                k += len(pr[cl])
            zz = np.stack(s, 1)[0]
            e = np.exp(zz - zz.max())
            e /= e.sum()
            z += float(logit(e[0] / (e[0] + e[1])))
            p_other = float(e[2])
        return float(sigmoid(z / c["temperature"])), p_other

    # -- disease arms -------------------------------------------------------------------
    def arm_scores(self, im: Image.Image, crop: str, feats: dict, cache: dict, which: set | None = None) -> dict:
        from leafdx.p3.cnnpred import load_arm
        from leafdx.robust.cnn_heads import predict_disease_logit

        heads.pin_lightgbm_openmp()  # per thread (web requests run in a thread pool)
        row = pd.DataFrame([{**feats, "crop": crop}])
        ci = np.array([[1.0 if crop == "turmeric" else 0.0]])
        out = {}
        for a, b in self.arms.items():
            if which is not None and a not in which:
                continue
            t0 = time.perf_counter()
            if self.man["arms"][a]["kind"] == "cnn":
                m = load_arm(b["checkpoint"], b["arch"], "cpu")
                z = float(predict_disease_logit(m, [im], [crop], self.cfg, dev=torch.device("cpu"))[0])
                T = b["temperature"]
            elif b["kind"] == "rule":
                r = b["rule"]
                z = 10 * (float(r.score(row).iloc[0]) - r.threshold[crop])
                T = b["temperature"]
            else:
                parts = []
                if b["kind"] in ("emb", "fusion"):
                    parts.append(self._emb(im, cache, b["backbone"]))
                if b["kind"] in ("feat", "fusion"):
                    parts.append(np.array([[float(feats[k]) for k in FEATURE_NAMES]]))
                parts.append(ci)
                X = np.column_stack(parts).astype(np.float32)
                h = b["heads"]["all"] if b["mode"] == "pooled" else b["heads"][crop]
                z = float(heads.raw_score(h, X)[0])
                T = b["temperature"]
            p = float(sigmoid(z / T))
            out[a] = {"label": "diseased" if p >= 0.5 else "healthy", "prob": p, "z": z, "name": ARM_LABELS.get(a, a),
                      "config": self.man["arms"][a]["config"], "ms": 1000 * (time.perf_counter() - t0)}
        return out

    # -- full decision ------------------------------------------------------------------
    def analyze(self, im: Image.Image, bgr: np.ndarray, stage1_fn, user_crop: str | None = None,
                arms: str = "all") -> dict:
        tr, cfg = self.trust, self.cfg
        t = {}
        cache: dict = {}
        t0 = time.perf_counter()
        p_tu, p_other = self.crop_probs(im, cache)
        t["crop"] = 1000 * (time.perf_counter() - t0)
        cset_mask = conformal_sets(np.array([[p_tu, 1 - p_tu]]), tr["crop_q_lac"])[0]
        crop_set = [c for c, k in zip(CROPS, cset_mask, strict=True) if k]
        t0 = time.perf_counter()
        ood_d = float(knn_distance(self._emb(im, cache, tr["ood_backbone"]), self.ood_ref, tr["ood_k"])[0])
        t["ood"] = 1000 * (time.perf_counter() - t0)
        crop = user_crop or ("turmeric" if p_tu >= 0.5 else "blackgram")
        t0 = time.perf_counter()
        s1 = stage1_fn(crop)
        t["stage1"] = 1000 * (time.perf_counter() - t0)
        feats = s1["features"]
        qst = quality_stats(s1["pre"]["resized"], s1["seg"]["leaf_mask"])
        q_ok, q_reason = quality_verdict(qst, float(s1["seg"]["coverage"]), tr["quality"])
        prod = self.man["production"]["disease"]
        base = {"A_rule", "B_svm", "C1_mobilenet_v2"}  # the v1 "by" fields (rule, SVM, CNN) always come back
        need = None if arms == "all" else base | ({prod} if prod != "STACK" else set(self.stacker["members"]))
        t0 = time.perf_counter()
        per_arm = self.arm_scores(im, crop, feats, cache, need)
        t["arms"] = 1000 * (time.perf_counter() - t0)
        if prod == "STACK":
            X = np.array([[logit(per_arm[a]["prob"]) for a in self.stacker["members"]]])
            p_dis = float(self.stacker["meta"].predict_proba(X)[0, 1])
            p_dis = float(sigmoid(logit(p_dis) / self.stacker.get("temperature", 1.0)))  # field_val stacker only
        else:
            p_dis = per_arm[prod]["prob"]
        dset_mask = conformal_sets(np.array([[p_dis, 1 - p_dis]]), tr["disease_q_lac"])[0]
        disease_set = [c for c, k in zip(("diseased", "healthy"), dset_mask, strict=True) if k]
        reason = None
        if not q_ok:
            reason = q_reason
        elif ood_d > tr["ood_threshold"]:
            reason = "this does not look like a turmeric or black gram leaf"
        elif user_crop is None and len(crop_set) != 1:
            reason = "crop uncertain"
        elif len(disease_set) != 1:
            reason = "disease status uncertain"
        label = "diseased" if p_dis >= 0.5 else "healthy"
        conf = max(p_dis, 1 - p_dis)
        prod_name = ARM_LABELS.get(prod, prod)
        return {
            "crop": crop, "crop_confidence": (p_tu if crop == "turmeric" else 1 - p_tu) if user_crop is None else None,
            "crop_probs": {"turmeric": p_tu, "blackgram": 1 - p_tu}, "crop_p_other_prompt": p_other,
            "crop_set": crop_set, "crop_source": "user" if user_crop else "model",
            "disease": label, "disease_prob": p_dis, "disease_confidence": conf, "disease_set": disease_set,
            "abstain": {"flag": reason is not None, "reason": reason},
            "ood": {"distance": ood_d, "threshold": tr["ood_threshold"], "flagged": ood_d > tr["ood_threshold"]},
            "quality": {**qst, "leaf_frac": float(s1["seg"]["coverage"]), "ok": q_ok, "reason": q_reason},
            "affected_pct": float(feats["affected_pct"]),
            "severity": severity(float(feats["affected_pct"]), cfg["p3"]["severity_cuts"]) if label == "diseased" else None,
            "severity_note": "heuristic grade from affected leaf area: mild < {mild}%, moderate {mild}-{moderate}%, "
                             "severe > {moderate}%".format(**cfg["p3"]["severity_cuts"]),
            "production_model": prod, "production_model_name": prod_name,
            "arms": per_arm, "reason": reason_string(feats, prod_name, conf),
            "segmenter": "classical", "timing_ms": t, "stage1": s1,
        }


@lru_cache(maxsize=4)
def _text(prompts: tuple, bb: str) -> np.ndarray:
    return clip_zero_shot_text(list(prompts), bb)


@lru_cache(maxsize=1)
def load() -> P3Model:
    return P3Model(json.loads(MANIFEST.read_text()))
