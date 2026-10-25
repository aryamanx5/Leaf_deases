"""P2-B04: hierarchical model -- separate CROP model (selected on wild_val) and a
crop-conditioned DISEASE CNN, both trained with background-swap augmentation.

Outputs: results/crop_model_selection.csv, results/robust_preds.csv (per image, per
candidate), models/crop_model.joblib, models/disease_cnn.pt, models/MODEL_CARD.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, log_loss  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from leafdx.cnn.augment import SwapDataset, build_banks, robust_transform  # noqa: E402
from leafdx.config import ROOT, load_config, path, seed_everything  # noqa: E402
from leafdx.robust.backbones import clip_zero_shot_text, embed  # noqa: E402
from leafdx.robust.cnn_heads import (  # noqa: E402
    crop_cnn_probs,
    crop_index,
    predict_disease,
    train_crop_cnn,
    train_disease_cnn,
)
from leafdx.robust.views import tta_views  # noqa: E402

REG = ROOT / "tests/fixtures/regressions/turmeric_google_01.jpg"


def _h(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def eval_sets(cfg) -> pd.DataFrame:
    s = pd.read_csv(path("splits", cfg))
    w = pd.read_csv(ROOT / "data/wild/splits.csv")
    rows = [s[s.split.isin(["val", "test"])].assign(set="clean_" + s.split, source="clean")[["path", "crop", "label", "set", "source"]],
            w.assign(set=w.split)[["path", "crop", "label", "set", "source"]]]
    if REG.exists():
        rows.append(pd.DataFrame([{"path": str(REG.relative_to(ROOT)), "crop": "turmeric", "label": "diseased",
                                   "set": "regression", "source": "user_reported"}]))
    return pd.concat(rows, ignore_index=True)


def load_img(p: str) -> Image.Image:
    return Image.open(ROOT / p).convert("RGB")


def train_views(train, banks, cfg) -> list:
    """K background-swap views per TRAIN image, ordered view-major (matches y_tr)."""
    ds = SwapDataset(train, banks, cfg, target_fn=lambda r: 0, seed=cfg["seed"])
    geo = T.Compose(robust_transform(cfg).transforms[:-3])
    return [geo(ds.sample_image(i)) for _ in range(cfg["robust"]["train_views_per_image"]) for i in range(len(ds))]


def cached_embed(name: str, key: str, make_images, cfg) -> np.ndarray:
    f = path("cache", cfg) / f"emb_{name}_{key}.npy"
    if f.exists():
        return np.load(f)
    e = embed(name, make_images())
    np.save(f, e)
    return e


def view_lists(ev: pd.DataFrame, cfg) -> tuple[list, np.ndarray]:
    ims, owner = [], []
    for k, p in enumerate(ev.path):
        v = tta_views(load_img(p), cfg, tiles=True)
        ims += v
        owner += [k] * len(v)
    return ims, np.array(owner)


def per_image(pv: np.ndarray, owner: np.ndarray, n: int) -> np.ndarray:
    return np.stack([pv[owner == k].mean(0) for k in range(n)])


def acc(p: np.ndarray, ev: pd.DataFrame, mask) -> float:
    y = np.array([crop_index(c) for c in ev.crop])
    return float((p[mask].argmax(1) == y[mask]).mean())


def main() -> int:
    cfg = load_config()
    seed_everything(cfg["seed"])
    r = cfg["robust"]
    s = pd.read_csv(path("splits", cfg))
    train = s[s.split == "train"].reset_index(drop=True)
    banks = build_banks(train, cfg)
    ev = eval_sets(cfg)
    m_val, m_wv = (ev.set == "clean_val").to_numpy(), (ev.set == "wild_val").to_numpy()
    y_ev = np.array([crop_index(c) for c in ev.crop])
    key_train = _h(len(train), r["train_views_per_image"], json.dumps(cfg["augment"], sort_keys=True), cfg["seed"],
                   json.dumps({k: cfg[k] for k in ("preprocess", "segment")}, sort_keys=True))
    key_eval = _h(*ev.path, json.dumps(cfg["segment"], sort_keys=True))
    tv_cache: dict = {}

    def get_train_views():
        if "v" not in tv_cache:
            tv_cache["v"] = train_views(train, banks, cfg)
        return tv_cache["v"]

    y_tr = np.array([crop_index(c) for c in train.crop] * r["train_views_per_image"])
    ev_views: dict = {}

    def get_ev_views():
        if "v" not in ev_views:
            ev_views["v"] = view_lists(ev, cfg)
        return ev_views["v"]

    owner_f = path("cache", cfg) / f"evowner_{key_eval}.npy"
    rows, preds = [], {}
    for name in r["crop_candidates"]:
        t0 = time.time()
        Xtr = cached_embed(name, key_train, get_train_views, cfg)
        Xev = cached_embed(name, key_eval, lambda: get_ev_views()[0], cfg)
        if not owner_f.exists():
            np.save(owner_f, get_ev_views()[1])
        owner = np.load(owner_f)
        Xval = Xev[np.isin(owner, np.flatnonzero(m_val))]
        o_val = owner[np.isin(owner, np.flatnonzero(m_val))]
        best = None
        for C in r["head_C"]:
            clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Xtr, y_tr)
            pv = np.stack([clf.predict_proba(Xval[o_val == k]).mean(0) for k in np.flatnonzero(m_val)])
            ll = log_loss(y_ev[m_val], pv, labels=[0, 1])
            if best is None or ll < best[0]:
                best = (ll, C, clf)
        _, C, clf = best
        p = per_image(clf.predict_proba(Xev), owner, len(ev))
        preds[f"{name}_head"] = p
        rows.append({"candidate": f"{name} + logistic head", "backbone": name, "C": C,
                     "clean_val_acc": acc(p, ev, m_val), "wild_val_acc": acc(p, ev, m_wv), "fit_s": time.time() - t0})
        joblib.dump({"backbone": name, "clf": clf, "C": C}, path("cache", cfg) / f"crophead_{name}.joblib")
        if name == "openclip_vitb32":
            txt = clip_zero_shot_text(r["clip_prompts"]["turmeric"] + r["clip_prompts"]["blackgram"])
            nt = len(r["clip_prompts"]["turmeric"])
            sims = 100 * Xev @ txt.T
            zl = np.stack([sims[:, :nt].mean(1), sims[:, nt:].mean(1)], 1)
            zp = np.exp(zl - zl.max(1, keepdims=True))
            zp /= zp.sum(1, keepdims=True)
            pz = per_image(zp, owner, len(ev))
            preds["openclip_zeroshot"] = pz
            rows.append({"candidate": "openclip_vitb32 zero-shot", "backbone": name, "C": None,
                         "clean_val_acc": acc(pz, ev, m_val), "wild_val_acc": acc(pz, ev, m_wv), "fit_s": 0.0})
            comb = p * pz
            comb /= comb.sum(1, keepdims=True)
            preds["openclip_head_x_zeroshot"] = comb
            rows.append({"candidate": "openclip_vitb32 head x zero-shot prior", "backbone": name, "C": C,
                         "clean_val_acc": acc(comb, ev, m_val), "wild_val_acc": acc(comb, ev, m_wv), "fit_s": 0.0})
        print(rows[-1] if name != "openclip_vitb32" else rows[-3:])

    # EfficientNet-B0 baseline, fine-tuned with the same augmentation
    val_df = ev[m_val].reset_index(drop=True)
    val_ims = [load_img(p) for p in val_df.path]
    ck = path("models", cfg) / "crop_cnn_effb0.pt"
    t0 = time.time()
    if "--reuse" in sys.argv and ck.exists():  # do not retrain an unchanged checkpoint
        from leafdx.cnn.model import build_model

        model, _ = build_model("efficientnet_b0", 2, pretrained=False)
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        info = {"reused_checkpoint": ck.name}
    else:
        model, info = train_crop_cnn(train, val_ims, val_df, banks, cfg, ck)
    pe = crop_cnn_probs(model, [load_img(p) for p in ev.path], cfg)
    preds["efficientnet_b0_finetune"] = pe
    rows.append({"candidate": "efficientnet_b0 fine-tuned (B03 aug)", "backbone": "efficientnet_b0", "C": None,
                 "clean_val_acc": acc(pe, ev, m_val), "wild_val_acc": acc(pe, ev, m_wv), "fit_s": time.time() - t0,
                 "train_info": json.dumps(info)})
    print(rows[-1])

    # CPU latency per image (TTA views incl. tiles), same images for every candidate,
    # each model warmed up first so loading time is not counted
    lat_imgs = [load_img(p) for p in ev[m_val].path[: r["latency_images"]]]
    embed("dinov2_vits14", lat_imgs[:1], dev_str="cpu")
    embed("openclip_vitb32", lat_imgs[:1], dev_str="cpu")
    crop_cnn_probs(model, lat_imgs[:1], cfg, dev=torch.device("cpu"))
    for row in rows:
        t0 = time.time()
        for im in lat_imgs:
            vs = tta_views(im, cfg, tiles=True)
            if row["backbone"] == "efficientnet_b0":
                crop_cnn_probs(model, [im], cfg, dev=torch.device("cpu"))
            else:
                embed(row["backbone"], vs, dev_str="cpu")
        row["cpu_ms_per_image"] = 1000 * (time.time() - t0) / len(lat_imgs)

    sel = pd.DataFrame(rows)
    sel["selected"] = False
    order = sel.sort_values(["wild_val_acc", "cpu_ms_per_image"], ascending=[False, True])
    best_i = order.index[0]
    sel.loc[best_i, "selected"] = True
    sel.to_csv(path("results", cfg) / "crop_model_selection.csv", index=False)
    chosen = sel.loc[best_i]
    key = {"dinov2_vits14 + logistic head": "dinov2_vits14_head", "openclip_vitb32 + logistic head": "openclip_vitb32_head",
           "openclip_vitb32 zero-shot": "openclip_zeroshot", "openclip_vitb32 head x zero-shot prior": "openclip_head_x_zeroshot",
           "efficientnet_b0 fine-tuned (B03 aug)": "efficientnet_b0_finetune"}[chosen.candidate]
    bundle = {"candidate": chosen.candidate, "key": key, "backbone": chosen.backbone}
    if chosen.backbone != "efficientnet_b0":
        bundle.update(joblib.load(path("cache", cfg) / f"crophead_{chosen.backbone}.joblib"))
        if "zero" in key:
            bundle["prompts"] = r["clip_prompts"]
    joblib.dump(bundle, path("models", cfg) / "crop_model.joblib")
    print(sel.drop(columns=[c for c in ("train_info",) if c in sel]).round(4).to_string())
    print("selected:", chosen.candidate)

    # Disease CNN (crop-conditioned)
    dck = path("models", cfg) / "disease_cnn.pt"
    dmodel, dinfo = train_disease_cnn(train, val_ims, val_df, banks, cfg, dck)
    lab = ev.label != "unknown"
    pdis = np.full(len(ev), np.nan)
    pdis[lab.to_numpy()] = predict_disease(dmodel, [load_img(p) for p in ev[lab].path], list(ev[lab].crop), cfg)
    out = ev.copy()
    for k, v in preds.items():
        out[f"p_turmeric__{k}"] = v[:, 0]
    out["p_diseased_given_crop"] = pdis
    out.to_csv(path("results", cfg) / "robust_preds.csv", index=False)
    dis = {}
    for st in ("clean_val", "clean_test", "wild_val", "wild_test"):
        m = (out.set == st) & lab
        yd = (out[m].label == "diseased").astype(int)
        dis[st] = {"n": int(m.sum()), "macro_f1": float(f1_score(yd, (out[m].p_diseased_given_crop >= 0.5).astype(int), average="macro")),
                   "accuracy": float(((out[m].p_diseased_given_crop >= 0.5).astype(int) == yd).mean())}
    json.dump({"disease_cnn": {"train": dinfo, "eval_tta": dis}, "crop_selected": chosen.candidate},
              open(path("results", cfg) / "robust_train.json", "w"), indent=2)
    print("disease:", json.dumps(dis, indent=1))
    write_model_card(cfg, sel, chosen, dinfo, dis)
    return 0


def write_model_card(cfg, sel, chosen, dinfo, dis) -> None:
    f = lambda x: f"{x:.3f}"  # noqa: E731
    lines = ["# Model card (generated by scripts/train_robust.py)", "",
             "## Crop model", "",
             f"Selected: **{chosen.candidate}** (highest wild_val crop accuracy; ties -> lower CPU latency).", "",
             "| Candidate | Clean val crop acc | Wild val crop acc | CPU ms / image (TTA) | Selected |", "|---|---|---|---|---|"]
    for _, r in sel.iterrows():
        lines.append(f"| {r.candidate} | {f(r.clean_val_acc)} | {f(r.wild_val_acc)} | {r.cpu_ms_per_image:.0f} | {'yes' if r.selected else ''} |")
    lines += ["", "Training data: TRAIN split only (see data/processed/splits.csv), "
              f"{cfg['robust']['train_views_per_image']} background-swap views per image for frozen-backbone heads. "
              "Inference: mean probability over TTA views (full, horizontal flip, 4 corner crops) plus 3x3 plant tiles "
              "for field scenes.", "",
              "## Disease model", "",
              f"{cfg['robust']['disease_cnn']['arch']} (ImageNet-pretrained) with one diseased/healthy logit per crop; "
              f"trained {dinfo['epochs_run']} epochs ({dinfo['minutes']:.1f} min on {dinfo['device']}), best epoch "
              f"{dinfo['epoch']} by clean VAL macro-F1 {f(dinfo['score'])}. Evaluated with TTA (no tiles):", "",
              "| Set | n (labelled) | Accuracy | Macro-F1 |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {f(v['accuracy'])} | {f(v['macro_f1'])} |" for k, v in dis.items()]
    lines += ["", "## Augmentation", "", "Background swap (leaf cut out with the Stage 1 mask, composited onto inpainted "
              "TRAIN backgrounds of either crop, procedural soil/foliage or plain colours), same-crop collages, "
              "RandomResizedCrop, rotation, colour jitter, blur, JPEG re-compression, random erasing (configs/default.yaml "
              "`augment`).", "",
              "## Limits", "", "- Training images come from one black gram and one turmeric dataset; field scenes of "
              "turmeric are absent from training and only simulated by compositing.",
              "- Wild labelled images are single leaves on white; field scenes are disease-unlabelled.",
              "- Confidence is calibrated separately (P2-B05); uncertain crop predictions are surfaced, not hidden.", ""]
    (path("models", cfg) / "MODEL_CARD.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
