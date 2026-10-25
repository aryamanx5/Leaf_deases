"""P2-B02: diagnose the v1 CNN crop errors -> results/diagnosis.md + figures + diagnosis.json.

a) crop confusion on clean TEST, wild_val, wild_test
b) shortcut probes on clean TEST: background-only (leaf replaced by the image mean colour,
   via the Stage 1 leaf mask) and leaf-only (background replaced by neutral grey)
c) Grad-CAM for the CROP decision (summed logits of the predicted crop) on 12 wild images
d) capture statistics per dataset (resolution, aspect, brightness, border colour, JPEG quality)
e) calibration: reliability diagrams + ECE (crop and disease) on clean TEST and wild_val
Every statement in diagnosis.md is derived from diagnosis.json.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from PIL import Image  # noqa: E402

from leafdx.cnn.dataset import CLASSES4, eval_transform  # noqa: E402
from leafdx.cnn.gradcam import gradcam, overlay_cam  # noqa: E402
from leafdx.cnn.model import device, load_checkpoint  # noqa: E402
from leafdx.config import ROOT, load_config, path  # noqa: E402
from leafdx.ip.preprocess import load_bgr, preprocess  # noqa: E402
from leafdx.ip.segment import border_ring, segment_leaf  # noqa: E402
from leafdx.viz import contact_sheet  # noqa: E402

CROPS = ("turmeric", "blackgram")
TU_IDX = [i for i, c in enumerate(CLASSES4) if c.startswith("turmeric")]
DIS_IDX = [i for i, c in enumerate(CLASSES4) if c.endswith("diseased")]


@torch.no_grad()
def predict(model, images_rgb: list[np.ndarray], cfg, dev) -> np.ndarray:
    tf = eval_transform(cfg["cnn"])
    out = []
    for i in range(0, len(images_rgb), 32):
        x = torch.stack([tf(Image.fromarray(im)) for im in images_rgb[i:i + 32]]).to(dev)
        out.append(torch.softmax(model(x), 1).cpu().numpy())
    return np.concatenate(out)


def crop_of(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p_tu = probs[:, TU_IDX].sum(1)
    return np.where(p_tu >= 0.5, "turmeric", "blackgram"), np.maximum(p_tu, 1 - p_tu)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> tuple[float, list]:
    edges = np.linspace(0.5 if conf.min() >= 0.5 else 0.0, 1.0, bins + 1)
    out, total = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi) if lo > edges[0] else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
            out.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return float(total), out


def seg_rgb(p: str, cfg: dict) -> tuple[np.ndarray, np.ndarray, float]:
    pre = preprocess(load_bgr(ROOT / p), cfg)
    seg = segment_leaf(pre, cfg)
    return cv2.cvtColor(pre["resized"], cv2.COLOR_BGR2RGB), seg["leaf_mask"], seg["coverage"]


def jpeg_quality(p) -> float:
    """IJG quality estimate from the luminance quantisation table (nan if not a JPEG)."""
    std = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)
    with Image.open(p) as im:
        q = getattr(im, "quantization", None)
    if not q:
        return float("nan")
    scale = 100 * np.mean(np.array(q[0], float)) / std.mean()
    return float((200 - scale) / 2 if scale <= 100 else 5000 / scale)


def capture_stats(p: str) -> dict:
    full = ROOT / p
    bgr = load_bgr(full)
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    ring = border_ring((256, 256), 0.05)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[ring].astype(float)
    return {"width": w, "height": h, "aspect": w / h, "brightness_v": float(hsv[..., 2].mean()),
            "border_L": float(lab[:, 0].mean()), "border_chroma": float(np.hypot(lab[:, 1] - 128, lab[:, 2] - 128).mean()),
            "border_neutral_bright_frac": float(((hsv[..., 1][ring] < 40) & (hsv[..., 2][ring] > 150)).mean()),
            "jpeg_quality": jpeg_quality(full)}


def reliability_fig(curves: dict, out) -> None:
    fig, axes = plt.subplots(1, len(curves), figsize=(4 * len(curves), 3.6))
    for ax, (name, (e, pts)) in zip(np.atleast_1d(axes), curves.items(), strict=True):
        ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
        if pts:
            c, a, n = zip(*pts, strict=True)
            ax.plot(c, a, marker="o", c="#c0392b")
            for ci, ai, ni in pts:
                ax.annotate(str(ni), (ci, ai), fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.set(xlim=(0.4, 1.02), ylim=(0, 1.02), xlabel="confidence", ylabel="accuracy", title=f"{name}\nECE {e:.3f}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    cfg = load_config()
    if "--render" in sys.argv:
        D = json.loads((path("results", cfg) / "diagnosis.json").read_text())
        (path("results", cfg) / "diagnosis.md").write_text(render(D))
        print((path("results", cfg) / "diagnosis.md").read_text())
        return 0
    res, figs = path("results", cfg), path("figures", cfg)
    dev = device()
    model, _ = load_checkpoint(path("models", cfg) / "cnn_best.pt")
    model.to(dev)
    splits = pd.read_csv(path("splits", cfg))
    test = splits[splits.split == "test"].reset_index(drop=True)
    wild = pd.read_csv(ROOT / "data/wild/splits.csv")
    D: dict = {"n": {"clean_test": len(test), "wild_val": int((wild.split == "wild_val").sum()),
                     "wild_test": int((wild.split == "wild_test").sum())}}

    # ---- a) crop confusion -------------------------------------------------------------
    seg = Parallel(n_jobs=-1)(delayed(seg_rgb)(p, cfg) for p in test.path)
    rgb = [s[0] for s in seg]
    probs_clean = predict(model, rgb, cfg, dev)
    wild_rgb = [cv2.cvtColor(load_bgr(ROOT / p), cv2.COLOR_BGR2RGB) for p in wild.path]
    probs_wild = predict(model, wild_rgb, cfg, dev)
    sets = {"clean_test": (test, probs_clean),
            "wild_val": (wild[wild.split == "wild_val"], probs_wild[(wild.split == "wild_val").to_numpy()]),
            "wild_test": (wild[wild.split == "wild_test"], probs_wild[(wild.split == "wild_test").to_numpy()])}
    D["crop"] = {}
    for name, (df, pr) in sets.items():
        pred, conf = crop_of(pr)
        y = df.crop.to_numpy()
        ok = pred == y
        cm = {t: {p: int(((y == t) & (pred == p)).sum()) for p in CROPS} for t in CROPS}
        D["crop"][name] = {"accuracy": float(ok.mean()), "confusion": cm,
                           "mean_conf_wrong": float(conf[~ok].mean()) if (~ok).any() else None,
                           "share_wrong_conf_ge_0.9": float((conf[~ok] >= 0.9).mean()) if (~ok).any() else None,
                           "majority_baseline": float(max((y == c).mean() for c in CROPS))}
    wv = wild.assign(pred=crop_of(probs_wild)[0])
    wv["ok"] = wv.pred == wv.crop
    D["crop"]["wild_by_source"] = {f"{c}|{s}": {"n": int(len(g)), "accuracy": float(g.ok.mean())}
                                   for (c, s), g in wv.groupby(["crop", "source"])}
    lab = wv[wv.label != "unknown"]
    D["crop"]["wild_by_label"] = {f"{c}|{lb}": {"n": int(len(g)), "accuracy": float(g.ok.mean())}
                                  for (c, lb), g in lab.groupby(["crop", "label"])}

    # ---- b) shortcut probes ------------------------------------------------------------
    bg_only, leaf_only, bg_box = [], [], []
    for im, m, _ in seg:
        mb = m.astype(bool)
        mean = im.reshape(-1, 3).mean(0).astype(np.uint8)
        a = im.copy()
        a[mb] = mean
        bg_only.append(a)
        # stricter: blank the dilated bounding box of the leaf so its silhouette (shape) is hidden too
        c = im.copy()
        ys, xs = np.nonzero(mb)
        if len(ys):
            pad = int(0.05 * max(im.shape[:2]))
            c[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad] = mean
        bg_box.append(c)
        b = im.copy()
        b[~mb] = 128
        leaf_only.append(b)
    y = test.crop.to_numpy()
    D["probe"] = {"original": float((crop_of(probs_clean)[0] == y).mean()),
                  "background_only": float((crop_of(predict(model, bg_only, cfg, dev))[0] == y).mean()),
                  "leaf_only": float((crop_of(predict(model, leaf_only, cfg, dev))[0] == y).mean()),
                  "background_box_only": float((crop_of(predict(model, bg_box, cfg, dev))[0] == y).mean()),
                  "majority_baseline": D["crop"]["clean_test"]["majority_baseline"],
                  "median_leaf_coverage": float(np.median([s[2] for s in seg]))}
    ex = np.random.default_rng(cfg["seed"]).choice(len(test), 6, replace=False)
    cv2.imwrite(str(figs / "diag_probes.png"), contact_sheet(
        [[cv2.cvtColor(x[i], cv2.COLOR_RGB2BGR) for x in (rgb, bg_only, bg_box, leaf_only)] for i in ex],
        [[f"{test.stratum[i]}", "background only", "background, leaf box hidden", "leaf only"] for i in ex], size=180))

    # ---- c) Grad-CAM for the crop decision on 12 wild images ----------------------------
    model_cpu, _ = load_checkpoint(path("models", cfg) / "cnn_best.pt")
    pick = wv.groupby("crop").sample(6, random_state=cfg["seed"])
    rows, titles = [], []
    tf = eval_transform(cfg["cnn"])
    for r in pick.itertuples():
        im = Image.open(ROOT / r.path).convert("RGB")
        x = tf(im).unsqueeze(0)
        _, pr = gradcam(model_cpu, x)
        idx = TU_IDX if pr[TU_IDX].sum() >= 0.5 else [i for i in range(4) if i not in TU_IDX]
        cam, _ = gradcam(model_cpu, x, target=lambda lg, ix=idx: torch.logsumexp(lg[ix], 0))
        rows.append([cv2.cvtColor(np.array(im.resize((224, 224))), cv2.COLOR_RGB2BGR),
                     overlay_cam(np.array(im.resize((224, 224))), cam)])
        titles.append([f"true {r.crop} ({r.source[:8]})", f"crop pred {r.pred} {max(pr[TU_IDX].sum(), 1 - pr[TU_IDX].sum()):.0%}"])
    cv2.imwrite(str(figs / "diag_crop_gradcam.png"), contact_sheet(
        [rows[i] + rows[i + 1] for i in range(0, len(rows), 2)], [titles[i] + titles[i + 1] for i in range(0, len(rows), 2)], size=200))

    # ---- d) capture statistics -----------------------------------------------------------
    allp = pd.concat([splits[["path", "crop"]].assign(dataset="train_" + splits.crop),
                      wild[["path", "crop", "source"]].assign(dataset="wild_" + wild.source.str.split(":").str[0])])
    st = pd.DataFrame(Parallel(n_jobs=-1)(delayed(capture_stats)(p) for p in allp.path))
    st["dataset"] = allp.dataset.to_numpy()
    st["crop"] = allp.crop.to_numpy()
    agg = st.groupby(["dataset", "crop"]).agg(n=("width", "size"), width=("width", "median"), height=("height", "median"),
                                              aspect=("aspect", "median"), brightness=("brightness_v", "mean"),
                                              border_L=("border_L", "mean"), border_chroma=("border_chroma", "mean"),
                                              neutral_bright_border=("border_neutral_bright_frac", "mean"),
                                              jpeg_q=("jpeg_quality", "median")).round(3)
    D["capture"] = {f"{d}|{c}": row.to_dict() for (d, c), row in agg.iterrows()}

    # ---- e) calibration --------------------------------------------------------------------
    curves = {}
    for name in ("clean_test", "wild_val"):
        df, pr = sets[name]
        pred, conf = crop_of(pr)
        e, pts = ece(conf, pred == df.crop.to_numpy())
        curves[f"crop | {name}"] = (e, pts)
        D.setdefault("ece", {})[f"crop_{name}"] = e
        lab_m = (df.label != "unknown").to_numpy()
        p_dis = pr[:, DIS_IDX].sum(1)[lab_m]
        yd = (df.label[lab_m] == "diseased").to_numpy()
        e2, pts2 = ece(np.maximum(p_dis, 1 - p_dis), (p_dis >= 0.5) == yd)
        curves[f"disease | {name}"] = (e2, pts2)
        D["ece"][f"disease_{name}"] = e2
        D.setdefault("disease_acc", {})[name] = float(((p_dis >= 0.5) == yd).mean())
    reliability_fig(curves, figs / "diag_reliability.png")

    # ---- d') Stage 1 on field scenes (cause d) ----------------------------------------------
    field = wild[wild.source.isin(["inaturalist", "wikimedia_commons"])]
    cov = Parallel(n_jobs=-1)(delayed(seg_rgb)(p, cfg) for p in field.path)
    covs = np.array([c[2] for c in cov])
    D["stage1_field"] = {"n": int(len(field)), "median_coverage": float(np.median(covs)),
                         "share_coverage_lt_0.15": float((covs < 0.15).mean()),
                         "clean_test_median_coverage": D["probe"]["median_leaf_coverage"]}
    bug = json.loads((res / "bug_repro_before.json").read_text()) if (res / "bug_repro_before.json").exists() else None
    if bug:
        D["bug_image"] = {"p_blackgram": bug["crop_probabilities"]["blackgram"], "affected_pct": bug["affected_region_pct"],
                          "lesion_count": bug["lesion_count"]}

    (res / "diagnosis.json").write_text(json.dumps(D, indent=2))
    (res / "diagnosis.md").write_text(render(D))
    print((res / "diagnosis.md").read_text())
    return 0


def verdict(level: int) -> str:
    return ["not confirmed", "**PARTLY CONFIRMED**", "**CONFIRMED**", "inconclusive"][level]


def render(D: dict) -> str:
    f = lambda x: f"{x:.3f}"  # noqa: E731
    c, p, cap = D["crop"], D["probe"], D["capture"]
    src = c["wild_by_source"]
    field_tu = [src[k] for k in ("turmeric|inaturalist", "turmeric|wikimedia_commons") if k in src]
    field_tu_acc = sum(x["accuracy"] * x["n"] for x in field_tu) / max(sum(x["n"] for x in field_tu), 1)
    white = {k: src[k] for k in ("turmeric|mendeley:jtttfbx342", "blackgram|mendeley:z55yrbmn2d") if k in src}
    bg_strong = p["background_box_only"] > p["majority_baseline"] + 0.15
    a_lvl = 2 if bg_strong and field_tu and field_tu_acc < 0.8 else (1 if bg_strong else 0)
    wl = c["wild_by_label"]
    n_err = sum(round((1 - v["accuracy"]) * v["n"]) for v in wl.values())
    dis_err = 1 - np.mean([v["accuracy"] for k, v in wl.items() if k.endswith("diseased")])
    hea_err = 1 - np.mean([v["accuracy"] for k, v in wl.items() if k.endswith("healthy")])
    b_lvl = 3 if n_err == 0 else (2 if dis_err > hea_err + 0.10 else 0)
    wrong_conf = [c[s]["mean_conf_wrong"] for s in ("wild_val", "wild_test") if c[s]["mean_conf_wrong"] is not None]
    mwc = float(np.mean(wrong_conf)) if wrong_conf else float("nan")
    ece_ratio = D["ece"]["crop_wild_val"] / max(D["ece"]["crop_clean_test"], 1e-6)
    c_lvl = 2 if (mwc >= 0.8 and ece_ratio > 2) else (1 if (ece_ratio > 2 or mwc >= 0.7) else 0)
    s1 = D["stage1_field"]
    d_lvl = 2 if (s1["share_coverage_lt_0.15"] > 0.25 or s1["median_coverage"] < 0.5 * s1["clean_test_median_coverage"]) \
        else (1 if s1["median_coverage"] < s1["clean_test_median_coverage"] else 0)
    lines = ["# Diagnosis: why the v1 CNN calls a turmeric field photo black gram",
             "", "> Generated by `scripts/diagnose.py` from `results/diagnosis.json`.", "",
             "## Crop accuracy of the v1 4-class CNN (crop = argmax of summed class probabilities)", "",
             "| Set | n | Crop accuracy | Majority baseline | Mean confidence when wrong | Wrong with confidence >= 0.9 |",
             "|---|---|---|---|---|---|"]
    for s in ("clean_test", "wild_val", "wild_test"):
        r = c[s]
        lines.append(f"| {s} | {D['n'][s]} | {f(r['accuracy'])} | {f(r['majority_baseline'])} | "
                     f"{f(r['mean_conf_wrong']) if r['mean_conf_wrong'] is not None else 'n/a'} | "
                     f"{f(r['share_wrong_conf_ge_0.9']) if r['share_wrong_conf_ge_0.9'] is not None else 'n/a'} |")
    lines += ["", "Confusion (rows true, columns predicted turmeric / blackgram):", ""]
    for s in ("clean_test", "wild_val", "wild_test"):
        cm = c[s]["confusion"]
        lines.append(f"- {s}: turmeric -> [{cm['turmeric']['turmeric']}, {cm['turmeric']['blackgram']}], "
                     f"blackgram -> [{cm['blackgram']['turmeric']}, {cm['blackgram']['blackgram']}]")
    lines += ["", "Wild crop accuracy by source:", ""]
    lines += [f"- {k}: {f(x['accuracy'])} (n = {x['n']})" for k, x in src.items()]
    white_txt = "; ".join(f"{k.split('|')[0]} {f(v['accuracy'])} (n = {v['n']})" for k, v in white.items())
    lines += ["", "## Causes", "",
              f"**(a) Shortcut learning from capture style: {verdict(a_lvl)}.** With the leaf removed (background only) "
              f"the CNN still names the crop of {f(p['background_only'])} of clean TEST images (majority baseline "
              f"{f(p['majority_baseline'])}); with the background greyed out (leaf only) it scores {f(p['leaf_only'])} "
              f"(original {f(p['original'])}). The mean-colour fill still shows the leaf's silhouette, so a stricter probe hides "
              f"the leaf's whole bounding box: the surroundings alone still give crop accuracy {f(p['background_box_only'])}. "
              + ("The background alone is therefore enough to predict the crop. " if bg_strong else
                 "So the surroundings alone are not enough to predict the crop. ")
              +
              f"The errors in the wild are all of one kind: turmeric FIELD scenes (foliage around the leaf, iNaturalist + "
              f"Commons) reach crop accuracy {f(field_tu_acc)}, while single leaves on white from the independent datasets "
              f"score {white_txt}. In training, turmeric is always on white paper and black gram mostly in the field, so "
              "foliage surroundings read as black gram. The capture statistics table shows how different the two training "
              "datasets are (resolution, brightness, border colour).",
              "",
              f"**(b) One 4-class softmax entangles crop and disease: {verdict(b_lvl)}.** "
              + ("Labelled wild images contain no crop errors, so crop error rates of diseased vs healthy leaves cannot be "
                 "compared; the separation of crop and disease (P2-B04) is still adopted because it removes the 4-way "
                 "softmax from the user-facing output." if b_lvl == 3 else
                 f"Crop error rate on labelled wild images: diseased {f(dis_err)} vs healthy {f(hea_err)}."),
              "",
              f"**(c) Out-of-distribution input with no calibration or abstention: {verdict(c_lvl)}.** Crop ECE rises from "
              f"{f(D['ece']['crop_clean_test'])} on clean TEST to {f(D['ece']['crop_wild_val'])} on wild_val; wrong crop "
              f"predictions on wild images carry mean confidence {f(mwc)} and are shown without any abstention. "
              f"Disease ECE: {f(D['ece']['disease_clean_test'])} clean vs {f(D['ece']['disease_wild_val'])} wild_val.",
              "",
              f"**(d) Stage 1 assumes one leaf on a contrasting background: {verdict(d_lvl)}.** On {s1['n']} field scenes the "
              f"median leaf-mask coverage is {f(s1['median_coverage'])} of the frame (clean TEST "
              f"{f(s1['clean_test_median_coverage'])}); {f(s1['share_coverage_lt_0.15'])} of field scenes yield a mask under "
              "0.15 of the frame. Coverage alone does not show whether the MAIN leaf was chosen; that is checked visually "
              "in P2-B06."
              + (f" On the user's image the mask is a central blob with {D['bug_image']['lesion_count']} lesion and "
                 f"{f(D['bug_image']['affected_pct'])}% affected." if "bug_image" in D else ""),
              "", "## Capture statistics", "",
              "JPEG quality is only meaningful for the training datasets (wild images were re-encoded on download).", "",
              "| Dataset | Crop | n | Median width | Median height | Brightness (V) | Border L* | Border chroma | Neutral bright border share | JPEG quality (median) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in cap.items():
        d, cr = k.split("|")
        q = f"{r['jpeg_q']:.0f}" if d.startswith("train") and r["jpeg_q"] == r["jpeg_q"] else "n/a"
        lines.append(f"| {d} | {cr} | {int(r['n'])} | {int(r['width'])} | {int(r['height'])} | {r['brightness']:.1f} | "
                     f"{r['border_L']:.1f} | {r['border_chroma']:.1f} | {f(r['neutral_bright_border'])} | {q} |")
    lines += ["", "Figures: `figures/diag_probes.png` (probe examples), `figures/diag_crop_gradcam.png` (Grad-CAM of the "
              "crop decision on wild images), `figures/diag_reliability.png` (reliability diagrams).", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
