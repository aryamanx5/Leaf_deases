"""T09: compare rule-based, SVM and CNN on the identical TEST split; leakage ablation.

Outputs (all generated):
  results/comparison.csv          long table of every reported number
  results/comparison_table.md     main metrics table
  results/comparison.md           findings addressing the research question
  results/figures/comparison_bars.png, confusion_matrices.png, roc_pr_curves.png,
  inference_time.png, errors_gallery.png, disagreements_gallery.png
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from leafdx.ablation import AUGMENTS, augment, imagenet_embeddings, run_ablation, scenario_splits  # noqa: E402
from leafdx.cnn.gradcam import gradcam_for_image  # noqa: E402
from leafdx.cnn.model import load_checkpoint  # noqa: E402
from leafdx.compare import METHODS, comparison_table, fmt, get, merge_test_predictions  # noqa: E402
from leafdx.config import ROOT, load_config, path  # noqa: E402
from leafdx.ip.decide import svm_matrix  # noqa: E402
from leafdx.ip.features import FEATURE_NAMES, stage1  # noqa: E402
from leafdx.ip.lesions import overlay_lesions  # noqa: E402
from leafdx.viz import contact_sheet  # noqa: E402

NAMES = {"rule_based": "Rule-based (IP)", "svm": "SVM (IP features)", "cnn": "CNN (disease MobileNetV2)"}
COLOURS = {"rule_based": "#8c6d31", "svm": "#2b7bba", "cnn": "#c0392b"}
ALPHA = 0.05
N_BOOT = 1000


# ----------------------------------------------------------------------------- figures
def fig_bars(tab, out):
    scopes = ["overall", "blackgram", "turmeric"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, metric in zip(axes, ("macro_f1", "accuracy"), strict=True):
        x = np.arange(len(scopes))
        for i, m in enumerate(METHODS):
            vals = [get(tab, "metrics", m, s, metric) for s in scopes]
            ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=NAMES[m], color=COLOURS[m])
            if metric == "macro_f1":
                lo = [get(tab, "metrics", m, s, "macro_f1_ci_low") for s in scopes]
                hi = [get(tab, "metrics", m, s, "macro_f1_ci_high") for s in scopes]
                ax.errorbar(x + (i - 1) * 0.27, vals, yerr=[np.subtract(vals, lo), np.subtract(hi, vals)],
                            fmt="none", ecolor="black", capsize=2, lw=0.8)
        ax.set_xticks(x, scopes)
        ax.set_title(f"TEST {metric}" + (" (95% bootstrap CI)" if metric == "macro_f1" else ""))
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_confusion(tab, out):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, m in zip(axes, METHODS, strict=True):
        cm = np.array([[get(tab, "confusion", m, "overall", k) for k in r] for r in (("tn", "fp"), ("fn", "tp"))])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["healthy", "diseased"])
        ax.set_yticks([0, 1], ["healthy", "diseased"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(NAMES[m], fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_roc_pr(df, tab, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    for m, (_, sc) in METHODS.items():
        fpr, tpr, _ = roc_curve(df.y, df[sc])
        pr, rc, _ = precision_recall_curve(df.y, df[sc])
        a1.plot(fpr, tpr, color=COLOURS[m], label=f"{NAMES[m]} AUC {fmt(get(tab, 'metrics', m, 'overall', 'roc_auc'), 'roc_auc')}")
        a2.plot(rc, pr, color=COLOURS[m], label=f"{NAMES[m]} AP {fmt(get(tab, 'metrics', m, 'overall', 'pr_auc'), 'pr_auc')}")
    a1.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
    a1.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC (TEST, diseased = positive)")
    a2.set(xlabel="recall", ylabel="precision", title="Precision-recall (TEST)")
    for a in (a1, a2):
        a.legend(frameon=False, fontsize=7)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_timing(tab, out):
    rows = tab[tab.section == "inference"]
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.barh(rows.method, rows.value, color="#555")
    for y, v in enumerate(rows.value):
        ax.text(v, y, f" {v:.1f} ms", va="center", fontsize=8)
    ax.set_xlabel("median ms / image (batch 1)")
    ax.set_title("Inference time")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def gallery_errors(df, cfg, out, per_method=8):
    rows, titles = [], []
    for m, (pc, _) in METHODS.items():
        wrong = df[df[pc] != df.y]
        pick = wrong.sample(min(per_method, len(wrong)), random_state=cfg["seed"]) if len(wrong) else wrong
        imgs = [cv2.imread(str(ROOT / p)) for p in pick.path]
        t = [f"{NAMES[m].split(' ')[0]}: {r.orig_class} -> {'dis' if r[pc] else 'healthy'}" for _, r in pick.iterrows()]
        imgs += [np.full((10, 10, 3), 255, np.uint8)] * (per_method - len(imgs))
        t += [""] * (per_method - len(t))
        rows.append(imgs)
        titles.append(t)
    cv2.imwrite(str(out), contact_sheet(rows, titles, size=150))


def gallery_disagreements(df, cfg, out, n=8):
    """SVM vs CNN disagreements: original | Stage 1 abnormal regions | CNN Grad-CAM."""
    dis = df[df.svm_pred != df.cnn_pred]
    pick = dis.sample(min(n, len(dis)), random_state=cfg["seed"]) if len(dis) else dis
    model, _ = load_checkpoint(path("models", cfg) / "cnn_best.pt")
    rows, titles = [], []
    for r in pick.itertuples():
        img = cv2.imread(str(ROOT / r.path))
        s1 = stage1(img, cfg, r.crop)
        over = overlay_lesions(s1["pre"]["resized"], s1["seg"]["leaf_mask"], s1["lesions"])
        cam, _ = gradcam_for_image(model, Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), cfg)
        rows.append([img, over, cam])
        titles.append([f"true {r.orig_class}", f"SVM {'dis' if r.svm_pred else 'healthy'} "
                       f"({s1['features']['affected_pct']:.1f}%)",
                       f"CNN {'dis' if r.cnn_pred else 'healthy'} {r.cnn_conf:.0%}"])
    if not rows:
        rows, titles = [[np.full((10, 10, 3), 255, np.uint8)]], [["no disagreements"]]
    cv2.imwrite(str(out), contact_sheet(rows, titles, size=190))


# ----------------------------------------------------------------------------- ablation
def _aug_features(p: str, aug: str, crop: str, cfg: dict, cache_dir) -> dict:
    key = hashlib.sha1(f"{p}|{aug}|{json.dumps(cfg['features'], sort_keys=True)}|{json.dumps(cfg['lesions'], sort_keys=True)}".encode()).hexdigest()
    f = cache_dir / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    out = stage1(augment(cv2.imread(str(ROOT / p)), aug), cfg, crop)["features"]
    f.write_text(json.dumps(out))
    return out


def ablation(cfg, feats: pd.DataFrame) -> pd.DataFrame:
    items = pd.concat([feats[["path", "crop", "label", "stratum", "cluster_id", "split"]].assign(aug=a) for a in AUGMENTS],
                      ignore_index=True)
    cache_dir = path("cache", cfg) / "ablation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    orig = feats.set_index("path")[FEATURE_NAMES]
    aug_rows = items[items.aug != "orig"]
    print(f"ablation: Stage 1 features for {len(aug_rows)} augmented copies (cached) ...")
    recs = Parallel(n_jobs=-1)(delayed(_aug_features)(r.path, r.aug, r.crop, cfg, cache_dir) for r in aug_rows.itertuples())
    F = pd.DataFrame(index=items.index, columns=FEATURE_NAMES, dtype=float)
    o = items.aug == "orig"
    F.loc[o] = orig.loc[items.loc[o, "path"]].to_numpy()
    F.loc[aug_rows.index] = pd.DataFrame(recs, columns=FEATURE_NAMES).to_numpy()
    X_feat = svm_matrix(pd.concat([F, items[["crop"]]], axis=1))
    print("ablation: frozen ImageNet MobileNetV2 embeddings ...")
    emb_cache = path("cache", cfg) / "ablation_embeddings.npy"
    if emb_cache.exists() and len(np.load(emb_cache)) == len(items):
        X_emb = np.load(emb_cache)
    else:
        X_emb = imagenet_embeddings(items, cfg)
        np.save(emb_cache, X_emb)
    splits = scenario_splits(items, items["split"], cfg["seed"], cfg["split"]["ratios"])
    svm_params = joblib.load(path("models", cfg) / "ip_svm.joblib")["best_params"]
    return run_ablation(items, X_feat, X_emb, splits, svm_params, cfg["seed"])


# ----------------------------------------------------------------------------- markdown
def md_table(tab) -> str:
    head = "| Method | Scope | n | Accuracy | Macro-F1 [95% CI] | Precision | Recall (diseased) | Recall (healthy) | ROC-AUC |\n"
    head += "|---|---|---|---|---|---|---|---|---|\n"
    lines = []
    for scope in ("overall", "blackgram", "turmeric"):
        for m in METHODS:
            def g(k, m=m, scope=scope):
                return fmt(get(tab, "metrics", m, scope, k), k)

            lines.append(f"| {NAMES[m]} | {scope} | {g('n')} | {g('accuracy')} | {g('macro_f1')} "
                         f"[{g('macro_f1_ci_low')}, {g('macro_f1_ci_high')}] | {g('precision')} | {g('recall')} "
                         f"| {g('healthy_recall')} | {g('roc_auc')} |")
    return head + "\n".join(lines) + "\n"


CROP_TXT = {"blackgram": "Black gram", "turmeric": "Turmeric"}


def render_md(tab, abl, class_crop: dict) -> str:
    class_txt = {c: f"{CROP_TXT[k].lower()} {c.replace('_', ' ')}" for c, k in class_crop.items()}
    G = lambda s, m, sc, k: fmt(get(tab, s, m, sc, k), k)  # noqa: E731
    V = lambda s, m, sc, k: get(tab, s, m, sc, k)  # noqa: E731
    f1 = {m: V("metrics", m, "overall", "macro_f1") for m in METHODS}
    order = sorted(METHODS, key=lambda m: -f1[m])
    mc = {k: V("mcnemar", "svm_vs_cnn", "overall", k) for k in ("a_right_b_wrong", "a_wrong_b_right", "p_value")}
    sig = mc["p_value"] < ALPHA
    lines = [
        "# Comparison: hand-designed image processing vs CNN",
        "",
        "> Generated by `scripts/run_comparison.py` from `results/comparison.csv`. Do not edit by hand.",
        "",
        "**Research question.** Can conventional digital image-processing techniques identify disease-related "
        "characteristics in turmeric and black gram leaves, and how does their performance compare with a CNN-based approach?",
        "",
        f"All three methods are scored on the identical leakage-free TEST split (n = {G('metrics', 'cnn', 'overall', 'n')}; "
        f"black gram {G('metrics', 'cnn', 'blackgram', 'n')}, turmeric {G('metrics', 'cnn', 'turmeric', 'n')}). "
        "Positive class = diseased. Macro-F1 averages the healthy and diseased F1 scores.",
        "",
        md_table(tab),
        "## Findings",
        "",
    ]
    findings = [
        "**Ranking (macro-F1, overall):** " + ", ".join(f"{NAMES[m]} {G('metrics', m, 'overall', 'macro_f1')}" for m in order) + ".",
        f"**SVM vs CNN (exact McNemar):** the SVM is right where the CNN is wrong on {G('mcnemar', 'svm_vs_cnn', 'overall', 'a_right_b_wrong')} "
        f"images and the reverse on {G('mcnemar', 'svm_vs_cnn', 'overall', 'a_wrong_b_right')}; p = {G('mcnemar', 'svm_vs_cnn', 'overall', 'p_value')}, "
        + ("a statistically significant difference" if sig else "so the difference is not statistically significant")
        + f" at alpha = {G('settings', 'all', 'overall', 'alpha')}.",
    ]
    # where IP matches / falls short, per crop
    for crop in ("blackgram", "turmeric"):
        s, c = V("metrics", "svm", crop, "macro_f1"), V("metrics", "cnn", crop, "macro_f1")
        s_lo, s_hi = V("metrics", "svm", crop, "macro_f1_ci_low"), V("metrics", "svm", crop, "macro_f1_ci_high")
        c_lo, c_hi = V("metrics", "cnn", crop, "macro_f1_ci_low"), V("metrics", "cnn", crop, "macro_f1_ci_high")
        overlap = s_lo <= c_hi and c_lo <= s_hi
        rel = "is higher than" if s > c else ("is lower than" if s < c else "equals")
        findings.append(f"**{CROP_TXT[crop]}:** the SVM on hand-designed features (macro-F1 {G('metrics', 'svm', crop, 'macro_f1')}, "
                        f"healthy recall {G('metrics', 'svm', crop, 'healthy_recall')}) {rel} the CNN "
                        f"({G('metrics', 'cnn', crop, 'macro_f1')}, healthy recall {G('metrics', 'cnn', crop, 'healthy_recall')})"
                        + ("; the 95% bootstrap CIs overlap." if overlap else "; the 95% bootstrap CIs do not overlap."))
    # per original class
    pc = tab[tab.section == "per_class"].pivot_table(index="scope", columns=["method", "metric"], values="value")
    gaps = {cls: pc.loc[cls, ("cnn", "accuracy")] - pc.loc[cls, ("svm", "accuracy")] for cls in pc.index}
    worst = max(gaps, key=gaps.get)
    best_ip = min(gaps, key=gaps.get)
    findings.append(f"**By original class**, the largest CNN advantage over the SVM is on *{class_txt[worst]}* "
                 f"(CNN {G('per_class', 'cnn', worst, 'accuracy')} vs SVM {G('per_class', 'svm', worst, 'accuracy')}, "
                 f"n = {G('per_class', 'svm', worst, 'n')}); "
                 + (f"the SVM fares best relative to the CNN on *{class_txt[best_ip]}* "
                    f"(SVM {G('per_class', 'svm', best_ip, 'accuracy')} vs CNN {G('per_class', 'cnn', best_ip, 'accuracy')}, "
                    f"n = {G('per_class', 'svm', best_ip, 'n')})." if gaps[best_ip] < 0 else
                    "the SVM does not beat the CNN on any original class (it ties on "
                    + ", ".join(f"*{class_txt[k]}*" for k, v in gaps.items() if v == 0) + ")."))
    findings.append(f"**Rule-based score vs SVM:** the transparent four-measurement rule reaches macro-F1 "
                 f"{G('metrics', 'rule_based', 'overall', 'macro_f1')} (healthy recall {G('metrics', 'rule_based', 'overall', 'healthy_recall')}) "
                 f"against {G('metrics', 'svm', 'overall', 'macro_f1')} (healthy recall {G('metrics', 'svm', 'overall', 'healthy_recall')}) "
                 "for the SVM, which combines the full Stage 1 feature vector with a learned boundary. The four transparent "
                 "measurements alone over-call disease on healthy leaves; the colour and texture statistics fix most of that.")
    findings.append(f"**Interpretability and cost:** the image-processing route returns the affected-area %, lesion count "
                 "and the segmented leaf / abnormal-region masks that explain each decision, at "
                 f"{G('inference', 'svm', 'overall', 'ms_per_image')} ms per image on CPU versus "
                 f"{G('inference', 'cnn_cpu', 'overall', 'ms_per_image')} ms (CPU) / {G('inference', 'cnn_accelerator', 'overall', 'ms_per_image')} ms (GPU) for the CNN. "
                 f"SVM and CNN agree on {G('agreement', 'svm_vs_cnn', 'overall', 'agreement_rate')} of TEST images and are then "
                 f"right {G('agreement', 'svm_vs_cnn', 'overall', 'accuracy_when_agree')} of the time; the "
                 f"{G('agreement', 'svm_vs_cnn', 'overall', 'n_disagree')} disagreements are shown with both kinds of evidence in "
                 "`figures/disagreements_gallery.png`.")
    findings.append(f"**Crop identification:** the crop model names the crop correctly on "
                 f"{G('metrics', 'cnn', 'overall', 'crop_accuracy')} of TEST images.")
    lines += [f"{i}. {t}" for i, t in enumerate(findings, start=1)]
    # ablation
    lines += ["", "## Leakage ablation", "",
              "Apparent TEST accuracy when the split ignores near-duplicates (`naive_random`) or when augmented copies "
              "are created before splitting (`augment_then_split`, as in pre-augmented public datasets). SVM = Stage 1 "
              "features with the T06 hyperparameters; CNN head = logistic regression on frozen ImageNet MobileNetV2 embeddings.",
              "",
              "| Scenario | Model | Accuracy | Macro-F1 | TEST items with a near-duplicate in TRAIN | Inflation vs clean (macro-F1) |",
              "|---|---|---|---|---|---|"]
    for _, r in abl.iterrows():
        infl = G("leakage_ablation", r.method, r.scenario, "macro_f1_inflation")
        lines.append(f"| {r.scenario} | {r.method} | {G('leakage_ablation', r.method, r.scenario, 'accuracy')} | "
                     f"{G('leakage_ablation', r.method, r.scenario, 'macro_f1')} | "
                     f"{G('leakage_ablation', r.method, r.scenario, 'test_items_with_cluster_in_train')} | {infl} |")
    a = lambda m, sc, k: G("leakage_ablation", m, sc, k)  # noqa: E731
    lines += ["",
              f"With original images only, near-duplicates are rare in these datasets: {a('svm', 'naive_random', 'test_items_with_cluster_in_train')} "
              f"of naive-random TEST items have a near-duplicate in TRAIN, and macro-F1 moves by {a('svm', 'naive_random', 'macro_f1_inflation')} (SVM) "
              f"and {a('cnn_head', 'naive_random', 'macro_f1_inflation')} (CNN head). Creating augmented copies before splitting puts a sibling of "
              f"{a('svm', 'augment_then_split', 'test_items_with_cluster_in_train')} of TEST items in TRAIN and inflates macro-F1 by "
              f"{a('svm', 'augment_then_split', 'macro_f1_inflation')} (SVM) and {a('cnn_head', 'augment_then_split', 'macro_f1_inflation')} (CNN head)"
              + (": the weaker method gains more, so the gap between methods shrinks, which is how pre-augmented benchmarks can flatter weaker methods. "
                 if V("leakage_ablation", "svm", "augment_then_split", "macro_f1_inflation")
                 > V("leakage_ablation", "cnn_head", "augment_then_split", "macro_f1_inflation") else ". ")
              +
              f"Side note: on the clean split the frozen-embedding CNN head ({a('cnn_head', 'clean', 'macro_f1')}) is on par with the "
              f"fine-tuned CNN ({G('metrics', 'cnn', 'overall', 'macro_f1')})."]
    lines += ["", "## What this answers", ""]
    gap = f1["cnn"] - f1["svm"]
    lines.append(
        "Hand-designed image processing does identify disease-related characteristics: abnormal-colour regions, lesion "
        f"counts and texture separate healthy from diseased leaves well enough for an SVM to reach macro-F1 "
        f"{G('metrics', 'svm', 'overall', 'macro_f1')} on unseen leaves. "
        + (f"The CNN is higher by {G('summary', 'cnn_minus_svm', 'overall', 'macro_f1_gap')} macro-F1"
           if gap > 0 else f"The CNN is lower by {G('summary', 'svm_minus_cnn', 'overall', 'macro_f1_gap')} macro-F1")
        + ("; the difference is statistically significant (McNemar). " if sig else "; the difference is not statistically significant (McNemar). ")
        + "Turmeric is close to solved by both routes on this paper-background dataset; black gram, photographed "
        "mostly in the field, is where hand-designed features fall short, mainly by calling healthy leaves diseased "
        "(see the healthy recall values above). The image-processing route buys an explicit, inspectable measurement "
        "of how much of each leaf is affected and why it was flagged; the CNN buys accuracy without feature design.")
    return "\n".join(lines) + "\n"


def check_md_numbers(md: str, tab: pd.DataFrame) -> list[str]:
    """Every multi-digit or decimal number in the markdown must appear in comparison.csv."""
    allowed = {fmt(v, m) for v, m in zip(tab.value, tab.metric, strict=True)}
    allowed |= {str(int(v)) for v in tab.value if float(v).is_integer()}
    tokens = re.findall(r"(?<![\w./])-?\d+(?:\.\d+)?(?![\w])", md)
    return [t for t in tokens if (len(t.lstrip("-")) > 1 or "." in t) and t not in allowed]


def main() -> int:
    cfg = load_config()
    res = path("results", cfg)
    figs = path("figures", cfg)
    ip_pred = pd.read_csv(res / "ip_predictions.csv")
    cnn_pred = pd.read_csv(res / "cnn_predictions.csv")
    df = merge_test_predictions(ip_pred, cnn_pred)
    ip_m = json.loads((res / "ip_metrics.json").read_text())
    cnn_m = json.loads((res / "cnn_metrics.json").read_text())
    timing = {
        "rule_based": ip_m["inference_time"]["total_ms_median"],
        "svm": ip_m["inference_time"]["total_ms_median"],
        "cnn_cpu": cnn_m["inference_time"]["cpu"]["ms_median"],
        "cnn_accelerator": cnn_m["inference_time"]["accelerator"]["ms_median"],
    }
    tab = comparison_table(df, timing, N_BOOT, cfg["seed"])

    feats = pd.read_csv(path("features", cfg))
    abl = ablation(cfg, feats)
    extra = []
    for _, r in abl.iterrows():
        clean = abl[(abl.scenario == "clean") & (abl.method == r.method)].iloc[0]
        for k in ("accuracy", "macro_f1", "test_items_with_cluster_in_train", "n_train", "n_test"):
            extra.append(("leakage_ablation", r.method, r.scenario, k, r[k]))
        extra.append(("leakage_ablation", r.method, r.scenario, "macro_f1_inflation", r.macro_f1 - clean.macro_f1))
        extra.append(("leakage_ablation", r.method, r.scenario, "accuracy_inflation", r.accuracy - clean.accuracy))
    f1s = {m: get(tab, "metrics", m, "overall", "macro_f1") for m in METHODS}
    extra += [("summary", "cnn_minus_svm", "overall", "macro_f1_gap", f1s["cnn"] - f1s["svm"]),
              ("summary", "svm_minus_cnn", "overall", "macro_f1_gap", f1s["svm"] - f1s["cnn"]),
              ("settings", "all", "overall", "alpha", ALPHA),
              ("settings", "all", "overall", "n_bootstrap", N_BOOT),
              ("settings", "all", "overall", "ci_level_pct", 95),
              ("settings", "all", "overall", "seed", cfg["seed"])]
    tab = pd.concat([tab, pd.DataFrame(extra, columns=tab.columns)], ignore_index=True)
    tab.to_csv(res / "comparison.csv", index=False)

    (res / "comparison_table.md").write_text(md_table(tab))
    class_crop = feats.drop_duplicates("orig_class").set_index("orig_class")["crop"].to_dict()
    md = render_md(tab, abl, class_crop)
    bad = check_md_numbers(md, tab)
    (res / "comparison.md").write_text(md)

    fig_bars(tab, figs / "comparison_bars.png")
    fig_confusion(tab, figs / "confusion_matrices.png")
    fig_roc_pr(df, tab, figs / "roc_pr_curves.png")
    fig_timing(tab, figs / "inference_time.png")
    gallery_errors(df, cfg, figs / "errors_gallery.png")
    gallery_disagreements(df, cfg, figs / "disagreements_gallery.png")

    print(md_table(tab))
    print(abl.round(3).to_string(index=False))
    print(f"McNemar SVM vs CNN p = {get(tab, 'mcnemar', 'svm_vs_cnn', 'overall', 'p_value'):.4f}")
    print(f"numbers in comparison.md not found in comparison.csv: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
