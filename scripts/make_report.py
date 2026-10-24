"""Generate results/REPORT.md from results files only (no hand-typed numbers).

Every number inserted goes through `R.num(...)`, which reads it from a results/data file
and records (formatted value, source) in results/report_numbers.json, so
scripts/verify_all.py can re-read each source and check the report.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from leafdx.config import ROOT, load_config, path

CFG = load_config()
RES = path("results", CFG)


class Recorder:
    def __init__(self):
        self.records: list[dict] = []
        self._json: dict[str, dict] = {}
        self._csv: dict[str, pd.DataFrame] = {}

    def js(self, file: str) -> dict:
        if file not in self._json:
            self._json[file] = json.loads((ROOT / file).read_text())
        return self._json[file]

    def csv(self, file: str) -> pd.DataFrame:
        if file not in self._csv:
            self._csv[file] = pd.read_csv(ROOT / file)
        return self._csv[file]

    def _emit(self, value: float, fmt: str, source: dict) -> str:
        s = format(int(round(value)), fmt) if fmt == "d" else format(value, fmt)
        self.records.append({"text": s, "value": float(value), "fmt": fmt, "source": source})
        return s

    def num(self, file: str, *keys, fmt: str = ".3f") -> str:
        v = self.js(file)
        for k in keys:
            v = v[k]
        return self._emit(float(v), fmt, {"file": file, "keys": list(keys)})

    def cmp(self, section: str, method: str, scope: str, metric: str, fmt: str = ".3f") -> str:
        t = self.csv("results/comparison.csv")
        v = t[(t.section == section) & (t.method == method) & (t.scope == scope) & (t.metric == metric)].value
        assert len(v) == 1, (section, method, scope, metric)
        q = {"section": section, "method": method, "scope": scope, "metric": metric}
        return self._emit(float(v.iloc[0]), fmt, {"file": "results/comparison.csv", "query": q})

    def cfg(self, *keys, fmt: str = "d") -> str:
        v = CFG
        for k in keys:
            v = v[k]
        return self._emit(float(v), fmt, {"config": list(keys), "file": "configs/default.yaml"})

    def count(self, file: str, fmt: str = "d", **where) -> str:
        df = self.csv(file)
        for k, v in where.items():
            df = df[df[k] == v]
        return self._emit(len(df), fmt, {"file": file, "count_where": where})


R = Recorder()
FEAT = "data/processed/features.csv"
SPL = "data/processed/splits.csv"
INV = "data/processed/inventory.csv"
IPM, CNM = "results/ip_metrics.json", "results/cnn_metrics.json"


def pct(file, *keys):
    return R.num(file, *keys, fmt=".1%")


def dataset_section() -> list[str]:
    out = ["## Data", "",
           "Public datasets (Mendeley Data, sha256-verified on download; see README):", "",
           f"- **Black gram:** {CFG['datasets']['blackgram']['citation']}",
           f"- **Turmeric:** {CFG['datasets']['turmeric']['citation']} - original images only (the pre-augmented archive is not used).",
           "", "| Crop | Original class | Label | Images | Train | Val | Test |", "|---|---|---|---|---|---|---|"]
    sp = R.csv(SPL)
    for (crop, cls, lab), _ in sp.groupby(["crop", "orig_class", "label"]):
        out.append(f"| {crop} | {cls} | {lab} | {R.count(SPL, crop=crop, orig_class=cls)} | "
                   + " | ".join(R.count(SPL, crop=crop, orig_class=cls, split=s) for s in ("train", "val", "test")) + " |")
    out += ["", f"Total readable leaf images: {R.count(INV, readable=True)} "
            f"(unreadable: {R.count(INV, readable=False)}). Rhizome/root classes of the turmeric dataset are excluded.", "",
            "**Leakage control.** Near-duplicates are clustered with perceptual hashes at two resolutions over all "
            "flip/rotation variants plus consecutive-file-number (burst shot) grouping; whole clusters are assigned to "
            f"train/val/test (stratified by crop x label, seed {R.cfg('seed')}). "
            f"Clusters: {R.num('data/processed/split_meta.json', 'n_clusters', fmt='d')}, of which "
            f"{R.num('data/processed/split_meta.json', 'n_multi_image_clusters', fmt='d')} contain more than one image; "
            f"clusters spanning two splits: {R.num('data/processed/split_meta.json', 'clusters_spanning_splits', fmt='d')}.", ""]
    out += ["**Capture conditions.** Share of images photographed on a plain card/paper/cloth (Stage 1 plain-background mode): "]
    for crop, lab in (("blackgram", "healthy"), ("blackgram", "diseased"), ("turmeric", "healthy"), ("turmeric", "diseased")):
        n_plain = R.count(FEAT, crop=crop, label=lab, bg_mode="plain")
        out.append(f"- {crop} {lab}: {n_plain} of {R.count(FEAT, crop=crop, label=lab)}")
    return out + [""]


def methods_section() -> list[str]:
    info = "results/cnn_train_info.json"
    return [
        "## Methods", "",
        "```",
        "              LEAF IMAGE",
        "                  |",
        "        +---------+---------+",
        "        v                   v",
        "  IMAGE PROCESSING         CNN",
        "        v                   v",
        "  regions + features   learned features",
        "        v                   v",
        "  rule score / SVM     CNN classifier",
        "        +---------+---------+",
        "                  v",
        "          HEALTHY / DISEASED",
        "```", "",
        "**Stage 1 - digital image processing (core).** Resize, median + Gaussian denoising, CLAHE on L*; leaf "
        "segmentation in two modes chosen from the image border (plain background: k-means background colour model "
        "in Lab + Otsu threshold on colour distance; field background: GrabCut with a centre prior); morphology and "
        "dominant-component selection with hole filling. Abnormal regions: yellow / brown / dark / pale colour rules "
        "relative to each leaf's own green reference, hysteresis growth, opening/closing and connected components. "
        "Features (listed in `results/feature_list.txt`): colour statistics over leaf and lesions (RGB/HSV/Lab), abnormal-colour %, GLCM contrast / "
        "energy / homogeneity / correlation / dissimilarity, edge density, lesion count / area / perimeter / "
        "circularity / mean and max size, leaf solidity. Decisions: (a) a fixed-weight rule score over affected-%, "
        "lesion count, abnormal-colour % and GLCM contrast with a per-crop threshold chosen on VAL; (b) an RBF SVM "
        "tuned by grouped cross-validation on TRAIN.", "",
        f"**Stage 2 - CNN.** {R.js(CNM)['checkpoint']['arch']} pretrained on ImageNet, new 4-class head "
        "(turmeric/black gram x healthy/diseased), train-only augmentation, class-weighted cross-entropy, AdamW, "
        "head-only stage then fine-tuning of the last blocks, early stopping on VAL macro-F1. "
        f"Epochs run: {R.num(info, 'epochs_run', fmt='d')}; best epoch {R.num(info, 'best', 'epoch', fmt='d')} "
        f"(VAL macro-F1 {R.num(info, 'best', 'macro_f1_bin')}); training time {R.num(info, 'minutes', fmt='.1f')} min "
        f"on `{R.js(info)['device']}`. P(diseased) = sum of the two diseased-class probabilities.", "",
        "![CNN training](figures/cnn_training.png)", ""]


def stage1_checks() -> list[str]:
    sc, lc = "results/segmentation_check.json", "results/lesion_check.json"
    return [
        "## Stage 1 checks", "",
        f"- Segmentation: {pct(sc, 'single_dominant_rate')} of {R.num(sc, 'n_images', fmt='d')} random train/val images "
        f"give a single dominant leaf mask covering {R.cfg('segment', 'quality', 'min_coverage', fmt='.0%')} to "
        f"{R.cfg('segment', 'quality', 'max_coverage', fmt='.0%')} of the frame.",
        f"- Abnormal-region detection ({R.num(lc, 'n_images', fmt='d')} train/val images): mean affected area "
        f"{R.num(lc, 'mean_affected_pct', 'healthy', fmt='.2f')}% on healthy vs {R.num(lc, 'mean_affected_pct', 'diseased', fmt='.2f')}% "
        "on diseased leaves.", "",
        "![segmentation](figures/segmentation_samples.png)", "",
        "![lesions](figures/lesion_samples_diseased.png)", ""]


def metrics_block(title: str, file: str, model_key: str | None, split: str, binary_key: str | None = None) -> list[str]:
    base = [model_key, split] if model_key else [split]
    if binary_key:
        base = [split, binary_key]
    rows = ["| Scope | n | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC |", "|---|---|---|---|---|---|---|---|"]
    for scope in ("overall", "blackgram", "turmeric"):
        keys = [*base, "overall"] if scope == "overall" else [*base, "per_crop", scope]
        rows.append(f"| {scope} | {R.num(file, *keys, 'n', fmt='d')} | " + " | ".join(
            R.num(file, *keys, k) for k in ("accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc")) + " |")
    return [f"**{title}**", "", *rows, ""]


def results_section() -> list[str]:
    out = ["## Results", "", "### Stage 1 decisions", ""]
    out += metrics_block("Rule-based score - VAL", IPM, "rule_based", "val")
    out += metrics_block("Rule-based score - TEST", IPM, "rule_based", "test")
    out += metrics_block("SVM - VAL", IPM, "svm", "val")
    out += metrics_block("SVM - TEST", IPM, "svm", "test")
    out += [f"SVM hyperparameters (grouped CV on TRAIN): C = {R.num(IPM, 'svm', 'best_params', 'svc__C', fmt='g')}, "
            f"gamma = {R.num(IPM, 'svm', 'best_params', 'svc__gamma', fmt='g')}; CV macro-F1 {R.num(IPM, 'svm', 'cv_macro_f1_train')}. "
            f"Rule thresholds from VAL: black gram {R.num(IPM, 'rule_based', 'thresholds_from_val', 'blackgram')}, "
            f"turmeric {R.num(IPM, 'rule_based', 'thresholds_from_val', 'turmeric')}.", "",
            "### CNN", ""]
    out += metrics_block("CNN healthy/diseased - VAL", CNM, None, "val", "binary")
    out += metrics_block("CNN healthy/diseased - TEST", CNM, None, "test", "binary")
    out += [f"4-class (crop x state) TEST accuracy {R.num(CNM, 'test', 'four_class', 'accuracy')}, macro-F1 "
            f"{R.num(CNM, 'test', 'four_class', 'macro_f1')}; crop identification accuracy "
            f"{R.num(CNM, 'test', 'four_class', 'crop_accuracy')}. Mean confidence when correct "
            f"{R.num(CNM, 'test', 'confidence', 'mean_conf_correct')} vs when wrong "
            + (R.num(CNM, 'test', 'confidence', 'mean_conf_wrong') if R.js(CNM)["test"]["confidence"]["mean_conf_wrong"] is not None else "n/a (no errors)")
            + ".", "", "![Grad-CAM](figures/gradcam_samples.png)", ""]
    return out


def comparison_section() -> list[str]:
    md = (RES / "comparison.md").read_text()
    body = md.split("## Findings", 1)[1]
    findings, rest = body.split("## Leakage ablation", 1)
    abl, answer = rest.split("## What this answers", 1)
    # comparison.md numbers are themselves checked against comparison.csv; re-record them here
    t = R.csv("results/comparison.csv")
    for v, m, sec, meth, sc in zip(t.value, t.metric, t.section, t.method, t.scope, strict=True):
        from leafdx.compare import fmt

        src = {"file": "results/comparison.csv", "query": {"section": sec, "method": meth, "scope": sc, "metric": m}}
        s = fmt(v, m)
        if s in md:
            R.records.append({"text": s, "value": float(v), "fmt": "compare.fmt", "source": src})
        if float(v).is_integer() and str(int(v)) in md:
            R.records.append({"text": str(int(v)), "value": float(v), "fmt": "d", "source": src})
    table = (RES / "comparison_table.md").read_text()
    return ["## Comparison: image processing vs CNN (identical TEST split)", "", table, "### Findings", findings.strip(), "",
            "![bars](figures/comparison_bars.png)", "", "![confusion](figures/confusion_matrices.png)", "",
            "![roc](figures/roc_pr_curves.png)", "", "![time](figures/inference_time.png)", "",
            "## Leakage ablation", abl.strip(), "",
            "## Error analysis", "",
            "Per-original-class TEST accuracy (share of images classified correctly as healthy/diseased):", "",
            *per_class_table(), "",
            "![errors](figures/errors_gallery.png)", "",
            "SVM vs CNN disagreements with both kinds of evidence (original | Stage 1 abnormal regions | Grad-CAM):", "",
            "![disagreements](figures/disagreements_gallery.png)", "",
            "## Answer to the research question", "", answer.strip(), ""]


def per_class_table() -> list[str]:
    t = R.csv("results/comparison.csv")
    classes = sorted(t[(t.section == "per_class")].scope.unique())
    rows = ["| Original class | n | Rule-based | SVM | CNN |", "|---|---|---|---|---|"]
    for c in classes:
        rows.append(f"| {c} | {R.cmp('per_class', 'svm', c, 'n', fmt='.0f')} | "
                    + " | ".join(R.cmp("per_class", m, c, "accuracy") for m in ("rule_based", "svm", "cnn")) + " |")
    return rows


def weakest_classes() -> str:
    t = R.csv("results/comparison.csv")
    pc = t[(t.section == "per_class") & (t.metric == "accuracy")]
    parts = []
    for m, name in (("svm", "SVM"), ("cnn", "CNN")):
        g = pc[pc.method == m].sort_values("value").head(2)
        parts.append(f"{name}: " + ", ".join(f"{c} ({R.cmp('per_class', m, c, 'accuracy')})" for c in g.scope))
    return "Weakest TEST classes by accuracy - " + "; ".join(parts) + "."


def limitations_section() -> list[str]:
    return [
        "## Limitations", "",
        "- Both datasets come from single regions (Andhra Pradesh, India for black gram; Jamalpur, Bangladesh for "
        "turmeric) and single capture campaigns; no cross-region test was possible, so the measured accuracies are "
        "within-dataset estimates.",
        "- All turmeric images are on white paper, while black gram mixes field and card photos; crop identity is "
        "therefore trivially predictable from the background, and the turmeric results say little about field photos.",
        "- Turmeric 'Dry Leaf' is counted as diseased (abnormal condition class in the source dataset).",
        "- Stage 1 colour rules were calibrated on TRAIN images of these two datasets; other cultivars, cameras or "
        "lighting may need recalibration. " + weakest_classes(),
        "- Abnormal-region detection under-measures faint powdery mildew on dark leaves and very small dark spots "
        "(see the lesion figure and the disagreement gallery); the SVM partly compensates through colour and texture "
        "statistics of the whole leaf.",
        "- Segmentation of field photos can include neighbouring foliage or miss part of a two-tone leaf.",
        "- The CNN was trained once (one seed) with a time budget; no architecture search.",
        "", "## Future work", "",
        "- Specific-disease classification (e.g. turmeric leaf spot / blotch, black gram yellow mosaic) instead of "
        "healthy vs diseased.",
        "- Cross-region testing with leaves photographed in other districts / seasons.",
        "- On-device deployment (the Stage 1 pipeline and MobileNetV2 both run on CPU).", ""]


def main() -> int:
    lines = ["# Image Processing + CNN for Disease Detection in Turmeric and Black Gram", "",
             "> Generated by `scripts/make_report.py` from files in `results/` and `data/processed/`. "
             "No number in this report is typed by hand; `results/report_numbers.json` lists the source of each.", "",
             "**Research question.** Can conventional digital image-processing techniques identify disease-related "
             "characteristics in turmeric and black gram leaves, and how does their performance compare with a "
             "CNN-based approach?", ""]
    lines += dataset_section() + methods_section() + stage1_checks() + results_section()
    lines += comparison_section() + limitations_section()
    report = "\n".join(lines)
    (RES / "REPORT.md").write_text(report)
    (RES / "report_numbers.json").write_text(json.dumps(R.records, indent=1))
    print(f"wrote results/REPORT.md ({len(report.splitlines())} lines, {len(R.records)} recorded numbers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

