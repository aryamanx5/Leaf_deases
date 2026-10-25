"""P2-B08: re-evaluate the Phase 2 system and write the robustness report (before vs after).

* Hierarchical model on clean VAL/TEST, wild_val/wild_test and the user-reported image
  (crop + status + OOD, disease under the predicted crop).
* v1 4-class CNN on wild_test (labelled) for the "before" disease column.
* Stage 1 rule/SVM on wild images.
* v2 CNN files in the v1 formats (results/cnn_predictions.csv, cnn_metrics.json,
  cnn_train_info.json) so the comparison and report machinery keep working; v1 originals
  are in results/v1_snapshot/.
* results/robustness.json + robustness.md (generated), bug_repro_after.json/png.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from leafdx.cnn.dataset import CLASSES4, DISEASED_IDX, eval_transform  # noqa: E402
from leafdx.cnn.model import load_checkpoint  # noqa: E402
from leafdx.config import ROOT, load_config, path  # noqa: E402
from leafdx.ip.decide import load_ip_models, predict_ip  # noqa: E402
from leafdx.ip.features import stage1  # noqa: E402
from leafdx.metrics import with_per_crop  # noqa: E402
from leafdx.robust.cnn_heads import predict_disease_logit  # noqa: E402
from leafdx.robust.infer import RobustModel, sigmoid  # noqa: E402
from leafdx.service import analyze_image, model_version, result_card  # noqa: E402
from leafdx.viz import contact_sheet  # noqa: E402

FIXTURE = ROOT / "tests/fixtures/regressions/turmeric_google_01.jpg"
TARGETS = {"wild_accepted_precision_min": 0.90, "clean_disease_f1_max_drop": 0.02}


def ece(p_pos, y, bins=10) -> float:
    conf = np.maximum(p_pos, 1 - p_pos)
    correct = (p_pos >= 0.5).astype(int) == y
    edges = np.linspace(0.5, 1.0, bins + 1)
    out = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        m = (conf >= lo) & ((conf <= hi) if i == bins - 1 else (conf < hi))
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def rows(cfg) -> pd.DataFrame:
    s = pd.read_csv(path("splits", cfg))
    w = pd.read_csv(ROOT / "data/wild/splits.csv")
    parts = [s[s.split.isin(["val", "test"])].assign(set="clean_" + s.split, source="clean"),
             w.assign(set=w.split)]
    df = pd.concat(parts, ignore_index=True)[["path", "crop", "label", "orig_class", "stratum", "set", "source", "split"]]
    if FIXTURE.exists():
        df = pd.concat([df, pd.DataFrame([{"path": str(FIXTURE.relative_to(ROOT)), "crop": "turmeric", "label": "unknown",
                                           "orig_class": "user_reported", "stratum": "turmeric_unknown",
                                           "set": "regression", "source": "user_reported", "split": "regression"}])],
                       ignore_index=True)
    return df


def predict_all(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cache = path("results", cfg) / "robust_eval.csv"
    if cache.exists():
        old = pd.read_csv(cache)
        if old.get("model_version", pd.Series([""])).iloc[0] == model_version() and set(old.path) == set(df.path):
            return old
    rm = RobustModel.load()
    out = []
    for r in df.itertuples():
        im = Image.open(ROOT / r.path).convert("RGB")
        p = rm.predict(im, None)
        z_true = predict_disease_logit(rm.disease_model, [im], [r.crop], cfg, dev=torch.device("cpu"))[0]
        out.append({"path": r.path, "crop_pred": p["crop"]["label"], "p_turmeric": p["crop"]["probs"]["turmeric"],
                    "crop_status": p["crop"]["status"], "ood_score": p["ood"]["ood_score"], "ood_flagged": p["ood"]["flagged"],
                    "p_dis_pred_crop": p["disease"]["prob"],
                    "p_dis_true_crop": float(sigmoid(z_true / rm.cal["T_disease"]))})
    res = df.merge(pd.DataFrame(out), on="path")
    res["model_version"] = model_version()
    res.to_csv(cache, index=False)
    return res


@torch.no_grad()
def v1_probs(paths: list[str], cfg: dict) -> np.ndarray:
    model, _ = load_checkpoint(path("models", cfg) / "cnn_best.pt")
    tf = eval_transform(cfg["cnn"])
    x = torch.stack([tf(Image.open(ROOT / p).convert("RGB")) for p in paths])
    return torch.softmax(model(x), 1).numpy()


def ip_on(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rule, svm = load_ip_models(path("models", cfg))
    out = []
    for r in df.itertuples():
        f = stage1(cv2.imread(str(ROOT / r.path)), cfg, r.crop)["features"]
        ip = predict_ip(f, r.crop, rule, svm)
        out.append({"path": r.path, "rule_pred": ip["rule_pred"], "rule_score": ip["rule_score"],
                    "svm_pred": ip["svm_pred"], "svm_score": ip["svm_score"]})
    return df.merge(pd.DataFrame(out), on="path")


def cnn_v2_files(ev: pd.DataFrame, cfg: dict, rm_timing: dict) -> dict:
    """v1-format prediction/metric files for the hierarchical CNN on clean VAL/TEST."""
    c = ev[ev.set.isin(["clean_val", "clean_test"])].copy()
    c["split"] = c.set.str.replace("clean_", "")
    c["y"] = (c.label == "diseased").astype(int)
    c["cnn_p_diseased"] = c.p_dis_pred_crop
    c["cnn_pred"] = (c.cnn_p_diseased >= 0.5).astype(int)
    c["cnn_conf"] = np.maximum(c.cnn_p_diseased, 1 - c.cnn_p_diseased)
    c["cnn_crop_pred"] = c.crop_pred
    c["cnn_pred4"] = c.crop_pred + "_" + np.where(c.cnn_pred == 1, "diseased", "healthy")
    c["cnn_conf4"] = c.cnn_conf
    keep = ["path", "split", "crop", "label", "orig_class", "stratum", "y", "cnn_p_diseased", "cnn_pred", "cnn_conf",
            "cnn_pred4", "cnn_conf4", "cnn_crop_pred", "crop_status", "p_turmeric"]
    c[keep].to_csv(path("results", cfg) / "cnn_predictions.csv", index=False)
    rt = json.loads((path("results", cfg) / "robust_train.json").read_text())
    metrics = {"checkpoint": {"arch": f"crop: {rt['crop_selected']}; disease: {cfg['robust']['disease_cnn']['arch']} "
                              "(one diseased logit per crop)", "epoch": rt["disease_cnn"]["train"]["epoch"],
                              "image_size": cfg["cnn"]["image_size"], "pretrained": True}}
    for s in ("val", "test"):
        part = c[c.split == s]
        y4 = part.stratum.map(CLASSES4.index).to_numpy()
        p4 = part.cnn_pred4.map(CLASSES4.index).to_numpy()
        ok = part.cnn_pred == part.y
        metrics[s] = {
            "binary": with_per_crop(part, "cnn_pred", "cnn_p_diseased"),
            "four_class": {"accuracy": float((y4 == p4).mean()),
                           "macro_f1": float(f1_score(y4, p4, average="macro", labels=range(4), zero_division=0)),
                           "confusion_matrix": confusion_matrix(y4, p4, labels=range(4)).tolist(), "classes": list(CLASSES4),
                           "crop_accuracy": float((part.cnn_crop_pred == part.crop).mean())},
            "confidence": {"mean_conf_correct": float(part.cnn_conf[ok].mean()) if ok.any() else None,
                           "mean_conf_wrong": float(part.cnn_conf[~ok].mean()) if (~ok).any() else None,
                           "median_conf": float(part.cnn_conf.median()),
                           "share_conf_ge_0.9": float((part.cnn_conf >= 0.9).mean())}}
    metrics["inference_time"] = rm_timing
    (path("results", cfg) / "cnn_metrics.json").write_text(json.dumps(metrics, indent=2))
    tr = rt["disease_cnn"]["train"]
    (path("results", cfg) / "cnn_train_info.json").write_text(json.dumps(
        {"best": {"epoch": tr["epoch"], "macro_f1_bin": tr["score"]}, "epochs_run": tr["epochs_run"],
         "minutes": tr["minutes"], "device": tr["device"], "pretrained_weights": True}, indent=2))
    return metrics


def time_model(cfg: dict, paths: list[str]) -> dict:
    rm = RobustModel.load()
    out = {}
    for key, dev in (("accelerator", None), ("cpu", "cpu")):
        rm.predict(Image.open(ROOT / paths[0]).convert("RGB"), None, dev)  # warm-up
        ms = []
        for p in paths:
            im = Image.open(ROOT / p).convert("RGB")
            t0 = time.perf_counter()
            rm.predict(im, None, dev)
            ms.append(1000 * (time.perf_counter() - t0))
        out[key] = {"device": dev or "mps (embeddings) + cpu (disease CNN)", "n_images": len(ms),
                    "ms_median": float(np.median(ms)), "ms_mean": float(np.mean(ms))}
    return out


def main() -> int:
    cfg = load_config()
    res = path("results", cfg)
    df = rows(cfg)
    ev = predict_all(df, cfg)
    s = pd.read_csv(path("splits", cfg))
    timing = time_model(cfg, list(s[s.split == "val"].sample(10, random_state=cfg["seed"]).path))
    m2 = cnn_v2_files(ev, cfg, timing)

    v1m = json.loads((res / "v1_snapshot/cnn_metrics.json").read_text())
    v1ip = json.loads((res / "v1_snapshot/ip_metrics.json").read_text())
    ipm = json.loads((res / "ip_metrics.json").read_text())
    cal = json.loads((res / "calibration.json").read_text())
    diag = json.loads((res / "diagnosis.json").read_text())
    lat = json.loads((res / "api_latency.json").read_text()) if (res / "api_latency.json").exists() else {}

    wt = ev[ev.set == "wild_test"].copy()
    wtl = wt[wt.label != "unknown"].copy()
    wtl["y"] = (wtl.label == "diseased").astype(int)
    p1 = v1_probs(list(wtl.path), cfg)
    wtl["v1_p_dis"] = p1[:, list(DISEASED_IDX)].sum(1)
    p1_all = v1_probs(list(wt.path), cfg)
    v1_crop_tu = p1_all[:, [i for i, k in enumerate(CLASSES4) if k.startswith("turmeric")]].sum(1)
    y_crop_wt = (wt.crop == "turmeric").astype(int).to_numpy()
    wtl = ip_on(wtl, cfg)

    acc_wt = wt.crop_pred == wt.crop
    accepted = wt.crop_status == "confident"
    after = {
        "crop_acc_clean_test": float((ev[ev.set == "clean_test"].crop_pred == ev[ev.set == "clean_test"].crop).mean()),
        "crop_acc_wild_test": float(acc_wt.mean()),
        "accepted_precision_wild_test": float(acc_wt[accepted].mean()) if accepted.any() else float("nan"),
        "coverage_wild_test": float(accepted.mean()),
        "abstention_wild_test": float(1 - accepted.mean()),
        "crop_ece_wild_test": cal["sets"]["wild_test"]["crop_ece_after"],
        "disease_f1_clean_test": m2["test"]["binary"]["overall"]["f1"],
        "disease_macro_f1_clean_test": m2["test"]["binary"]["overall"]["macro_f1"],
        "disease_macro_f1_wild_test": float(f1_score(wtl.y, (wtl.p_dis_pred_crop >= 0.5).astype(int), average="macro")),
        "svm_macro_f1_clean_test": ipm["svm"]["test"]["overall"]["macro_f1"],
        "rule_macro_f1_clean_test": ipm["rule_based"]["test"]["overall"]["macro_f1"],
        "svm_macro_f1_wild_test": float(f1_score(wtl.y, wtl.svm_pred, average="macro")),
        "rule_macro_f1_wild_test": float(f1_score(wtl.y, wtl.rule_pred, average="macro")),
        "latency_ms_cnn_cpu": timing["cpu"]["ms_median"],
        "api_p50_ms": lat.get("default", {}).get("p50_ms"),
    }
    before = {
        "crop_acc_clean_test": diag["crop"]["clean_test"]["accuracy"],
        "crop_acc_wild_test": diag["crop"]["wild_test"]["accuracy"],
        "accepted_precision_wild_test": diag["crop"]["wild_test"]["accuracy"],  # no abstention in v1
        "coverage_wild_test": 1.0,
        "abstention_wild_test": 0.0,
        "crop_ece_wild_test": ece(v1_crop_tu, y_crop_wt),
        "disease_f1_clean_test": v1m["test"]["binary"]["overall"]["f1"],
        "disease_macro_f1_clean_test": v1m["test"]["binary"]["overall"]["macro_f1"],
        "disease_macro_f1_wild_test": float(f1_score(wtl.y, (wtl.v1_p_dis >= 0.5).astype(int), average="macro")),
        "svm_macro_f1_clean_test": v1ip["svm"]["test"]["overall"]["macro_f1"],
        "rule_macro_f1_clean_test": v1ip["rule_based"]["test"]["overall"]["macro_f1"],
        "svm_macro_f1_wild_test": None, "rule_macro_f1_wild_test": None,
        "latency_ms_cnn_cpu": v1m["inference_time"]["cpu"]["ms_median"], "api_p50_ms": None,
    }
    # user-reported image, before and after
    bug_b = json.loads((res / "bug_repro_before.json").read_text())
    reg = analyze_image(FIXTURE.read_bytes(), None) if FIXTURE.exists() else None
    if reg:
        imgs = reg.pop("images")
        reg.pop("features", None)
        (res / "bug_repro_after.json").write_text(json.dumps(reg, indent=2, default=float))
        panels = [cv2.cvtColor(imgs[k], cv2.COLOR_RGB2BGR) for k in ("original", "segmented_leaf", "abnormal_regions", "gradcam")]
        cv2.imwrite(str(path("figures", cfg) / "bug_repro_after.png"), contact_sheet(
            [panels], [["original", "segmented leaf", f"abnormal {reg['measurements']['affected_region_pct']:.1f}% "
                        f"n={reg['measurements']['lesion_count']}", "Grad-CAM (disease)"]], size=300))
    regression = {
        "before": {"crop": bug_b["crop"], "p_blackgram": bug_b["crop_probabilities"]["blackgram"], "status": "no abstention",
                   "cnn_disease": bug_b["cnn_prediction"], "affected_pct": bug_b["affected_region_pct"],
                   "lesion_count": bug_b["lesion_count"], "svm": bug_b["ip_prediction"]["svm"]},
        "after": None if reg is None else {
            "crop": reg["crop"]["label"], "p_turmeric": reg["crop"]["probs"]["turmeric"], "status": reg["crop"]["status"],
            "cnn_disease": reg["disease"]["label"], "p_diseased": reg["disease"]["by"]["cnn"]["prob"],
            "svm": reg["disease"]["by"]["svm"]["label"], "rule": reg["disease"]["by"]["rule"]["label"],
            "affected_pct": reg["measurements"]["affected_region_pct"], "lesion_count": reg["measurements"]["lesion_count"],
            "card": result_card(reg)},
    }
    targets = {
        "wild_test_accepted_precision": {"value": after["accepted_precision_wild_test"],
                                         "target": TARGETS["wild_accepted_precision_min"],
                                         "met": bool(after["accepted_precision_wild_test"] >= TARGETS["wild_accepted_precision_min"])},
        "regression_not_confident_blackgram": {"value": None if reg is None else f"{reg['crop']['label']} ({reg['crop']['status']})",
                                               "met": reg is not None and not (reg["crop"]["label"] == "blackgram"
                                                                               and reg["crop"]["status"] == "confident")},
        "clean_disease_f1_drop": {"value": before["disease_f1_clean_test"] - after["disease_f1_clean_test"],
                                  "target": TARGETS["clean_disease_f1_max_drop"],
                                  "met": bool(before["disease_f1_clean_test"] - after["disease_f1_clean_test"]
                                              <= TARGETS["clean_disease_f1_max_drop"])},
    }
    wild_n = {"wild_test": int(len(wt)), "wild_test_labelled": int(len(wtl))}
    # full metric set for rule / SVM / CNN on labelled wild_test (Results section toggle)
    from leafdx.metrics import binary_metrics  # noqa: PLC0415

    wtl["cnn_pred"] = (wtl.p_dis_pred_crop >= 0.5).astype(int)
    comp_wild = {m: binary_metrics(wtl.y, wtl[pc], wtl[sc]) for m, (pc, sc) in
                 {"rule_based": ("rule_pred", "rule_score"), "svm": ("svm_pred", "svm_score"),
                  "cnn": ("cnn_pred", "p_dis_pred_crop")}.items()}
    wtl.to_csv(res / "wild_test_predictions.csv", index=False)
    out = {"before": before, "after": after, "targets": targets, "regression": regression, "n": wild_n,
           "comparison_wild_test": comp_wild,
           "wild_test_by_source": {src: {"n": int(len(g)), "crop_acc": float((g.crop_pred == g.crop).mean()),
                                         "coverage": float((g.crop_status == "confident").mean())}
                                   for src, g in wt.groupby("source")}}
    (res / "robustness.json").write_text(json.dumps(out, indent=2, default=float))
    (res / "robustness.md").write_text(render(out))
    print((res / "robustness.md").read_text())
    return 0


def render(o: dict) -> str:
    f = lambda x: "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))  # noqa: E731
    b, a = o["before"], o["after"]
    rows_ = [("Crop accuracy, clean TEST", "crop_acc_clean_test"), ("Crop accuracy, wild_test (all)", "crop_acc_wild_test"),
             ("Accepted-crop precision, wild_test", "accepted_precision_wild_test"),
             ("Coverage (accepted share), wild_test", "coverage_wild_test"), ("Abstention rate, wild_test", "abstention_wild_test"),
             ("Crop ECE, wild_test", "crop_ece_wild_test"), ("Disease F1 (diseased class), clean TEST", "disease_f1_clean_test"),
             ("Disease macro-F1, clean TEST", "disease_macro_f1_clean_test"),
             ("Disease macro-F1, wild_test (labelled)", "disease_macro_f1_wild_test"),
             ("SVM macro-F1, clean TEST", "svm_macro_f1_clean_test"), ("Rule macro-F1, clean TEST", "rule_macro_f1_clean_test"),
             ("SVM macro-F1, wild_test (labelled)", "svm_macro_f1_wild_test"),
             ("Rule macro-F1, wild_test (labelled)", "rule_macro_f1_wild_test"),
             ("CNN latency, CPU ms / image", "latency_ms_cnn_cpu"), ("API p50 latency, ms (full analysis)", "api_p50_ms")]
    L = ["# Robustness: before (v1) vs after (Phase 2)", "",
         "> Generated by `scripts/evaluate_robust.py` from `results/robustness.json`. 'Before' = the v1 4-class CNN and "
         "v1 Stage 1 (results/v1_snapshot); 'after' = calibrated crop model + crop-conditioned disease CNN + Phase 2 "
         "Stage 1. wild_test was used only for this report.", "",
         f"wild_test: {o['n']['wild_test']} images ({o['n']['wild_test_labelled']} with a disease label).", "",
         "| Metric | Before (v1) | After (Phase 2) |", "|---|---|---|"]
    L += [f"| {name} | {f(b[k])} | {f(a[k])} |" for name, k in rows_]
    L += ["", "## Targets", "", "| Target | Value | Required | Met |", "|---|---|---|---|"]
    t = o["targets"]
    L.append(f"| wild_test crop precision on accepted predictions | {f(t['wild_test_accepted_precision']['value'])} | >= "
             f"{f(t['wild_test_accepted_precision']['target'])} | {'yes' if t['wild_test_accepted_precision']['met'] else 'NO'} |")
    L.append(f"| user-reported turmeric image is not a confident black gram | {t['regression_not_confident_blackgram']['value']} | "
             f"turmeric, or uncertain | {'yes' if t['regression_not_confident_blackgram']['met'] else 'NO'} |")
    L.append(f"| clean-TEST disease F1 drop vs v1 | {f(t['clean_disease_f1_drop']['value'])} | <= "
             f"{f(t['clean_disease_f1_drop']['target'])} | {'yes' if t['clean_disease_f1_drop']['met'] else 'NO'} |")
    r = o["regression"]
    L += ["", "## The user-reported image (turmeric field photo from the bug report)", "",
          f"- Before: crop **{r['before']['crop']}** (P(black gram) {f(r['before']['p_blackgram'])}, shown as certain); "
          f"CNN {r['before']['cnn_disease']}; Stage 1 {f(r['before']['affected_pct'])}% affected in "
          f"{r['before']['lesion_count']} region (central blob); SVM {r['before']['svm']}."]
    if r["after"]:
        ra = r["after"]
        L.append(f"- After: crop **{ra['crop']}** (P(turmeric) {f(ra['p_turmeric'])}, status {ra['status']}); disease CNN "
                 f"**{ra['cnn_disease']}** (P(diseased) {f(ra['p_diseased'])}); Stage 1 {f(ra['affected_pct'])}% affected in "
                 f"{ra['lesion_count']} regions; SVM **{ra['svm']}**, rule **{ra['rule']}**.")
        if ra["cnn_disease"] != ra["svm"]:
            L.append("- The two routes disagree on disease for this photo; the app shows both with a DISAGREE badge. "
                     "Nothing was tuned on this image (it is test-only).")
        L += ["", "Result card:", "", "```", *ra["card"], "```"]
    L += ["", "## wild_test by source", "", "| Source | n | Crop accuracy | Coverage |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {f(v['crop_acc'])} | {f(v['coverage'])} |" for k, v in o["wild_test_by_source"].items()]
    L += ["", "Figures: `figures/bug_repro_before.png`, `figures/bug_repro_after.png`.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
