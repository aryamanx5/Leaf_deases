"""P3-M00: baseline audit.

1. CLAHE: confirm it is in the Stage 1 code path (source grep + a functional check).
2. data/wild/labels.csv (path, crop, healthy_or_diseased, source, labelled_by) for every
   wild image plus the user-reported image; per-crop counts with a small-sample flag.
3. Leakage report: 64/256-bit pHash groups over clean TRAIN/VAL/TEST + wild + user images
   (same union-find rules as the split, cross-crop joins allowed); count groups whose
   members fall in more than one role. Must be 0.

Writes results/p3/audit.json and results/p3/leakage_report.md.
"""

from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from leafdx.config import ROOT, load_config, path
from leafdx.ip.features import stage1
from leafdx.ip.preprocess import load_bgr, preprocess
from leafdx.p3.data import image_table
from leafdx.split import cluster_images, phash_pair

MIN_WILD_PER_CROP = 40


def clahe_audit(cfg: dict) -> dict:
    pre_src = (ROOT / "src/leafdx/ip/preprocess.py").read_text()
    feat_src = (ROOT / "src/leafdx/ip/features.py").read_text()
    img = load_bgr(ROOT / pd.read_csv(path("splits", cfg)).path.iloc[0])
    on = preprocess(img, cfg)
    off = preprocess(img, {**cfg, "preprocess": {**cfg["preprocess"], "clahe_enabled": False}})
    import cv2

    lab_f, lab_e = (cv2.cvtColor(on[k], cv2.COLOR_BGR2LAB).astype(int) for k in ("filtered", "enhanced"))
    return {
        "clahe_function_present": "createCLAHE" in pre_src and "def clahe_l" in pre_src,
        "applied_on_lab_L": "lab[..., 0] = clahe.apply" in pre_src,
        "enhanced_used_by_features": bool(re.search(r'pre\["enhanced"\]', feat_src)),
        "stage1_calls_preprocess": "preprocess(img, cfg)" in feat_src,
        "functional_L_changed_mean_abs": float(np.abs(lab_e[..., 0] - lab_f[..., 0]).mean()),
        "functional_ab_changed_mean_abs": float(np.abs(lab_e[..., 1:] - lab_f[..., 1:]).mean()),
        "flag_off_enhanced_equals_filtered": bool((off["enhanced"] == off["filtered"]).all()),
        "stage1_runs": "features" in stage1(img, cfg, "turmeric"),
        "note": "CLAHE runs before feature extraction (texture/edge features). Segmentation and colour "
                "features deliberately use the denoised, non-CLAHE image because CLAHE shifts colour "
                "statistics; this is the v1 design (PROGRESS.md T03).",
    }


def labels_csv(cfg: dict) -> pd.DataFrame:
    pd.read_csv(ROOT / cfg["wild"]["root"] / "SOURCES.csv")
    t = image_table(cfg)
    w = t[t.split.isin(["wild", "user"])]

    def who(r):
        if r.source.startswith("mendeley"):
            return f"dataset authors ({r.source}, class '{r.orig_class}')"
        if r.source == "user_reported":
            return "user (crop stated by the user; disease status not stated)"
        return f"crop from the {r.source} record (taxon/title); disease status not verified"

    out = pd.DataFrame({"path": w.path, "crop": w.crop, "healthy_or_diseased": w.label, "source": w.source,
                        "labelled_by": w.apply(who, axis=1)})
    out.to_csv(ROOT / cfg["wild"]["root"] / "labels.csv", index=False)
    return out


def leakage(cfg: dict) -> tuple[dict, pd.DataFrame]:
    t = image_table(cfg)
    hashes = Parallel(n_jobs=-1)(delayed(phash_pair)(ROOT / p) for p in t.path)
    hs, hl = (np.stack(h) for h in zip(*hashes, strict=True))
    # cross-crop hash joins allowed; the burst rule (consecutive file numbers) only within one
    # source class of one dataset, as in the split
    t["phash_group"] = cluster_images(t.assign(crop="all", orig_class=t.source + ":" + t.crop + ":" + t.orig_class),
                                      hs, hl, cfg["split"]["dedup"], use_stems=False)
    roles = t.groupby("phash_group").split.agg(lambda s: tuple(sorted(set(s))))
    span = roles[roles.map(len) > 1]
    multi = t.groupby("phash_group").size()
    return {
        "n_images": len(t),
        "by_role": t.split.value_counts().to_dict(),
        "n_phash_groups": int(t.phash_group.nunique()),
        "n_multi_image_groups": int((multi > 1).sum()),
        "groups_spanning_roles": int(len(span)),
        "spanning_detail": [{"group": int(g), "roles": list(r),
                             "paths": t.loc[t.phash_group == g, "path"].tolist()} for g, r in span.items()],
        "clean_split_groups_spanning_splits_(split_meta)": json.loads(
            (path("processed", cfg) / "split_meta.json").read_text())["clusters_spanning_splits"],
    }, t


def main() -> int:
    cfg = load_config()
    out = path("results", cfg) / "p3"
    out.mkdir(exist_ok=True)
    cl = clahe_audit(cfg)
    lab = labels_csv(cfg)
    counts = lab.groupby(["crop", "healthy_or_diseased"]).size().unstack(fill_value=0)
    per_crop = lab[lab.source != "user_reported"].crop.value_counts().to_dict()
    lk, _ = leakage(cfg)
    small = {c: n < MIN_WILD_PER_CROP for c, n in per_crop.items()}
    audit = {"clahe": cl, "wild_counts": {c: counts.loc[c].to_dict() for c in counts.index},
             "wild_per_crop": per_crop, "wild_small_sample": small, "leakage": lk}
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    lines = ["# Leakage report (P3-M00)", "",
             f"Images hashed: {lk['n_images']} ({', '.join(f'{k} {v}' for k, v in lk['by_role'].items())}).",
             f"Near-duplicate groups (64/256-bit pHash over 8 flip/rotation variants, split rules, cross-crop): "
             f"{lk['n_phash_groups']}, of which {lk['n_multi_image_groups']} have more than one image.", "",
             f"**Groups spanning more than one role (train / val / test / wild / user): {lk['groups_spanning_roles']}**", ""]
    for d in lk["spanning_detail"]:
        lines.append(f"- group {d['group']}: {d['roles']}: {', '.join(d['paths'])}")
    lines += ["", f"Clean split clusters spanning splits (data/processed/split_meta.json): "
              f"{lk['clean_split_groups_spanning_splits_(split_meta)']}.", "",
              "Augmentation is applied at load time to TRAIN rows only (LeakageError guards in "
              "leafdx.cnn.augment); no augmented copy is written to disk.", ""]
    (out / "leakage_report.md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in audit.items() if k != "leakage"}, indent=1))
    print("groups spanning roles:", lk["groups_spanning_roles"])
    ok = lk["groups_spanning_roles"] == 0 and cl["clahe_function_present"] and cl["enhanced_used_by_features"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
