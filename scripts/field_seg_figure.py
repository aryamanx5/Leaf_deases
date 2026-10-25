"""P2-B06: contact sheet of 20 wild field scenes: original | foliage | chosen leaf | lesions."""

from __future__ import annotations

import json
import sys

import cv2
import pandas as pd

from leafdx.config import ROOT, load_config, path
from leafdx.ip.lesions import detect_lesions, overlay_lesions
from leafdx.ip.preprocess import load_bgr, preprocess
from leafdx.ip.segment import segment_leaf
from leafdx.viz import contact_sheet, overlay_mask


def main() -> int:
    cfg = load_config()
    w = pd.read_csv(ROOT / "data/wild/splits.csv")
    field = w[w.source.isin(["inaturalist", "wikimedia_commons"])].sample(20, random_state=cfg["seed"])
    rows, titles, meta = [], [], []
    for i, r in enumerate(field.itertuples()):
        pre = preprocess(load_bgr(ROOT / r.path), cfg)
        seg = segment_leaf(pre, cfg)
        les = detect_lesions(pre, seg["leaf_mask"], cfg, r.crop)
        rows.append([pre["resized"], seg["foliage_mask"],
                     overlay_mask(pre["resized"], seg["leaf_mask"] == 0, (0, 0, 0), 0.6, seg["leaf_mask"]),
                     overlay_lesions(pre["resized"], seg["leaf_mask"], les)])
        titles.append([f"#{i} {r.crop[:4]} {r.path.split('/')[-1][:14]}", "foliage", f"{seg['method'][-18:]} {seg['coverage']:.2f}",
                       f"{les['affected_pct']:.1f}% n={les['lesion_count']}"])
        meta.append({"i": i, "path": r.path, "method": seg["method"], "coverage": seg["coverage"]})
    out = path("figures", cfg) / "field_segmentation.png"
    cv2.imwrite(str(out), contact_sheet(rows, titles, size=180))
    (path("results", cfg) / "field_segmentation.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
