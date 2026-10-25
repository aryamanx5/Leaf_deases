"""P2-B00: run the CURRENT pipeline on the user-reported image; save JSON + 4-panel figure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from leafdx.config import ROOT, load_config, path
from leafdx.pipeline import analyze
from leafdx.viz import contact_sheet

FIXTURE = ROOT / "tests/fixtures/regressions/turmeric_google_01.jpg"


def main(tag: str = "before") -> int:
    cfg = load_config()
    if not FIXTURE.exists():
        print(f"missing {FIXTURE}; see BLOCKERS.md")
        return 1
    res = analyze(FIXTURE.read_bytes(), crop=None)
    imgs = res.pop("images")
    out = {k: v for k, v in res.items() if k != "features"}
    out["features"] = {k: round(v, 4) for k, v in res["features"].items()}
    (path("results", cfg) / f"bug_repro_{tag}.json").write_text(json.dumps(out, indent=2, default=float))
    panels = [cv2.cvtColor(imgs[k], cv2.COLOR_RGB2BGR) for k in
              ("original", "segmented_leaf", "abnormal_regions_overlay", "gradcam") if imgs.get(k) is not None]
    titles = ["original", "segmented leaf", f"abnormal {res['affected_region_pct']:.1f}% n={res['lesion_count']}",
              "Grad-CAM"][: len(panels)]
    cv2.imwrite(str(path("figures", cfg) / f"bug_repro_{tag}.png"), contact_sheet([panels], [titles], size=300))
    print(json.dumps({k: out[k] for k in ("crop", "crop_probabilities", "cnn_class_probabilities", "cnn_prediction",
                                         "cnn_confidence", "affected_region_pct", "lesion_count", "ip_prediction")},
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
