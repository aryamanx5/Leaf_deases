"""Per-arm CPU inference time (ms/image), measured sequentially on 10 images (5 clean TEST,
5 wild), one image at a time, after a warm-up call. Components: Stage 1 (segmentation +
57 features), one embedding per backbone, CNN with its 6 TTA views, head prediction.
Writes results/p3/timing.json.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ["LEAFDX_DEVICE"] = "cpu"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from leafdx.config import ROOT, load_config, path  # noqa: E402
from leafdx.ip.features import stage1  # noqa: E402
from leafdx.ip.preprocess import load_bgr  # noqa: E402
from leafdx.p3.data import image_table, load_rgb  # noqa: E402
from leafdx.robust.backbones import embed  # noqa: E402
from leafdx.robust.cnn_heads import predict_disease_logit  # noqa: E402


def ms(fn, items) -> float:
    fn(items[0])
    t0 = time.perf_counter()
    for it in items:
        fn(it)
    return 1000 * (time.perf_counter() - t0) / len(items)


def main() -> int:
    cfg = load_config()
    t = image_table(cfg)
    rows = list(t[t.split == "test"].sample(5, random_state=0).itertuples()) + \
        list(t[(t.split == "wild") & (t.label != "unknown")].sample(5, random_state=0).itertuples())
    imgs = [(load_rgb(r.path), r.crop, r.path) for r in rows]
    comp = {"stage1": ms(lambda x: stage1(load_bgr(ROOT / x[2]), cfg, x[1]), imgs)}
    man = json.loads((path("results", cfg) / "p3" / "embed_manifest.json").read_text())
    for bb, st in man["backbones"].items():
        if st["status"] == "ok":
            comp[f"embed_{bb}"] = ms(lambda x, bb=bb: embed(bb, [x[0]], dev_str="cpu"), imgs)
    from leafdx.p3.cnnpred import load_arm

    tr = json.loads((path("results", cfg) / "p3_cnn_train.json").read_text())
    for arm in cfg["p3"]["cnn_arms"]["arms"]:
        rec = tr.get(f"{arm['name']}_swap", {})
        if rec.get("checkpoint") and (ROOT / rec["checkpoint"]).exists():
            m = load_arm(rec["checkpoint"], arm["arch"], "cpu")
            comp[f"cnn_{arm['name']}"] = ms(lambda x, m=m: predict_disease_logit(m, [x[0]], [x[1]], cfg,
                                                                                 dev=torch.device("cpu")), imgs)
    arms = {"A_rule": ["stage1"], "B_svm": ["stage1"], "B2_lgbm": ["stage1"],
            "D_dinov2": ["embed_dinov2_vits14"], "E_openclip": ["embed_openclip_vitb32"], "E_bioclip2": ["embed_bioclip2"],
            **{a["name"]: [f"cnn_{a['name']}"] for a in cfg["p3"]["cnn_arms"]["arms"]}}
    out = {"device": "cpu", "torch_threads": torch.get_num_threads(), "n_images": len(imgs), "components_ms": comp,
           "arms": {}}
    for a, parts in arms.items():
        if all(p in comp for p in parts):
            out["arms"][a] = {"ms_per_image": float(np.sum([comp[p] for p in parts])), "components": parts}
    sel = json.loads((path("results", cfg) / "disease_models.json").read_text())["arms"] \
        if (path("results", cfg) / "disease_models.json").exists() else {}
    fb = {"dinov2": "embed_dinov2_vits14", "openclip": "embed_openclip_vitb32", "bioclip2": "embed_bioclip2"}
    fsel = sel.get("F_fusion", {}).get("selected_config", "F_fusion_dinov2")
    fpart = next((v for k, v in fb.items() if f"_{k}[" in fsel or fsel.endswith(k)), "embed_dinov2_vits14")
    if fpart in comp:
        out["arms"]["F_fusion"] = {"ms_per_image": comp["stage1"] + comp[fpart], "components": ["stage1", fpart]}
    (path("results", cfg) / "p3" / "timing.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
