"""P3-M09: copy the selected heads into models/p3/arms and write models/manifest.json
(production models, weights hashes, training date, dataset hash, licences).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys

import joblib

from leafdx.config import ROOT, load_config, path
from leafdx.p3.embcache import weights_hash

LICENCES = {
    "dinov2_vits14": "Apache-2.0 (facebookresearch/dinov2)",
    "openclip_vitb32": "MIT (OpenCLIP code); LAION-2B weights released under MIT",
    "bioclip2": "MIT (imageomics/bioclip-2 model card)",
    "mobilenet_v2": "BSD-3-Clause (torchvision ImageNet weights)",
    "efficientnet_b0": "BSD-3-Clause (torchvision ImageNet weights)",
    "convnext_tiny": "BSD-3-Clause (torchvision ImageNet weights)",
}
ARCH = {a["name"]: a["arch"] for a in load_config()["p3"]["cnn_arms"]["arms"]}


def sha(f) -> str:
    return hashlib.sha1((ROOT / f).read_bytes()).hexdigest()


def main() -> int:
    cfg = load_config()
    res = path("results", cfg)
    D = json.loads((res / "disease_models.json").read_text())
    S = json.loads((res / "stacking.json").read_text())
    T = json.loads((path("models", cfg) / "p3" / "trust.json").read_text())
    C = json.loads((res / "crop_models.json").read_text())
    adir = path("models", cfg) / "p3" / "arms"
    adir.mkdir(parents=True, exist_ok=True)
    arms = {}
    tr = json.loads((res / "p3_cnn_train.json").read_text())
    log = json.loads((res / "p3" / "disease_configs.json").read_text())
    for a, v in D["arms"].items():
        if "selected_config" not in v:
            continue
        cid = v["selected_config"]
        if a.startswith("C"):
            variant = cid.split("[")[1].rstrip("]")
            ck = tr[f"{a}_{variant}"]["checkpoint"]
            f = adir / f"{a}.joblib"
            joblib.dump({"checkpoint": ck, "arch": ARCH[a], "temperature": log[cid]["temperature"]}, f)
            arms[a] = {"config": cid, "kind": "cnn", "file": str(f.relative_to(ROOT)), "checkpoint": ck,
                       "weights_sha1": sha(ck), "licence": LICENCES[ARCH[a]]}
        else:
            src = ROOT / ".cache" / "p3models" / f"{cid}.joblib"
            f = adir / f"{a}.joblib"
            shutil.copyfile(src, f)
            b = joblib.load(f)
            arms[a] = {"config": cid, "kind": b["kind"], "file": str(f.relative_to(ROOT)), "weights_sha1": sha(f)}
            if b.get("backbone"):
                arms[a]["backbone"] = b["backbone"]
                arms[a]["backbone_weights_hash"] = weights_hash(b["backbone"])
                arms[a]["licence"] = LICENCES[b["backbone"]]
    crop_f = path("models", cfg) / "p3" / "crop_head.joblib"
    cb = joblib.load(crop_f)
    if C["chosen"].endswith("+bioclip2_zeroshot"):  # product of experts: keep the head's own info for inference
        head = C["chosen"].split("+")[0]
        cb["head_info"] = C["candidates"][head]["info"]
        joblib.dump(cb, crop_f)
    stack_f = path("models", cfg) / "p3" / "stacker.joblib"
    prod = S["production_choice"]
    splits = path("splits", cfg)
    man = {
        "schema": 1,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "training_date": dt.datetime.fromtimestamp((res / "p3" / "preds_disease.csv.gz").stat().st_mtime).isoformat(
            timespec="seconds"),
        "dataset_hash": {"splits.csv_sha1": sha(splits.relative_to(ROOT)),
                         "features.csv_sha1": sha(path("features", cfg).relative_to(ROOT))},
        "production": {"disease": prod, "disease_config": "STACK" if prod == "STACK" else arms[prod]["config"],
                       "crop": C["chosen"], "reason": S["production_reason"]},
        "crop": {"config": C["chosen"], "file": str(crop_f.relative_to(ROOT)), "weights_sha1": sha(crop_f.relative_to(ROOT))},
        "arms": arms,
        "stacker": {"file": str(stack_f.relative_to(ROOT)), "members": S["members"]} if stack_f.exists() else None,
        "trust": {**T, "ood_reference": "models/p3/ood_reference.npy"},
        "licences": LICENCES,
        "data_licences": {"turmeric": "Mendeley g46dvrcvwn v2 (CC BY 4.0)", "blackgram": "BPLD Mendeley zfcv9fmrgv v3 (CC BY 4.0)"},
    }
    mdir = path("models", cfg).parent / "models"
    # after a field-validated switch (scripts/p3_reselect_eval.py) the Phase 3 bundle lives in manifest_phase3.json
    target = mdir / ("manifest_phase3.json" if (mdir / "manifest_phase3.json").exists() else "manifest.json")
    target.write_text(json.dumps(man, indent=1))
    print(json.dumps(man["production"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
