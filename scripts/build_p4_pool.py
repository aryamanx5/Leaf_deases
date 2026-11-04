"""Phase 4 data pool (docs/PLAN_phase4_more_data.md): data/p4/labels.csv with a fixed p4_train / p4_val / p4_test split.

Pool = data/field_val (re-used in place) + new CC BY 4.0 field images (config `p4.sources`):
  * z55yrbmn2d originals not in wild / wild_ext / field_val, and from no capture minute any of them uses;
  * 45djgf3p96 leaf boxes from field scenes used by neither wild_ext nor field_val;
  * PhilMedic (tsvdyhbphs) turmeric leaves (crop label only) and a few leaves of each other species ("other").
Near-duplicates of any clean / wild / wild_ext image are dropped. Groups = union of pHash cluster and source group
(capture minute, field scene, PhilMedic leaf pair); the split is a StratifiedGroupKFold over crop x label x source.
data/wild and data/wild_ext are never written or used for anything but the exclusion check.

    python scripts/build_p4_pool.py
"""

from __future__ import annotations

import io
import json
import random
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from leafdx.config import ROOT, load_config
from leafdx.data import HttpRangeFile, normalise_class
from leafdx.p3.data import image_table
from leafdx.p4 import minute
from leafdx.split import UnionFind, cluster_images, phash_pair

sys.path.insert(0, str(ROOT / "scripts"))
import build_wild_ext as bwe  # noqa: E402  (shared helpers: _save, _row, p2_stem, mendeley_boxes)

API = "https://data.mendeley.com/public-api/datasets/{id}"
UA = {"User-Agent": "leafdx-research/0.1 (university research prototype; contact via repository)"}
SPLIT_OF_FOLD = {0: "p4_train", 1: "p4_train", 2: "p4_train", 3: "p4_val", 4: "p4_test"}


def save(im: Image.Image, dest: Path, max_side: int) -> None:
    """bwe._save, but an image whose EXIF block is malformed (PIL cannot rewrite it) is saved unrotated."""
    try:
        bwe._save(im, dest, max_side)
    except (TypeError, ValueError, SyntaxError):
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, quality=92)


def local_or_remote_zip(s: dict) -> zipfile.ZipFile:
    lz = s.get("local_zip")
    if lz and (ROOT / lz).exists():
        return zipfile.ZipFile(ROOT / lz)
    d = requests.get(API.format(id=s["id"]), timeout=60, headers=UA).json()
    f = next(x for x in d["files"] if x["filename"].endswith(".zip"))
    return zipfile.ZipFile(HttpRangeFile(f["content_details"]["download_url"]))


def z55_rows(s, out, used_stems, used_minutes, max_side) -> list[dict]:
    z = local_or_remote_zip(s)
    rows = []
    want = {normalise_class(c): (c, lab) for c, (lab, _n) in s["classes"].items()}
    for i in sorted(z.infolist(), key=lambda i: i.filename):
        parts = i.filename.split("/")
        if i.is_dir() or "__MACOSX" in parts or not parts[-1].lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        key = normalise_class(parts[-2])
        if key not in want:
            continue  # insect (pest) and anything unexpected
        cls, label = want[key]
        stem, mn = Path(parts[-1]).stem, minute(parts[-1])
        if bwe.p2_stem(s["id"], cls, parts[-1]) in used_stems or stem in used_stems or (mn and mn in used_minutes):
            continue
        dest = out / s["crop"] / label / f"{bwe.p2_stem(s['id'], cls, parts[-1])}.jpg"
        if not dest.exists():
            save(Image.open(io.BytesIO(z.read(i))), dest, max_side)
        rows.append(bwe._row(dest, s["crop"], label, cls, s["id"], s["licence"], f"{s['id']}:{mn or stem}",
                             s["citation"], f"https://data.mendeley.com/datasets/{s['id']}/{s['version']}"))
    return rows


def philmedic_rows(s, out, rng, max_side) -> list[dict]:
    d = requests.get(API.format(id=s["id"]), timeout=120, headers=UA).json()
    by_cls: dict[str, list] = {}
    for f in d["files"]:
        m = re.match(r"^(Cl(?:a)?ss\d+)_(\d+)\.jpe?g$", f["filename"], re.I)
        if m:
            by_cls.setdefault(m.group(1).replace("Clss", "Class"), []).append((int(m.group(2)), f))
    jobs, rows = [], []
    for cls, items in sorted(by_cls.items()):
        items.sort(key=lambda t: t[0])
        crop = s["target"].get(cls)
        pick = items if crop else rng.sample(items, min(s["other_per_class"], len(items)))
        for num, f in pick:
            dest = (out / crop / "unknown" if crop else out / "other") / f"{s['id']}_{cls}_{num}.jpg"
            jobs.append((f["content_details"]["download_url"], dest))
            r = bwe._row(dest, crop or "other", "unknown", cls, s["id"], s["licence"],
                         f"{s['id']}:{cls}:leaf{(num + 1) // 2}",  # each leaf is photographed front and back
                         s["citation"], f"https://data.mendeley.com/datasets/{s['id']}/{s['version']}")
            r["labelled_by"] = (f"dataset authors (mendeley:{s['id']}, species class '{cls}'"
                                + ("; Curcuma longa, disease status not labelled)" if crop else "; not turmeric / black gram)"))
            rows.append(r)

    def fetch(url, dest):
        import time

        for attempt in range(6):
            if dest.exists():
                return
            try:
                r = requests.get(url, timeout=180, headers=UA)
                if r.ok and r.headers.get("content-type", "").startswith("image"):
                    save(Image.open(io.BytesIO(r.content)), dest, max_side)
                    return
            except (requests.RequestException, OSError):
                pass
            time.sleep(2 ** attempt)
        print(f"   skipped after 6 attempts: {dest.name}", flush=True)

    Parallel(n_jobs=4, prefer="threads")(delayed(fetch)(u, dd) for u, dd in jobs)
    return [r for r in rows if (ROOT / r["path"]).exists()]


def main() -> int:
    cfg = load_config()
    w = cfg["p4"]
    out = ROOT / w["root"]
    rng = random.Random(w["seed"])

    # ---- what must stay out: wild (+ its sources list) and wild_ext; field_val joins the pool ------------------
    wild_src = pd.read_csv(ROOT / cfg["wild"]["root"] / "SOURCES.csv")
    excl = [pd.read_csv(ROOT / f) for f in w["exclude_sets"]]
    reuse = pd.concat([pd.read_csv(ROOT / f) for f in w["reuse"]], ignore_index=True)
    test_paths = list(wild_src.path) + [p for e in excl for p in e.path]
    test_groups = [g for e in excl for g in e.src_group]
    pool_groups = list(reuse.src_group)
    used_stems = {Path(p).stem for p in test_paths + list(reuse.path)}
    used_stems |= {Path(g.split(":")[-1]).stem for g in test_groups + pool_groups if g.startswith("z55yrbmn2d:")}
    test_minutes = {minute(x) for x in test_paths + test_groups if minute(x)}
    pool_minutes = {minute(x) for x in list(reuse.path) + pool_groups if minute(x)}
    used_scenes = {g.split(":", 1)[1] for g in test_groups + pool_groups if g.startswith("45djgf3p96:")}

    rows = []
    for s in w["sources"]:
        print("==", s["id"], s["kind"], flush=True)
        if s["kind"] == "mendeley_zip":
            rows += z55_rows(s, out, used_stems, test_minutes, w["max_side"])  # pool minutes: grouped, not dropped
        elif s["kind"] == "mendeley_boxes":
            rows += bwe.mendeley_boxes(s, out, rng, w["max_side"], used_scenes)
        elif s["kind"] == "mendeley_prefix":
            rows += philmedic_rows(s, out, rng, w["max_side"])
        print(f"   {len(rows)} new images so far", flush=True)
    new = pd.DataFrame(rows).drop_duplicates("path")
    new["from"] = "new"
    fv = reuse.drop(columns=[c for c in ("dup_of_existing", "group") if c in reuse.columns]).assign(**{"from": "field_val"})
    # field_val was built with Phase 3's minute rule, which missed IMGyyyymmddhhmmss names: drop its images whose
    # capture minute a test image shares (they would put same-minute shots of test leaves into training)
    same_minute = np.array([minute(p) in test_minutes for p in fv.path])
    n_fv_same_minute = int(same_minute.sum())
    fv = fv[~same_minute]
    df = pd.concat([fv, new], ignore_index=True)
    # one capture-minute group per z55 image, whichever split it came from
    is_z55 = df.source == "mendeley:z55yrbmn2d"
    df.loc[is_z55, "src_group"] = ["z55yrbmn2d:" + (minute(p) or Path(p).stem) for p in df.loc[is_z55, "path"]]

    # ---- near-duplicates: drop pool images that match any clean / wild / wild_ext image -------------------------
    t = image_table(cfg)
    t = pd.concat([t, *[e.assign(split="test_set") for e in excl]], ignore_index=True)
    allp = pd.concat([t[["path", "crop"]].assign(orig=t.source.astype(str) + ":" + t.orig_class.astype(str), kind="existing"),
                      df[["path", "crop"]].assign(orig=df.source + ":" + df.orig_class.astype(str), kind="pool")],
                     ignore_index=True)
    hashes = Parallel(n_jobs=-1)(delayed(phash_pair)(ROOT / p) for p in allp.path)
    hs, hl = (np.stack(h) for h in zip(*hashes, strict=True))
    allp["cl"] = cluster_images(allp.assign(crop="all", orig_class=allp.orig), hs, hl, cfg["split"]["dedup"],
                                use_stems=False)
    existing = set(allp.loc[allp.kind == "existing", "cl"])
    cl = allp.loc[allp.kind == "pool", "cl"].to_numpy()
    df["dup_of_existing"] = [c in existing for c in cl]
    # split groups = source group (capture minute / field scene / leaf pair) + STRONG pHash duplicates within a crop.
    # The medium / burst rules above chain similar-looking field leaves into one giant cluster (592 images): right
    # for dropping near-duplicates of test images, too coarse for splitting the pool.
    pool_idx = np.flatnonzero((allp.kind == "pool").to_numpy())
    strong = {**cfg["split"]["dedup"], "medium_d64": -1, "medium_d256": -1, "burst_index_gap": -1}
    cl_s = cluster_images(df.assign(orig_class=df.source), hs[pool_idx], hl[pool_idx], strong, use_stems=False)
    uf = UnionFind(len(df))
    for key in (df.src_group.to_numpy(), cl_s):
        first: dict = {}
        for i, k in enumerate(key):
            if k in first:
                uf.union(first[k], i)
            else:
                first[k] = i
    df["group"] = [f"p{uf.find(i)}" for i in range(len(df))]
    keep = df[~df.dup_of_existing].reset_index(drop=True)

    # ---- grouped, stratified split --------------------------------------------------------------------------------
    strat = keep.crop + "|" + keep.label + "|" + keep.source
    sgk = StratifiedGroupKFold(n_splits=w["folds"], shuffle=True, random_state=w["seed"])
    keep["p4_split"] = ""
    for k, (_, te) in enumerate(sgk.split(keep, strat, keep.group)):
        keep.loc[te, "p4_split"] = SPLIT_OF_FOLD[k]
    assert keep.groupby("group").p4_split.nunique().max() == 1, "a group spans two splits"
    keep.to_csv(out / "labels.csv", index=False)

    summary = {
        "n": len(keep), "n_groups": int(keep.group.nunique()),
        "dropped_near_duplicates_of_existing": int(df.dup_of_existing.sum()),
        "by_split": {s: {f"{a}/{b}": int(v) for (a, b), v in g.groupby(["crop", "label"]).size().items()}
                     for s, g in keep.groupby("p4_split")},
        "by_source": keep.source.value_counts().to_dict(), "by_from": keep["from"].value_counts().to_dict(),
        "excluded": {"test_capture_minutes": len(test_minutes), "pool_capture_minutes_reused": len(pool_minutes),
                     "field_scenes": len(used_scenes), "field_val_rows_sharing_a_test_capture_minute": n_fv_same_minute},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
