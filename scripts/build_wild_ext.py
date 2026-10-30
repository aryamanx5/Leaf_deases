"""Extended field test set (evaluation ONLY): data/wild_ext/.

Sources (config `wild_ext`, all CC BY 4.0, recorded per image in data/wild_ext/labels.csv):
  * more ORIGINAL images from the two wild-set datasets (images already in data/wild are skipped),
  * single-leaf crops from the bounding boxes of a black gram FIELD dataset (healthy / yellow mosaic) plus whole
    field scenes (crop label only),
  * leaves of five OTHER plant species (crop "other"): tests that the app refuses non-target leaves.
Pest classes (aphids, insect, flea beetle) are excluded, as in Phase 2. Images are stored downscaled.
Near-duplicates of any train / val / test / wild image are dropped (same pHash rules as the split, burst rule only
within one source class); near-duplicates inside wild_ext share a group (for grouped bootstrap).
Nothing here is used for training, tuning, calibration or selection.

    python scripts/build_wild_ext.py
"""

from __future__ import annotations

import io
import json
import random
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed
from PIL import Image, ImageOps

from leafdx.config import ROOT, load_config
from leafdx.data import HttpRangeFile, normalise_class
from leafdx.p3.data import image_table
from leafdx.split import cluster_images, phash_pair

API = "https://data.mendeley.com/public-api/datasets/{id}"


def _save(im: Image.Image, dest, max_side: int) -> None:
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, quality=92)


def p2_stem(sid: str, cls: str, name: str) -> str:
    """File stem build_wild_set.py gave a Mendeley image (to skip images already in data/wild)."""
    return re.sub(r"[^\w]+", "_", f"{sid}_{cls}_{name}").strip("_")


def _row(dest, crop, label, orig, src, licence, group, citation, url):
    return {"path": str(dest.relative_to(ROOT)), "crop": crop, "label": label, "orig_class": orig,
            "source": f"mendeley:{src}", "licence": licence, "src_group": group, "citation": citation, "url": url,
            "labelled_by": f"dataset authors (mendeley:{src}, class '{orig}')"}


def mendeley_files(s, out, used, rng, max_side):
    d = requests.get(API.format(id=s["id"]), timeout=60).json()
    by_cls: dict[str, list] = {}
    for f in d["files"]:
        m = re.match(r"^([a-z_]+?)(?:_disease)?_\(\d+\)\.jpe?g$", f["filename"], re.I)
        if m and not f["filename"].startswith("aug_"):
            by_cls.setdefault(normalise_class(m.group(1)), []).append(f)
    rows, jobs = [], []
    for cls, (label, n) in s["classes"].items():
        pool = sorted((f for f in by_cls.get(normalise_class(cls), []) if p2_stem(s["id"], cls, f["filename"]) not in used),
                      key=lambda f: f["filename"])
        for f in rng.sample(pool, min(n, len(pool))):
            dest = out / s["crop"] / label / f"{p2_stem(s['id'], cls, f['filename'])}.jpg"
            jobs.append((f["content_details"]["download_url"], dest))
            rows.append(_row(dest, s["crop"], label, cls, s["id"], s["licence"], f"{s['id']}:{f['filename']}",
                             s["citation"], f"https://data.mendeley.com/datasets/{s['id']}/{s['version']}"))

    def fetch(url, dest):
        import time

        for attempt in range(5):  # the file server rate-limits bursts (returns an HTML page): back off and retry
            if dest.exists():
                return True
            try:
                r = requests.get(url, timeout=180)
                if r.ok and r.headers.get("content-type", "").startswith("image"):
                    _save(Image.open(io.BytesIO(r.content)), dest, max_side)
                    return True
            except (requests.RequestException, OSError):
                pass
            time.sleep(2 ** attempt)
        print(f"   skipped after 5 attempts: {dest.name}", flush=True)
        return False

    Parallel(n_jobs=4, prefer="threads")(delayed(fetch)(u, d) for u, d in jobs)  # 3-4 MB originals: download in parallel
    return [r for r in rows if (ROOT / r["path"]).exists()]


def _zip(s, name=None):
    d = requests.get(API.format(id=s["id"]), timeout=60).json()
    f = next(x for x in d["files"] if (name is None and x["filename"].endswith(".zip")) or x["filename"] == name)
    return zipfile.ZipFile(HttpRangeFile(f["content_details"]["download_url"]))


def mendeley_zip(s, out, used, rng, max_side):
    z = _zip(s)
    by_cls: dict[str, list] = {}
    for i in z.infolist():
        if not i.is_dir():
            by_cls.setdefault(normalise_class(i.filename.split("/")[-2]), []).append(i)
    rows = []
    for cls, (label, n) in s["classes"].items():
        pool = sorted((i for i in by_cls.get(normalise_class(cls), [])
                       if p2_stem(s["id"], cls, i.filename.split("/")[-1]) not in used), key=lambda i: i.filename)
        for i in rng.sample(pool, min(n, len(pool))):
            stem = re.sub(r"[^\w]+", "_", f"{s['id']}_{cls}_{i.filename.split('/')[-1]}").strip("_")
            dest = out / s["crop"] / label / f"{stem}.jpg"
            if not dest.exists():
                _save(Image.open(io.BytesIO(z.read(i))), dest, max_side)
            # capture-time groups: files named YYYYMMDD_HHMMSS; shots within the same minute share a group
            ts = re.match(r"(\d{8}_\d{4})", i.filename.split("/")[-1])
            rows.append(_row(dest, s["crop"], label, cls, s["id"], s["licence"],
                             f"{s['id']}:{cls}:{ts.group(1) if ts else i.filename}", s["citation"],
                             f"https://data.mendeley.com/datasets/{s['id']}/{s['version']}"))
    return rows


def mendeley_boxes(s, out, rng, max_side):
    z = _zip(s)
    names = {i.filename.split("/")[-1]: i for i in z.infolist() if i.filename.lower().endswith(".jpg")}
    lab = pd.read_csv(io.BytesIO(z.read(next(i for i in z.namelist() if i.endswith("labels.csv")))))
    lab = lab[lab.image_name.isin(names)]
    rows = []
    cache: dict[str, Image.Image] = {}

    def img(n):
        if n not in cache:
            cache[n] = ImageOps.exif_transpose(Image.open(io.BytesIO(z.read(names[n])))).convert("RGB")
        return cache[n]

    for code, (label, n) in s["boxes"].items():
        b = lab[(lab.label_name == code) & (lab[["bbox_width", "bbox_height"]].min(axis=1) >= s["min_box_side"])]
        b = b.sample(min(n, len(b)), random_state=rng.randrange(1 << 30)).sort_values(["image_name", "bbox_x"])
        for r in b.itertuples():
            dest = out / "blackgram" / label / f"{s['id']}_{code}_{r.image_name[:-4]}_{r.bbox_x}_{r.bbox_y}.jpg"
            if not dest.exists():
                im = img(r.image_name)
                px, py = s["pad"] * r.bbox_width, s["pad"] * r.bbox_height
                box = (max(0, r.bbox_x - px), max(0, r.bbox_y - py), min(im.width, r.bbox_x + r.bbox_width + px),
                       min(im.height, r.bbox_y + r.bbox_height + py))
                _save(im.crop(tuple(int(v) for v in box)), dest, max_side)
            rows.append(_row(dest, "blackgram", label, f"box {code}", s["id"], s["licence"],
                             f"{s['id']}:{r.image_name}", s["citation"],
                             f"https://data.mendeley.com/datasets/{s['id']}/{s['version']}"))
    for n in rng.sample(sorted(names), min(s["scenes"], len(names))):
        dest = out / "blackgram" / "scene" / f"{s['id']}_scene_{n[:-4]}.jpg"
        if not dest.exists():
            _save(img(n).copy(), dest, max_side)
        rows.append(_row(dest, "blackgram", "unknown", "field scene (several leaves, mixed labels)", s["id"],
                         s["licence"], f"{s['id']}:{n}", s["citation"],
                         f"https://data.mendeley.com/datasets/{s['id']}/{s['version']}"))
    return rows


def mendeley_zip_other(s, out, rng, max_side):
    z = _zip(s, s["zip"])
    by_dir: dict[str, list] = {}
    for i in z.infolist():
        if not i.is_dir() and i.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            by_dir.setdefault(i.filename.split("/")[-2], []).append(i)
    rows = []
    for folder, items in sorted(by_dir.items()):
        for i in rng.sample(sorted(items, key=lambda x: x.filename), min(s["per_folder"], len(items))):
            stem = re.sub(r"[^\w]+", "_", f"{s['id']}_{folder}_{i.filename.split('/')[-1]}").strip("_")
            dest = out / "other" / f"{stem}.jpg"
            if not dest.exists():
                _save(Image.open(io.BytesIO(z.read(i))), dest, max_side)
            r = _row(dest, "other", "unknown", folder, s["id"], s["licence"], f"{s['id']}:{folder}:{i.filename}",
                     s["citation"], f"https://data.mendeley.com/datasets/{s['id']}/{s['version']}")
            r["labelled_by"] = f"dataset authors (mendeley:{s['id']}, species folder '{folder}'; not turmeric / black gram)"
            rows.append(r)
    return rows


def main() -> int:
    cfg = load_config()
    w = cfg["wild_ext"]
    out = ROOT / w["root"]
    rng = random.Random(w["seed"])
    wild_src = pd.read_csv(ROOT / cfg["wild"]["root"] / "SOURCES.csv")
    used = {p.split("/")[-1][:-4] for p in wild_src.path}  # stems of images already in data/wild
    rows = []
    for s in w["sources"]:
        print("==", s["id"], s["kind"], flush=True)
        if s["kind"] == "mendeley_files":
            rows += mendeley_files(s, out, used, rng, w["max_side"])
        elif s["kind"] == "mendeley_zip":
            rows += mendeley_zip(s, out, used, rng, w["max_side"])
        elif s["kind"] == "mendeley_boxes":
            rows += mendeley_boxes(s, out, rng, w["max_side"])
        else:
            rows += mendeley_zip_other(s, out, rng, w["max_side"])
        print(f"   {len(rows)} images so far", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("path").reset_index(drop=True)

    # near-duplicates against every existing image (train / val / test / wild / user) and inside wild_ext
    t = image_table(cfg)
    allp = pd.concat([t[["path", "crop"]].assign(orig=t.source + ":" + t.orig_class, kind="existing"),
                      df[["path", "crop"]].assign(orig=df.source + ":" + df.orig_class, kind="ext")], ignore_index=True)
    hashes = Parallel(n_jobs=-1)(delayed(phash_pair)(ROOT / p) for p in allp.path)
    hs, hl = (np.stack(h) for h in zip(*hashes, strict=True))
    allp["cl"] = cluster_images(allp.assign(crop="all", orig_class=allp.orig), hs, hl, cfg["split"]["dedup"], use_stems=False)
    existing = set(allp.loc[allp.kind == "existing", "cl"])
    cl = allp.loc[allp.kind == "ext", "cl"].to_numpy()
    df["dup_of_existing"] = [c in existing for c in cl]
    # bootstrap group = union of pHash cluster and source group (same field image / same capture minute)
    df["group"] = "x" + pd.Series(cl).astype(str)
    df["group"] = df.groupby("src_group")["group"].transform("min")
    keep = df[~df.dup_of_existing].reset_index(drop=True)
    keep.to_csv(out / "labels.csv", index=False)
    summary = {"n": len(keep), "dropped_near_duplicates_of_existing": int(df.dup_of_existing.sum()),
               "by_crop_label": {f"{a}/{b}": int(v) for (a, b), v in keep.groupby(["crop", "label"]).size().items()},
               "by_source": keep.source.value_counts().to_dict(), "n_groups": int(keep.group.nunique())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
