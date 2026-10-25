"""P2-B01: build the wild / out-of-distribution evaluation set (never used for training).

Sources (all openly licensed, recorded in data/wild/SOURCES.csv):
  * a seeded sample of ORIGINAL images from two independent Mendeley datasets (different
    country, photographers and devices than the training data; labels from the authors),
  * CC0 / CC-BY / CC-BY-SA iNaturalist field photos (label "unknown").
Images are stored downscaled. Near-duplicates of TRAINING images (same pHash rules as
the main split) are dropped; the rest is split 50/50 into wild_val / wild_test by cluster.
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

from leafdx.config import ROOT, load_config, path
from leafdx.data import HttpRangeFile, normalise_class
from leafdx.split import cluster_images, grouped_stratified_split, phash_pair

API = "https://data.mendeley.com/public-api/datasets/{id}"
UA = {"User-Agent": "leafdx-research/0.1 (university research prototype; contact via repository)"}


def save_image(data: bytes, dest, max_side: int) -> bool:
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        return False
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, quality=92)
    return True


def mendeley_files(spec: dict) -> list[dict]:
    """(class, filename, fetch) for the dataset's original images."""
    d = requests.get(API.format(id=spec["id"]), timeout=60).json()
    out = []
    zips = [f for f in d["files"] if f["filename"].endswith(".zip")]
    if zips:
        z = zipfile.ZipFile(HttpRangeFile(zips[0]["content_details"]["download_url"]))
        for info in z.infolist():
            if info.is_dir():
                continue
            cls = normalise_class(info.filename.split("/")[-2])
            out.append({"cls": cls, "name": info.filename.split("/")[-1],
                        "url": f"https://data.mendeley.com/datasets/{spec['id']}/{spec['version']}",
                        "fetch": (lambda i=info, zz=z: zz.read(i))})
    else:
        for f in d["files"]:
            m = re.match(r"^([a-z_]+?)(?:_disease)?_\(\d+\)\.jpe?g$", f["filename"], re.I)
            if not m or f["filename"].startswith("aug_"):
                continue
            url = f["content_details"]["download_url"]
            out.append({"cls": normalise_class(m.group(1)), "name": f["filename"], "url": url,
                        "fetch": (lambda u=url: requests.get(u, timeout=120).content)})
    return out


def from_mendeley(spec: dict, cfg: dict, rng: random.Random) -> list[dict]:
    files = mendeley_files(spec)
    rows = []
    for cls, (label, n) in spec["classes"].items():
        pool = sorted((f for f in files if f["cls"] == normalise_class(cls)), key=lambda f: f["name"])
        for f in rng.sample(pool, min(n, len(pool))):
            stem = re.sub(r"[^\w]+", "_", f"{spec['id']}_{cls}_{f['name']}").strip("_")
            dest = ROOT / cfg["wild"]["root"] / spec["crop"] / label / f"{stem}.jpg"
            if dest.exists() or save_image(f["fetch"](), dest, cfg["wild"]["max_side"]):
                rows.append({"path": str(dest.relative_to(ROOT)), "crop": spec["crop"], "label": label,
                             "orig_class": normalise_class(cls).replace(" ", "_"), "source": f"mendeley:{spec['id']}",
                             "url": f["url"], "author": "dataset authors (see citation)", "licence": spec["licence"],
                             "group": f"{spec['id']}:{f['name']}", "notes": spec["citation"]})
    return rows


def from_inaturalist(cfg: dict) -> list[dict]:
    w = cfg["wild"]["inaturalist"]
    rows = []
    for crop, taxon in w["taxa"].items():
        r = requests.get("https://api.inaturalist.org/v1/observations", timeout=60, params={
            "taxon_name": taxon, "photo_license": ",".join(w["licences"]), "per_page": 100,
            "order_by": "id", "order": "asc"}).json()
        n = 0
        for obs in r["results"]:
            for ph in obs.get("photos", [])[:1]:   # first photo per observation (one group each)
                lic = (ph.get("license_code") or "").lower()
                if lic not in w["licences"] or n >= w["max_per_crop"]:
                    continue
                url = ph["url"].replace("square", "large")
                dest = ROOT / cfg["wild"]["root"] / crop / "unknown" / f"inat_{obs['id']}_{ph['id']}.jpg"
                if dest.exists() or save_image(requests.get(url, timeout=60).content, dest, cfg["wild"]["max_side"]):
                    rows.append({"path": str(dest.relative_to(ROOT)), "crop": crop, "label": "unknown",
                                 "orig_class": "unknown", "source": "inaturalist",
                                 "url": f"https://www.inaturalist.org/observations/{obs['id']}",
                                 "author": ph.get("attribution", ""), "licence": lic.upper(),
                                 "group": f"inat:{obs['id']}", "notes": taxon})
                    n += 1
    return rows


def from_commons(cfg: dict) -> list[dict]:
    c = cfg["wild"]["commons"]
    ok_lic = ("CC0", "CC BY", "CC-BY", "PUBLIC DOMAIN")
    rows, seen = [], set()
    for crop in ("turmeric", "blackgram"):
        spec, n = c[crop], 0
        for q in spec["queries"]:
            r = requests.get("https://commons.wikimedia.org/w/api.php", timeout=60, headers=UA, params={
                "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6, "gsrlimit": 50,
                "gsrsearch": q, "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 1280}).json()
            for p in sorted(r.get("query", {}).get("pages", {}).values(), key=lambda p: p["title"]):
                title, ii = p["title"], p["imageinfo"][0]
                meta = ii.get("extmetadata", {})
                lic = meta.get("LicenseShortName", {}).get("value", "")
                if (title in seen or n >= c["max_per_crop"] or not title.lower().endswith((".jpg", ".jpeg", ".png"))
                        or not any(k in lic.upper() for k in ok_lic) or re.search(r"NC|ND", lic.upper())
                        or not re.search(spec["include"], title) or re.search(spec["exclude"], title)):
                    continue
                seen.add(title)
                stem = re.sub(r"[^\w]+", "_", title[5:].rsplit(".", 1)[0])[:60]
                dest = ROOT / cfg["wild"]["root"] / crop / "unknown" / f"commons_{stem}.jpg"
                data = None if dest.exists() else requests.get(ii["thumburl"], timeout=60,
                                                                headers=UA).content
                if dest.exists() or save_image(data, dest, cfg["wild"]["max_side"]):
                    artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
                    rows.append({"path": str(dest.relative_to(ROOT)), "crop": crop, "label": "unknown",
                                 "orig_class": "unknown", "source": "wikimedia_commons", "url": ii["descriptionurl"],
                                 "author": artist, "licence": lic, "group": f"commons:{title}", "notes": title})
                    n += 1
    return rows


def main() -> int:
    cfg = load_config()
    rng = random.Random(cfg["wild"]["sample_seed"])
    rows = []
    for spec in cfg["wild"]["mendeley"]:
        got = from_mendeley(spec, cfg, rng)
        print(f"mendeley {spec['id']} ({spec['crop']}): {len(got)} images")
        rows += got
    inat = from_inaturalist(cfg)
    print(f"inaturalist: {len(inat)} images")
    rows += inat
    com = from_commons(cfg)
    print(f"wikimedia commons: {len(com)} images")
    rows += com
    df = pd.DataFrame(rows)
    excl = {e["path"]: e["reason"] for e in cfg["wild"]["exclude"]}
    df["excluded"] = df.path.map(excl).fillna("")

    # near-duplicates of TRAINING images (and among wild images)
    train = pd.read_csv(path("splits", cfg))
    allp = pd.concat([train[["path", "crop"]].assign(orig_class=train.orig_class, kind="train"),
                      df[["path", "crop", "orig_class"]].assign(kind="wild")], ignore_index=True)
    hashes = Parallel(n_jobs=-1)(delayed(phash_pair)(ROOT / p) for p in allp.path)
    hs, hl = (np.stack(h) for h in zip(*hashes, strict=True))
    # cross-crop joins are allowed here: a wild image may duplicate a training image of either crop
    allp["cluster"] = cluster_images(allp.assign(crop="all"), hs, hl, cfg["split"]["dedup"], use_stems=False)
    train_clusters = set(allp.loc[allp.kind == "train", "cluster"])
    wild_cl = allp.loc[allp.kind == "wild", "cluster"].to_numpy()
    df["dup_of_train"] = [c in train_clusters for c in wild_cl]
    df.loc[df.dup_of_train & (df.excluded == ""), "excluded"] = "near-duplicate of a training image"
    df["cluster_id"] = wild_cl
    # keep source groups (one iNat observation / one burst) together too
    df["cluster_id"] = df.groupby("group")["cluster_id"].transform("min")

    keep = df[df.excluded == ""].copy()
    keep["stratum"] = keep.crop + "_" + keep.label
    keep["split"] = grouped_stratified_split(keep, {"wild_val": 0.5, "wild_test": 0.5}, cfg["seed"])
    out = ROOT / cfg["wild"]["root"]
    df.drop(columns=["group"]).to_csv(out / "SOURCES.csv", index=False)
    keep[["path", "crop", "label", "orig_class", "source", "cluster_id", "stratum", "split"]].to_csv(out / "splits.csv", index=False)
    print(f"near-duplicates of training images dropped: {int(df.dup_of_train.sum())}; excluded after vetting: "
          f"{int(((df.excluded != '') & ~df.dup_of_train).sum())}")
    print(keep.groupby(["crop", "label", "split"]).size().unstack(fill_value=0).to_string())
    print("per crop:", keep.crop.value_counts().to_dict())
    json.dump({"n": len(keep), "per_crop": keep.crop.value_counts().to_dict(),
               "dups_dropped": int(df.dup_of_train.sum())}, open(out / "summary.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
