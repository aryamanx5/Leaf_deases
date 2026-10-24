"""Fetch the BPLD (black gram) and turmeric datasets from Mendeley Data.

Downloads into data/downloads/<crop>/, verifies the sha256 reported by the Mendeley
public API, extracts, and copies images into data/raw/<crop>/<healthy|diseased>/<orig_class>/.
If the network is unavailable, prints manual steps and validates what data/raw/ holds.
"""

from __future__ import annotations

import argparse
import sys

from leafdx.config import load_config, path
from leafdx.data import download, mendeley_files, place_images, safe_extract

MANUAL = """
Manual download:
  1. Open https://data.mendeley.com/datasets/{id}/{version} and click "Download All".
  2. Put the zip(s) {files} in data/downloads/{crop}/
  3. Re-run: python scripts/download_data.py
"""


def fetch(crop: str, spec: dict, cfg: dict) -> dict:
    dl_dir = path("downloads", cfg) / crop
    records = {r["filename"]: r for r in mendeley_files(spec["mendeley_id"], spec["version"])}
    counts: dict[str, int] = {}
    for name in spec["files"]:
        rec = records[name]
        print(f"[{crop}] {name}: {rec['size'] / 1e6:.1f} MB, sha256 {rec['sha256'][:12]}...")
        zpath = download(rec["url"], dl_dir / name, rec["sha256"])
        print(f"[{crop}] sha256 verified: {zpath}")
        out = dl_dir / (zpath.stem + "_extracted")
        if not out.exists():
            safe_extract(zpath, out)
        for k, v in place_images(out, path("raw", cfg), crop, spec["class_map"]).items():
            counts[k] = counts.get(k, 0) + v
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", choices=["turmeric", "blackgram"], default=None)
    args = ap.parse_args()
    cfg = load_config()
    ok = True
    for crop, spec in cfg["datasets"].items():
        if args.crop and crop != args.crop:
            continue
        try:
            counts = fetch(crop, spec, cfg)
            print(f"[{crop}] images per original class: {counts}")
        except Exception as e:  # network / API failure -> manual instructions
            ok = False
            print(f"[{crop}] download failed: {e}", file=sys.stderr)
            print(MANUAL.format(id=spec["mendeley_id"], version=spec["version"],
                                files=spec["files"], crop=crop), file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
