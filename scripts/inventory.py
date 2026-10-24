"""Scan data/raw/ and write data/processed/inventory.csv; print counts per crop/label."""

from __future__ import annotations

import sys

from leafdx.config import ROOT, load_config, path
from leafdx.data import build_inventory


def main() -> int:
    cfg = load_config()
    inv = build_inventory(path("raw", cfg), cfg["crops"], cfg["labels"], rel_to=ROOT)
    out = path("inventory", cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    inv.to_csv(out, index=False)
    if inv.empty:
        print("No images found under data/raw/. See README / BLOCKERS.md for download steps.")
        return 1
    print(f"Wrote {out.relative_to(ROOT)} ({len(inv)} files)")
    print("\nCounts per crop/label (readable only):")
    good = inv[inv.readable]
    print(good.groupby(["crop", "label"]).size().to_string())
    print("\nCounts per original class:")
    print(good.groupby(["crop", "label", "orig_class"]).size().to_string())
    print("\nImage sizes (w x h) per crop:")
    print(good.groupby("crop")[["width", "height"]].describe().loc[:, (slice(None), ["min", "50%", "max"])].to_string())
    bad = inv[~inv.readable]
    print(f"\nUnreadable images: {len(bad)} (excluded downstream)")
    for p in bad.path:
        print("  ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
