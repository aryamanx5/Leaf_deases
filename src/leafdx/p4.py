"""Phase 4 helpers shared by the pool builder, verify_all and the tests (docs/PLAN_phase4_more_data.md)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_MIN_UNDERSCORE = re.compile(r"(\d{8})_(\d{4})\d{2}")             # 20241104_110935, IMG_20241101_220925
_MIN_COMPACT = re.compile(r"(?<!\d)(\d{8})(\d{4})\d{2}(?!\d)")    # IMG20241104113250


def minute(name: str) -> str | None:
    """Capture minute 'yyyymmddhhmm' from a camera file name, or None.

    Phase 3's rule only knew the underscore forms, so IMGyyyymmddhhmmss shots were grouped by file, not by minute."""
    m = _MIN_UNDERSCORE.search(str(name)) or _MIN_COMPACT.search(str(name))
    return m.group(1) + m.group(2) if m else None


def overlap_with_tests(pool: pd.DataFrame, wild_paths: list[str], ext: pd.DataFrame) -> dict[str, int]:
    """Images, file names, black gram field scenes and z55yrbmn2d capture minutes the pool shares with wild / wild_ext."""
    test_paths = set(wild_paths) | set(ext.path)
    names = {Path(p).name for p in test_paths}
    scenes = {g for g in ext.src_group if str(g).startswith("45djgf3p96:")}
    z55 = [x for x in [*test_paths, *ext.src_group] if "z55yrbmn2d" in str(x)]
    minutes = {minute(x) for x in z55} - {None}
    pz = [x for x in [*pool.path, *pool.src_group] if "z55yrbmn2d" in str(x)]
    return {"paths": len(set(pool.path) & test_paths),
            "file_names": len({Path(p).name for p in pool.path} & names),
            "scenes": len({g for g in pool.src_group if str(g).startswith("45djgf3p96:")} & scenes),
            "capture_minutes": len(({minute(x) for x in pz} - {None}) & minutes)}
