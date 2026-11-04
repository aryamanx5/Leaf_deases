"""Phase 4 data rules (docs/PLAN_phase4_more_data.md): capture-minute parsing and test-set disjointness."""

import pandas as pd
import pytest

from leafdx.config import ROOT
from leafdx.p4 import minute, overlap_with_tests

POOL = ROOT / "data/p4/labels.csv"


@pytest.mark.parametrize("name,expected", [
    ("20241104_110935.jpg", "202411041109"),
    ("IMG_20241101_220925.jpg", "202411012209"),
    ("IMG20241104113250.jpg", "202411041132"),  # the form Phase 3's rule missed
    ("z55yrbmn2d_healthy_IMG20241104113250_jpg.jpg", "202411041132"),
    ("45djgf3p96_LA_BG0006_255_37.jpg", None),
    ("tsvdyhbphs_Class15_3.jpg", None),
])
def test_minute(name, expected):
    assert minute(name) == expected


def test_overlap_counts_each_kind():
    ext = pd.DataFrame({"path": ["data/wild_ext/blackgram/healthy/z55yrbmn2d_healthy_IMG20241104113250_jpg.jpg",
                                 "data/wild_ext/blackgram/healthy/45djgf3p96_LA_BG0006_1_2.jpg"],
                        "src_group": ["z55yrbmn2d:healthy:x/IMG20241104113250.jpg", "45djgf3p96:BG0006.jpg"]})
    clean = pd.DataFrame({"path": ["data/p4/blackgram/healthy/z55yrbmn2d_healthy_IMG20241104120000_jpg.jpg"],
                          "src_group": ["z55yrbmn2d:202411041200"]})
    assert not any(overlap_with_tests(clean, [], ext).values())
    leak = pd.DataFrame({"path": ["data/p4/blackgram/healthy/z55yrbmn2d_healthy_IMG20241104113255_jpg.jpg",
                                  "data/p4/blackgram/healthy/45djgf3p96_LA_BG0006_9_9.jpg"],
                         "src_group": ["z55yrbmn2d:202411041132", "45djgf3p96:BG0006.jpg"]})
    ov = overlap_with_tests(leak, [], ext)
    assert ov["capture_minutes"] == 1 and ov["scenes"] == 1


@pytest.mark.skipif(not POOL.exists(), reason="Phase 4 pool not built (scripts/build_p4_pool.py)")
def test_pool_disjoint_and_grouped():
    pool = pd.read_csv(POOL)
    ov = overlap_with_tests(pool, list(pd.read_csv(ROOT / "data/wild/SOURCES.csv").path),
                            pd.read_csv(ROOT / "data/wild_ext/labels.csv"))
    assert not any(ov.values()), ov
    assert set(pool.p4_split) == {"p4_train", "p4_val", "p4_test"}
    assert (pool.groupby("group").p4_split.nunique() == 1).all()
    assert pool.licence.eq("CC BY 4.0").all()
