"""The /analyze sample gallery (web/public/data/site.json "samples") never shows a training or validation photo."""

import json

import pandas as pd
import pytest

from leafdx.config import ROOT, load_config, path

CFG = load_config()
SITE = ROOT / "web/public/data/site.json"
CLASSES = {("turmeric", c) for c in ("healthy_leaf", "leaf_blotch", "dry_leaf")} | {
    ("blackgram", c) for c in ("healthy", "anthracnose", "leaf_crinckle", "powdery_mildew", "yellow_mosaic")}

pytestmark = pytest.mark.skipif(not SITE.exists() or not path("splits", CFG).exists(), reason="site data not exported")


@pytest.fixture(scope="module")
def samples():
    return json.loads(SITE.read_text())["samples"]


def test_samples_come_from_test_splits_only(samples):
    split = pd.read_csv(path("splits", CFG)).set_index("path").split.to_dict()
    split |= pd.read_csv(ROOT / CFG["wild"]["root"] / "splits.csv").set_index("path").split.to_dict()
    bad = [(s["id"], split.get(s["source"])) for s in samples if split.get(s["source"]) not in ("test", "wild_test")]
    assert not bad, f"samples outside the test splits: {bad}"


def test_samples_cover_every_class(samples):
    lab = {(s["crop"], s["orig_class"]) for s in samples if s.get("set", "test") == "test"}
    assert CLASSES <= lab, f"missing classes: {CLASSES - lab}"
    assert len(samples) >= 24


def test_samples_one_per_cluster_and_files_exist(samples):
    ids = [s["id"] for s in samples]
    assert len(ids) == len(set(ids))
    lab = pd.read_csv(path("splits", CFG)).set_index("path").cluster_id.to_dict()
    clusters = [lab[s["source"]] for s in samples if s["source"] in lab]
    assert len(clusters) == len(set(clusters)), "two samples from one near-duplicate cluster"
    for s in samples:
        assert (ROOT / "web/public" / s["src"].lstrip("/")).exists()
        assert (ROOT / "web/public" / s["thumb"].lstrip("/")).exists()
