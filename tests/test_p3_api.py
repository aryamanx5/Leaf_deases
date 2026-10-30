"""Phase 3 API contract (P3-M09) and smoke test on three wild images (P3-V01).

Needs the Phase 3 bundle (models/manifest.json); the wild images and the user-reported image are
downloaded / user-supplied and not committed, so those tests skip when the files are absent.
"""

import pandas as pd
import pytest

from leafdx.config import ROOT, load_config, path

CFG = load_config()
MAN = ROOT / "models" / "manifest.json"
USER = ROOT / "data/wild/user_reported/turmeric_google_01.jpg"
pytestmark = pytest.mark.skipif(not MAN.exists(), reason="Phase 3 bundle not built (models/manifest.json)")
V3_FIELDS = ("crop_confidence", "crop_set", "disease_confidence", "disease_set", "abstain", "affected_pct", "severity",
             "arms", "timings_ms", "segmenter", "production_model", "explanation")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


def _wild(n: int = 3) -> list:
    w = pd.read_csv(ROOT / CFG["wild"]["root"] / "splits.csv")
    w = w[(w.label != "unknown") & w.path.map(lambda p: (ROOT / p).exists())]
    return [w[w.crop == "turmeric"].iloc[0], w[w.crop == "blackgram"].iloc[0], w[w.crop == "blackgram"].iloc[-1]][:n]


def _check_v3(j: dict) -> None:
    for k in V3_FIELDS:
        assert k in j, k
    assert set(j["crop_set"]) <= {"turmeric", "blackgram"}
    assert set(j["disease_set"]) <= {"healthy", "diseased"}
    assert isinstance(j["abstain"]["flag"], bool)
    assert j["abstain"]["flag"] == (j["abstain"]["reason"] is not None)
    assert j["severity"] in (None, "mild", "moderate", "severe")
    if j["disease"]["label"] == "healthy":
        assert j["severity"] is None
    assert j["production_model"] in j["arms"] or j["production_model"] == "STACK"
    for a in j["arms"].values():
        assert 0 <= a["prob"] <= 1 and a["label"] in ("healthy", "diseased")
    assert j["segmenter"] == "classical"


def test_analyze_returns_v3_fields_on_a_test_image(client):
    s = pd.read_csv(path("splits", CFG))
    data = (ROOT / s[(s.split == "test") & (s.crop == "blackgram")].path.iloc[0]).read_bytes()
    r = client.post("/api/analyze", files={"file": ("leaf.jpg", data, "image/jpeg")}, data={"crop": "auto"})
    assert r.status_code == 200, r.text
    _check_v3(r.json())


def test_arms_production_mode_and_bad_value(client):
    s = pd.read_csv(path("splits", CFG))
    data = (ROOT / s[s.split == "test"].path.iloc[0]).read_bytes()
    r = client.post("/api/analyze", files={"file": ("leaf.jpg", data, "image/jpeg")}, data={"arms": "production"})
    assert r.status_code == 200, r.text
    assert len(r.json()["arms"]) >= 1
    bad = client.post("/api/analyze", files={"file": ("leaf.jpg", data, "image/jpeg")}, data={"arms": "some"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "bad_arms"


@pytest.mark.parametrize("i", [0, 1, 2])
def test_smoke_wild_images(client, i):
    rows = _wild()
    if len(rows) <= i:
        pytest.skip("wild images not downloaded (scripts/build_wild_set.py)")
    row = rows[i]
    r = client.post("/api/analyze", files={"file": ("w.jpg", (ROOT / row.path).read_bytes(), "image/jpeg")},
                    data={"crop": "auto"})
    assert r.status_code in (200, 422), r.text  # 422 = politely refused as not a leaf
    if r.status_code == 200:
        _check_v3(r.json())


def test_explain_endpoint(client):
    s = pd.read_csv(path("splits", CFG))
    data = (ROOT / s[(s.split == "test") & (s.label == "diseased")].path.iloc[0]).read_bytes()
    r = client.post("/api/explain", files={"file": ("leaf.jpg", data, "image/jpeg")}, data={"crop": "auto"})
    assert r.status_code == 200, r.text
    j = r.json()
    for k in ("gradcam_pp", "vit_patch_map", "lesion_overlay"):
        assert j[k].startswith("data:image/webp;base64,")
    assert "affected area" in j["explanation"] and "confidence" in j["explanation"]


@pytest.mark.skipif(not USER.exists(), reason="user-reported image not present (not committed)")
def test_user_google_turmeric_is_turmeric_or_abstains():
    from leafdx.service import analyze_image

    res = analyze_image(USER.read_bytes(), crop=None, with_gradcam=False)
    assert res["crop"]["label"] == "turmeric" or res["abstain"]["flag"], (res["crop"], res["abstain"])
