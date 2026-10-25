"""FastAPI service tests (TestClient). Need the Phase 2 models; skip otherwise."""

import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from leafdx.config import ROOT, load_config, path

CFG = load_config()
NEEDED = ["crop_model.joblib", "disease_cnn.pt", "calibration.json", "ood_reference.npy", "ip_svm.joblib"]
pytestmark = pytest.mark.skipif(not all((path("models", CFG) / f).exists() for f in NEEDED),
                                reason="Phase 2 models not trained yet")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


def _sample(crop="turmeric") -> bytes:
    s = pd.read_csv(path("splits", CFG))
    return (ROOT / s[(s.split == "test") & (s.crop == crop)].path.iloc[0]).read_bytes()


def test_health_and_meta(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["model_versions"]["bundle"]
    assert "calibration" in client.get("/api/meta").json()


def test_analyze_auto_crop(client):
    r = client.post("/api/analyze", files={"file": ("leaf.jpg", _sample("turmeric"), "image/jpeg")}, data={"crop": "auto"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["crop"]["source"] == "model" and j["crop"]["status"] in ("confident", "uncertain")
    assert abs(sum(j["crop"]["probs"].values()) - 1) < 1e-6
    assert j["disease"]["label"] in ("healthy", "diseased")
    assert set(j["disease"]["by"]) == {"cnn", "svm", "rule"}
    assert j["images"]["original"].startswith("data:image/webp;base64,")
    assert j["measurements"]["lesion_count"] >= 0 and j["timings_ms"]["total"] > 0
    # per-stage trace (ids match GET /api/code) and the plain-text result card
    code_stages = client.get("/api/code").json()["stages"]
    assert [st["id"] for st in j["trace"]["stages"]] == [c for c in code_stages if c != "card"]
    assert j["trace"]["input"]["filename"] == "leaf.jpg" and j["trace"]["input"]["crop_given"] == "auto"
    assert j["trace"]["stages"][3]["values"]["lesion_count"] == j["measurements"]["lesion_count"]
    assert len(j["result_card"]) == 4 and j["result_card"][1].startswith("Prediction: ")


def test_user_selected_crop(client):
    r = client.post("/api/analyze", files={"file": ("leaf.jpg", _sample("blackgram"), "image/jpeg")},
                    data={"crop": "blackgram", "roi_x": "0.5", "roi_y": "0.5"})
    j = r.json()
    assert r.status_code == 200 and j["crop"]["status"] == "user_selected" and j["crop"]["label"] == "blackgram"
    assert j["measurements"]["segmentation_mode"] == "user_roi_watershed"


def test_uncertain_path(client):
    from leafdx.robust.infer import RobustModel

    rm = RobustModel.load()
    old = rm.cal["tau_crop"]
    rm.cal["tau_crop"] = 1.01  # no prediction can reach this -> must abstain
    try:
        j = client.post("/api/analyze", files={"file": ("leaf.jpg", _sample(), "image/jpeg")}).json()
    finally:
        rm.cal["tau_crop"] = old
    assert j["crop"]["status"] == "uncertain"


def test_rejects_non_image_bad_crop_and_oversize(client):
    r = client.post("/api/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_type"
    r = client.post("/api/analyze", files={"file": ("x.jpg", b"not really a jpeg", "image/jpeg")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_image"
    r = client.post("/api/analyze", files={"file": ("x.jpg", _sample(), "image/jpeg")}, data={"crop": "rice"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_crop"
    r = client.post("/api/analyze", files={"file": ("big.jpg", b"0" * (10 * 1024 * 1024 + 10), "image/jpeg")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_large"
    r = client.post("/api/analyze", files={"file": ("x.jpg", _sample(), "image/jpeg")}, data={"roi_box": "0.1,0.2"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_roi"


def test_not_a_plant(client):
    buf = io.BytesIO()
    Image.fromarray(np.full((300, 300, 3), (40, 120, 200), np.uint8)).save(buf, format="PNG")
    r = client.post("/api/analyze", files={"file": ("sky.png", buf.getvalue(), "image/png")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "not_a_plant"
