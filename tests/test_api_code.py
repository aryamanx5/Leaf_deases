"""GET /api/code: the real source of a fixed whitelist of pipeline functions, nothing else."""

import pytest

from leafdx.config import ROOT


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


def test_code_returns_every_stage_with_real_line_numbers(client):
    r = client.get("/api/code")
    assert r.status_code == 200
    body = r.json()
    assert body["stages"] == ["denoise", "colour", "segment", "lesions", "measure", "ip_decision", "cnn", "compare", "card"]
    assert {i["stage"] for i in body["items"]} == set(body["stages"])
    for i in body["items"]:
        assert set(i) == {"stage", "name", "file", "start_line", "end_line", "source"}
        assert i["file"].startswith("src/leafdx/") and i["file"].endswith(".py")
        lines = (ROOT / i["file"]).read_text().splitlines(keepends=True)
        assert "".join(lines[i["start_line"] - 1:i["end_line"]]) == i["source"]  # exactly what is on disk
        assert i["source"].lstrip().startswith(("def ", "@"))


def test_code_filters_by_stage(client):
    r = client.get("/api/code", params={"stage": "segment"})
    assert r.status_code == 200
    assert [i["name"] for i in r.json()["items"]] == ["otsu_threshold", "threshold_distance", "segment_leaf"]


@pytest.mark.parametrize("stage", ["../../api/main", "/etc/passwd", "src/leafdx/config.py", "load_config", "", "SEGMENT"])
def test_code_rejects_anything_outside_the_whitelist(client, stage):
    r = client.get("/api/code", params={"stage": stage})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "unknown_stage"


def test_code_ignores_path_like_parameters(client):
    base = client.get("/api/code").json()
    for params in ({"file": "/etc/passwd"}, {"path": "../../.env"}, {"name": "load_config"}):
        assert client.get("/api/code", params=params).json() == base
    assert client.get("/api/code/../../etc/passwd").status_code == 404
    assert client.post("/api/code").status_code == 405
