"""User-reported regression: an in-field turmeric photo was called black gram with high confidence.

The fixture is a Google image supplied by the user (test-only, not committed); the test skips if absent.
"""

import pytest

from leafdx.config import ROOT, load_config, path

FIXTURE = ROOT / "tests/fixtures/regressions/turmeric_google_01.jpg"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists() or not (path("models", load_config()) / "cnn_best.pt").exists(),
    reason="regression fixture or models missing")


@pytest.mark.xfail(reason="v1 4-class CNN calls this turmeric field photo black gram (P2-B00); fixed in P2-B08",
                   strict=True)
def test_turmeric_field_photo_is_not_confident_blackgram():
    from leafdx.pipeline import analyze

    res = analyze(FIXTURE.read_bytes(), crop=None)
    assert not (res["crop"] == "blackgram" and res["crop_probabilities"]["blackgram"] > 0.5)
