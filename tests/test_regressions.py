"""User-reported regression: an in-field turmeric photo was called black gram with high confidence (v1).

The fixture is a Google image supplied by the user (test-only, not committed); the test skips if absent.
Fixed in P2-B08: the crop must be turmeric, or explicitly uncertain -- never a confident black gram.
"""

import pytest

from leafdx.config import ROOT, load_config, path

FIXTURE = ROOT / "tests/fixtures/regressions/turmeric_google_01.jpg"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists() or not (path("models", load_config()) / "crop_model.joblib").exists(),
    reason="regression fixture or Phase 2 models missing")


def test_turmeric_field_photo_is_not_confident_blackgram():
    from leafdx.service import analyze_image

    res = analyze_image(FIXTURE.read_bytes(), crop=None, with_gradcam=False)
    c = res["crop"]
    assert c["label"] == "turmeric" or c["status"] == "uncertain", c
    assert not (c["label"] == "blackgram" and c["status"] == "confident")
    # Stage 1 now measures the main leaf, not a central blob
    assert res["measurements"]["lesion_count"] > 5
