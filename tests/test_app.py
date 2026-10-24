"""Headless smoke test of the Streamlit app (Streamlit AppTest)."""

import pytest
from streamlit.testing.v1 import AppTest

from leafdx.config import load_config, path

pytestmark = pytest.mark.skipif(not (path("models", load_config()) / "cnn_best.pt").exists(),
                                reason="models not trained yet")


def test_app_renders_sample_without_exceptions():
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=180).run()
    assert not at.exception, at.exception
    codes = [c.value for c in at.code]
    assert any(v.startswith("Crop: ") and "Prediction: " in v and "Affected Region: " in v
               and "CNN Confidence: " in v for v in codes), codes
    assert len(at.tabs) == 2
