"""FastAPI service for the web app.

  GET  /api/health   -> {status, model_versions}
  GET  /api/meta     -> metrics summary read from results/*.json
  POST /api/analyze  -> multipart file + crop=auto|turmeric|blackgram + optional roi_x, roi_y or roi_box
  GET  /api/code     -> source of the whitelisted pipeline functions (optional ?stage=<id>)

Run: uvicorn api.main:app --port 8000   (from the repo root, PYTHONPATH=src)
"""

from __future__ import annotations

import base64
import io
import json
import time
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from api.code import STAGE_IDS, code_items
from api.schemas import AnalyzeResponse, ErrorResponse
from leafdx.config import ROOT, load_config, path
from leafdx.pipeline import InvalidImageError, NotALeafError
from leafdx.service import analyze_image, model_version

MAX_BYTES = 10 * 1024 * 1024
MAX_SIDE = 768
OK_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff", "application/octet-stream"}

app = FastAPI(title="leafdx API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5180", "http://127.0.0.1:5180",
                                                  "http://localhost:4180", "http://127.0.0.1:4180"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])
_LAT: list[float] = []


def err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=ErrorResponse(error={"code": code, "message": message}).model_dump())


@app.exception_handler(Exception)
async def unhandled(_: Request, e: Exception) -> JSONResponse:  # structured error, never a stack trace
    return err(500, "internal_error", f"{type(e).__name__}")


def to_webp_data_url(rgb: np.ndarray | None) -> str | None:
    if rgb is None:
        return None
    im = Image.fromarray(rgb)
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, format="WEBP", quality=85)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model_versions": {"bundle": model_version(), "schema": AnalyzeResponse.model_fields["schema_version"].default}}


@app.get("/api/meta")
def meta() -> dict:
    res = path("results", load_config())
    out = {}
    for name in ("calibration.json", "robust_train.json", "robustness.json", "ip_metrics.json", "cnn_metrics.json"):
        f = res / name
        if f.exists():
            out[name.replace(".json", "")] = json.loads(f.read_text())
    if (res / "comparison.csv").exists():
        import pandas as pd

        t = pd.read_csv(res / "comparison.csv")
        out["comparison_overall"] = t[(t.section == "metrics") & (t.scope == "overall")].pivot(
            index="method", columns="metric", values="value").to_dict(orient="index")
    out["p50_latency_ms_this_process"] = float(np.median(_LAT)) if _LAT else None
    return out


@app.get("/api/code")
def code(stage: str | None = None):
    """Read-only: the real source of each pipeline stage, from a fixed whitelist (api/code.py)."""
    if stage is not None and stage not in STAGE_IDS:
        return err(404, "unknown_stage", f"stage must be one of {', '.join(STAGE_IDS)}")
    return {"stages": list(STAGE_IDS), "items": [i for i in code_items() if stage is None or i["stage"] == stage]}


@app.post("/api/analyze", response_model=AnalyzeResponse, responses={400: {"model": ErrorResponse},
                                                                     413: {"model": ErrorResponse},
                                                                     415: {"model": ErrorResponse},
                                                                     422: {"model": ErrorResponse}})
async def analyze(file: Annotated[UploadFile, File()], crop: Annotated[str, Form()] = "auto",
                  roi_x: Annotated[float | None, Form()] = None, roi_y: Annotated[float | None, Form()] = None,
                  roi_box: Annotated[str | None, Form()] = None):
    if crop not in ("auto", "turmeric", "blackgram"):
        return err(400, "bad_crop", "crop must be auto, turmeric or blackgram")
    if file.content_type and file.content_type.lower() not in OK_TYPES:
        return err(415, "unsupported_type", f"expected an image upload, got {file.content_type}")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        return err(413, "too_large", "upload is larger than 10 MB")
    roi = None
    if roi_box:
        try:
            box = [float(v) for v in roi_box.split(",")]
            assert len(box) == 4 and all(0 <= v <= 1 for v in box)
            roi = {"box": tuple(box)}
        except (ValueError, AssertionError):
            return err(400, "bad_roi", "roi_box must be 'x,y,w,h' with values in 0-1")
    elif roi_x is not None and roi_y is not None:
        if not (0 <= roi_x <= 1 and 0 <= roi_y <= 1):
            return err(400, "bad_roi", "roi_x and roi_y must be in 0-1")
        roi = {"x": roi_x, "y": roi_y}
    t0 = time.perf_counter()
    try:
        res = analyze_image(data, None if crop == "auto" else crop, roi)
    except InvalidImageError as e:
        return err(400, "invalid_image", str(e))
    except NotALeafError as e:
        return err(422, "not_a_plant", str(e))
    _LAT.append(1000 * (time.perf_counter() - t0))
    res["images"] = {k: to_webp_data_url(v) for k, v in res["images"].items()}
    res["trace"]["input"]["filename"] = (file.filename or "upload")[:120]  # echoed back only, never stored
    res.pop("features", None)
    return AnalyzeResponse(**res)


ROOT_DIR = ROOT  # re-exported for tests
