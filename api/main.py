"""FastAPI service for the web app.

  GET  /api/health   -> {status, model_versions}
  GET  /api/meta     -> metrics summary read from results/*.json
  POST /api/analyze  -> multipart file + crop=auto|turmeric|blackgram + optional roi_x, roi_y or roi_box
                        + arms=all|production (per-arm predictions for every arm, or only the ones needed)
  POST /api/explain  -> multipart file + crop: Grad-CAM++, DINOv2 patch map, lesion overlay, reason string
  GET  /api/code     -> source of the whitelisted pipeline functions (optional ?stage=<id>)

Run: uvicorn api.main:app --port 8000   (from the repo root, PYTHONPATH=src)
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from api.code import STAGE_IDS, code_items
from api.schemas import AnalyzeResponse, ErrorResponse, ExplainResponse
from leafdx.config import ROOT, load_config, path
from leafdx.pipeline import InvalidImageError, NotALeafError
from leafdx.service import analyze_image, model_version

MAX_BYTES = 10 * 1024 * 1024
MAX_SIDE = 768
OK_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff", "application/octet-stream"}

def warm_up() -> None:
    """Load the Phase 3 bundle (backbones, heads, CNN arms) once at start-up, so the first request does not pay for
    it (cold loading BioCLIP 2 + every CNN arm on CPU takes tens of seconds). Skipped with LEAFDX_NO_WARMUP=1."""
    from leafdx.p3 import production as p3

    if os.environ.get("LEAFDX_NO_WARMUP") or not p3.available():
        return
    from leafdx.p3.cnnpred import load_arm
    from leafdx.robust.backbones import embed

    m = p3.load()
    dummy = Image.new("RGB", (224, 224), (60, 120, 40))
    for bb in ({v.get("backbone") for v in m.man["arms"].values()} | {m.trust["ood_backbone"], "bioclip2"}) - {None}:
        embed(bb, [dummy])
    for v in m.arms.values():
        if "checkpoint" in v:
            load_arm(v["checkpoint"], v["arch"], "cpu")


@asynccontextmanager
async def lifespan(_: FastAPI):
    warm_up()
    yield


app = FastAPI(title="leafdx API", version="3.0", lifespan=lifespan)
# Extra browser origins (e.g. the GitHub Pages site) come from CORS_ORIGINS, comma separated; origins have no path.
CORS_ORIGINS = ["http://localhost:5180", "http://127.0.0.1:5180", "http://localhost:4180", "http://127.0.0.1:4180",
                "https://aryamanx5.github.io",
                *[o.strip().rstrip("/") for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]]
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])
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
                  roi_box: Annotated[str | None, Form()] = None, arms: Annotated[str, Form()] = "all"):
    if crop not in ("auto", "turmeric", "blackgram"):
        return err(400, "bad_crop", "crop must be auto, turmeric or blackgram")
    if arms not in ("all", "production"):
        return err(400, "bad_arms", "arms must be all or production")
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
        res = analyze_image(data, None if crop == "auto" else crop, roi, arms=arms)
    except InvalidImageError as e:
        return err(400, "invalid_image", str(e))
    except NotALeafError as e:
        return err(422, "not_a_plant", str(e))
    _LAT.append(1000 * (time.perf_counter() - t0))
    res["images"] = {k: to_webp_data_url(v) for k, v in res["images"].items()}
    res["trace"]["input"]["filename"] = (file.filename or "upload")[:120]  # echoed back only, never stored
    res.pop("features", None)
    return AnalyzeResponse(**res)


@app.post("/api/explain", response_model=ExplainResponse, responses={400: {"model": ErrorResponse},
                                                                     413: {"model": ErrorResponse},
                                                                     415: {"model": ErrorResponse},
                                                                     503: {"model": ErrorResponse}})
async def explain(file: Annotated[UploadFile, File()], crop: Annotated[str, Form()] = "auto"):
    """Where the models looked: Grad-CAM++ (best fine-tuned CNN arm), DINOv2 patch-similarity map, Stage 1
    lesion overlay, plus the plain-language reason string."""
    import uuid

    from leafdx.p3 import production as p3

    if not p3.available():
        return err(503, "no_bundle", "the Phase 3 model bundle (models/manifest.json) is not installed")
    if crop not in ("auto", "turmeric", "blackgram"):
        return err(400, "bad_crop", "crop must be auto, turmeric or blackgram")
    if file.content_type and file.content_type.lower() not in OK_TYPES:
        return err(415, "unsupported_type", f"expected an image upload, got {file.content_type}")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        return err(413, "too_large", "upload is larger than 10 MB")
    try:
        from leafdx.service import explain_image

        r = explain_image(data, None if crop == "auto" else crop)
    except InvalidImageError as e:
        return err(400, "invalid_image", str(e))
    for k in ("gradcam_pp", "vit_patch_map", "lesion_overlay"):
        r[k] = to_webp_data_url(r[k])
    return ExplainResponse(request_id=uuid.uuid4().hex[:12], **r)


ROOT_DIR = ROOT  # re-exported for tests
