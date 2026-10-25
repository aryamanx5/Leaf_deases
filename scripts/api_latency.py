"""P2-B07: p50 latency of POST /api/analyze over TEST images (TestClient, one warm-up request)."""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from leafdx.config import ROOT, load_config, path


def main() -> int:
    cfg = load_config()
    s = pd.read_csv(path("splits", cfg))
    paths = list(s[s.split == "test"].groupby("crop").sample(6, random_state=cfg["seed"]).path)
    c = TestClient(app)
    c.post("/api/analyze", files={"file": ("w.jpg", (ROOT / paths[0]).read_bytes(), "image/jpeg")})
    ms, server = [], []
    for p in paths:
        t0 = time.perf_counter()
        r = c.post("/api/analyze", files={"file": ("x.jpg", (ROOT / p).read_bytes(), "image/jpeg")})
        ms.append(1000 * (time.perf_counter() - t0))
        server.append(r.json()["timings_ms"])
    dev = os.environ.get("LEAFDX_DEVICE", "default (MPS for embeddings)")
    out = {"device": dev, "n": len(ms), "p50_ms": float(np.median(ms)), "p90_ms": float(np.percentile(ms, 90)),
           "server_p50_by_stage_ms": {k: float(np.median([t[k] for t in server])) for k in server[0]}}
    f = path("results", cfg) / "api_latency.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    d["cpu" if dev == "cpu" else "default"] = out
    f.write_text(json.dumps(d, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
