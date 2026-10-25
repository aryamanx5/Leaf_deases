#!/usr/bin/env bash
# Start the FastAPI service from the repo root (used by `make dev` and the preview launcher).
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src:.
exec .venv/bin/uvicorn api.main:app --host 127.0.0.1 --port "${PORT:-8010}"
