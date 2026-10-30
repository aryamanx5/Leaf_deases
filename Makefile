PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
export PYTHONPATH := src

.PHONY: setup data splits features train-ip train-cnn compare report app test lint verify all \
	wild wild-ext robust p3 site-data api web-install dev web-build web-preview e2e

setup:
	test -d .venv || python3.11 -m venv .venv
	$(PIP) install -q -r requirements.txt
	$(PIP) install -q -e .

data:
	$(PY) scripts/download_data.py
	$(PY) scripts/inventory.py

splits:
	$(PY) scripts/make_splits.py

features:
	$(PY) scripts/extract_features.py

train-ip:
	$(PY) scripts/train_ip.py

train-cnn:
	$(PY) scripts/train_cnn.py

compare:
	$(PY) scripts/run_comparison.py

report:
	$(PY) scripts/make_report.py

app:
	.venv/bin/streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check .

verify:
	$(PY) scripts/verify_all.py

all: data splits features train-ip train-cnn compare report

# ---- Phase 2: robust models, API, web ----------------------------------------------------------
API_PORT ?= 8010

wild:            # openly licensed field photos (test/val only, never trained on)
	$(PY) scripts/build_wild_set.py

robust:          # diagnosis -> hierarchical crop + disease models -> calibration/OOD -> evaluation
	$(PY) scripts/diagnose.py
	$(PY) scripts/train_robust.py
	$(PY) scripts/calibrate.py
	$(PY) scripts/evaluate_robust.py

wild-ext:        # extended field test set (evaluation only): download + dedupe, then score the frozen Phase 3 bundle
	$(PY) scripts/build_wild_ext.py
	$(PY) scripts/p3_wild_ext.py
	$(PY) scripts/p3_report.py

p3:              # Phase 3: audit -> embeddings -> CNN arms -> crop ID -> disease arms -> stack -> trust -> report
	$(PY) scripts/p3_audit.py
	$(PY) scripts/p3_embed.py
	$(PY) scripts/p3_train_cnns.py
	$(PY) scripts/p3_crop.py
	$(PY) scripts/p3_disease.py
	$(PY) scripts/p3_metrics.py && $(PY) scripts/p3_timing.py && $(PY) scripts/p3_metrics.py
	$(PY) scripts/p3_stack.py
	$(PY) scripts/p3_trust.py
	$(PY) scripts/p3_robustness.py
	$(PY) scripts/p3_finalize.py
	$(PY) scripts/p3_report.py

site-data:       # every number and image the website shows (web/public/data, web/public/media)
	$(PY) scripts/make_heal_assets.py
	$(PY) scripts/export_site_data.py

api:
	PORT=$(API_PORT) scripts/run_api.sh

web-install:
	cd web && npm ci

dev:             # FastAPI on :$(API_PORT) + Vite on :5180 (proxies /api); Ctrl-C stops both
	@test -d web/node_modules || $(MAKE) web-install
	@trap 'kill 0' INT TERM EXIT; PORT=$(API_PORT) scripts/run_api.sh & cd web && npm run dev

web-build:       # typecheck + production bundle in web/dist
	@test -d web/node_modules || $(MAKE) web-install
	cd web && npm run build

web-preview:     # serve web/dist on :4180 (needs `make api` for /analyze)
	cd web && npm run preview -- --port 4180

e2e:             # Playwright against `make dev` (API + Vite must be running)
	cd web && npx playwright test
