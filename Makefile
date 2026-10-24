PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
export PYTHONPATH := src

.PHONY: setup data splits features train-ip train-cnn compare report app test lint verify all

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
