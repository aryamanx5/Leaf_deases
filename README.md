# Image Processing + CNN for Disease Detection in Turmeric and Black Gram

A research prototype that takes a photo of a turmeric or black gram (urad) leaf and decides whether it is
**healthy or diseased**, in two complementary ways on the same image:

* **Stage 1, digital image processing (core focus).** Explicit, inspectable steps: denoising and CLAHE,
  leaf segmentation (HSV/Lab colour thresholds, Otsu, GrabCut, morphology), abnormal-region detection
  (yellow / brown / dark / pale tissue relative to the leaf's own green), connected-component lesion analysis,
  colour + GLCM texture + shape features, then (a) a transparent rule-based score and (b) an SVM.
* **Stage 2, CNN.** MobileNetV2 transfer learning. Phase 1 used one 4-class head (crop x state); Phase 2 replaced it
  with a separate, calibrated crop model that can answer "unsure", and a crop-conditioned disease CNN (see below).

The two are compared on an identical, leakage-free test split.

**Research question:** *Can conventional digital image-processing techniques identify disease-related
characteristics in turmeric and black gram leaves, and how does their performance compare with a CNN-based approach?*
The measured answer is in [`results/REPORT.md`](results/REPORT.md) and [`results/comparison.md`](results/comparison.md);
all numbers there are produced by the code (this README deliberately contains none).

## Quick start

```bash
make setup            # .venv (Python 3.10+, developed on 3.11) + pinned requirements
make all              # data -> splits -> features -> train-ip -> train-cnn -> compare -> report
make app              # Streamlit demo at http://localhost:8501
make verify           # lint, tests, leakage check, artefacts, report numbers vs sources
```

Individual steps: `make data`, `make splits`, `make features`, `make train-ip`, `make train-cnn`,
`make compare`, `make report`, `make test`, `make lint`.

Hardware used for the reported run: Apple M4 (16 GB). The CNN uses CUDA or Apple MPS automatically if present and
falls back to CPU (set a smaller `cnn.time_budget_min` / fewer epochs in `configs/default.yaml` for slow CPUs).

## Data

`make data` (= `scripts/download_data.py` + `scripts/inventory.py`) downloads both public datasets through the
Mendeley Data public API, verifies the sha256 reported by the API, extracts them and places the images under
`data/raw/<crop>/<healthy|diseased>/<original_class>/`:

| Crop | Dataset | Used classes |
|---|---|---|
| Black gram | Talasila et al., *Black gram Plant Leaf Disease (BPLD) dataset*, Data in Brief 2022, [doi:10.17632/zfcv9fmrgv.3](https://data.mendeley.com/datasets/zfcv9fmrgv/3) | Healthy -> healthy; Anthracnose, Leaf Crinkle, Powdery Mildew, Yellow Mosaic -> diseased |
| Turmeric | *A data-driven approach to turmeric disease detection: Dataset for plant condition classification*, Data in Brief 2025, [doi:10.17632/g46dvrcvwn.2](https://data.mendeley.com/datasets/g46dvrcvwn/2) | Healthy Leaf -> healthy; Leaf Blotch, Dry Leaf -> diseased; rhizome classes excluded |

Only the *original* turmeric images are used; the dataset's pre-augmented archive is not, because augmented
copies must never be split across train and test. **Manual download** (if the API is unreachable): open the two
links above, "Download All", put the zip files in `data/downloads/blackgram/` and `data/downloads/turmeric/`, and run
`make data` again. The class mapping lives in `configs/default.yaml`.

`make splits` clusters near-duplicates (two-resolution perceptual hashes over flips/rotations plus burst-shot
filename grouping) and splits whole clusters 70/15/15, stratified by crop x label, seed fixed in the config.

## Demo app

`make app` opens a Streamlit page with two tabs:

* **Analyze a leaf:** upload a photo or pick a test-set sample, choose the crop or let the CNN decide, and get the
  result card (Crop / Prediction / Affected Region / CNN Confidence), the evidence strip
  *Original -> Segmented Leaf -> Detected Abnormal Regions*, the CNN Grad-CAM, and "Image Processing says X |
  CNN says Y" with an agreement badge.
* **Results:** the generated comparison (`results/comparison.md`, `comparison.csv`) and figures.

Programmatic use:

```python
from leafdx.pipeline import analyze, result_card
res = analyze("leaf.jpg", crop=None)   # crop: "turmeric", "blackgram" or None (CNN decides)
print(result_card(res))
```

## Phase 2: field photos, calibrated crop, API

Phase 1's CNN called a turmeric photo from the web "black gram" with high confidence: every training turmeric
leaf lies on white paper, so the network had partly learned the background. Phase 2 fixed this without training on
any web photo:

* `make wild`: an openly licensed field-photo set (`data/wild/`, licences in `SOURCES.csv`), used only for
  validation and the final robustness test, never for training.
* `make robust`: diagnosis (`results/diagnosis.md`), background-swap augmentation, a crop model (OpenCLIP features +
  zero-shot prior) with temperature scaling, an out-of-distribution check and an "unsure" answer, a crop-conditioned
  disease CNN, and the before/after report `results/robustness.md`. Model details: `models/MODEL_CARD.md`.
* Stage 1 now handles field scenes (foliage mask, watershed fallback, optional click on the leaf).
* `make api`: FastAPI on `http://127.0.0.1:8010` with `GET /api/health`, `GET /api/meta` and
  `POST /api/analyze` (multipart `file`, optional `crop` = auto | turmeric | blackgram, optional `roi_x`/`roi_y`
  in 0-1). Responses follow the versioned schema in `api/schemas.py`; errors are `{"error": {"code", "message"}}`
  (`not_a_plant`, `too_large`, `unsupported_type`, `invalid_image`, `bad_crop`, `bad_roi`).

## Web app

`web/` is the project site, **Patra**: a scroll-story landing page (React 18 + TypeScript + Vite + Tailwind, GSAP
ScrollTrigger + Lenis) and an `/analyze` page that calls the API.

```bash
make dev           # API on :8010 + Vite on http://localhost:5180 (proxies /api); Ctrl-C stops both
make web-build     # typecheck + production bundle in web/dist
make web-preview   # serve web/dist on http://localhost:4180 (run `make api` alongside for /analyze)
make e2e           # Playwright: /analyze upload, healing interaction, axe, reduced motion, scroll long tasks
make site-data     # regenerate web/public/data/site.json + web/public/media from results/ and models/
```

* Every number on the site is read from `web/public/data/site.json`, which `scripts/export_site_data.py` builds
  from `results/` (cited background statistics live in `web/src/content/citations.ts` with their sources).
  `scripts/verify_all.py` fails if site.json is older than the results it reads or if a section component
  contains a hard-coded metric.
* Images are real pipeline outputs for test-set leaves; the closing "healing leaves" section inpaints the lesions
  found by our own Stage 1 mask (a visualisation, labelled as such on the page).
* Optional settings in `web/.env.local` (see `web/.env.example`): `VITE_API_BASE` (API origin when not proxied),
  `VITE_DEPLOYED_APP_URL` (the "go to the deployed app" button, hidden when empty), `VITE_GITHUB_URL`.
* Lighthouse (desktop) and axe results are recorded in `results/web_quality.json` and `PROGRESS.md`.

**Quick start from a fresh clone** (the trained models in `models/` and the 28 sample photos are committed, so
no data download or training is needed to try the site): `make setup` then `make dev`, and open
http://localhost:5180. Needs Python 3.11, Node and `make` (macOS/Linux; on Windows use WSL). The first analysis
downloads the DINOv2 and OpenCLIP backbones (about 700 MB, needs internet) and takes a few seconds longer.

**Rebuilding everything** (re-download the datasets and retrain): `make setup`, `make data`, `make all`,
`make wild`, `make robust`, `make site-data`, `make dev`. Tested on a fresh clone: `make setup` (clean venv) and
`make web-build` (clean `npm ci`) succeed, and the API loads its models. The full download-and-train chain was
run in the development checkout, not repeated in the fresh clone.

**Online:** the landing page is on GitHub Pages and its "Run the model" buttons open the older Streamlit demo
(`app/streamlit_app.py`, same models, simpler interface), because the React `/analyze` page needs the FastAPI
backend (about 2 GB RAM) and no free host was available. `Dockerfile` builds that backend for any Docker host.

## Repository map

```
configs/default.yaml      every threshold / hyperparameter (no magic numbers in code)
src/leafdx/
  data.py split.py        download helpers, inventory, near-duplicate clustering + grouped split
  ip/preprocess.py        resize, median/Gaussian filter, CLAHE, colour spaces
  ip/segment.py           leaf segmentation (plain-background threshold / field GrabCut)
  ip/lesions.py           abnormal-region detection, morphology, connected components
  ip/features.py          57-feature vector (colour, GLCM, edges, lesion shape) + stage1()
  ip/decide.py            rule-based score and SVM
  cnn/                    dataset (train-only augmentation), model, train, evaluate, gradcam,
                          augment.py (background swap)
  robust/                 crop model, crop-conditioned disease CNN, calibration, OOD, inference
  service.py              analyze_image() behind the API
  compare.py ablation.py  method comparison, McNemar, bootstrap CIs, leakage ablation
  pipeline.py             analyze(image, crop) for the app
  viz.py metrics.py       figures, shared metrics
scripts/                  thin entry points (one per Makefile target) + verify_all.py
app/streamlit_app.py      demo
api/                      FastAPI service (main.py) and response schema (schemas.py)
web/                      Patra site: src/sections (landing), src/pages/Analyze.tsx, e2e/ (Playwright)
tests/                    pytest (synthetic test images live in tests/synthetic.py)
results/                  generated metrics, figures, comparison.md, REPORT.md
models/                   generated IP models, CNN weights, crop model, calibration, MODEL_CARD.md
PROGRESS.md               task log with evidence for every step
```

## Limitations

* Both datasets come from single regions and capture campaigns; there is no cross-region test, so the
  results are within-dataset estimates.
* All turmeric photos are on white paper; black gram mixes field and plain-card photos. Crop identity is easy to
  read from the background, and turmeric results say little about field photos.
* Turmeric "Dry Leaf" is treated as diseased.
* Stage 1 colour thresholds were calibrated on the training images of these datasets.
* Abnormal-region detection under-measures faint powdery mildew on dark leaves and very small dark spots; the
  weakest classes of each method are listed (from measured numbers) in the report's limitations and per-class table.
* Disease-type classification is out of scope (future work).

* Web hero (DONE-WITH-CAVEAT): the spec asks for scroll-scrubbed footage of an agronomist inspecting a turmeric leaf.
  None was obtainable under an open licence (and ffmpeg is not installed), so the hero uses a CC BY still with a
  slow zoom. Details and how to lift it: `BLOCKERS.md`.
