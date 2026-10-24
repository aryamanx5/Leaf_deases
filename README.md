# Image Processing + CNN for Disease Detection in Turmeric and Black Gram

A research prototype that takes a photo of a turmeric or black gram (urad) leaf and decides whether it is
**healthy or diseased**, in two complementary ways on the same image:

* **Stage 1, digital image processing (core focus).** Explicit, inspectable steps: denoising and CLAHE,
  leaf segmentation (HSV/Lab colour thresholds, Otsu, GrabCut, morphology), abnormal-region detection
  (yellow / brown / dark / pale tissue relative to the leaf's own green), connected-component lesion analysis,
  colour + GLCM texture + shape features, then (a) a transparent rule-based score and (b) an SVM.
* **Stage 2, CNN.** MobileNetV2 transfer learning with a 4-class head (crop x state), which also identifies the crop.

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
  cnn/                    dataset (train-only augmentation), model, train, evaluate, gradcam
  compare.py ablation.py  method comparison, McNemar, bootstrap CIs, leakage ablation
  pipeline.py             analyze(image, crop) for the app
  viz.py metrics.py       figures, shared metrics
scripts/                  thin entry points (one per Makefile target) + verify_all.py
app/streamlit_app.py      demo
tests/                    pytest (synthetic test images live in tests/synthetic.py)
results/                  generated metrics, figures, comparison.md, REPORT.md
models/                   generated ip_rule.joblib, ip_svm.joblib, cnn_best.pt
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

No known caveats are open in `BLOCKERS.md` (the file is absent when nothing is blocked).
