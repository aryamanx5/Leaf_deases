# PROGRESS

## Task table

| ID | Task | Status | Evidence pointer |
|----|------|--------|------------------|
| T00 | Bootstrap | DONE | Evidence log 2026-10-24 T00 |
| T01 | Data acquisition + inventory | DONE | Evidence log T01 |
| T02 | Leakage-free split | TODO | |
| T03 | Stage 1a: pre-processing + leaf segmentation | TODO | |
| T04 | Stage 1b: disease region detection | TODO | |
| T05 | Stage 1c: features | TODO | |
| T06 | Stage 1d: decisions (rule-based + SVM) | TODO | |
| T07 | Stage 2a: CNN data + model | TODO | |
| T08 | Stage 2b: CNN training + evaluation | TODO | |
| T09 | Comparison | TODO | |
| T10 | Unified inference pipeline | TODO | |
| T11 | Demo app | TODO | |
| T12 | Tests, lint, quality | TODO | |
| T13 | Documentation + report | TODO | |
| T14 | Final verification | TODO | |

## Evidence log

(newest first)

### 2026-10-24 T01 Data acquisition + inventory
- Sources (located via web search + Mendeley public API, sha256 verified against the API):
  black gram BPLD `zfcv9fmrgv` v3 `BPLD Dataset.zip` (113.2 MB, sha256 52c798005373...);
  turmeric `g46dvrcvwn` v2 `Turmeric Plant Disease.zip` (258.9 MB, sha256 2c58ac33c2e4...).
- `python scripts/download_data.py` -> blackgram `{'anthracnose': 230, 'healthy': 221, 'leaf crinckle': 152, 'powdery mildew': 180, 'yellow mosaic': 224}`; turmeric `{'dry leaf': 203, 'healthy leaf': 197, 'leaf blotch': 199, 'rhizome disease root (excluded)': 182, 'rhizome healthy root (excluded)': 282}`
- `python scripts/inventory.py` -> `Wrote data/processed/inventory.csv (1606 files)`; blackgram diseased 786 / healthy 221; turmeric diseased 402 / healthy 197; `Unreadable images: 0`
- Image sizes: blackgram 512x512 (one 512x513), turmeric 1000x1000.
- `pytest` -> `9 passed, 1 skipped`

### 2026-10-24 T00 Bootstrap
- `make setup && make test` -> `2 passed in 0.12s`
- `python -c "import cv2, skimage, sklearn, torch, torchvision, streamlit"` -> `imports OK` (cv2 5.0.0, skimage 0.26.0, sklearn 1.9.1, torch 2.14.1, torchvision 0.29.1, streamlit 1.66.1; MPS available)
- `make lint` -> `All checks passed!`

## Decisions

- Python 3.11 (Homebrew) venv in `.venv/`; system Python is 3.14, which is newer than the stack is tested against.
- Hardware: Apple M4, 16 GB RAM. PyTorch MPS backend is available and used automatically for the CNN; CPU fallback otherwise.

- Turmeric dataset = Mendeley g46dvrcvwn v2 (Data in Brief 59, 2025, 111435), the paper cited in the decisions summary. Only the ORIGINAL zip is used; the 2.6 GB pre-augmented zip is skipped (augmentation must be train-only at load time, R2).
- Turmeric "Dry Leaf" is mapped to diseased (it is an abnormal plant condition class in the source dataset, not healthy). Rhizome/root classes are excluded: the project is about leaves. Mapping lives in configs/default.yaml.
- BPLD archive has 1007 files (folder names say 1000; extra files in Healthy/Leaf Crinckle/Yellow Mosaic). All 1007 are kept; near-duplicates are handled by the T02 clustering.
- Class imbalance: blackgram is 78% diseased, turmeric 67% diseased (dry leaf + blotch vs healthy). Handled with class weights, not oversampling.
- Background/label correlation risk (seen while inspecting black gram images): some BPLD healthy and powdery-mildew images are on a plain grey card or blue/white cloth, others in the field. Logged for the report's limitations; checked quantitatively later.

## NEXT

T02: run scripts/make_splits.py on the real inventory, inspect cluster sizes, confirm tests/test_split.py passes including the real-file check.
