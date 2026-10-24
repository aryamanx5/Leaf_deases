# PROGRESS

## Task table

| ID | Task | Status | Evidence pointer |
|----|------|--------|------------------|
| T00 | Bootstrap | DONE | Evidence log 2026-10-24 T00 |
| T01 | Data acquisition + inventory | TODO | |
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

### 2026-10-24 T00 Bootstrap
- `make setup && make test` -> `2 passed in 0.12s`
- `python -c "import cv2, skimage, sklearn, torch, torchvision, streamlit"` -> `imports OK` (cv2 5.0.0, skimage 0.26.0, sklearn 1.9.1, torch 2.14.1, torchvision 0.29.1, streamlit 1.66.1; MPS available)
- `make lint` -> `All checks passed!`

## Decisions

- Python 3.11 (Homebrew) venv in `.venv/`; system Python is 3.14, which is newer than the stack is tested against.
- Hardware: Apple M4, 16 GB RAM. PyTorch MPS backend is available and used automatically for the CNN; CPU fallback otherwise.

## NEXT

T01: locate the BPLD and turmeric Mendeley datasets on the web, write scripts/download_data.py and scripts/inventory.py.
