# PROGRESS

## Task table

| ID | Task | Status | Evidence pointer |
|----|------|--------|------------------|
| T00 | Bootstrap | DONE | Evidence log 2026-10-24 T00 |
| T01 | Data acquisition + inventory | DONE | Evidence log T01 |
| T02 | Leakage-free split | DONE | Evidence log T02 |
| T03 | Stage 1a: pre-processing + leaf segmentation | DONE | Evidence log T03 |
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

### 2026-10-24 T03 Pre-processing + leaf segmentation
- `pytest tests/test_segment.py` -> `16 passed` (synthetic leaf on paper/grey/cloth/field x 3 seeds, IoU > 0.9 each; lesions stay inside the leaf mask; blank image does not crash). Extra check over 8 seeds: field IoU 0.977-0.991, paper/grey/cloth >= 0.999.
- `python scripts/ip_figures.py --stage segment` -> `single dominant leaf mask (coverage 5-95%): 99/100 = 99.0%`; by stratum bg_dis 1.0, bg_h 1.0, tu_dis 1.0, tu_h 0.917; methods plain_bg_threshold 54, field_grabcut 40, field_grabcut_core 6 (results/segmentation_check.json). Sampled from train+val only.
- results/figures/segmentation_samples.png (24 images, 6 per crop x label) opened and inspected: turmeric (paper background, incl. curled dry leaves) segmented cleanly; black gram card/cloth shots clean; field shots mostly correct, a few partial (crinkled leaf) or slightly over-inclusive masks.

### 2026-10-24 T02 Leakage-free split
- `python scripts/make_splits.py` -> `clusters_spanning_splits: 0, clusters_with_mixed_labels: 0, n_clusters: 1548, n_multi_image_clusters: 50`; cluster sizes `{1: 1498, 2: 45, 3: 3, 4: 1, 5: 1}`
- Split counts (test/train/val): blackgram_diseased 118/550/118, blackgram_healthy 32/155/34, turmeric_diseased 60/282/60, turmeric_healthy 29/138/30
- Re-running gives byte-identical splits.csv (md5 919846308c8edef2ad552c38fe36c8c9 both runs).
- `pytest tests/test_split.py` -> `6 passed` (synthetic flipped/noisy duplicates clustered; no cluster spans splits; every stratum in every split; seed-reproducible; real splits.csv leakage-free).
- Class weights (train, balanced) in data/processed/split_meta.json: binary diseased 0.676 / healthy 1.920; 4-class bg_dis 0.511, bg_h 1.815, tu_dis 0.997, tu_h 2.038.

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

- Dedup rule (T02): a single 64-bit pHash threshold was not good enough on these datasets. With threshold 6 it merged DIFFERENT leaves posed alike on a plain card (3 mixed-label clusters, e.g. healthy 167h with anthracnose 92a) while missing true burst shots at distance 8 (e.g. 134h/135h). I inspected labelled pairs and switched to: 256-bit pHash <= 54, or (64-bit <= 6 and 256-bit <= 70), or consecutive file numbers (gap <= 3) in the same class with 64-bit <= 12. All hashes are taken over the 8 flip/rot90 variants. Result: 50 duplicate groups, 0 mixed-label. The burst rule over-merges a few distinct leaves (e.g. healthy 30h-36h chain); that only costs a little stratification balance and never causes leakage.

- Segmentation (T03) is two-mode, chosen per image by the share of bright non-plant pixels in a 5% border ring (bimodal on real data: field shots < 0.2, card/paper/cloth > 0.6; all turmeric >= 0.28; switch at 0.25). Plain mode: k-means (k<=3) background colours in Lab, colour distance with L down-weighted (shadows), Otsu threshold. Field mode: GrabCut with a centre prior, run with loose and with core-colour seeding, keep the more solid mask. Otsu on saturation was tried first and rejected: blue cloth / grey card backgrounds are more saturated than dull dark leaves.
- Known segmentation limits: two-tone (yellow mosaic) leaves against dark foliage can fragment; frame-filling field leaves lose a thin border; a beige card passes the plant-colour test (GrabCut still separates it).

## NEXT

T04: implement ip/lesions.py (abnormal colour mask inside the leaf, per-crop ranges in config, adaptive to the leaf's own green), synthetic lesion tests, lesion contact sheets (24 diseased + 24 healthy) and healthy vs diseased mean affected-%.
