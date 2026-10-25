# PROGRESS

## Task table

| ID | Task | Status | Evidence pointer |
|----|------|--------|------------------|
| T00 | Bootstrap | DONE | Evidence log 2026-10-24 T00 |
| T01 | Data acquisition + inventory | DONE | Evidence log T01 |
| T02 | Leakage-free split | DONE | Evidence log T02 |
| T03 | Stage 1a: pre-processing + leaf segmentation | DONE | Evidence log T03 |
| T04 | Stage 1b: disease region detection | DONE | Evidence log T04 |
| T05 | Stage 1c: features | DONE | Evidence log T05 |
| T06 | Stage 1d: decisions (rule-based + SVM) | DONE | Evidence log T06 |
| T07 | Stage 2a: CNN data + model | DONE | Evidence log T07 |
| T08 | Stage 2b: CNN training + evaluation | DONE | Evidence log T08 |
| T09 | Comparison | DONE | Evidence log T09 |
| T10 | Unified inference pipeline | DONE | Evidence log T10 |
| T11 | Demo app | DONE | Evidence log T11 |
| T12 | Tests, lint, quality | DONE | Evidence log T12 |
| T13 | Documentation + report | DONE | Evidence log T13 |
| T14 | Final verification | DONE | Evidence log T14 |

## Evidence log

(newest first)

### 2026-10-25 T14 Final verification
- `python scripts/verify_all.py` (full) -> exit 0:
  `lint (ruff) PASS`; `all tests incl. pipeline + app smoke PASS 47 passed`; `split leakage PASS 0 clusters span splits; 1606 images`; `artefacts PASS all present`; `REPORT.md numbers vs sources PASS identical to a fresh render; 498 numbers re-read from sources`; `comparison.md numbers in comparison.csv PASS`; `OVERALL: PASS`.
- `verify_all.py --quick` -> PASS (lint, unit tests without app/pipeline, leakage).
- Checker self-test: editing one number in REPORT.md (1548 -> 1549) and, separately, swapping a number for another valid one (0.935 -> 0.991) both make the report check FAIL; regenerating restores PASS. The second case led to the fresh-render comparison being added.
- Re-ran `make data splits features train-ip compare report`: splits.csv byte-identical (md5 919846308c8edef2ad552c38fe36c8c9), IP metrics identical; only wall-clock timings changed (Stage 1 + decision median 173.8 -> 119.6 ms/image; timing is load-dependent). Comparison and report regenerated from the new timings. The CNN was not retrained in this re-run (MPS is not bit-deterministic; models/cnn_best.pt from T08 is used).

### 2026-10-25 T13 Documentation + report
- `make report` -> `wrote results/REPORT.md (212 lines, 498 recorded numbers)`; deleting REPORT.md + report_numbers.json and re-running `make report` gives a byte-identical file (`cmp` -> identical).
- REPORT.md covers: research question, datasets + per-class split counts, leakage control, capture conditions, methods + pipeline diagram, Stage 1 checks, IP and CNN VAL/TEST tables, comparison table + findings, leakage ablation, per-class error analysis + galleries, measured answer, limitations (weakest classes read from comparison.csv), future work. Every number goes through a recorder that stores its source in results/report_numbers.json.
- Review fixes before acceptance: a typed limitation ("leaf crinkle / powdery mildew are hardest") contradicted the measured per-class table (SVM 1.000 / 0.971) and was replaced by the measured weakest classes; README limitation corrected the same way; plain-background counts now include colour-fallback images (new `bg_mode` column).
- README.md: setup, data download + manual placement, `make all`, app, programmatic use, repo map, limitations (no metric values in README by design).

### 2026-10-25 T12 Tests, lint, quality
- `make test` -> `47 passed` (split integrity, data/inventory, preprocess + segmentation, lesions, features, decisions, CNN smoke, pipeline, Streamlit AppTest).
- `make lint` -> `All checks passed!`
- Dead code removed: pipeline.encode_png, ablation.stage1_matrix, cnn.dataset.diseased_prob (+ its test); unused imports cleaned by ruff.

### 2026-10-25 T11 Demo app
- `streamlit run app/streamlit_app.py` started (built-in browser preview on :8501, no server errors in the log). Screenshots inspected: Analyze tab shows the card `Crop: Black Gram / Prediction: Diseased / Affected Region: 22.8% / CNN Confidence: 100%` for TEST sample 114p.jpg (powdery mildew), the Original -> Segmented Leaf -> Detected Abnormal Regions strip + Grad-CAM, "Image Processing says Diseased (SVM) | CNN says Diseased (confidence 100%)" with an AGREE badge and rule components; Results tab renders comparison.md, comparison.csv and figures.
- `pytest tests/test_app.py` (Streamlit AppTest, headless) -> `1 passed` (no exceptions, four-line card present, two tabs).

### 2026-10-24 T10 Unified inference pipeline
- `pytest tests/test_pipeline.py` -> `3 passed` (synthetic leaf with user crop; 5 real TEST images with `affected_region_pct == features.csv affected_pct` (abs 1e-9) + auto crop via CNN; corrupt bytes / 20x20 image -> InvalidImageError, plain blue card / grey frame -> NotALeafError, unknown crop -> ValueError).
- Example `analyze(<black gram yellow mosaic TEST image>)` printed the four-line card (see app check in T11).

### 2026-10-24 T09 Comparison + leakage ablation
- `python scripts/run_comparison.py` -> identical TEST split (n=239; black gram 150, turmeric 89). Macro-F1 [95% bootstrap CI]: rule-based 0.839 [0.780, 0.892], SVM 0.935 [0.896, 0.968], CNN 0.978 [0.954, 0.995]; ROC-AUC 0.924 / 0.981 / 0.999.
- Per crop macro-F1: black gram rule 0.737, SVM 0.910, CNN 0.960; turmeric rule 0.949, SVM 0.963, CNN 1.000.
- `McNemar SVM vs CNN p = 0.0574` (SVM right/CNN wrong 3, reverse 11); SVM-CNN agreement 0.941, accuracy when agreeing 0.996, 14 disagreements.
- Inference: Stage 1 + decision 173.8 ms/image CPU; CNN 48.2 ms CPU / 35.8 ms MPS.
- Leakage ablation (macro-F1): clean SVM 0.935 / CNN-head 0.984; naive_random 0.931 / 0.984 (5.4% of TEST items have a near-duplicate in TRAIN); augment_then_split 0.974 / 0.991 (94.9% have a sibling in TRAIN) -> inflation +0.039 SVM, +0.007 CNN head.
- `numbers in comparison.md not found in comparison.csv: []`
- Files: results/comparison.csv, comparison.md, comparison_table.md, figures comparison_bars / confusion_matrices / roc_pr_curves / inference_time / errors_gallery / disagreements_gallery (bars, ROC/PR and disagreement gallery opened: SVM misses small dark anthracnose spots and thin turmeric blotch margins; SVM catches faint mildew the CNN misses).

### 2026-10-24 T07 CNN data + model
- `python scripts/train_cnn.py --smoke` -> MobileNetV2 ImageNet weights downloaded (mobilenet_v2-7ebf99e0.pth); `epoch 1 [head] ... (10s)`; `smoke: trained 64 images, 1 epoch on mps, pretrained=True; checkpoint cnn_smoke.pt reloaded (epoch 1)`; 4 predictions printed.
- `pytest tests/test_cnn.py` -> `3 passed` (forward + freeze + checkpoint round trip; dataset/transforms on synthetic images; P(diseased) aggregation).

### 2026-10-24 T07 CNN data + model
- `python scripts/train_cnn.py --smoke` -> MobileNetV2 ImageNet weights downloaded (mobilenet_v2-7ebf99e0.pth); `epoch 1 [head] ... (10s)`; `smoke: trained 64 images, 1 epoch on mps, pretrained=True; checkpoint cnn_smoke.pt reloaded (epoch 1)`; 4 predictions printed.
- `pytest tests/test_cnn.py` -> `3 passed` (forward + freeze + checkpoint round trip; dataset/transforms on synthetic images; P(diseased) aggregation).

### 2026-10-24 T06 Decisions (rule-based + SVM)
- `python scripts/train_ip.py` (train 1125 / val 242 / test 239):
  - SVM grouped 5-fold CV on TRAIN: best `{'svc__C': 30, 'svc__gamma': 0.01}` CV macro-F1 0.934
  - rule_based val: acc 0.913, macro-F1 0.882, AUC 0.914; thresholds from VAL blackgram 0.110, turmeric 0.235
  - rule_based test: `acc 0.883 P 0.903 R 0.944 F1 0.923 macroF1 0.839 AUC 0.924 CM [[43, 18], [10, 168]]`; per-crop macro-F1 blackgram 0.737, turmeric 0.949
  - svm val: acc 0.955, macro-F1 0.942, AUC 0.986
  - svm test: `acc 0.950 P 0.972 R 0.961 F1 0.966 macroF1 0.935 AUC 0.981 CM [[56, 5], [7, 171]]`; per-crop macro-F1 blackgram 0.910, turmeric 0.963
  - inference (sequential, 40 VAL images): Stage 1 median 161 ms + decision 6.3 ms per image
- Saved models/ip_rule.joblib, models/ip_svm.joblib, results/ip_metrics.json, results/ip_predictions.csv.
- `pytest tests/test_decide.py` -> `2 passed` (rule scorer on toy data; end-to-end predict on a synthetic image with the saved models).

### 2026-10-24 T05 Features
- `python scripts/extract_features.py` -> `Wrote data/processed/features.csv: 1606 rows x 57 features in 248s`; `non-finite values: 0; rows with missing features: 0; seg_failed: 0`; stage1 median 603 ms/image under parallel load. Cache: data/processed/cache/features_<config-hash>/.
- results/feature_list.txt lists the 57 features in order + degenerate-case handling.
- Segmentation mode by stratum: blackgram_diseased plain 203 / field 583; blackgram_healthy plain 67 / field 154; turmeric all plain except 1.
- `pytest tests/test_features.py` -> `4 passed` (after fixing the first version of the diseased-vs-healthy test, whose two painted lesions overlapped and were correctly detected as one) (fixed order + finite; no-lesion and failed-segmentation cases finite; synthetic diseased > healthy affected-%; real file one row per image, all finite).

### 2026-10-24 T04 Disease region detection
- `pytest tests/test_lesions.py` -> `9 passed` (painted brown/yellow/dark lesions on paper and field backgrounds: area within 15% of truth and count == 3; healthy synthetic leaf 0 lesions; min-area filter + circularity).
- `python scripts/ip_figures.py --stage lesions` (60 random train+val images per crop x label) -> mean affected-%: blackgram healthy 2.46 vs diseased 16.42; turmeric healthy 1.30 vs diseased 62.84; overall `healthy 1.88 vs diseased 39.63` (results/lesion_check.json).
- Per class mean affected-%: anthracnose 8.55, dry_leaf 78.99, leaf_blotch 40.22, leaf_crinckle 8.15, powdery_mildew 5.72, yellow_mosaic 35.25, healthy (bg) 2.46, healthy_leaf (tu) 1.30.
- results/figures/lesion_samples_{diseased,healthy}.png (24 + 24) opened and inspected: yellow mosaic, anthracnose spots, turmeric blotch and dry/necrotic tissue detected; healthy leaves ~0%. Weak: powdery mildew on dark leaves (faint film, 0.2-1.9%), bright turmeric midribs sometimes flagged as pale.

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

- Lesion colour rules (T04) were calibrated on TRAIN leaf-pixel statistics (30 images per original class): healthy hue median black gram ~50 / turmeric ~41; yellow-mosaic, blotch and dry pixels concentrate at H 20-30, necrosis at H < 20. Rules are relative to each leaf's own green reference. Hysteresis growth (Lab distance to the reference green) recovers soft lesion edges; its floor (38) was chosen on a train grid where the affected-% AUC was flat (bg 0.87-0.88, tu ~0.98) and healthy inflation was the deciding factor. Pale (mildew) rule grid: chroma < 0.65 x leaf chroma and L > leaf L + 15 (mildew 8.6% vs healthy 0.9% pale on train).
- Leaf crinkle is mostly a shape/texture disease (little colour change); its signal is expected to come from GLCM texture and dark fold shading, not abnormal colour.

- Background/label correlation (measured, T05): in black gram, 25.8% of diseased (203/786) vs 30.3% of healthy (67/221) images were taken on a plain card/cloth, so the plain-vs-field background is only weakly tied to the label. All turmeric images are on white paper (no background shortcut available within turmeric, but crop identity is trivially predictable from background).
- Leaf area as a fraction of the frame and the segmentation method are NOT used as features (capture artefacts, not leaf properties); they are kept as metadata columns.

- Rule-based score (T06): four inputs with fixed equal weights (not fitted), each min-max scaled per crop by its TRAIN 1st-99th percentile; directions are the domain hypothesis "more abnormal = more diseased", confirmed on TRAIN medians for all four in both crops. Only the per-crop threshold is learned (VAL macro-F1 grid). Macro-F1 is used for both the rule threshold and the SVM CV because healthy is the minority class.
- SVM score = decision_function (sklearn 1.9 deprecates SVC(probability=True)); it is used for ROC curves. Single SVM for both crops with an is_turmeric indicator, mirroring the single 4-class CNN.

- CNN (T07): single 4-class model (turmeric/blackgram x healthy/diseased); P(diseased) = sum of the two diseased-class probabilities, binary decision at 0.5. Eval transform = resize of the whole image to 224x224 (no centre crop, leaves are often off-centre). Hue jitter kept at 0.02 because colour is the disease signal.

- Comparison (T09): every reported number lives in the long-format results/comparison.csv (section, method, scope, metric, value); comparison.md is rendered from it and checked automatically (no number in the markdown may be missing from the CSV). Bootstrap CIs (1000 resamples) and an exact McNemar test qualify the differences; wording ("higher/lower, CIs overlap") is generated from the numbers.
- Leakage ablation uses a frozen ImageNet MobileNetV2 + logistic-regression head instead of the fine-tuned CNN, because the fine-tuned CNN has already seen clean-TRAIN images that a naive split would place in TEST. Augmented copies exist only in the cache, never in data/raw.

- Pipeline headline prediction (T10): when SVM and CNN agree, that label; when they disagree, the method with the higher measured TEST macro-F1 in results/comparison.csv decides (CNN at present), and the card says the methods disagree. Both individual predictions are always shown.

## NEXT

All tasks T00-T14 DONE and verify_all.py exits 0. Possible follow-ups (out of current scope): disease-type classification, cross-region test data, second CNN seed / EfficientNet-B0 run.

## Phase 2

Prompt: claude_loop_prompt_v2.txt + speed_rules_prompt.txt (speed rules: batch small tasks, targeted tests, full verify_all at phase ends).

| ID | Task | Status | Evidence |
|----|------|--------|----------|
| P2-B00 | Reproduce the bug | DONE | P2 log B00 |
| P2-B01 | Wild / OOD evaluation set | DONE | P2 log B01 |
| P2-B02 | Diagnose | DONE | P2 log B02 |
| P2-B03 | Background-swap + robust augmentation | DONE | P2 log B03 |
| P2-B04 | Hierarchical crop + disease models | DONE | P2 log B04 |
| P2-B05 | Calibration, abstention, OOD | DONE | P2 log B05 |
| P2-B06 | Stage 1 for field images | DONE | P2 log B06 |
| P2-B07 | FastAPI service | DONE | P2 log B07 |
| P2-B08 | Re-evaluate + robustness report | DONE | P2 log B08 |
| P2-F00 | Web scaffold | DONE | P2 log F00 |
| P2-F01 | Real content export | DONE | P2 log F01 |
| P2-F02 | Healing-leaves assets | DONE | P2 log F02 |
| P2-F03 | Shell: nav, smooth scroll, rail, routing | TODO | |
| P2-F04 | 01 Hero | TODO | |
| P2-F05 | 02 Problem | TODO | |
| P2-F06 | 03 Crops | TODO | |
| P2-F07 | 04 Stage 1 scrollytelling | TODO | |
| P2-F08 | 05 CNN | TODO | |
| P2-F09 | 06 Results | TODO | |
| P2-F10 | 07 System diagram | TODO | |
| P2-F11 | 08 Healing leaves + CTAs | TODO | |
| P2-F12 | /analyze page | TODO | |
| P2-F13 | Responsiveness, a11y, performance | TODO | |
| P2-V01 | One-command dev + build | TODO | |
| P2-V02 | Extend verify_all | TODO | |

### Phase 2 evidence log (newest first)

#### 2026-10-25 P2-F01 Real content export
- `python scripts/export_site_data.py` -> `wrote web/public/data/site.json (18 kB); media files: 216`: dataset counts, clean + wild comparison (rule/SVM/CNN), leakage ablation, robustness before/after + targets, calibration, latency; showcase TEST leaves (turmeric Leaf Blotch00009, black gram anthracnose 115a) with every REAL Stage 1 intermediate (original, denoised, Lab a*, colour distance, threshold, leaf mask, lesions, segmented) + params (median/Gaussian 5, CLAHE 2.0, Otsu distance, open/close, min lesion area) + readouts; 16 early (features[3]) + 9 late (features[13]) MobileNetV2 disease-CNN feature maps, Grad-CAM, calibrated P(crop) and P(diseased|crop); crop panels (healthy/diseased TEST photos); heal metadata; hero still (Starr, CC BY 2.0) with leaf outline + midrib/venation anchors from our Stage 1. WebP + AVIF at 640/1280 (+ hero 960/1600/2400).
- Viewed the exported media sheet: intermediates and feature maps are real and legible. Note: Stage 1 reports 38% "affected" on the healthy, dew-covered hero leaf (dew highlights / yellow light) -> the hero shows only the outline and morphology callouts, no disease measurement.
- web/src/content/citations.ts holds the four sourced statistics; all other numbers come from site.json.

#### 2026-10-25 P2-B08 Re-evaluate + robustness report
- `python scripts/evaluate_robust.py` -> results/robustness.json + robustness.md (before v1 / after Phase 2): crop accuracy clean TEST 1.000 / 0.996; wild_test 0.927 / 1.000; accepted-crop precision wild_test 0.927 / 1.000; coverage 1.000 / 0.817 (abstention 0.183, all OOD-flagged); crop ECE wild_test 0.059 / 0.015; disease F1 clean TEST 0.989 / 0.997; disease macro-F1 clean 0.978 / 0.994, wild_test 0.887 / 0.935; SVM clean macro-F1 0.935 / 0.966; rule 0.839 / 0.845; SVM wild 0.903, rule wild 0.772; CNN CPU latency 48 / 208 ms; API p50 449 ms.
- TARGETS: wild_test accepted precision 1.000 >= 0.90 (met); regression image -> turmeric (confident), not black gram (met); clean-TEST disease F1 drop -0.008 <= 0.02 (met).
- Regression image after: crop turmeric P=1.000; Stage 1 36 lesion regions, 8.7% affected; SVM + rule diseased; disease CNN healthy (P(diseased) 0.117) -> methods disagree, shown with a DISAGREE badge; nothing tuned on this image. results/figures/bug_repro_after.png viewed (main leaf segmented, specks outlined).
- tests/test_regressions.py xfail removed -> passes. v2 CNN files written in v1 formats (results/cnn_predictions.csv, cnn_metrics.json, cnn_train_info.json; v1 originals in results/v1_snapshot/). Comparison re-run (CNN 0.994, SVM 0.966, rule 0.845 macro-F1; McNemar p 0.0625) and REPORT regenerated with a robustness section.
- `make test` -> `66 passed`; `python scripts/verify_all.py` (end of Phase A) -> OVERALL: PASS (545 report numbers re-read).

#### 2026-10-25 P2-B07 FastAPI service
- api/main.py (+ api/schemas.py, pydantic, schema_version "1") over src/leafdx/service.py: GET /api/health, GET /api/meta, POST /api/analyze (multipart file, crop=auto|turmeric|blackgram, roi_x/roi_y or roi_box); 10 MB limit, content-type + decode validation, EXIF orientation, CORS for Vite (5173/4173), structured error JSON (400 invalid_image/bad_crop/bad_roi, 413 too_large, 415 unsupported_type, 422 not_a_plant, 500 internal_error); images as data:image/webp;base64 <= 768 px.
- `pytest tests/test_api.py` -> `6 passed` (health/meta; auto crop; user-selected crop + ROI click -> user_roi_watershed; uncertain path (tau forced above 1); non-image / corrupt / bad crop / oversize / bad roi; not-a-plant).
- `python scripts/api_latency.py` (12 TEST images, warm): p50 449 ms (MPS for embeddings), CPU-only (LEAFDX_DEVICE=cpu) p50 599 ms; server stages p50: crop+OOD+disease 139 ms, Stage 1 58 ms, IP decision 7 ms, Grad-CAM 191 ms (results/api_latency.json).
- Streamlit app moved to the hierarchical output (crop status, "It's turmeric / It's black gram" re-run on uncertain, no 4-class table); `pytest tests/test_app.py` -> `1 passed`.
- IP decisions retrained on the new Stage 1 features first (features re-extracted, 111 s): SVM test macro-F1 0.966 (v1 0.935), rule 0.845 (v1 0.839); SVM now stores Platt scaling fitted on VAL.

#### 2026-10-25 P2-B05 Calibration, abstention, OOD
- `python scripts/calibrate.py` -> results/calibration.json + models/calibration.json: T_crop 0.442, T_disease 0.682 (both < 1: TTA averaging made raw probabilities under-confident); OOD = DINOv2 kNN (k=10) cosine distance, threshold 0.346 (99th pct of clean VAL); tau_crop 0.500 (accepted precision on wild_val is already 1.000 at every tau, so abstention comes from the OOD gate).
- Crop ECE before -> after: clean test 0.018 -> 0.007, wild_val 0.030 -> 0.014, wild_test 0.034 -> 0.015. Disease ECE: clean test 0.018 -> 0.007, wild_test 0.076 -> 0.054.
- wild_test: crop accuracy 1.000, coverage 0.817, accepted precision 1.000, abstention 0.183 (all OOD-flagged).
- Regression image: calibrated P(turmeric) 1.000, OOD 0.255 < 0.346 -> status confident (was confident black gram 0.73). Disease head calls it healthy (disease ECE 0.88 on this one image): reported, not tuned (R2).
- `pytest tests/test_robust.py` -> `6 passed` (confident / uncertain-low-prob / uncertain-OOD / user_selected paths, kNN distance, temperature fit).

#### 2026-10-25 P2-B04 Hierarchical crop + disease models
- `python scripts/train_robust.py` -> results/crop_model_selection.csv (wild_val crop accuracy, CPU ms/image with TTA after warm-up): DINOv2 ViT-S/14 head 0.976 / 154 ms; OpenCLIP ViT-B/32 head 0.976 / 117 ms; OpenCLIP zero-shot 0.880 / 120 ms; **OpenCLIP head x zero-shot prior 0.988 / 113 ms (selected)**; EfficientNet-B0 fine-tuned with B03 aug 0.964 / 420 ms. (v1 4-class CNN wild_val crop accuracy: 0.916.)
- Disease CNN (MobileNetV2, one diseased logit per crop, B03 augmentation), 16 epochs / 3.6 min on MPS, best epoch 11; with TTA: clean test macro-F1 0.994 (v1 0.978), wild_val 0.968, wild_test 0.935. models/MODEL_CARD.md generated; models/crop_model.joblib, models/disease_cnn.pt saved.
- Two runs fixed on the way: float64 sample weights rejected by MPS; first latency numbers included one-time model loading on CPU (now warmed up before timing).

#### 2026-10-25 P2-F02 Healing-leaves assets
- `python scripts/make_heal_assets.py` -> `21x9: affected 23.9% SSIM(non-lesion) 1.0000`, `9x16: affected 22.8% SSIM(non-lesion) 1.0000` (0.99998); web/public/media/heal/: diseased + healed WebP/AVIF (1280/2520 and 720/1080 widths, 36-146 kB), lesion + leaf masks at half resolution, heal_<variant>.json with affected_pct_initial.
- Leaves (TRAIN/VAL only): black gram anthracnose 111a + 124a, turmeric leaf blotch 00031, black gram yellow mosaic 183y; cut out with our Stage 1 leaf mask on a darkened, blurred field backdrop (healthy 150h).
- Healing recipe (spec): necrotic (brown/dark) -> Telea inpainting on a leaf-extended image (no background sampling) + borrowed surface grain; large lesions (>(60 px)^2 at processing scale) cloned along the leaf's principal (vein) axis with low-frequency brightness matched; chlorotic/pale -> a*/b* moved to the leaf's healthy median with lightness lifted 60% (veins kept); feathered blend.
- Preview results/figures/heal_preview.jpg viewed after each of 5 iterations: v1 pulled white paper into edge lesions and smeared large blotches (rejected); final: spots lift away naturally on the anthracnose leaves, the turmeric tip regrows veins. Caveats: at full zoom the turmeric tip shows faint repetition from the clone shifts; the yellow-mosaic leaf changes little because our mask finds only part of its pale patches (by design only detected lesions heal).

#### 2026-10-25 P2-F00 Web scaffold
- web/: React 18.3.1 + TypeScript 5.9 (strict, noUncheckedIndexedAccess) + Vite 6.4 + Tailwind 3.4 + lucide-react, motion, GSAP 3.15, Lenis 1.3, react-router 6.30; ESLint 9 flat config + Prettier; Playwright 1.64 + @axe-core/playwright; Vite proxy /api -> 127.0.0.1:8000; alias @ -> src; tokens (Section 5.3) as CSS variables + Tailwind theme; Inter + Playfair Display italic + JetBrains Mono; Appendix A keyframes.
- `npm run typecheck` / `npm run lint` clean; `npm run build` -> `index-*.js 161.26 kB (gzip 52.83 kB)`, built in 1.13 s; routes "/" and "/analyze" render.
- web/src/content/citations.ts: the four Problem statistics traced to sources (Chopada & Rakholiya 2015 citing Panja et al. 2000; Vadivel et al. 2019 EJPB citing Nene 1972; Spices Board brochure (~80%, proposal states >75%); ANGRAU Blackgram Outlook 2022 (~29% of pulse area)).

#### 2026-10-25 P2-B06 Stage 1 for field images
- Segmentation: foliage mask (Excess-Green + Otsu OR green a* with saturation gate) always returned; plain mode now requires a SMOOTH background (median L* gradient of background ring <= 13; measured: training plain backgrounds median 2.6-3.8, p90 11.4; wild field scenes p10 12.6+) or a very uniform border (<= 5, studio wall); GrabCut collapse in field mode now falls back to a marker-controlled watershed (leaf marker = core disc at the centre or the user's click/box, background markers = non-foliage + colour-far pixels); user ROI (click or box) segments by watershed directly.
- Lesions: black/white top-hat specks at 1x + 0.5x on a lightly smoothed image (the 5x5 median of pre-processing erased few-pixel specks), adaptive min area max(8 px, 0.0002 x leaf), opening 2 / closing 3.
- `pytest tests/test_lesions.py -k sixty` -> 60 painted speckles (54 distinct after overlap): area recall 0.859, count 52 vs 54; v1 segmentation/lesion/feature tests pass (`26 passed`).
- `python scripts/field_seg_figure.py` -> results/figures/field_segmentation.png (20 wild field scenes: original | foliage | chosen leaf | lesions), viewed in two halves: main leaf/plant chosen in 16/20. Misses: #0 soil enclosed between leaves kept, #7 house wall + sky make a smooth "plain" border (whole frame), #9 colour fallback covers the frame, #12 watershed region mixes leaves and background.
- Bug image Stage 1: before 1 lesion / 5.7% on a central blob; after `field_grabcut+foliage_watershed`, coverage 0.596 (main leaf + part of a neighbour), 36 lesions, 8.7% affected.
- Side effect on training data (ip_figures --stage lesions, 60/stratum): mean affected healthy bg 2.46 -> 3.93, healthy tu 1.30 -> 2.72, diseased bg 16.42 -> 19.15, tu 62.84 -> 64.30; mean lesion count on healthy black gram 2.97 -> 11.73 (speck detector also fires on leaf texture). Decision features are re-learned in P2-B08.

#### 2026-10-25 P2-B03 Background-swap + robust augmentation
- `python scripts/augment_preview.py` -> `banks: 1125 leaf cut-outs (turmeric 420, black gram 705), 1125 backgrounds` (TRAIN only, cached npz keyed on config + paths); results/figures/augment_samples.png viewed: turmeric leaves on soil / procedural foliage / inpainted black gram field backgrounds, black gram leaves on white paper and grey, same-crop collages, blur/JPEG/jitter visible.
- `pytest tests/test_augment.py` -> `4 passed` (banks and SwapDataset raise LeakageError on non-TRAIN rows; real banks contain only TRAIN paths; swap keeps canvas + target leaf; paste clips at borders).

#### 2026-10-25 P2-B02 Diagnose
- `python scripts/diagnose.py` -> results/diagnosis.md + diagnosis.json. v1 crop accuracy: clean TEST 1.000, wild_val 0.916, wild_test 0.927; ALL wild crop errors are turmeric field scenes (iNaturalist 0.500, Commons 0.722); turmeric on white 1.000 (n=55), black gram on white 1.000 (n=70).
- Probes (clean TEST): background-only (mean-colour silhouette) 0.954, background with leaf box hidden 0.841, leaf-only 1.000, majority 0.628 -> (a) shortcut CONFIRMED (foliage surroundings read as black gram). (b) inconclusive (no crop errors on labelled wild images). (c) PARTLY: crop ECE 0.004 clean -> 0.078 wild_val, wrong crop calls mean confidence 0.754, no abstention. (d) PARTLY: field-scene median leaf coverage 0.264 vs 0.380 clean, 17.5% under 0.15.
- Viewed figures: diag_crop_gradcam.png (on turmeric field scenes predicted black gram 85% / 69%, the crop Grad-CAM sits on grass and leaf litter, not the plant; on leaves on white it sits on the leaf), diag_probes.png, diag_reliability.png.

#### 2026-10-25 P2-B01 Wild / OOD evaluation set
- `python scripts/build_wild_set.py` -> kept 165 (turmeric 89, black gram 76): Mendeley jtttfbx342 turmeric (Bangladesh, CC BY 4.0) 55 originals, Mendeley z55yrbmn2d black gram (Bangladesh, CC BY 4.0) 70 originals, iNaturalist 44 (CC0/CC-BY/CC-BY-SA), Wikimedia Commons 25; 29 excluded after visual vetting (flowers/rhizomes/no leaf visible); `near-duplicates of training images dropped: 0`; 0 clusters span wild_val/wild_test; SOURCES.csv has url, author, licence for every file.
- Split (val/test): bg diseased 20/20, bg healthy 15/15, bg unknown 3/3, tu diseased 15/15, tu healthy 13/12, tu unknown 17/17.
- Viewed contact sheets: the two Mendeley sets are single leaves on WHITE (black gram cut out on white, turmeric on white paper) -> a direct test of a "white background = turmeric" shortcut. Field scenes are the iNaturalist/Commons images (turmeric 34, black gram 6, disease unlabelled).

#### 2026-10-25 P2-B00 Reproduce the bug
- User image found in ~/Downloads ("images (1).jpeg", in-field turmeric with brown speckles), copied to data/wild/user_reported/turmeric_google_01.jpg and tests/fixtures/regressions/ (gitignored: Google image, test-only).
- `python scripts/bug_repro.py before` -> 4-class probs turmeric_healthy 0.030, turmeric_diseased 0.241, blackgram_healthy 0.021, blackgram_diseased 0.709; CNN Diseased (0.949); affected 5.7%, 1 lesion; rule 0.153 vs 0.110 -> Diseased; SVM Diseased. Saved results/bug_repro_before.json + figures/bug_repro_before.png (viewed: segmented leaf is a small central blob; one lesion outlined while dozens of specks are visible; Grad-CAM on the speckled blade).
- tests/test_regressions.py -> `1 xfailed` (strict xfail until P2-B08).

### Phase 2 decisions

- Appendix B (hero prompt) is empty -> Section 8 fallback hero.

- Wild set sources: iNaturalist has only 4 CC-licensed Vigna mungo observations, so two independent CC BY 4.0 Mendeley datasets (Bangladesh) supply labelled cross-dataset images; iNaturalist + Commons supply field scenes labelled "unknown". Images fetched from a remote zip by HTTP range (data.HttpRangeFile) to avoid a 1.4 GB download. Downloaded images are gitignored; `scripts/build_wild_set.py` rebuilds them.

- Diagnosis probe added beyond the spec: "background with leaf bounding box hidden", because the spec's mean-colour fill keeps the leaf silhouette (shape leaks crop). Verdict for (a) uses the stricter probe.

- P2-B06: a superpixel (SLIC + RAG boundary merge) approach was tried first and rejected after viewing: it fragmented field scenes and was worse than v1 GrabCut on soil backgrounds. The final change is targeted: v1 behaviour is kept unless the background is textured (mode switch) or GrabCut collapses (watershed fallback).

- Phase 2 "current system" = Phase 2 Stage 1 + retrained rule/SVM + hierarchical CNN. comparison.csv / REPORT.md describe it; the v1 study's files are preserved verbatim in results/v1_snapshot/ (from commit 0aacce8) and feed the "before" column of robustness.md.

## NEXT (Phase 2)

P2-F01: run scripts/export_site_data.py, then build landing sections F03-F11 and view screenshots.
