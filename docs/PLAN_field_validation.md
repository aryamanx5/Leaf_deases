# Plan: re-select the production model on a field validation split (pre-registered)

Written 2026-10-30, before any field-validation result exists. The selection rule below is fixed now; it is not
changed after looking at results. Any deviation is logged in PROGRESS.md with its reason.

## Why

Phase 3 chose the production disease model (F_fusion, BioCLIP 2 + 57 features) on clean VAL + VAL-swap: lab-style
photos plus synthetic backgrounds. On the extended field test set (1199 leaves) four arms score higher (EfficientNet-B0,
stack, OpenCLIP head, ConvNeXt-T), and disease conformal coverage drops to 0.69 against a 0.90 target. Choosing a
model on the test set would make its numbers optimistic, so a separate field VALIDATION split is needed.

## Data: `field_val` (new) vs the frozen test sets

| Set | Role | Content |
|---|---|---|
| clean train / val / test | as before | unchanged |
| `data/wild` (165) and `data/wild_ext` (1529) | **frozen TEST**: final report only | unchanged; file hashes recorded in a test manifest before any field_val result |
| `data/field_val` (new) | **VALIDATION**: selection, temperatures, conformal and OOD thresholds, stacker | images never in any other set |

Planned `field_val` content (all CC BY 4.0, same licences as now):

* turmeric, jtttfbx342 unused originals: about 250 (healthy ~80, blotch ~100, leaf spot ~65; everything left);
* black gram, z55yrbmn2d unused originals: 100 per class (healthy, Cercospora, leaf crinkle, yellow mosaic);
* black gram field leaves, 45djgf3p96: box crops (healthy / yellow mosaic, min side 128 px) only from the 202
  scenes that contribute nothing to wild_ext, so no field scene is split between validation and test;
* other species, 89rfgtxbdc: about 180 unused images (negatives for the OOD threshold and the "other" refusal rate).

Leakage rules: pHash near-duplicate check against train / val / test / wild / wild_ext (must be 0 shared groups);
45djgf3p96 grouped by scene and z55yrbmn2d by capture minute; a `verify_all` check that field_val and the test sets
share no image, scene or capture group.

## Selection rule (fixed now)

1. Candidates: the 39 existing configurations (A-F, both regimes and data variants, C1-C3 swap / plain) plus a stack
   re-fitted on field_val. Base models are NOT retrained: field_val only selects and calibrates.
2. Primary score: macro-F1 on field_val, pooled; per-crop macro-F1 reported.
3. Tie rule: if the best candidate's grouped-bootstrap macro-F1 CI on field_val overlaps another candidate's point
   estimate, pick the one with lower CPU ms/image (simpler / faster first), as the add-on's decision default says.
4. The stack competes only if its field_val score comes from grouped 5-fold cross-fitting on field_val (each fold's
   meta-learner never sees that fold).
5. Calibration on field_val (grouped 5-fold check of stability): temperature per chosen model; LAC conformal
   thresholds for crop and disease at alpha 0.10; kNN OOD threshold = the quantile of field_val target distances that
   keeps 95 % of them, reported together with the refusal rate on field_val "other" leaves.
6. The quality gate thresholds stay as they are (not tuned).
7. Crop model: kept (bioclip2_head, 0.997 on wild_ext) unless field_val crop accuracy is below 0.98, then the crop
   candidates are re-ranked by the same rule.

## Evaluation (once, after selection is final)

* New vs old production on wild_ext and wild: macro-F1 with grouped bootstrap CIs, paired-bootstrap difference CI and
  exact McNemar. The new model is called "better" only if the paired CI excludes 0 on wild_ext.
* Trust layer on wild_ext with the new thresholds: conformal coverage (target 0.90), answered share, accuracy on
  answered, other-species refusal rate.
* Leave-one-source-out sanity check: re-run the selection with each field_val source removed; report if the winner
  changes (selection that depends on one source is flagged).
* The production switch happens only if the new model is not worse on wild_ext (paired CI upper bound >= 0); otherwise
  F_fusion stays and the result is reported.

## Deliverables

* `scripts/build_field_val.py`, `scripts/p3_reselect.py` (selection + calibration on field_val only),
  `scripts/p3_wild_ext.py` re-run with the new bundle; `models/manifest.json` updated only on a switch, old manifest
  kept as `models/manifest_phase3.json`.
* `results/reselection.json`; a "Field-validated re-selection" section in `results/final_report.md` (generated);
  PROGRESS.md log; `verify_all` disjointness check. Everything stays in the same commit history on its own branch.

## Known limits (stated up front)

* field_val, wild and wild_ext share the same few Bangladeshi sources: re-selection may favour models that suit those
  photographers and devices. The leave-one-source-out check measures this, it does not remove it.
* Turmeric validation is small (~250 leaves); its per-crop numbers will have wide intervals.
* Labels remain the dataset authors'; no agronomist re-check. A truly independent final test still needs new photos
  from another region or the user's own fields, labelled by an expert.
