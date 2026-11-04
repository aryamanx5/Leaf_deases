# Plan: Phase 4, train on more field data (pre-registered)

Written 2026-11-04, before any Phase 4 data was split or any Phase 4 model was trained. The rules below are fixed
now and are not changed after looking at results; any deviation is logged in PROGRESS.md with its reason.

## Why

Every Phase 3 model was trained on lab-style photos only (turmeric on white paper, black gram on cards / cloth).
On field photos the production model drops from 1.000 (clean TEST) to 0.917 macro-F1 (wild_ext), and black gram
field leaves are the weakest part (0.895). Openly licensed field photos of both crops exist that no model has been
trained on. Phase 4 asks: **does adding field photos to training close that gap, measured on field photos that no
model has seen?**

## What stays frozen

* The Phase 3 / field-validated production bundle (`models/manifest.json` at commit 9ade0ed) and every number in
  `results/final_report.md`. The paper's main result is unchanged by Phase 4. The Phase 4 bundle is written to
  `models/manifest_p4.json` and becomes production only under the switch rule below.
* `data/wild` and `data/wild_ext` stay **test only**: no Phase 4 image shares a file, a field scene or a capture
  minute with them, and none of them is used to fit, select or calibrate anything.
  Caveat stated up front: Phase 4 trains on *other images from the same sources* as wild / wild_ext, so after
  Phase 4 those two sets measure "new images, same sources", not "new sources". Phase 3 numbers keep the stronger
  claim.

## Data: the Phase 4 pool (`data/p4`, all CC BY 4.0)

| Source | Crop | Labels | Content |
|---|---|---|---|
| `data/field_val` (Phase 3 validation split; its role ends here) | both + other | healthy / diseased; other species | 911 leaves + 158 other-species leaves |
| Mendeley z55yrbmn2d v3, originals not in wild / wild_ext / field_val (insect class excluded) | black gram | healthy / diseased | about 550 |
| Mendeley 45djgf3p96 v2, leaf boxes (LA healthy, LC yellow mosaic; LB flea beetle excluded) from field scenes used by neither wild_ext nor field_val | black gram | healthy / diseased | every box with min side >= 128 px |
| PhilMedic, Mendeley tsvdyhbphs v1 (Philippines), class 15 Curcuma longa | turmeric | crop only (not disease-labelled) | 100 |
| PhilMedic, other 38 species | other | other plant | 4 per species |

Turmeric disease-labelled field data stays small (only the field_val turmeric leaves): every open turmeric leaf
dataset found (Mendeley g46dvrcvwn, jtttfbx342; Kaggle and Hugging Face copies of them) is already fully used.

Leakage rules: pHash near-duplicate check against train / val / test / wild / wild_ext; z55yrbmn2d grouped by
capture minute, 45djgf3p96 by field scene, PhilMedic by leaf (front and back photo of one leaf share a group);
a `verify_all` check that the Phase 4 pool shares no image, scene or capture minute with wild / wild_ext.

## Split of the pool (fixed now)

Grouped split (groups as above, merged with pHash clusters), stratified by crop x label x source, seed 4:
**p4_train 60 % / p4_val 20 % / p4_test 20 %**. `p4_test` is the held-out field test for later: it is read
exactly once, by the evaluation script, after the selection below is final.

## Training and selection (fixed now)

1. Training rows: clean TRAIN (+ its background-swap views for `swap` variants) **plus p4_train**. Crop-only and
   other-species rows train the crop head and the OOD reference only, never a disease head.
2. Candidates: every frozen-backbone / feature configuration of Phase 3 (B_svm, B2_lgbm, D_dinov2, E_openclip,
   E_bioclip2, F_fusion x 3 backbones; pooled / percrop; orig / swap), re-fitted on the rows above. Fine-tuned CNN
   arms (C1-C3) are not retrained in this phase.
3. Head choice inside a configuration: mean of p4_val macro-F1 and clean VAL macro-F1 (both must stay good).
4. Configuration choice: highest p4_val pooled macro-F1; tie rule as in Phase 3 re-selection (any candidate whose
   point estimate reaches the best one's grouped-bootstrap lower 95 % bound counts as tied; the fastest wins).
5. Calibration on p4_val: temperature, LAC conformal thresholds (crop and disease, alpha 0.10), kNN OOD threshold
   (95 % of p4_val target leaves kept; OOD reference = Phase 3 reference + p4_train target leaves).
   Quality gate unchanged.
6. Crop head: BioCLIP 2 logistic head re-fitted on clean TRAIN + p4_train target rows (incl. PhilMedic turmeric).

## Evaluation (once) and switch rule (fixed now)

Scored once on clean TEST, p4_test, wild and wild_ext: Phase 4 bundle vs current production, macro-F1 with
grouped-bootstrap CIs, paired-bootstrap difference and exact McNemar.

**Switch to Phase 4 in `models/manifest.json` only if** the paired macro-F1 difference on p4_test has a 95 % CI
lower bound > 0 **and** the clean TEST macro-F1 drops by no more than 0.02. Otherwise the current production stays and
the result is reported. Either way the old bundle is kept as `models/manifest_field.json`.

## Known limits (stated up front)

* All disease-labelled field data are from Bangladesh (the same few sources as wild / wild_ext).
* Turmeric field training data are small (about 100 leaves in p4_train); turmeric gains, if any, will be uncertain.
* Labels are the dataset authors'; no agronomist re-check.
