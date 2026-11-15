# Plan: Phase 5, sharper answers and fewer needless refusals (pre-registered)

Written 2026-11-15, before any Phase 5 number was computed. The rules below are fixed now; any deviation is logged
in PROGRESS.md with its reason.

## Why

The deployed bundle (Phase 4, `models/manifest.json`) refuses leaves it would have got right. Seen on the site: a
clean-test turmeric leaf with leaf blotch came back "Not sure" although the disease model was about 97 % confident
and correct. Two causes, both visible in `models/manifest.json` and `results/p4/selection.json` without touching a
test set:

1. **The refusal thresholds are tighter than the models need.** Phase 4 calibrated LAC conformal thresholds at
   alpha = 0.10. With a disease model that is right about 98 % of the time on p4_val, a 90 % coverage target can only
   be met by returning the *empty* set for the least confident tenth of the leaves, right or wrong. The resulting
   thresholds: answer only when the disease probability is at least 0.967 and the crop probability at least 0.994.
   On p4_test that refused 15 % of the leaves while every answered leaf was right.
2. **One of several equally good models was deployed.** Phase 4's tie rule picked the fastest of 24 statistically
   tied configurations (one DINOv2 head). Averaging tied models built on different backbones is a standard way to get
   probabilities that sit further from 0.5 on the leaves they get right, without retraining anything.

Phase 5 asks: **can the system answer more leaves, with more confidence, while staying as accurate on the leaves it
answers?** No model is retrained. No new data is used.

## What stays frozen

* Every number in `results/final_report.md` (Phase 3) and `results/p4_report.md` (Phase 4).
* The Phase 4 bundle is kept as `models/manifest_p4.json`. The Phase 5 bundle is written to
  `models/manifest_p5.json` and becomes production only under the switch rule below.
* The kNN out-of-distribution gate and the image-quality gate: unchanged (same reference, same thresholds).
* The crop model: unchanged. Only its refusal threshold is re-derived.

## Data used for every choice

p4_val (333 disease-labelled field leaves, plus its crop-only and other-species rows for the crop threshold) and
clean VAL. These are the selection and calibration sets of Phase 4. p4_test, clean TEST, data/wild and
data/wild_ext are not read by the selection script.

## Candidates (fixed list; all are Phase 4 fits read from `.cache/p4configs`, nothing is re-fitted)

| Id | Disease model |
|---|---|
| M0 | the deployed head, `D_dinov2[pooled,orig]` |
| M1 | mean of the temperature-scaled logits of the three embedding heads `D_dinov2`, `E_openclip`, `E_bioclip2`, each `[pooled,orig]` |
| M2 | M1 plus the three fusion heads `F_fusion_{dinov2,openclip,bioclip2}[pooled,orig]` (six members; the 57 Stage 1 features they need are computed for every request anyway) |
| M3 | `D_dinov2` + `E_bioclip2`, `[pooled,orig]` (the two backbones the OOD gate and the crop model already embed) |

Each member gets its own temperature on p4_val; the ensemble logit is the plain mean of the members' scaled logits
(no learned weights); one more temperature is fitted on p4_val for the ensemble.

## Selection (fixed now)

1. A candidate is admissible if its p4_val macro-F1 is not below M0's and its clean VAL macro-F1 is not below M0's
   by more than 0.01.
2. Among admissible candidates the lowest p4_val negative log-likelihood (after temperature scaling) wins: it is the
   proper score for "how much probability does the model put on the right answer". Candidates within 0.005 nats of
   the best count as tied and the one with the fewest members wins. If only M0 is admissible, M0 stays.

## Refusal thresholds (fixed now)

Selective prediction with a stated risk target replaces "LAC set at alpha 0.10 must be a single label".

* **Disease.** Answer when the confidence max(p, 1 - p) is at least tau_d. tau_d is the lowest value in
  [0.70, current threshold] for which, among p4_val leaves at or above it, the one-sided 95 % Clopper-Pearson upper
  bound on the error rate is at most 2 %. The floor 0.70 is the site's own "Low" cut-off: the system never gives a
  firm answer where the page would call it a guess. If no such value exists below the current threshold, the current
  threshold stays.
* **Crop.** Answer when the crop confidence is at least tau_c. tau_c is the lowest value in [0.50, current threshold]
  for which (a) the same bound on the crop error among answered p4_val target leaves is at most 1 % and (b) the share
  of p4_val other-species leaves refused (OOD gate or crop gate) is not below its current value. If no such value
  exists, the current threshold stays.

The thresholds are stored in the manifest in the existing fields (`disease_q_lac` = 1 - tau_d, `crop_q_lac` =
1 - tau_c), so the inference code is unchanged; the manifest records how they were derived.

## Evaluation and switch rule (fixed now)

`scripts/p5_eval.py` scores the deployed bundle ("old") and the Phase 5 bundle ("new") on clean TEST, p4_test, wild
and wild_ext: macro-F1 with grouped-bootstrap CIs, paired difference, exact McNemar; and on p4_test the trust
figures (share answered, accuracy on answered, other-species refusal), mean confidence on right and on wrong answers,
and calibration error.

**Switch to Phase 5 in `models/manifest.json` only if all of these hold on p4_test / clean TEST:**

1. macro-F1 (new - old) on p4_test: point estimate at least -0.005 and 95 % CI lower bound above -0.02;
2. share of p4_test leaves answered is higher than old;
3. end-to-end accuracy on answered p4_test leaves is at least 0.97;
4. clean TEST macro-F1 drops by no more than 0.02;
5. other-species refusal on p4_test is not lower than old by more than one leaf.

Otherwise the deployed bundle stays and the result is reported.

## Known limits (stated up front)

* **p4_test has been read before.** Phase 4 scored it once to decide its own switch. Phase 5 reads it a second time,
  for a different, pre-registered decision; nothing is fitted or selected on it. A second look is weaker evidence
  than a first: the Phase 5 numbers are reported with that caveat, and Phase 4's remain the once-only result.
* Looser thresholds trade refusals for answers. Phase 4 answered 85 % of p4_test leaves and all of those answers were
  right; Phase 5 will answer more and can be wrong on some. The risk target above (at most 2 % wrong among answered,
  95 % bound on p4_val) is the price that is accepted, and the measured rate is reported.
* An ensemble needs every member's backbone: with M1 or M2 the OpenCLIP embedding is required for the production
  answer (the API already computes it when all arms are requested).
* Same data limits as Phase 4: all disease-labelled field data are from Bangladesh; turmeric field leaves are few.
