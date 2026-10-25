# Patra UI upgrade — loop progress

Loop prompt: `../patra_ui_loop_prompt.txt`. One task per iteration. A box is ticked only when its acceptance passed.

## Task list

- [x] T0  Baseline: "before" screenshots of landing + /analyze at 390 and 1440; typecheck, lint, pytest, e2e counts
- [x] T4  More samples (data first, so the UI has material)
- [x] T3  Code + data panel (+ API endpoint + pytest)
- [x] T1  Plain-language layer
- [x] T2  Front-end redesign of /analyze (assemble T1, T3, T4 into the new layout)
- [x] TB  Hero farmer girl (section 5)
- [x] TF  Final audit (section 8)

## Environment notes (read before each iteration)

- Another copy of the app was already running on the configured ports when the loop started (uvicorn on 8010,
  vite on 5180, both started 2026-10-25 14:51 from the same launch configs). So `preview_start` moved
  patra-web to **5181** and patra-api to **8011**. The web dev server's `/api` proxy still points to **8010**
  (vite.config.ts default), i.e. the older uvicorn, which has **no --reload**.
  **Consequence for T3:** after adding `GET /api/code`, the 8010 process will not know it. Either restart that
  uvicorn, or start the web server with `VITE_API_PROXY=http://127.0.0.1:8011`. The processes were left alone
  because they may belong to another session.
  **Update (T3):** stopping the stale uvicorn (pid 11920, started by another Claude preview) was refused by the
  permission classifier, so it is still running the old code. **User action needed: restart the API on 8010**
  (or stop it, so `preview_start patra-api` takes 8010). Until then:
  - e2e: run with `E2E_API_URL=http://127.0.0.1:8011` (new opt-in in `e2e/analyze.spec.ts`; it routes `/api` to
    that server). Without it, the new "How this was calculated" test fails against the old API: no `trace`,
    and `/api/code` returns 404.
  - Screenshot scripts route `/api/**` to 8011 the same way.
- e2e runs against the dev server via `E2E_BASE_URL=http://localhost:5181`.
- Screenshot helper used for T0: a one-off Playwright script (copy kept at the session scratchpad, not in the
  repo); `web/e2e/shots.mjs` does the same for landing sections only.

## Iteration log

### Iteration 1 — T0 baseline (2026-10-25)

Screenshots: `web/test-results/ui-loop/before/`
- `landing-{1440,390}-<section>-05.png` for problem, crops, pipeline, cnn, results, system, run
  (section scrolled to 50 % of its sticky span); hero `field` at 0 / 0.5 / 1.
- `analyze-{1440,390}-empty.png` and `analyze-{1440,390}-result.png` (full page; result = first sample chip,
  "Turmeric · leaf blotch").
- Console errors / HTTP >= 400 during the capture: none at either width.

What I saw:
- Hero: full-bleed turmeric leaf photo, mono eyebrow, "Read the leaf" Playfair italic + "before the harvest
  does", paragraph bottom-left, Starr credit line, Scroll cue, nav pill, "Run the model" button, "01 FIELD"
  section rail on the right.
- /analyze (1440): two columns. Left: preview, upload zone, 4 sample chips, crop segmented control, "Point at
  the leaf" checkbox. Right: 8-step pipeline strip, plain result card (crop / prediction / affected region /
  CNN confidence), "Image processing vs CNN" panel with rule components and SVM/CNN bars, AGREE badge.
  Below: Evidence (4 images, captions, colour legend), Measurements table with GLCM rows and a "Download
  report (JSON)" button, request id line, disclaimer.
  Note: jargon visible today (CNN, SVM, GLCM, OOD, Grad-CAM) with no explanation, which is what T1 fixes.
- /analyze (390): stacked; chips wrap one per line; the empty state is a plain bordered text box.

Baseline checks: see "Baseline counts" below.

## Baseline counts

All green on 2026-10-25, before any change:

| Check | Result |
|---|---|
| `npm --prefix web run typecheck` | pass (exit 0) |
| `npm --prefix web run lint` (max-warnings 0) | pass (exit 0) |
| `.venv/bin/python -m pytest -q` | **66 passed**, 3 warnings (xFormers notices) |
| `npx playwright test` (E2E_BASE_URL=5181) | **9 passed**: analyze ×4, heal ×1, a11y ×3, perf ×1 |

Note for TB/TF: `web/e2e/perf.spec.ts` checks only "no long tasks while scrolling" (baseline: 0 long tasks
over 15,358 px). It does not compute a Lighthouse score, so the "before" score was recorded
separately below.

### Lighthouse "before" (recorded in T0 so TB has a reference)

Production build (`vite build` into the scratchpad, `vite preview` on port 4199), landing `/`, performance only,
Lighthouse defaults (mobile emulation, simulated throttling), 3 runs:

| Run | Perf | LCP | CLS | TBT |
|---|---|---|---|---|
| 1 | 79 | 4204 ms | 0.000 | 8 ms |
| 2 | 84 | 4204 ms | 0.000 | 0 ms |
| 3 | 84 | 4204 ms | 0.000 | 0 ms |

**Baseline = 84 (median).** TB must keep it at ≥ 79 with CLS 0, measured the same way.
JSON report: `web/test-results/ui-loop/before/lighthouse-landing.json`.

### Iteration 2 — T4 more samples (2026-10-25)

**Data** (`scripts/export_site_data.py`)
- `samples()` keeps its name and output shape; selection moved to a new `sample_picks(cfg)`.
- 3 photos per class × 8 classes = 24 from `split == "test"`, plus 4 labelled photos from the wild test set
  (`wild_test`, Mendeley CC BY 4.0: turmeric healthy, turmeric blotch, black gram healthy, black gram yellow
  mosaic). **28 total.**
- Deterministic: candidates sorted by path, `drop_duplicates("cluster_id")`, then `np.random.default_rng(seed=42)`.
  At most one photo per cluster. Black gram alternates field / plain backgrounds (2 field + 1 plain per class).
  Turmeric has only plain-background photos in the dataset, so all its samples are plain.
- Excluded: the `pick_showcase` photos **and** the Crops-section photos. To reuse the Crops-section selection,
  `crop_panels()` was split into `crop_panel_picks()` (selection) + `crop_panels()` (saving), with identical output.
- Each sample gets 3 new optional fields: `bg` (plain|field), `set` (test|wild_test), and `credit` (wild photos only).
- `make site-data` regenerated everything. A diff of site.json against the previous copy shows that only
  `samples` and `generated_at` changed, so the landing data is untouched. `web/public/media/samples/` is
  56 files, 3.5 MB (was 0.74 MB, so +2.8 MB). The 8 old sample files were removed by the script.
- New `tests/test_site_samples.py` (3 tests): every sample's source is in `test` / `wild_test` (never
  train/val); all 8 classes are present and there are ≥ 24 samples; ids are unique; one sample per cluster;
  jpg + thumb exist.
- All 28 samples were run through the API: **28 / 28 correct** (crop and healthy/diseased).

**UI**
- New `web/src/components/analyze/Samples.tsx`:
  - `SampleGallery` replaces the chip row in `DropZone`. It has filter buttons (All / Turmeric / Black gram /
    Healthy / Diseased) that toggle with aria-pressed and are 44 px tall. Tiles are a horizontal snap-scroll
    row below 640 px and a grid of 6 columns at sm and 5 at lg. Order is a stable hash shuffle (FNV-1a +
    murmur finaliser), so neighbouring tiles don't give away the class. `data-testid="sample-<id>"` is kept.
  - `SampleCheck` is shown above the result card after a sample run, with a tick or cross:
    "Photo's real label: X · Model said: Y". `data-testid="label-check"`.
- `Input.tsx`: `Picked` has a new optional `sample` field, and `DropZone` takes `revealed`.
  `Analyze.tsx`: tracks the `revealed` sample ids, renders `SampleCheck`, and adds `min-w-0` on both grid columns.
- `site.ts`: optional `bg`, `set` and `credit` on `Sample`. `format.ts`: `className()` gives plain class names
  (e.g. `leaf_crinckle` becomes "leaf crinkle").
- `e2e/analyze.spec.ts`: the sample test now also checks that the label is hidden before the run, checks
  `label-check` after it, and runs axe in that state. The test was renamed to match.

**Decisions (defaults chosen without asking)**
- The spec says both "tile shows crop and class" and "do NOT show the true label before the run". I chose:
  - A tile shows the crop plus the kind of photo: "Plain", "Outdoor" (test set, outdoor background), or
    "Field photo" with a FIELD badge (wild set).
  - The class name replaces the kind of photo **after** that tile has been run.
  - The Healthy / Diseased filters are kept as specified. Using them is an explicit choice to see the labels.
- When the user overrides the crop, "Model said" reads "Black gram (your choice) · healthy", and only the
  healthy/diseased call is judged, because the crop came from the user, not the model.
- Wild photos show their dataset citation + licence under the label check.

**Look / fix rounds (4)**
1. At 390 the page was ~3000 px wide: the horizontal tile row expanded its grid column (`min-width: auto`).
   Fixed with `min-w-0` on the columns. Also, "Field photo" was used for both the test-set outdoor photos
   and the wild set; the test-set ones were renamed "Outdoor".
2. "Plain backdr…" was truncated in the lg grid; shortened to "Plain". The tile order clustered by crop
   (weak hash); replaced with FNV-1a + murmur. With the crop overridden, "Model said: Black gram" credited the
   model with the user's choice; this now reads "(your choice)".
3. Axe found `label-content-name-mismatch` (aria-label didn't contain the visible text). The aria-label was
   dropped; the name is now built from the visible text plus sr-only words.
4. At 390 the page was 1545 px wide again: the new `sr-only` spans (position: absolute) escaped the scroll
   row's clipping. Fixed by making the tile `relative`.

Final state: 0 px horizontal overflow at 390 / 768 / 1024 / 1440 / 1920, both idle and after a run.
No console errors or HTTP ≥ 400. Keyboard: Tab to a tile, Enter runs it, focus ring is 2 px solid. All
touch targets are ≥ 44 px (tiles ≥ 137 px tall). The cross state was checked with a mocked response
(`*-label-check-wrong-mocked` round, not kept), because the model gets all 28 samples right.

Screenshots: `web/test-results/ui-loop/T4/{390,768,1024,1440,1920}-{empty,gallery,result}.png`.

Checks: typecheck pass · lint pass · pytest **69 passed** (66 + 3 new) · ruff pass · e2e **9 passed**.

Still imperfect (for T2):
- At lg the gallery is 6 rows (~870 px), so the left column runs much longer than the result. T2's sticky
  input column needs a shorter gallery: for example 2–3 rows with "Show all 28", or a fixed-height scroller.
- On mobile, the result appears below a tall input stack. The existing scroll-into-view still runs; this was
  not changed here.

### Iteration 3 — T3 "How this was calculated" panel (2026-10-25)

**API** (additive; schema version stays "1")
- New `api/code.py`: a hard-coded whitelist `STAGES` of 10 functions across 9 stage ids:
  denoise → `preprocess.denoise`, colour → `preprocess.preprocess`, segment → `segment.segment_leaf`,
  lesions → `lesions.detect_lesions`, measure → `features.extract_features`,
  ip_decision → `decide.predict_ip` + `RuleBasedScorer.components`, cnn → `RobustModel.predict`,
  compare → `service.analyze_image`, card → `service.result_card`.
  Source comes from `inspect.getsourcelines` on the imported objects. Nothing from the request touches the
  file system. The result is cached with lru_cache, so the line numbers are those at server start.
- `GET /api/code[?stage=<id>]` returns `{stages, items: [{stage, name, file, start_line, end_line, source}]}`.
  An unknown stage gives 404 `unknown_stage`.
- `POST /api/analyze` gains two optional fields:
  - `trace`: input info (filename echoed back and never stored; original + analysed size; crop given; ROI;
    leaf-coloured share) and 8 stages with this run's values and ms. Stage ids match `/api/code`.
  - `result_card`: the 4 lines from `service.result_card`, exactly as the old Streamlit `st.code` card.
- To get real per-stage times and values (small, additive changes):
  - `features.stage1` now also returns `ms_steps`, timing preprocess / segment / lesions / features.
  - `lesions.detect_lesions` now also returns `min_area` (the adaptive minimum spot size it already computed).
  - `service.run_trace()` builds the trace; NaN becomes null.
- Tests:
  - New `tests/test_api_code.py` (9 tests):
    - every stage returns 200 with the right shape;
    - for every item, `source` equals the file's lines `start_line..end_line` exactly;
    - the `?stage` filter works;
    - 6 path-traversal or non-whitelisted values (`../../api/main`, `/etc/passwd`, a file path, a function
      name, empty, wrong case) all give 404;
    - `file=` / `path=` / `name=` params are ignored;
    - `/api/code/../../etc/passwd` gives 404, and POST gives 405.
  - `tests/test_api.py` now also checks that trace ids match `/api/code`, that the filename is echoed, that
    the trace lesion count equals measurements, and that the card has 4 lines.

**Web**
- `web/src/lib/highlight.ts`: a dependency-free Python highlighter (one regex pass, about 45 lines).
  Token colours reuse existing tokens: turmeric, leaf, muted, chlorosis, mildew, paper.
- `web/src/lib/api.ts`: optional `trace` and `result_card` types, `CodeItem`, `fetchCode()` (cached for the
  session; a failure clears the cache). `analyze()` takes an optional filename, so samples send `<id>.jpg`.
- New `web/src/components/analyze/HowCalculated.tsx`, mounted under Measurements in `Analyze.tsx`:
  - Collapsible, collapsed by default. The open state is remembered in localStorage (`patra:how-open`,
    wrapped in try/catch). Uses aria-expanded and aria-controls.
  - Real ARIA tabs (Steps / Code / Data) with roving tabindex and ←/→ keys. On mobile the tabs and chips
    scroll horizontally.
  - **Steps**: a vertical timeline of the 8 stages. Each has a plain-language sentence, this run's values
    as chips (segmentation method in words, Otsu threshold, leaf share, spot count, minimum spot size,
    rule score vs threshold, SVM margin and chance, CNN chances, OOD distance vs limit, agreement), its time
    in ms, and a "View code" button that jumps to that stage in Code.
  - **Code**: stage chips (the selected chip is kept in view). Each function shows its name, `file:start–end`,
    a Copy button, and highlighted source with real line numbers, scrolling inside its own box
    (`data-lenis-prevent`, max-h 480). Loading shows a skeleton; on failure, "could not be loaded" with
    "Try again".
  - **Data**: tables for Input, Measurements, Rule-based score (4 components in words), Probabilities and
    checks, and Time on the server; the plain-text result card with "Copy card"; and the raw response as a
    collapsible JSON tree (images shortened to `"data:image/webp;base64,…" (N kB image)`), with "Copy JSON" and
    "Download JSON" (`patra-run-<request_id>.json`, images left out, as the note on the page says).
- e2e: new test in `e2e/analyze.spec.ts`, covering the panel collapsed by default, 8 steps, "View code"
  showing `def segment_leaf(` at the right file line with the right number of lines, the result-card text,
  copy JSON (clipboard parsed, no images), the download filename, and the remembered open state.

**Look / fix rounds (2 + a final check)**
1. Problems seen: segmentation method shown as "plain bg threshold"; rule components as raw keys
   ("abn colour pct"); sub-rows not indented; stage chips too long for the row; "included above" wrong for
   the compare step; at 390 the file path broke mid-name ("segment.p / y:211"). Fixed: `segMethod()` gives
   "plain backdrop, colour threshold" or "busy background, GrabCut"; plain names for the rule parts;
   indented sub-rows; a short chip label per stage; per-stage wording for steps with no timing ("timed with
   step 1", "instant"); the path on its own line.
   The odd-looking tab buttons in the first data screenshot were a capture mid-transition, not a bug.
2. At 390, after "View code" the selected chip was out of view. The row now scrolls to it (setting
   scrollLeft only, so the page does not jump).
3. Final check at 390 and 1440:
   - axe (wcag2a/aa, 21a/aa) on the panel, every tab: **0 violations**, so the token colours pass contrast;
   - arrow-key tab navigation works;
   - the code-load failure state shows, and its retry works.
   Earlier passes at 390 / 768 / 1440 / 1920: every one of the 10 code blocks shows exactly
   `end - start + 1` lines, and its first line matches the file on disk; 0 px horizontal overflow; no console
   errors or HTTP ≥ 400.

Screenshots: `web/test-results/ui-loop/T3/{390,768,1440,1920}-{collapsed,steps,code-segment,data}.png`,
`{390,1440}-code-cnn.png`, `{390,1440}-code-compare.png`, `{390,1440}-code-error.png`.

Checks: typecheck pass · lint pass · ruff pass · pytest **78 passed** (69 + 9 new) ·
e2e **10 passed** with `E2E_API_URL=http://127.0.0.1:8011`. Without it, 9 passed and 1 failed: the new panel
test, against the stale 8010 API (see Environment notes).

Still imperfect:
- The Measurements card still has its own "Download report (JSON)" (`patra-report-<id>.json`), and the panel has
  "Download JSON" (`patra-run-<id>.json`) with the same content. T2 should keep one of them.
- The trace's CNN step time includes the crop model, the OOD check and the disease CNN together (one timer in
  service.py). Grad-CAM time appears only in the Data tab's timings.
- Simple / Detailed is T1's job. Today the panel is always available, collapsed.

### Iteration 4 — T1 plain-language layer (2026-10-25)

**Verdict card** (`web/src/components/analyze/Verdict.tsx`): the first block of the result, in both modes.
- The crop ("Black gram leaf", "you chose the crop", or "Crop: not sure"), then "This leaf looks *diseased* /
  *healthy*."
- How much is damaged:
  - "About 3% of the leaf shows damage, in 16 separate spots."
  - On a healthy call, the same numbers are worded as "unusual colour … usually glare, veins or small marks".
  - With 0 spots: "No damaged spots were found".
- Agreement: "Both methods agree, so this is a stronger answer." / "The two methods disagree, so treat this as a
  maybe and check the leaf yourself." Both have a "?" explaining the two methods.
- **How sure**: High / Medium / Low as a word and a 3-segment bar. The rule is in `web/src/lib/verdict.ts`:
  - **Low**: CNN confidence < 0.7, or the methods disagree and it is < 0.9, or the crop is unsure.
  - **High**: ≥ 0.9, both methods agree and the photo is not flagged as unusual.
  - **Medium**: everything else.
- **The numbers are real.** The cut-offs and how often each level was right come from the new
  `confidence_levels()` in `scripts/export_site_data.py`. It reads `results/robust_preds_calibrated.csv`
  (calibrated CNN) + `ip_predictions.csv` (SVM) on the held-out clean validation + test photos (481):
  **High 453/453 right, Medium 20/23, Low 1/5.** Before choosing the cut-offs, I checked accuracy by
  confidence: ≥ 0.9 was 96–100% right on val, test and wild; 0.7–0.9 about 70%; < 0.7 about 40%. The
  confidence "?" quotes these numbers. In `site.json` only `confidence_levels` was added; everything else is
  byte-identical apart from `generated_at`.
- When the photo is flagged as unusual, a warning with a "?" (OOD) appears. When the crop is unsure, the
  "It's turmeric / It's black gram" buttons now live here, removed from the 4-line card so they are not
  duplicated.
- **"What should I do?"** depends on the result:
  - Low: take another photo (one leaf, close up, daylight).
  - Diseased: check other leaves; if several look alike, show the plant to an agronomist or agricultural
    extension officer before treating.
  - Healthy: nothing to act on; keep checking.
  - Always ends with "Patra is a research prototype, not a diagnosis." No treatment advice.

**"?" help** (`Help.tsx` + `web/src/content/glossary.ts`, 10 entries: CNN, SVM, rule-based score,
segmentation, lesions, Grad-CAM, OOD, GLCM, the two methods, confidence).
- A disclosure button (aria-expanded, aria-controls, aria-label "What does … mean?"), 18 px visible with a
  44 px hit area. Escape closes it and returns focus to the button; so does clicking outside.
- It is kept inside the viewport using `documentElement.clientWidth`, and moved with `left`, not a transform.

**Simple / Detailed** toggle at the top of the result column (radiogroup, 44 px).
- Simple by default, remembered in localStorage `patra:view` (try/catch).
- **Simple** shows: verdict, sample label check, progress, "What the model saw" (4 images, plain captions,
  legend) and a "See how this was calculated" button. That button switches to Detailed, opens the panel and
  scrolls to it.
- **Detailed** adds everything that existed before: the 4-line card, Two readings compared, Measurements, and
  How this was calculated.

**Plain labels everywhere**, with the technical term second:
- Evidence: "Your photo", "Leaf cut-out (segmentation)", "Damaged spots (lesions)", "Where the AI looked
  (Grad-CAM)", each with a one-line caption. The heading is now "What the model saw".
- The overlay legend now reads: "yellow = yellowing", "orange-red = brown dead tissue", "magenta = dark
  tissue", "blue = pale, mildew-like", "green line = edge of the leaf". The lightbox uses the same words.
- Compare is now "Two readings compared": "Image processing: rule-based score", "Machine-learning reading
  (SVM)", "Neural network (CNN)".
- Measurements: "Damaged area (affected region)", "Texture contrast (GLCM)", "Unusual photo score (OOD
  distance)", and so on, with "?" where it helps. The segmentation method is in words (`segMethod()`, moved
  to `lib/format.ts` and shared with the T3 panel).
- Idle text and the offline error carry no jargon (developer hint only in Detailed). The upload hint is now
  "a photo from your phone or camera · up to 10 megabytes".

**Honest progress** (`Progress` in `Result.tsx`, replacing the old `Steps` that advanced on a 320 ms timer):
- 4 phases: Finding the leaf / Spotting damaged areas / Asking the neural network / Comparing both answers.
- While the single request runs, all four show "working" next to a **real elapsed-time counter**. When the
  answer arrives, each phase gets its real time from `trace`, falling back to `timings_ms`.
- Detailed adds the technical step names. The live region announces only start and finish (the counter is
  aria-hidden).
- Note: the server actually runs the CNN first. The UI lists the four phases in the spec's order and does not
  claim an order while busy.

**e2e changes** (the DOM legitimately changed):
- The 4-line-card test switches to Detailed first; the lightbox names changed ("Open Damaged spots larger",
  "Damaged spots").
- The gallery test waits for `verdict`, and the crop override now checks `verdict-crop`.
- The panel test enters through "See how this was calculated".
- The a11y test now runs axe in both Simple and Detailed.
- New test "plain-language layer": Simple by default, verdict first in the result, the confidence word, "What
  should I do?", no tables or 4-line card in Simple, "?" opens with Enter and closes with Escape with focus
  returned, Detailed shows the tables and is remembered after reload.

**Look / fix rounds (3)**
1. Problem: at 390 the "?" popover ran off-screen. Cause: in phone emulation, the overflowing popover widened
   the layout viewport, so `innerWidth` read 463 and the clamp fed on itself. Fixed by using `clientWidth`.
   Also: Chrome counted the untransformed box as page overflow (`scrollWidth` reached 584); fixed by
   positioning with `left`. Re-checked by opening every visible "?" (5 in Simple, 17 in Detailed) at 390,
   768 and 1440: all inside the viewport, `scrollWidth` equal to the viewport width.
   **Lesson for later tasks:** check overflow with `scrollWidth > viewport width`, not
   `scrollWidth - innerWidth`, because on mobile emulation `innerWidth` grows with overflow. The page itself
   was re-verified at 390 (idle, result, Simple, Detailed with Code and Data tabs) at 390 px.
2. Problems: "CNN Confidence?:" broke the old 4-line card format, so the "?" was moved after the value. Axe
   flagged contrast on the Simple/Detailed button caught mid-fade, so the colour transition was removed from
   that control. The idle upload hint showed JPEG/PNG/WebP/MB with no explanation, so it was reworded.
3. Problem: on mobile, the auto-scroll to the result hid the "Your result" row and the toggle under the
   fixed nav. Fixed with `scroll-mt-24`; the toggle now sits at y=96, below the nav's bottom at 70.

**Acceptance evidence**
- An acronym scan of all visible text in `main` (Simple mode), idle and with a result: the only acronyms are
  "AI" and "Grad-CAM", and both have a "?" in the same block. Uppercase captions such as "YOUR RESULT"
  are styling, not acronyms, and were excluded.
- The verdict is the first block after the "Your result" header, asserted in e2e.
- Mocked variants (screenshots):
  - methods disagree, confidence 0.8: Low, "treat this as a maybe";
  - unusual photo: Medium with the warning;
  - crop unsure: Low with the crop buttons;
  - offline: plain error with Retry.

Screenshots: `web/test-results/ui-loop/T1/`. These include `{390,1440}-simple-diseased`, `-simple-healthy`,
`-detailed`, `-busy` (2.5 s delayed response), `-help-open`, `-verdict-{low-disagree,ood,crop-unsure}` and
`-offline`, plus `{390,768,1440,1920}-idle` / `-simple-viewport` and `390-after-autoscroll`.

Checks: typecheck pass · lint pass · ruff pass · pytest **78 passed** · e2e **11 passed**
(`E2E_API_URL=http://127.0.0.1:8011`; the 8010 API is still the stale one, see Environment notes).

Still imperfect (for T2):
- On a phone, while the request runs, the progress is below the inputs and off-screen. The existing
  scroll-into-view fires only once the result arrives. T2's 3-step flow should bring "2 Reading it" into
  view at the start.
- The Detailed view now has two JSON downloads (Measurements "Download report" and the panel's "Download
  JSON"), as noted in T3.
- The crop selector (Auto / Turmeric / Black gram) buttons are about 30 px tall, below 44 px. They are
  unchanged here; tidying the controls is part of T2.
- The panel's Steps titles in Detailed still name HSV/Lab, CLAHE, GLCM inline (explained in the sentence
  next to them, without "?"). Detailed only.

### Iteration 5 — T2 front-end redesign of /analyze (2026-10-25)

**3-step flow**
- A step indicator (`Stepper` in `Analyze.tsx`): 1 Choose a leaf → 2 Reading it → 3 Your result. It uses
  `aria-current="step"` and marks finished steps with a tick. On phones the connector lines are hidden so it
  fits one line at 390.
- Each column has a numbered heading ("01 Choose a leaf", "02 Reading it" / "03 Your result"). Both heading
  rows are 52 px tall, so they line up.
- **Sticky input column on desktop**, but only while it fits under the nav. `useFitsViewport` (a
  ResizeObserver plus a resize listener) sets `lg:sticky lg:top-24` only when the column height is at most
  innerHeight − 112. Without this, the end of a sticky column taller than the screen could never be reached.
  - To make it fit once a photo is chosen: preview `lg:max-h-[28vh]`; crop selector and "Point at the leaf"
    above a one-line "Upload another photo"; gallery folded to one row (5 at lg, 6 at sm) with
    "Show all 28 photos" (filters appear only after expanding).
  - Measured: **sticky at 1440×900 (column 770 px of 788) and 1920×1080 (820 of 968)**, and the photo stays
    in view while scrolling through the result. At 1024×768 the column (720 px) does not fit 656, so it scrolls
    normally (intended fallback).
- Phones and tablets stack the columns. When a run starts, the result column scrolls into view **at the start**
  (step 2 is visible while the server works; T1 left this off-screen). On desktop it also scrolls when the
  result column's top is not on screen, e.g. after picking a tile from lower down. Checked: the progress is in
  the viewport during the request at all five widths.

**Result order** (right column; both modes unless marked)
1. Verdict, sample label check, progress summary.
2. "What the model saw":
   - a new inline **before/after drag slider** (your photo ↔ damaged spots), driven by a transparent range
     input over the picture, so pointer, touch and arrow keys all work; the focus ring is drawn on the frame
     with `has-[:focus-visible]`;
   - the colour legend;
   - the 4-picture strip in order (01 Your photo → 02 Leaf cut-out → 03 Damaged spots → 04 Where the AI
     looked), each opening the existing lightbox.
3. **"Two readings compared"** side by side: Image processing | Neural network, each with its answer in words
   and an Agree/Disagree badge. Detailed adds the SVM chance, the rule score against its threshold with its
   4 parts, and the CNN chance.
4. Detailed only: the 4-line card, Measurements, "How this was calculated". Simple shows the "See how this was
   calculated" call to action instead.
- **One JSON download.** Measurements' "Download report (JSON)" was removed; the panel's Data tab has Copy and
  Download JSON. Same content, without images.

**States** (new `web/src/components/analyze/States.tsx`)
- **Empty**: "Pick a test photo or upload your own leaf" plus a worked example built from the site's real
  showcase turmeric leaf (original → cut-out → damaged spots → Grad-CAM, the real Stage 1 outputs from
  `site.json`), plus three photo tips. The squares are reserved before `site.json` loads.
- **Loading**: `ResultSkeleton` with the same footprint as the verdict and the before/after picture (CSS
  shimmer), under the honest progress narration.
- **Errors**: a friendly title, a hint and the server's own message for each code. All the existing codes are
  kept: offline, too_large, not_a_plant, unsupported_type, invalid_image. There are always two actions:
  **Retry** (except too_large, where retrying cannot help) and **Choose another photo** (scrolls to step 1 and
  opens the file picker). `data-code` is on the box.

**Controls**
- Crop selector: 44 px targets; no fade, so axe never measures it mid-transition.
- "Point at the leaf" is now a real `role="switch"` with the line "Several leaves in the photo? Turn this on
  and tap the one to check." (aria-describedby).
- Upload zone: drag/drop, paste and file picker unchanged ("browse" is now visibly a link). Before a photo is
  chosen, the crop and point options are folded into an "Options: crop and pointing at one leaf" disclosure.
- Fixed in passing: the switch knob and the progress pulse dot used `bg-text/70`. The `text` token is a plain
  CSS variable, so Tailwind generates no opacity variant and they were invisible. Changed to `bg-white/70`.

**Motion**: `.reveal` (fade-up 0.45 s, staggered 0/80/160 ms) on result blocks and `.shimmer` on placeholders,
both in `index.css` and both `animation: none` under prefers-reduced-motion (checked: the computed
animation-name is `none` with reduced motion emulated). Hover zooms have `motion-reduce:transition-none`.

**Layout shift**
- Measured with a PerformanceObserver: **CLS 0.0000 while a result replaces the skeleton** (1440 and 390,
  with a 1.5 s delayed response).
- On first load, CLS was 0.03–0.05 because the gallery and the empty-state pictures arrived with `site.json`
  and pushed content down. Fixed by reserving the squares and putting the gallery last in its column. It is
  now **0.003 (1440) / 0.027 (390) / 0.005 (1024)**; what remains is text reflow when the web fonts swap in,
  which happens site-wide.

**Look / fix rounds (4)**
1. Problems: the input column was never sticky (926 px at 1440); the crop control stretched to full width;
   during the request, the progress was off-screen at 1024 and 1440 after picking a tile from lower down.
   Fixed: the compact column, `self-start` on the crop control, and the unified scroll-into-view described
   above.
2. Problems: the column headings were misaligned; the switch knob was invisible; on phones the stepper wrapped
   onto two lines with stray connectors. Fixed.
3. Problem: the column at 1440 was 788 px against 788 available (too tight). The preview went from 30vh to
   28vh, giving 770 px.
4. Problem: load CLS. Fixed with the reserved squares and the new column order.

**Checks at 390 / 768 / 1024 / 1440 / 1920**:
- **0 px horizontal overflow** (scrollWidth equal to the viewport) in idle, result, Detailed and error states;
- no console errors besides the deliberate 422 from the not-a-leaf upload;
- axe (wcag2a/aa) with **0 serious/critical** in the empty, loading, Simple result, Detailed result and
  offline-error states, with and without reduced motion;
- all controls ≥ 44 px; visible focus rings, including on the slider frame.

**e2e**
- `analyze.spec.ts`: the "verdict first" check now compares document order (the verdict sits inside an
  animated wrapper).
- New test "3-step flow": the stepper goes from Choose a leaf to Your result, the empty state shows, and the
  inline slider moves with the arrow keys.
- All existing data-testids kept: sample-*, file-input, result-card, card-*, agreement, error, verdict,
  how-*, copy-json, download-json, label-check.
- New ids: stepper, input-column, result-column, empty-state, result-skeleton, before-after, point-toggle,
  samples-show-all.

Screenshots: `web/test-results/ui-loop/T2/{390,768,1024,1440,1920}-{idle,idle-full,busy,result,result-full,result-scrolled,detailed-full,error}.png`
(40 files).

Checks: typecheck pass · lint pass · pytest **78 passed** · e2e **12 passed**
(`E2E_API_URL=http://127.0.0.1:8011`; the 8010 API is still stale).

Still imperfect:
- At lg widths of about 1024–1100 px, gallery captions such as "Black gram" wrap onto two lines in the
  5-column folded row. Readable, a little cramped.
- At 1024×768 the input column is not sticky (it does not fit). Intended, but the spec's "sticky on desktop"
  holds only from about 900 px of viewport height up.
- The idle gallery is still 6 rows tall at 1440. The column is not sticky before a photo is chosen, so it
  only makes the page longer.

### Iteration 6 — TB hero: the farmer girl (2026-10-25)

**Media** (the two URLs from the loop prompt; downloaded to the scratchpad first, verified, then copied)
- `farmer.jpg`: 1356×1528 JPEG, 276 kB, decodes (Pillow).
- `farmer.mp4`: H.264 1020×1150, 24 fps, 241 frames, **10.04 s**, 3.8 MB.
  - **ffprobe/ffmpeg are not installed**, so it was verified with OpenCV's FFmpeg backend: all 241 frames
    decode.
  - Loop seam: mean |last − first| = 3.1, against 17.3 between neighbouring frames. Frame *i* equals frame
    *n−1−i* (diff 2.8), which confirms it was authored forward-then-reversed.
- **Problem found:** `farmer.mp4` has a **single keyframe** (MP4 `stss` table read directly). Every
  `currentTime` seek would decode up to 240 frames, so the scroll-scrub would stutter, especially backwards.
  - Re-encoded with macOS AVFoundation through a small Swift tool (scratchpad, not in the repo; no new
    dependency): H.264 High, **a keyframe every 6 frames**, no B-frames, moov at the front.
  - `farmer-1020.mp4`: 1020×1150, 3.9 MB, PSNR 34.8 dB against the original (visually identical in a crop
    comparison).
  - `farmer-480.mp4`: 480×542, 1.7 MB; for screens < 768 px, Save-Data or 2g.
  - The original `farmer.mp4` is kept as the source but **is not loaded by the page**.
- Stills: `farmer-960.{webp,avif}` and `farmer-1356.{webp,avif}`. The spec named 1600, but the source is
  1356 px wide, so it was not upscaled. Also `farmer-ambient.webp` (96 px, pre-blurred, 548 B).
- **Credit: unknown, so no credit line in the hero** (spec 5.4). *User action:* if the media has an author or
  licence, it should be credited. The footer still reads "Hero photo: Forest and Kim Starr…" from
  `site.hero.credit`. The footer is out of scope (spec C), so it was left unchanged, but that credit now
  names a photo the hero no longer shows; consider updating it.

**Hero** (`web/src/sections/Hero.tsx`, rewritten; `Landing.tsx`: `<Hero />` no longer takes `site`)
- Removed: the leaf image layers, the outline clip-path, the SVG outline, the 3 callouts and the cover-fit
  maths. Kept: `site.hero` in site.json and its types (the footer uses its credit), and the leaf files.
- **Full framing, no cover-crop**: `frameFor()` computes the frame in JS (ResizeObserver).
  - Side layout (wide screens, or any width/height ≥ 1.2): a tall frame from under the nav to 24 px above the
    bottom, at the media's aspect ratio, rounded 24 px, centred at 58 % of the width. The headline (left)
    overlaps only its plant-filled left edge; a left-edge gradient inside the frame keeps the text legible.
  - Stacked layout (portrait): the largest frame under the nav that leaves room for the copy (290 / 400 /
    480 px by breakpoint). The copy overlaps only the blurred foreground at the frame's bottom.
  - `object-position: 50% 22%`.
  - Ambient: the tiny pre-blurred poster, scaled and darkened (radial ink overlay). It is static, so there is
    no per-frame full-screen blur repaint.
- **Copy kept exactly**: the eyebrow, "Read the leaf" (Playfair italic) and "before the harvest does", the
  paragraph, the Scroll cue, the nav and Run the model (both untouched), and the same headline fade formula.
  The second line is now set as two lines ("before the / harvest does") so it fits beside the frame; the words
  are unchanged. On phones the paragraph sits under the headline.
- **Idle loop**:
  - the video's `src` is set only after `load` and requestIdleCallback, so the **poster is the LCP**
    (confirmed by Lighthouse);
  - muted, playsInline, loop; it fades in over 0.8 s on `loadeddata`, over the poster, with no flash.
- **Scroll-scrub** (`--p` from the existing ScrollTrigger):
  - p 0→0.85 maps to the forward half of the clip (0→5.02 s: the camera orbits until she faces the lens).
  - A 0.22 lerp in requestAnimationFrame; it does not seek while a seek is still in progress.
  - The video is paused while scrubbing. 600 ms after the last scroll update, the idle loop resumes from the
    current frame.
  - Starting mid-loop: a frame from the reverse half is swapped for its identical forward twin (palindrome),
    and the first 0.15 of scroll eases from the current frame into the position mapping, so there is no jump.
  - It pauses off screen (IntersectionObserver with a −1 px margin, so a hero that only touches the viewport
    edge counts as off screen) and when the tab is hidden.
- **Exit**:
  - the same filter as the old hero (saturate 1−0.85·late, brightness 1−0.35·late, late = p 0.7→0.9) on the
    frame and the ambient;
  - plus an ink fade over the lower stage, so the seam into Problem is as dark as before (bottom 2 % of rows,
    luminance before / after: 1440: 24.7 / 16.8; 390: 21.6 / 14.8; the ink is about 13);
  - a subtle scale from 1 to 1.04 over the scroll.
- **Reduced motion**: no `<video>` element, **0 mp4 requests**, `--p` fixed at 0.18, a 100dvh section.
- **A dependency the old hero provided, now kept explicit**:
  - The old hero's large eager image held the `load` event until the fonts and site data had settled, so
    ScrollTrigger's automatic refresh measured the final layout.
  - The new hero loads light: `load` fired at 255 ms, before the content rendered at 307 ms and before the
    font reflow at 342–356 ms. So the scroll-linked Crops reveal (390) and the CNN tiles (1440) were measured
    about 26 px off: 2.5 % and 0.85 % of pixels differed from before, the same in two runs and after waiting.
    A forced refresh made them identical.
  - Fix in Hero.tsx: a ResizeObserver on `document.body` calls `ScrollTrigger.refresh()` once the page height
    has been stable for 200 ms, during the first 4 s.

**Look / fix rounds (5)**
1. Problem: the video never loaded (`src` empty). Cause: the `<video>` is inside the frame, which renders
   after the first measurement, so the effect ran with no element and never re-ran. Fixed with a `hasFrame`
   dependency. (Ruled out first: both the bundled Chromium and Chrome report `canPlayType` H.264 as
   "probably".)
2. Problem: the video kept playing with Problem exactly at the top (edge-touching counted as intersecting).
   Fixed with the −1 px rootMargin.
3. Problem: at 768×1024 the paragraph ended 9 px from the bottom (headline at 96 px). The stacked copy
   reserve was raised to 480 px at ≥ 768. Text now clears the bottom by ≥ 83 px at 390 / 768 / 820×1180 /
   1024×1366.
4. Problem: the sections below differed from before (scroll-trigger measurement race). Fixed with the
   refresh-on-settle above.
5. Problem: the seam into Problem was brighter than before (pale foreground leaves). Added the exit ink fade.

**Acceptance evidence** (`web/test-results/ui-loop/TB/`)
- First screen, `{1440x900,1920x1080,1280x720,768x1024,390x844}-p0.png`:
  - the frame is entirely inside the viewport at all five sizes;
  - geometric check: **no hero text box overlaps her face or her hands** (regions measured on the poster and
    mapped onto the frame on the page);
  - 0 px horizontal overflow; no console errors and no 4xx (hero media included).
  - She is whole in every shot: face, both hands, the leaves she parts, the hanging plants.
- Scroll positions at 1440, `1440-p025/p05/p075/p1.png` + `scrub-sheet.png`:
  - p 0 / .25 / .5 / .75 / 1 show video time loop / 1.24 / 2.73 / 4.16 / 5.02 s;
  - mean frame difference between neighbouring positions is 43–53, so the frames differ;
  - they look coherent: the camera orbits and she turns to the lens, then desaturates at p = 1.
- Reverse: video time at p .75 was 4.69 s and at p .4 was 2.62 s. After scrolling stops, the idle loop
  resumes (+0.36 s in 0.5 s). Off screen (at Crops): paused.
- Seam in the browser, `seam-first/last/middle.png`: |first − last| = 2.5, against 57.7 between first and
  middle. No visible jump.
- Sections below, re-shot exactly like T0 and compared against `before/` in **two runs**:
  - Problem, Crops, Pipeline, CNN, Results, System, Run at 1440 and 390 are **pixel-identical (0.00 %)**;
  - one 390 Run capture was 0.01 %, which is the live healing canvas;
  - only `field-*` (the hero) differs.
- **Lighthouse** (production build + `vite preview`, 3 runs, same method as T0):
  - **85 / 94 / 94, median 94**, against 84 before;
  - LCP 2.85 s (before 4.2 s), and the LCP element is the farmer poster (`farmer-960.avif`);
  - CLS 0.001, TBT 0–1 ms, FCP 2.10 s and SI 2.10 s (both the same as before).
- New `e2e/hero.spec.ts` (4 tests):
  - the frame is whole on screen and the clip plays at 1440 and 390 (1020 vs 480 source chosen);
  - scrolling back reverses the clip, and it pauses off screen;
  - reduced motion shows the photo only, with no mp4 requested.

Checks: typecheck pass · lint pass · pytest **78 passed** · e2e **16 passed** (analyze 7, hero 4, a11y 3, heal 1,
perf 1; `E2E_API_URL=http://127.0.0.1:8011`).

Still imperfect:
- Video in the scrub frames is softer than the poster on very large screens (1020 px source for a frame up
  to 860 px wide at 1920, about 1.2× on 2× screens). A sharper clip would need a higher-resolution original.
- `web/public/media/hero/` keeps `farmer.mp4` (3.8 MB, unused by the page) and the old `leaf-*` files
  (unused). Neither was deleted; remove them before deploying if size matters.
- The footer credit line (see Media) and the missing credit for the new media need the user.

### Iteration 7 — TF final audit (2026-10-25) — ticked in iteration 8 after the blocker was lifted

| Section 8 item | Result | Evidence |
|---|---|---|
| `npm --prefix web run typecheck` / `lint` / `build` | **pass** | all exit 0 (build 1.4 s, index chunk 507 kB / 173 kB gzip) |
| `pytest -q` | **pass**, 78 passed | + ruff clean on api, src, scripts, tests |
| `npm --prefix web run e2e` **as written** | **FAIL, 1 test** | It runs against the dev server on 5180, whose `/api` proxy reaches the **stale uvicorn on 8010** (pid 11920, started 14:51 by another Claude session, no `--reload`). That server predates T3, so `/api/code` is 404 and responses have no `trace`. The "How this was calculated" test then sees 0 steps. Not a code defect. |
| e2e against the current API (`E2E_BASE_URL=http://localhost:5181 E2E_API_URL=http://127.0.0.1:8011`) | **pass, 16/16**, 4 consecutive full runs | analyze 7, hero 4, a11y 3, heal 1, perf 1; perf 0 long tasks each time |
| Landing below the hero unchanged | **pass** | TB: Problem, Crops, Pipeline, CNN, Results, System, Run at 1440 and 390 are pixel-identical to `before/` (0.00 %) in two runs |
| /analyze Simple readable, Detailed has all numbers, code + data panel works | **pass** | T1/T3 evidence; re-swept below |
| ≥ 24 samples, 8 classes, test split only | **pass** | 28 samples, 8 classes, `tests/test_site_samples.py` |
| /analyze five widths clean | **pass** | sweep below: 0 px overflow at 390 / 768 / 1024 / 1440 / 1920 |
| Hero: whole at five sizes, scrub both ways, seamless loop, reduced motion | **pass** | TB evidence + `e2e/hero.spec.ts` |
| No new console errors, no 404s | **pass** | sweep below |
| No committed secrets | **pass** | not a git repo, so nothing is committed; a pattern scan of every file the loop touched found no secrets (only prose about colour "tokens") |
| No uploaded photo persisted | **pass** | code: the only `save()` on the analyze path writes to an in-memory `BytesIO`; nothing logs uploads. Empirical: uploaded `privacy-probe-7f3a91.jpg` to the current API; afterwards no file newer than the upload in the project (excluding test output and caches) or in `/tmp` / `$TMPDIR`, and no file named after it anywhere. The filename is only echoed back in the response. |

**Console / 404 sweep** at 390, 768, 1024, 1440 and 1920. Each width covered:
- the landing page scrolled top to bottom;
- /analyze: a sample run in Simple, the confidence "?" opened, then "See how this was calculated" →
  Detailed, the panel's Code and Data tabs.

Results at every width: **0 console errors, 0 page errors, 0 HTTP ≥ 400** (the deliberate 422 of the
not-a-leaf test is excluded), max scrollWidth equal to the viewport width. Screenshots:
`web/test-results/ui-loop/TF/{w}-{landing-hero,analyze-simple,analyze-detailed}.png`.

**Flakes found during the audit, and what was done**
1. The axe checks on /analyze (`a11y.spec.ts` and the gallery test in `analyze.spec.ts`) intermittently
   reported `color-contrast` on the verdict card. Cause: T2's 0.45 s fade-up `.reveal`; axe measured text
   mid-fade at partial opacity. The page itself passes; T1 and T2 measured it settled. Fix (test side): both
   now wait until every finite CSS animation has finished (`document.getAnimations()`, ignoring infinite
   ones) before running axe. Afterwards both tests passed 12/12 repeated with 4 parallel workers.
2. `perf.spec.ts` occasionally saw one to three 50–110 ms long tasks, **only on the dev server while 4 test
   workers ran at once**. Investigated:
   - isolated runs: 0 long tasks in 3/3 (dev) and 3/3 (production build); production build under the full
     4-worker suite: 0;
   - same parallel load, 8 runs each: current code 1/8 (63 ms); with the hero's settle-refresh disabled 4/8;
     **with the old leaf hero swapped back in temporarily: 1/8 (109, 60, 63 ms)**.
   - So it is pre-existing dev-server contention, not caused by this work. The temporary swap was restored
     and byte-compared against the saved copy.
   The spec's own header says it is meaningful against the production build (`PERF_URL`), where it is clean.

**Blocker (also in BLOCKERS.md):** item 1 of section 8 ("`npm --prefix web run e2e` passes") is not met as
written until the API on port 8010 runs current code. Stopping pid 11920 was attempted in T3 and refused by
the permission classifier (it belongs to another session). **The user needs to restart or stop it.** For
example: stop the other session's preview, or `kill 11920` and then start `patra-api` from the launch
config, which then takes 8010. After that, `npm --prefix web run e2e` is expected to pass 16/16, as the same
suite does against the current API. **The next iteration should re-run only that command; if it passes,
tick TF and finish.**

## Honest summary of the whole upgrade (as of iteration 7)

Done and verified:
- T4: 28 test-split samples covering all 8 classes, a gallery with an honest label check.
- T3: GET /api/code (whitelist only) + trace + the Steps / Code / Data panel.
- T1: Simple / Detailed, a verdict with held-out-backed confidence, and "?" help.
- T2: the 3-step layout, sticky input, before/after slider, empty / loading / error states.
- TB: the farmer hero, framed whole, with a scrubbed loop; sections below pixel-identical; Lighthouse 84 → 94.

Not done or needs the user:
- the stale API on 8010 (above);
- a credit for the farmer media (unknown, so the hero has none), and the footer still credits the old leaf
  photo (footer out of scope);
- unused files that could be removed before deploying: `web/public/media/hero/farmer.mp4` (3.8 MB, the
  single-keyframe original) and `leaf-*`;
- the loop prompt's ffprobe check was done with OpenCV's FFmpeg backend, because ffmpeg/ffprobe are not
  installed.

### Iteration 8 — TF blocker lifted, audit complete (2026-10-25)

- With the user's explicit approval: stopped the stale uvicorn on 8010 (pid 11920), stopped this session's
  `patra-api` on 8011 and started `patra-api` again from `.claude/launch.json`. It now listens on **8010**.
  - `/api/health` returns ok;
  - `/api/code` returns 200 directly and through both dev-server proxies (5180 and 5181).
- `npm --prefix web run e2e`, exactly as section 8 states it: **exit code 0, 16 passed**, in two consecutive
  runs (analyze 7, hero 4, a11y 3, heal 1, perf 1; perf 0 long tasks).
- No code changed since iteration 7, so the other section 8 results stand:
  - typecheck, lint, build: pass;
  - pytest 78 passed;
  - the sections below the hero are pixel-identical to before;
  - 0 console errors and 0 404s at five widths;
  - no secrets;
  - no upload persisted.
- The environment note at the top of this file is now historical: the web servers' `/api` proxy reaches
  current code on 8010, and `E2E_API_URL` is no longer needed.

**All section 6 boxes are ticked and the section 8 audit passes.**

Items left for the user (not part of the acceptance list, recorded honestly):
- a credit for the farmer photo/video (unknown, so the hero has no credit line);
- the footer still credits the old leaf photo (footer out of scope);
- optional cleanup before deploying: `web/public/media/hero/farmer.mp4` (3.8 MB original, not loaded by the
  page) and the unused `leaf-*` hero files.


### Iteration 9 — user feedback fixes (2026-10-25)

1. **Upload box**: no longer a 4:3 box; it is compact (padding instead of aspect ratio). 157 px tall at 1440,
   down from about 500.
2. **Gallery**: shows the first 6 photos, then a **"More options (22 more photos)"** button that expands to
   all 28 ("Fewer options" collapses). The grid is 3 columns in the side column at ≥ 1024, 6 at sm, and a
   swipe row on phones. e2e helpers press "More options" when a test needs a photo outside the first 6.
3. **Hero, bigger and sharper**:
   - **Bigger**: the side-layout frame is 0.96 wide-to-tall (cover-crops only a few percent of
     foreground/glass) and runs from 80 px to 16 px from the bottom. At 1440 it is 772×804 (was 699×788);
     at 1920 it is 945×984 (was 859×968).
   - **Sharper**: new `farmer-1530.mp4`, a 1.5× Lanczos upscale + light unsharp mask (Core Image, Swift tool
     in the scratchpad), 9 Mbit/s, keyframe every 6 frames, 11.7 MB. Edge sharpness (Laplacian variance on
     the face) is 74 against 27 for the browser-upscaled 1020 clip.
   - Clip choice by frame width × devicePixelRatio: > 1100 device px gets 1530, otherwise 1020; Save-Data or
     2g gets 480. Phones now get 1020 instead of 480.
   - Re-checked at 1440@2x, 1920, 1280×720@2x, 768@2x and 390@3x: face and hands inside the frame, no text
     over them, no overflow.
   - Note: the source is 1020 px, so this is not true 4K detail. It removes the softness of browser
     upscaling.
4. **Footer**: Team, Faculty mentor, Institution and the hero photo credit were removed. Only **Data** remains,
   and the field-photo set now has links: the Mendeley DOIs for the turmeric and black gram field photos,
   iNaturalist (turmeric, black gram) and Wikimedia Commons (turmeric, black gram). The DOI and Commons links
   return 200; iNaturalist returns 403 to command-line clients only. `Footer` no longer takes `site`.
5. **"Every spot has a signature" healing**:
   - Problem: the healed image only repaired the Stage 1 lesion mask, so specks and halos the detector
     missed stayed brown after hovering.
   - `scripts/make_heal_assets.py` now adds every leaf pixel clearly off the leaf's own healthy colour
     nearby, compared with a local reference: brown or dark-and-not-greener counts as dead tissue,
     yellowing as chlorosis. Yellowing is also checked against the leaf's greenest tissue, for mosaic
     leaves. Chlorosis healing is stronger (lightness 0.6 → 0.9, chroma variation 0.35 → 0.2).
   - The black gram anthracnose leaves now heal clean; the mosaic leaf is much greener with no clone streaks
     (`results/figures/heal_preview.jpg`).
   - The meter mask grew accordingly (affected 23.9 % → 36.6 % for 21×9, 34.5 → 35.0 % for 9×16). The
     alignment check (SSIM outside the mask) stays 1.0000.
   - The heal e2e test now parks the pointer on the cleanest spot anywhere on the canvas.
6. **Two readings compared**:
   - Each side shows its answer, a chance-of-disease bar with a percentage, and what it based the answer on.
   - Image processing: damaged area and spot count, unusual colour split by kind, and the hand-written rule
     with its score against the threshold.
   - Neural network: the crop and how sure it was, the photo check, and where it looked.
   - Detailed adds the rule-part meters and the crop and photo-check meters.
   - **"This server did not send the step-by-step trace"** came from the stale API on 8010 (fixed in
     iteration 8). Through 5180 → 8010 the panel now shows 8 steps and the code loads. A page opened before
     the restart needs a reload and a new run.

Checks: typecheck pass · lint pass · build pass · pytest 78 passed · ruff pass · `npm --prefix web run e2e`
**16 passed, exit 0** (one earlier run hit the documented dev-server perf flake under 4 parallel workers).
The hero scrub test now waits for the scroll and video to settle instead of a fixed 450 ms.
