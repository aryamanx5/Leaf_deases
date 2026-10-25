# Blockers and caveats

## P2-F04 Hero: no "woman inspecting turmeric leaves" footage (DONE-WITH-CAVEAT)

**Superseded on 2026-10-25 (UI loop, task TB):** the landing hero now uses user-supplied media (a grower in a
greenhouse: photo + 10 s clip, `web/public/media/hero/farmer*`), scroll-scrubbed. See UI_PROGRESS.md, iteration 6.
The leaf photo below is no longer shown in the hero; its files and `site.hero` data remain (the footer credit
still reads `site.hero.credit`).

**Spec (loop prompt v2, Section 8 fallback):** a 5-7 s shot of an agronomist lifting a turmeric leaf, converted to
~120 scroll-scrubbed frames with ffmpeg; if no footage is obtainable, use a still with a slow zoom and log it here,
without a stock-photo look-alike.

**What happened:**
* Appendix B (the user's hero prompt) was left empty, so the Section 8 fallback applies.
* No footage was supplied in `web/assets-src/hero/`.
* No openly licensed (CC0 / CC BY / CC BY-SA) clip matching the description was found.
* No AI video model is available to this loop.
* `ffmpeg` is not installed on the build machine, so a frame sequence could not be produced even with footage.

**What ships instead:**
* A real, openly licensed still: *Curcuma longa leaves with morning dew* by Forest and Kim Starr (CC BY 2.0,
  Wikimedia Commons; credited on the page and in `web/assets-src/hero/starr_6608.json`).
* It has the slow hero zoom, the scroll-scrubbed 1 -> 1.35 push toward the midrib and hairline callouts.
* The leaf outline drawn at the end is our own Stage 1 segmentation of this photo.
* No disease callouts are drawn: the leaf is healthy, and Stage 1's lesion rules over-fire on its dew and
  specular highlights, so the page shows only morphology (midrib, venation, outline).

**To lift the caveat:**
1. Put a licensed clip in `web/assets-src/hero/`.
2. Install ffmpeg.
3. Implement `scripts/make_hero_frames.sh` and the canvas frame scrubber in `web/src/sections/Hero.tsx`. Its
   scroll timeline (0-25 / 25-70 / 70-100 %) is already in place.

## UI loop TF: `npm --prefix web run e2e` fails 1 test because the API on port 8010 is stale (RESOLVED 2026-10-25)

**Resolved:** with the user's approval, pid 11920 was stopped and `patra-api` restarted from the launch config
on 8010. `npm --prefix web run e2e` now exits 0 with 16 passed (two runs).

* The dev server's `/api` proxy (vite.config.ts) targets 127.0.0.1:8010. That port is held by a uvicorn started
  at 14:51 on 2026-10-25 by another Claude session (pid 11920, no `--reload`), so it serves the code from
  before T3: `GET /api/code` returns 404 and `/api/analyze` responses have no `trace`.
* As a result, the e2e test "How this was calculated" sees 0 steps. Everything else passes. The full suite
  passes 16/16 against a current API (`E2E_BASE_URL=http://localhost:5181 E2E_API_URL=http://127.0.0.1:8011`).
* Tried: `kill 11920` (T3), refused by the permission classifier because the process belongs to another
  session.
* To lift: stop or restart that server (e.g. `kill 11920`, then start `patra-api` from .claude/launch.json),
  then re-run `npm --prefix web run e2e`.

