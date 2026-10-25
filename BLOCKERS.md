# Blockers and caveats

## P2-F04 Hero: no "woman inspecting turmeric leaves" footage (DONE-WITH-CAVEAT)

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
