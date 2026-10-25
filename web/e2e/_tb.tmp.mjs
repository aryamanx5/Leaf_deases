import { chromium } from '@playwright/test';
const out = 'test-results/ui-loop/TB';
const browser = await chromium.launch();
// face and hands on the poster, as fractions of the media (measured on farmer.jpg)
const FACE = [0.46, 0.10, 0.74, 0.34], HANDS = [0.60, 0.56, 0.94, 0.72];
const sizes = (process.argv[2] ?? '1440x900,1920x1080,1280x720,768x1024,390x844').split(',').map(s => s.split('x').map(Number));
for (const [w, h] of sizes) {
  const m = w < 768;
  const ctx = await browser.newContext({ viewport: { width: w, height: h }, isMobile: m, hasTouch: m });
  const page = await ctx.newPage();
  const errs = []; page.on('console', x => { if (x.type() === 'error') errs.push(x.text()); }); page.on('response', r => { if (r.status() >= 400) errs.push(`${r.status()} ${r.url()}`); });
  await page.goto('http://localhost:5181/', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => { const v = document.querySelector('[data-testid="hero-video"]'); return v && v.readyState >= 2; }, null, { timeout: 30000 });
  await page.waitForTimeout(1800); // hero intro animations + 0.8 s fade
  await page.screenshot({ path: `${out}/${w}x${h}-p0.png` });
  const geo = await page.evaluate(([FACE, HANDS]) => {
    const f = document.querySelector('[data-testid="hero-frame"]').getBoundingClientRect();
    const toPage = ([a, b, c, d]) => ({ l: f.left + a * f.width, t: f.top + b * f.height, r: f.left + c * f.width, b: f.top + d * f.height });
    const face = toPage(FACE), hands = toPage(HANDS);
    const texts = [...document.querySelectorAll('#field h1 span.block, #field p')].filter(e => e.offsetParent).flatMap(e => {
      const range = document.createRange(); range.selectNodeContents(e); return [...range.getClientRects()].map(r => ({ txt: e.textContent.trim().slice(0, 20), l: r.left, t: r.top, r: r.right, b: r.bottom }));
    });
    const hit = (a, b) => a.l < b.r && b.l < a.r && a.t < b.b && b.t < a.b;
    const v = document.querySelector('[data-testid="hero-video"]');
    return { frame: [Math.round(f.left), Math.round(f.top), Math.round(f.width), Math.round(f.height)], inViewport: f.left >= 0 && f.top >= 0 && f.right <= innerWidth && f.bottom <= innerHeight,
      overFace: texts.filter(t => hit(t, face)).map(t => t.txt), overHands: texts.filter(t => hit(t, hands)).map(t => t.txt),
      video: v.currentSrc.split('/').pop(), paused: v.paused, opacity: getComputedStyle(v).opacity, sw: document.documentElement.scrollWidth };
  }, [FACE, HANDS]);
  const res = { size: `${w}x${h}`, ...geo };
  if (w === 1440) {
    const pos = async (p) => page.evaluate((p) => { const s = document.getElementById('field'); const top = s.getBoundingClientRect().top + scrollY; return Math.round(top + (s.offsetHeight - innerHeight) * p); }, p);
    const t = () => page.evaluate(() => +document.querySelector('[data-testid="hero-video"]').currentTime.toFixed(3));
    const go = async (p) => { await page.mouse.move(700, 400); const y = await pos(p); const now = await page.evaluate(() => scrollY); await page.mouse.wheel(0, y - now); await page.waitForTimeout(450); };
    res.scrub = [];
    for (const p of [0.25, 0.5, 0.75, 1]) {
      await go(p);
      res.scrub.push({ p, t: await t(), paused: await page.evaluate(() => document.querySelector('[data-testid="hero-video"]').paused), P: await page.evaluate(() => getComputedStyle(document.getElementById('field')).getPropertyValue('--p')) });
      await page.screenshot({ path: `${out}/1440-p${String(p).replace('.', '')}.png` });
    }
    await go(0.75); const tBack1 = await t(); await go(0.4); const tBack2 = await t();
    res.reverse = { at075: tBack1, at040: tBack2, reversed: tBack2 < tBack1 };
    await page.waitForTimeout(900);
    const a = await t(); await page.waitForTimeout(500); const b = await t();
    res.idleResume = { paused: await page.evaluate(() => document.querySelector('[data-testid="hero-video"]').paused), advanced: +(b - a).toFixed(3) };
    await page.evaluate(() => document.getElementById('problem').scrollIntoView()); await page.waitForTimeout(1200);
    res.offscreenPaused = await page.evaluate(() => document.querySelector('[data-testid="hero-video"]').paused);
  }
  res.errs = errs;
  console.log(JSON.stringify(res));
  await ctx.close();
}
await browser.close();
