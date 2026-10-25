// TB: the landing hero. The grower is framed whole (no cover-crop), the clip loads after first paint and is
// scrubbed by scroll (scroll back reverses it), it pauses off screen, and reduced motion shows the photo only.
import { expect, test } from '@playwright/test';

const frameInViewport = () => {
  const f = document.querySelector('[data-testid="hero-frame"]')!.getBoundingClientRect();
  return f.width > 100 && f.left >= 0 && f.top >= 0 && f.right <= innerWidth && f.bottom <= innerHeight;
};
const videoTime = () => (document.querySelector('[data-testid="hero-video"]') as HTMLVideoElement).currentTime;

for (const vp of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
  test(`hero frame is whole on screen and the clip plays (${vp.width})`, async ({ page }) => {
    await page.setViewportSize(vp);
    await page.goto('/', { waitUntil: 'networkidle' });
    expect(await page.evaluate(frameInViewport)).toBe(true);
    await expect(page.getByRole('img', { name: /grower in a greenhouse/ })).toBeVisible();
    await expect(page.getByRole('heading', { level: 1 })).toContainText('Read the leaf');
    await page.waitForFunction(() => (document.querySelector('[data-testid="hero-video"]') as HTMLVideoElement).readyState >= 2, null, { timeout: 30_000 });
    const src = await page.evaluate(() => (document.querySelector('[data-testid="hero-video"]') as HTMLVideoElement).currentSrc);
    expect(src).toMatch(/farmer-(1020|1530)\.mp4$/); // sharp enough for the frame on this screen; 480 only with Save-Data
  });
}

test('scrolling scrubs the clip forward, back up reverses it, off screen it pauses', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => (document.querySelector('[data-testid="hero-video"]') as HTMLVideoElement).readyState >= 2, null, { timeout: 30_000 });
  const to = async (p: number) => {
    const y = await page.evaluate((p) => { const s = document.getElementById('field')!; return Math.round(s.offsetTop + (s.offsetHeight - innerHeight) * p); }, p);
    await page.mouse.move(700, 400);
    await page.mouse.wheel(0, y - (await page.evaluate(() => scrollY)));
    // wait for the smooth scroll to arrive and the scrub to settle (a fixed delay is too short under load)
    await expect.poll(() => page.evaluate((p) => Math.abs(parseFloat(getComputedStyle(document.getElementById('field')!).getPropertyValue('--p')) - p), p),
      { timeout: 10_000 }).toBeLessThan(0.06);
    let last = -1;
    await expect.poll(async () => { const t = await page.evaluate(videoTime); const still = Math.abs(t - last) < 0.01; last = t; return still; },
      { timeout: 10_000, intervals: [150] }).toBe(true);
  };
  await to(0.7);
  const deep = await page.evaluate(videoTime);
  await to(0.3);
  const back = await page.evaluate(videoTime);
  expect(back).toBeLessThan(deep);
  await page.evaluate(() => document.getElementById('crops')!.scrollIntoView());
  await expect.poll(() => page.evaluate(() => (document.querySelector('[data-testid="hero-video"]') as HTMLVideoElement).paused)).toBe(true);
});

test('reduced motion: the photo only, no clip downloaded', async ({ browser }) => {
  const ctx = await browser.newContext({ reducedMotion: 'reduce', viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  const mp4: string[] = [];
  page.on('request', (r) => { if (r.url().endsWith('.mp4')) mp4.push(r.url()); });
  await page.goto('/', { waitUntil: 'networkidle' });
  await expect(page.getByRole('img', { name: /grower in a greenhouse/ })).toBeVisible();
  await expect(page.getByTestId('hero-video')).toHaveCount(0);
  expect(mp4).toEqual([]);
  await ctx.close();
});
