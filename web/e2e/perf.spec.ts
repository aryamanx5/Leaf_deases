// P2-F03..F11 acceptance: smooth scroll through the whole landing page with no long main-thread
// tasks (> 50 ms). Meaningful against the production build: PERF_URL=http://localhost:4180.
import { expect, test } from '@playwright/test';

test('no long tasks while scrolling the landing page', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(process.env.PERF_URL ?? '/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  await page.evaluate(() => {
    const w = window as unknown as { __long: number[] };
    w.__long = [];
    new PerformanceObserver((l) => l.getEntries().forEach((e) => w.__long.push(e.duration))).observe({ type: 'longtask' });
  });
  const total = await page.evaluate(() => document.documentElement.scrollHeight - window.innerHeight);
  // wheel scrolling (what Lenis smooths), ~120 px per event like a trackpad flick
  for (let y = 0; y < total; y += 120) {
    await page.mouse.wheel(0, 120);
    await page.waitForTimeout(16);
  }
  await page.waitForTimeout(1500);
  const long = await page.evaluate(() => (window as unknown as { __long: number[] }).__long);
  console.log(JSON.stringify({ scrolled_px: total, long_tasks_ms: long.map((d) => Math.round(d)) }));
  expect(long.filter((d) => d > 50)).toEqual([]);
});
