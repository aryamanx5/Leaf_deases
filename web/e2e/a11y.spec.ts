// P2-F13: axe-core on every landing section and on /analyze (idle + with a result): 0 serious/critical.
// Also checks the reduced-motion variant: no pinned sections, the healing section becomes a slider.
import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';

const SECTIONS = ['field', 'problem', 'crops', 'pipeline', 'cnn', 'results', 'system', 'run'];
const site = JSON.parse(readFileSync(new URL('../public/data/site.json', import.meta.url), 'utf8')) as { samples: { src: string }[] };

async function serious(page: Page) {
  const r = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze();
  return r.violations
    .filter((v) => v.impact === 'serious' || v.impact === 'critical')
    .map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).slice(0, 3).join(' | ')}`);
}

test('landing: no serious axe violations in any section', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1000);
  const found: string[] = [];
  for (const id of SECTIONS) {
    await page.evaluate((id) => {
      const el = document.getElementById(id)!;
      const span = Math.max(0, el.offsetHeight - window.innerHeight);
      window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY + span); // end of the section: everything revealed
    }, id);
    await page.waitForTimeout(1300);
    found.push(...(await serious(page)).map((v) => `${id} ${v}`));
  }
  expect([...new Set(found)]).toEqual([]);
});

test('/analyze: no serious axe violations, idle and with a result', async ({ page }) => {
  await page.goto('/analyze');
  await page.waitForTimeout(600);
  expect(await serious(page)).toEqual([]);
  await page.getByTestId('file-input').setInputFiles(new URL(`../public${site.samples[0]!.src}`, import.meta.url).pathname);
  await expect(page.getByTestId('result-card')).toBeVisible({ timeout: 60_000 });
  expect(await serious(page)).toEqual([]);
});

test('reduced motion: nothing pinned, healing becomes a before/after slider', async ({ browser }) => {
  const ctx = await browser.newContext({ reducedMotion: 'reduce', viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  await page.goto('/', { waitUntil: 'networkidle' });
  // no scroll-scrubbed tracks (inline heights in vh) and nothing sticky
  const tracks = await page.evaluate(() => Array.from(document.querySelectorAll<HTMLElement>('main [style*="vh"]')).map((e) => e.style.height).filter((h) => parseFloat(h) > 100));
  expect(tracks).toEqual([]);
  // a sticky element only pins if its container is taller than the viewport
  const pinned = await page.evaluate(() =>
    Array.from(document.querySelectorAll<HTMLElement>('main .sticky')).filter((e) => e.parentElement!.offsetHeight > window.innerHeight * 1.1).length);
  expect(pinned).toBe(0);
  await page.evaluate(() => document.getElementById('run')!.scrollIntoView());
  const slider = page.getByRole('slider');
  await expect(slider).toBeVisible();
  const before = await page.getByTestId('meter').textContent();
  await slider.fill('1');
  await expect(page.getByTestId('meter')).not.toHaveText(before!);
  await page.screenshot({ path: 'test-results/shots/reduced-run.png' });
  await ctx.close();
});
