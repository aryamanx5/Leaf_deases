// The sections and tools added on top of the original story: film interlude, field guide, threshold lab,
// evidence dots, timeline, questions, command palette; and on /analyze the camera, at-a-glance tiles and history.
import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';

type SiteJson = {
  samples: { id: string; src: string }[];
  confidence_levels: { high: { n: number }; medium: { n: number }; low: { n: number } };
  dataset: { by_class: Record<string, number> };
};
const site = JSON.parse(readFileSync(new URL('../public/data/site.json', import.meta.url), 'utf8')) as SiteJson;

// a fake camera device for the whole file (launch options are per worker), so the capture path runs for real
test.use({
  launchOptions: { args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] },
  permissions: ['camera'],
});

const toSection = async (page: Page, id: string, offset = 0) => {
  await page.evaluate(([id, offset]) => {
    const el = document.getElementById(id as string)!;
    window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY + (offset as number));
  }, [id, offset] as const);
};

test.describe('landing additions', () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/', { waitUntil: 'networkidle' });
  });

  test('film clips are fetched only when their section is near', async ({ page }) => {
    const film = () => page.evaluate(() => performance.getEntriesByType('resource').map((e) => e.name).filter((n) => /\/media\/film\/.*\.mp4/.test(n)));
    expect(await film()).toEqual([]); // nothing at the top of the page
    await toSection(page, 'interlude', 300);
    await expect.poll(film, { timeout: 15_000 }).toEqual([expect.stringMatching(/field-(720|540)\.mp4$/)]);
    await expect.poll(() => page.evaluate(() => {
      const v = document.querySelector<HTMLVideoElement>('#interlude video')!;
      return v.readyState >= 2 && !v.paused;
    }), { timeout: 15_000 }).toBe(true);
    // off screen it pauses again
    await toSection(page, 'crops', 200);
    await expect.poll(() => page.evaluate(() => document.querySelector<HTMLVideoElement>('#interlude video')!.paused)).toBe(true);
  });

  test('field guide: choosing a class swaps photos, text and the real photo count', async ({ page }) => {
    await toSection(page, 'atlas');
    const atlas = page.locator('#atlas');
    await atlas.getByRole('button', { name: 'Powdery mildew' }).click();
    await expect(atlas.getByRole('heading', { level: 3 })).toHaveText('Powdery mildew');
    await expect(atlas).toContainText('Erysiphe polygoni');
    await expect(atlas).toContainText(`${site.dataset.by_class['blackgram/powdery_mildew']} photos of this class`);
    await expect(atlas.locator('img[alt*="powdery mildew"]')).toBeVisible();
    await expect(atlas.locator('img')).toHaveCount(3);
  });

  test('threshold lab: Otsu’s cut is found in the browser and the slider moves the mask', async ({ page }) => {
    await toSection(page, 'lab', 100);
    const t = page.getByTestId('lab-t');
    await page.getByTestId('lab-snap').click();
    const best = Number(await t.textContent());
    expect(best).toBeGreaterThan(5);
    expect(best).toBeLessThan(250);
    await expect(page.locator('#lab')).toContainText('Otsu’s pick');
    const ink = () => page.evaluate(() => {
      const c = document.querySelector<HTMLCanvasElement>('[data-testid="lab-canvas"]')!;
      const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
      let s = 0;
      for (let i = 0; i < d.length; i += 16) s += d[i]! + d[i + 1]! + d[i + 2]!;
      return s;
    });
    const atBest = await ink();
    await page.getByTestId('lab-range').fill('250'); // almost nothing is bright enough to count as leaf
    await expect(t).toHaveText('250');
    expect(await ink()).toBeLessThan(atBest * 0.6);
    await expect(page.locator('#lab')).toContainText('Your cut');
    // keyboard: the slider is a real range input
    await page.getByTestId('lab-range').press('ArrowLeft');
    await expect(t).toHaveText('249');
  });

  test('evidence: one dot per held-out photo', async ({ page }) => {
    await toSection(page, 'proof');
    const lv = site.confidence_levels;
    await expect(page.locator('[data-testid="proof-dots"] circle')).toHaveCount(lv.high.n + lv.medium.n + lv.low.n);
    await expect(page.getByTestId('proof-dots')).toHaveClass(/dots-in/);
  });

  test('timeline: the veins are drawn as the section scrolls by', async ({ page }) => {
    const painted = () => page.evaluate(() => {
      const c = document.querySelector<HTMLCanvasElement>('[data-testid="veins"]')!;
      if (!c.width) return 0;
      const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data;
      let n = 0;
      for (let i = 3; i < d.length; i += 4) if (d[i]! > 90) n++;
      return n;
    });
    await toSection(page, 'journey', -500);
    await page.waitForTimeout(1200);
    const early = await painted();
    await toSection(page, 'journey', 500);
    await expect.poll(painted, { timeout: 10_000 }).toBeGreaterThan(early + 2000);
    await expect(page.getByTestId('journey-phases').locator('li')).toHaveCount(4);
  });

  test('questions open and close with the keyboard', async ({ page }) => {
    await toSection(page, 'notes', 700);
    const q = page.getByTestId('faq').locator('summary', { hasText: 'Is this a diagnosis?' });
    await q.focus();
    await page.keyboard.press('Enter');
    await expect(page.getByTestId('faq')).toContainText('research prototype');
    await expect(page.getByTestId('faq').locator('details[open]')).toHaveCount(1);
    await page.keyboard.press('Enter');
    await expect(page.getByTestId('faq').locator('details[open]')).toHaveCount(0);
  });

  test('command palette: opens with the shortcut, filters, jumps, restores focus', async ({ page }) => {
    await page.keyboard.press('ControlOrMeta+k');
    const dialog = page.getByRole('dialog', { name: 'Jump to a part of the site' });
    await expect(dialog).toBeVisible();
    await page.keyboard.type('thresh');
    await expect(dialog.getByRole('option')).toHaveCount(1);
    await page.keyboard.press('Enter');
    await expect(dialog).toHaveCount(0);
    await expect.poll(() => page.evaluate(() => Math.abs(document.getElementById('lab')!.getBoundingClientRect().top)), { timeout: 10_000 }).toBeLessThan(40);
    // the nav button opens it too, and Escape gives focus back to the button
    const button = page.getByRole('button', { name: 'Jump to a part of the site' });
    await button.click();
    await expect(dialog).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(dialog).toHaveCount(0);
    await expect(button).toBeFocused();
  });

  test('film reel: one panel is open, and only its clip is fetched', async ({ page }) => {
    await toSection(page, 'reel', 100);
    const panels = page.getByTestId('reel').getByRole('button');
    await expect(panels).toHaveCount(5);
    await expect(panels.nth(0)).toHaveAttribute('aria-pressed', 'true');
    const fetched = () => page.evaluate(() => Array.from(document.querySelectorAll<HTMLVideoElement>('#reel video')).filter((v) => v.currentSrc).length);
    await expect.poll(fetched, { timeout: 15_000 }).toBe(1);
    await panels.nth(2).focus(); // keyboard users open a panel by tabbing to it
    await expect(panels.nth(2)).toHaveAttribute('aria-pressed', 'true');
    await expect(panels.nth(0)).toHaveAttribute('aria-pressed', 'false');
    await expect.poll(fetched, { timeout: 15_000 }).toBe(2);
    await expect.poll(() => page.evaluate(() => Array.from(document.querySelectorAll<HTMLVideoElement>('#reel video')).filter((v) => !v.paused).length)).toBe(1);
  });

  test('theme: the button cycles palettes, the choice survives a reload, the palette can reset it', async ({ page }) => {
    const theme = () => page.evaluate(() => document.documentElement.dataset.theme ?? 'forest');
    const ink = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    const forestInk = await ink();
    await page.getByTestId('theme-button').click();
    await expect.poll(theme).toBe('monsoon');
    await expect.poll(ink).not.toBe(forestInk);
    await page.reload({ waitUntil: 'networkidle' });
    expect(await theme()).toBe('monsoon');
    await page.keyboard.press('ControlOrMeta+k');
    await page.keyboard.type('forest theme');
    await page.keyboard.press('Enter');
    await expect.poll(theme).toBe('forest');
    await expect.poll(ink).toBe(forestInk);
  });

  test('intro curtain: shown once per visit, never blocks input, then gone', async ({ page }) => {
    await page.evaluate(() => sessionStorage.clear());
    await page.reload();
    await expect(page.getByTestId('intro')).toBeVisible();
    // the page under it already takes input
    await page.keyboard.press('ControlOrMeta+k');
    await expect(page.getByRole('dialog', { name: 'Jump to a part of the site' })).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByTestId('intro')).toHaveCount(0, { timeout: 6000 });
    await page.reload();
    await expect(page.getByTestId('intro')).toHaveCount(0);
  });

  test('no serious axe violations in the new sections', async ({ page }) => {
    const found: string[] = [];
    for (const id of ['interlude', 'atlas', 'lab', 'proof', 'reel', 'journey', 'notes', 'band']) {
      await page.evaluate((id) => {
        const el = document.getElementById(id)!;
        window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY + Math.max(0, el.offsetHeight - window.innerHeight));
      }, id);
      await page.waitForTimeout(1300);
      await page.waitForFunction(() =>
        document.getAnimations().every((a) => a.playState !== 'running' || a.effect?.getComputedTiming().endTime === Infinity));
      const r = await new AxeBuilder({ page }).include(`#${id}`).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze();
      found.push(...r.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical')
        .map((v) => `${id} ${v.id}: ${v.nodes.map((n) => n.target.join(' ')).slice(0, 3).join(' | ')}`));
    }
    expect(found).toEqual([]);
  });

  test('no horizontal overflow on a phone, new sections included', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/', { waitUntil: 'networkidle' });
    for (const id of ['interlude', 'atlas', 'lab', 'proof', 'reel', 'journey', 'notes', 'band']) {
      await toSection(page, id, 200);
      await page.waitForTimeout(250);
      expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth), id).toBeLessThanOrEqual(0);
    }
  });
});

test.describe('/analyze additions', () => {
  test('camera: a frame from the live preview goes down the upload path', async ({ page }) => {
    let posted = 0;
    await page.route('**/api/analyze', async (route) => {
      posted = route.request().postDataBuffer()?.length ?? 0;
      await route.continue();
    });
    await page.goto('/analyze');
    await page.getByTestId('camera-open').click();
    const dialog = page.getByRole('dialog', { name: 'Photograph a leaf' });
    await expect(dialog).toBeVisible();
    await expect(page.getByTestId('camera-shutter')).toBeEnabled({ timeout: 15_000 });
    await page.getByTestId('camera-shutter').click();
    await expect(dialog).toHaveCount(0);
    await expect(page.getByAltText('Your uploaded photo.')).toBeVisible();
    await expect.poll(() => posted, { timeout: 15_000 }).toBeGreaterThan(2000); // a real JPEG was sent
    // the stream is released when the dialog closes
    expect(await page.evaluate(() => document.querySelectorAll('video').length)).toBe(0);
    // the fake camera shows a test pattern, not a leaf: any answer or a clear refusal is fine, a crash is not
    await expect(page.getByTestId('verdict').or(page.getByTestId('error'))).toBeVisible({ timeout: 60_000 });
  });

  test('at-a-glance tiles and the visit history', async ({ page }) => {
    await page.goto('/analyze');
    await page.getByTestId('sample-gallery').getByRole('button').first().click(); // the gallery shows a shuffled first few
    await expect(page.getByTestId('verdict')).toBeVisible({ timeout: 60_000 });
    const glance = page.getByTestId('glance');
    await expect(glance).toBeVisible();
    await expect(glance).toContainText(/separate spots?/);
    await expect(page.getByTestId('history')).toHaveCount(0); // one result is not a history
    const first = await page.getByTestId('verdict-title').textContent();

    await page.getByTestId('file-input').setInputFiles(new URL(`../public${site.samples[4]!.src}`, import.meta.url).pathname);
    await expect(page.getByTestId('history').getByRole('button')).toHaveCount(2, { timeout: 60_000 });
    // going back to the first result needs no new request
    let calls = 0;
    await page.route('**/api/analyze', (route) => {
      calls++;
      return route.continue();
    });
    await page.getByTestId('history').getByRole('button').nth(1).click();
    await expect(page.getByTestId('verdict-title')).toHaveText(first!);
    expect(calls).toBe(0);
  });
});
