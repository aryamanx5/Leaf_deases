// P2-F12: /analyze against the running API (make dev). Uploads a test-set sample and checks the
// four-line card, the evidence strip and the comparison panel; then the not-a-plant error state.
import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';

/** A flat grey RGB PNG (no leaf-coloured pixels). */
function greyPng(n: number): Buffer {
  const crcTable = Array.from({ length: 256 }, (_, k) => {
    let c = k;
    for (let i = 0; i < 8; i++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
  });
  const crc = (b: Buffer) => {
    let c = 0xffffffff;
    for (const x of b) c = crcTable[(c ^ x) & 0xff]! ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };
  const chunk = (type: string, data: Buffer) => {
    const t = Buffer.concat([Buffer.from(type), data]);
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const c = Buffer.alloc(4);
    c.writeUInt32BE(crc(t));
    return Buffer.concat([len, t, c]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(n, 0);
  ihdr.writeUInt32BE(n, 4);
  ihdr.set([8, 2, 0, 0, 0], 8);
  const row = Buffer.concat([Buffer.from([0]), Buffer.alloc(n * 3, 128)]);
  const raw = Buffer.concat(Array.from({ length: n }, () => row));
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

/** The gallery shows 6 photos until "More options" is pressed. */
async function showSample(page: Page, id: string) {
  if (!(await page.getByTestId(`sample-${id}`).count())) await page.getByTestId('samples-show-all').click();
}
async function chooseSample(page: Page, id: string) {
  await showSample(page, id);
  await page.getByTestId(`sample-${id}`).click();
}

const site = JSON.parse(readFileSync(new URL('../public/data/site.json', import.meta.url), 'utf8')) as {
  samples: { id: string; crop: string; src: string }[];
};

for (const vp of [{ name: 'desktop', width: 1440, height: 900 }, { name: 'mobile', width: 390, height: 844 }]) {
  test(`upload renders the 4-line card (${vp.name})`, async ({ page }) => {
    await page.setViewportSize(vp);
    await page.goto('/analyze');
    await page.getByRole('radio', { name: 'Detailed' }).click(); // the 4-line card is part of the Detailed view
    const s = site.samples.find((x) => x.crop === 'turmeric')!;
    await page.getByTestId('file-input').setInputFiles(new URL(`../public${s.src}`, import.meta.url).pathname);
    const card = page.getByTestId('result-card');
    await expect(card).toBeVisible({ timeout: 60_000 });
    await expect(page.getByTestId('card-crop')).toContainText(/Crop:\s*(Turmeric \(\d+%\)|Unsure — Turmeric \d+% \/ Black gram \d+%)/);
    await expect(page.getByTestId('card-prediction')).toContainText(/Prediction:\s*(Healthy|Diseased)/);
    await expect(page.getByTestId('card-affected-region')).toContainText(/Affected Region:\s*\d+\.\d%/);
    await expect(page.getByTestId('card-cnn-confidence')).toContainText(/CNN Confidence:\s*\d+%/);
    await expect(page.getByTestId('agreement')).toHaveText(/Agree|Disagree/);
    await expect(page.getByRole('img', { name: /^Damaged spots/ })).toBeVisible();
    await expect(page.getByText(/4-class|four-class/i)).toHaveCount(0); // never the old 4-class table
    await page.screenshot({ path: `test-results/shots/${vp.name}-analyze-result.png`, fullPage: true });
    // lightbox: before/after slider between Original and Abnormal regions, Esc closes
    await page.getByRole('button', { name: 'Open Damaged spots larger' }).click();
    const dlg = page.getByRole('dialog');
    await expect(dlg.getByRole('slider', { name: 'Before / after position' })).toBeVisible();
    await page.screenshot({ path: `test-results/shots/${vp.name}-analyze-lightbox.png` });
    await page.keyboard.press('Escape');
    await expect(dlg).toHaveCount(0);
  });
}

test('sample gallery: label hidden, revealed after the run; crop override re-runs as user-selected', async ({ page }) => {
  await page.goto('/analyze');
  const s = site.samples.find((x) => x.crop === 'blackgram')!;
  await showSample(page, s.id);
  await expect(page.getByTestId(`sample-${s.id}`)).not.toContainText(/anthracnose|mildew|mosaic|crinkle|healthy/i); // label hidden before the run
  await chooseSample(page, s.id);
  await expect(page.getByTestId('verdict')).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId('label-check')).toContainText(/Photo’s real label: Black gram · .+ Model said: (Turmeric|Black gram) · (healthy|diseased)/);
  // the result fades up (0.45 s); axe measures contrast at the current opacity, so let finite animations finish
  await page.waitForFunction(() =>
    document.getAnimations().every((a) => a.playState !== 'running' || a.effect?.getComputedTiming().endTime === Infinity));
  const axe = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze();
  expect(axe.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical').map((v) => v.id)).toEqual([]);
  await page.getByRole('radio', { name: 'Black gram' }).click();
  await expect(page.getByTestId('verdict-crop')).toContainText('you chose the crop', { timeout: 60_000 });
});

test('not-a-plant upload shows the error state with retry', async ({ page }) => {
  await page.goto('/analyze');
  const png = greyPng(256);
  await page.getByTestId('file-input').setInputFiles({ name: 'grey.png', mimeType: 'image/png', buffer: png });
  const err = page.getByTestId('error');
  await expect(err).toBeVisible({ timeout: 60_000 });
  await expect(err).toContainText('This does not look like a leaf');
  await expect(err.getByRole('button', { name: 'Retry' })).toBeVisible();
});

test('"How this was calculated": steps with this run\'s numbers, real code, copy and download the data', async ({ page, context }) => {
  // E2E_API_URL: point /api at a specific API (e.g. when the dev server's proxy target runs older code)
  const api = process.env.E2E_API_URL;
  if (api) await page.route('**/api/**', async (r) => r.fulfill({ response: await r.fetch({ url: r.request().url().replace(/^https?:\/\/[^/]+/, api) }) }));
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/analyze');
  const s = site.samples.find((x) => x.crop === 'turmeric')!;
  await chooseSample(page, s.id);
  await expect(page.getByTestId('verdict')).toBeVisible({ timeout: 60_000 });
  await page.getByTestId('show-workings').click(); // Simple -> Detailed, panel opened and scrolled to
  await expect(page.getByRole('radio', { name: 'Detailed' })).toHaveAttribute('aria-checked', 'true');
  const toggle = page.getByTestId('how-toggle');
  await expect(toggle).toHaveAttribute('aria-expanded', 'true');
  await toggle.click();
  await expect(toggle).toHaveAttribute('aria-expanded', 'false'); // collapsed by default
  await toggle.click();
  const steps = page.getByTestId('how-steps').locator(':scope > li');
  await expect(steps).toHaveCount(8);
  await expect(steps.nth(3)).toContainText(/spots found: \d+/);
  await page.getByRole('button', { name: /View code for Cut the leaf out/ }).click();
  await expect(page.getByTestId('code-block').first()).toContainText('def otsu_threshold('); // the Otsu threshold comes first
  const block = page.getByTestId('code-block').filter({ hasText: 'def segment_leaf(' });
  await expect(block).toContainText('def segment_leaf(');
  const [, file, start, end] = (await block.locator('figcaption span.block').nth(1).innerText()).match(/(\S+):(\d+)–(\d+)/)!;
  const src = readFileSync(new URL(`../../${file}`, import.meta.url), 'utf8').split('\n');
  expect(src[Number(start) - 1]).toContain('def segment_leaf(');
  await expect(block.locator('code > span')).toHaveCount(Number(end) - Number(start) + 1);
  await page.getByTestId('how-tab-data').click();
  await expect(page.getByTestId('result-card-text')).toContainText('Prediction: ');
  await page.getByTestId('copy-json').click();
  const copied = JSON.parse(await page.evaluate(() => navigator.clipboard.readText())) as { request_id: string; images?: unknown };
  expect(copied.images).toBeUndefined();
  const [dl] = await Promise.all([page.waitForEvent('download'), page.getByTestId('download-json').click()]);
  expect(dl.suggestedFilename()).toBe(`patra-run-${copied.request_id}.json`);
  // the open state is remembered
  await page.reload();
  await chooseSample(page, s.id);
  await expect(page.getByTestId('how-toggle')).toHaveAttribute('aria-expanded', 'true', { timeout: 60_000 });
});

test('plain-language layer: Simple by default, verdict first, confidence in words, "?" help, Detailed remembered', async ({ page }) => {
  await page.goto('/analyze');
  await expect(page.getByRole('radio', { name: 'Simple' })).toHaveAttribute('aria-checked', 'true');
  const s = site.samples.find((x) => x.crop === 'blackgram')!;
  await chooseSample(page, s.id);
  const verdict = page.getByTestId('verdict');
  await expect(verdict).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId('verdict-title')).toHaveText(/^This leaf looks (healthy|diseased)\.$/);
  await expect(page.getByTestId('confidence-level')).toHaveText(/^(High|Medium|Low)/);
  await expect(page.getByTestId('what-to-do')).toContainText('not a diagnosis');
  // the verdict comes before every other block of the result
  const after = await verdict.evaluate((v) => {
    const others = document.querySelectorAll('[data-testid="result-column"] [data-testid]:not([data-testid="verdict"]):not([data-testid="mode-toggle"])');
    return Array.from(others).filter((o) => !v.contains(o) && !(v.compareDocumentPosition(o) & Node.DOCUMENT_POSITION_FOLLOWING)).length;
  });
  expect(after).toBe(0);
  // Simple shows no tables and no 4-line card
  await expect(page.locator('main table')).toHaveCount(0);
  await expect(page.getByTestId('result-card')).toHaveCount(0);
  // "?" help opens with the keyboard and closes with Escape
  const help = verdict.locator('[data-help="methods"]');
  await help.focus();
  await page.keyboard.press('Enter');
  await expect(verdict.getByRole('note')).toContainText('two independent ways');
  await page.keyboard.press('Escape');
  await expect(verdict.getByRole('note')).toHaveCount(0);
  await expect(help).toBeFocused();
  // Detailed shows the numbers and is remembered
  await page.getByRole('radio', { name: 'Detailed' }).click();
  await expect(page.getByTestId('result-card')).toBeVisible();
  await expect(page.locator('main table').first()).toBeVisible();
  await page.reload();
  await expect(page.getByRole('radio', { name: 'Detailed' })).toHaveAttribute('aria-checked', 'true');
});

test('3-step flow: the stepper follows the run; the inline before/after slider works by keyboard', async ({ page }) => {
  await page.goto('/analyze');
  const current = page.getByTestId('stepper').locator('[aria-current="step"]');
  await expect(current).toContainText('Choose a leaf');
  await expect(page.getByTestId('empty-state')).toBeVisible();
  const s = site.samples.find((x) => x.crop === 'turmeric')!;
  await chooseSample(page, s.id);
  await expect(page.getByTestId('verdict')).toBeVisible({ timeout: 60_000 });
  await expect(current).toContainText('Your result');
  const slider = page.getByRole('slider', { name: 'Compare your photo with its damaged spots' });
  const clip = () => page.getByTestId('before-after').locator('img').nth(1).evaluate((el) => (el as HTMLElement).style.clipPath);
  const before = await clip();
  await slider.focus();
  await page.keyboard.press('ArrowLeft');
  await page.keyboard.press('ArrowLeft');
  expect(await clip()).not.toBe(before);
});
