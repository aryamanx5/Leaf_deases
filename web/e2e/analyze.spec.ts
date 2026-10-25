// P2-F12: /analyze against the running API (make dev). Uploads a test-set sample and checks the
// four-line card, the evidence strip and the comparison panel; then the not-a-plant error state.
import { expect, test } from '@playwright/test';
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

const site = JSON.parse(readFileSync(new URL('../public/data/site.json', import.meta.url), 'utf8')) as {
  samples: { id: string; crop: string; src: string }[];
};

for (const vp of [{ name: 'desktop', width: 1440, height: 900 }, { name: 'mobile', width: 390, height: 844 }]) {
  test(`upload renders the 4-line card (${vp.name})`, async ({ page }) => {
    await page.setViewportSize(vp);
    await page.goto('/analyze');
    const s = site.samples.find((x) => x.crop === 'turmeric')!;
    await page.getByTestId('file-input').setInputFiles(new URL(`../public${s.src}`, import.meta.url).pathname);
    const card = page.getByTestId('result-card');
    await expect(card).toBeVisible({ timeout: 60_000 });
    await expect(page.getByTestId('card-crop')).toContainText(/Crop:\s*(Turmeric \(\d+%\)|Unsure — Turmeric \d+% \/ Black gram \d+%)/);
    await expect(page.getByTestId('card-prediction')).toContainText(/Prediction:\s*(Healthy|Diseased)/);
    await expect(page.getByTestId('card-affected-region')).toContainText(/Affected Region:\s*\d+\.\d%/);
    await expect(page.getByTestId('card-cnn-confidence')).toContainText(/CNN Confidence:\s*\d+%/);
    await expect(page.getByTestId('agreement')).toHaveText(/Agree|Disagree/);
    await expect(page.getByRole('img', { name: /^Detected abnormal regions/ })).toBeVisible();
    await expect(page.getByText(/4-class|four-class/i)).toHaveCount(0); // never the old 4-class table
    await page.screenshot({ path: `test-results/shots/${vp.name}-analyze-result.png`, fullPage: true });
    // lightbox: before/after slider between Original and Abnormal regions, Esc closes
    await page.getByRole('button', { name: 'Open Detected abnormal regions larger' }).click();
    const dlg = page.getByRole('dialog');
    await expect(dlg.getByRole('slider', { name: 'Before / after position' })).toBeVisible();
    await page.screenshot({ path: `test-results/shots/${vp.name}-analyze-lightbox.png` });
    await page.keyboard.press('Escape');
    await expect(dlg).toHaveCount(0);
  });
}

test('sample chip + crop override re-runs as user-selected', async ({ page }) => {
  await page.goto('/analyze');
  const s = site.samples.find((x) => x.crop === 'blackgram')!;
  await page.getByTestId(`sample-${s.id}`).click();
  await expect(page.getByTestId('result-card')).toBeVisible({ timeout: 60_000 });
  await page.getByRole('radio', { name: 'Black gram' }).click();
  await expect(page.getByTestId('card-crop')).toContainText('selected by you', { timeout: 60_000 });
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
