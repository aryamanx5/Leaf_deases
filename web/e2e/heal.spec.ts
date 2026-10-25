// Section 08 healing interaction (spec 7.3): pixels near the pointer path move toward the
// healed image, a region > 2R away is untouched, healing decays back, and the meter follows.
import { expect, test, type Page } from '@playwright/test';

type Pt = { x: number; y: number };

async function setup(page: Page) {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/', { waitUntil: 'networkidle' });
  await page.waitForSelector('#run');
  await page.waitForTimeout(1000); // Lenis + ScrollTrigger settle, else they restore the old position
  await page.evaluate(() => window.scrollTo(0, document.getElementById('run')!.getBoundingClientRect().top + window.scrollY));
  await page.waitForFunction(() => {
    const c = document.querySelector<HTMLCanvasElement>('#run canvas');
    return !!c?.dataset.healed && !!c.dataset.r;
  });
  await page.waitForTimeout(800); // entry animations, first frames
}

/** Mean abs RGB difference of the live canvas to the healed / diseased reference, per box. */
async function diffs(page: Page, boxes: Pt[], half: number) {
  return page.evaluate(
    async ({ boxes, half }) => {
      const c = document.querySelector<HTMLCanvasElement>('#run canvas')!;
      const sec = document.getElementById('run')!.getBoundingClientRect();
      const load = (src: string) =>
        new Promise<HTMLImageElement>((res) => {
          const im = new Image();
          im.onload = () => res(im);
          im.src = src;
        });
      const healedSrc = c.dataset.healed!;
      const [hea, dis] = await Promise.all([load(healedSrc), load(healedSrc.replace('healed', 'diseased'))]);
      const W = c.width, H = c.height;
      const ref = (im: HTMLImageElement) => {
        const o = document.createElement('canvas');
        o.width = W;
        o.height = H;
        const s = Math.max(W / im.naturalWidth, H / im.naturalHeight);
        const g = o.getContext('2d')!;
        g.drawImage(im, (W - im.naturalWidth * s) / 2, (H - im.naturalHeight * s) / 2, im.naturalWidth * s, im.naturalHeight * s);
        return g;
      };
      const gh = ref(hea), gd = ref(dis), live = c.getContext('2d')!;
      return boxes.map(({ x, y }) => {
        const bx = Math.round(x - sec.left - half), by = Math.round(y - sec.top - half), n = 2 * half;
        const a = live.getImageData(bx, by, n, n).data, h = gh.getImageData(bx, by, n, n).data, d = gd.getImageData(bx, by, n, n).data;
        let toH = 0, toD = 0;
        for (let i = 0; i < a.length; i += 4)
          for (let k = 0; k < 3; k++) {
            toH += Math.abs(a[i + k]! - h[i + k]!);
            toD += Math.abs(a[i + k]! - d[i + k]!);
          }
        const m = (a.length / 4) * 3;
        return { toHealed: toH / m, toDiseased: toD / m };
      });
    },
    { boxes, half },
  );
}

const meter = async (page: Page) => parseFloat((await page.getByTestId('meter').textContent())!.replace('%', ''));

test('healing follows the pointer, stays local, decays back; meter tracks it', async ({ page }) => {
  await setup(page);
  const R = Number(await page.locator('#run canvas').getAttribute('data-r'));
  // pick a lesion-dense spot P, a lesion-bearing far spot F (> 2R from the path) and a clean spot Q (> 2R from P) to park on
  const spots = await page.evaluate(async (R) => {
    const c = document.querySelector<HTMLCanvasElement>('#run canvas')!;
    const sec = document.getElementById('run')!.getBoundingClientRect();
    const src = c.dataset.healed!;
    const variant = src.includes('9x16') ? '9x16' : '21x9';
    const im = await new Promise<HTMLImageElement>((res) => {
      const i = new Image();
      i.onload = () => res(i);
      i.src = `/media/heal/lesion_mask_${variant}.png`;
    });
    const W = c.width, H = c.height;
    const o = document.createElement('canvas');
    o.width = W;
    o.height = H;
    const s = Math.max(W / im.naturalWidth, H / im.naturalHeight);
    const g = o.getContext('2d')!;
    g.drawImage(im, (W - im.naturalWidth * s) / 2, (H - im.naturalHeight * s) / 2, im.naturalWidth * s, im.naturalHeight * s);
    const d = g.getImageData(0, 0, W, H).data;
    const density = (x: number, y: number, r: number) => {
      let n = 0, t = 0;
      for (let yy = y - r; yy < y + r; yy += 4) for (let xx = x - r; xx < x + r; xx += 4) {
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        t++;
        if (d[(yy * W + xx) * 4]! > 127) n++;
      }
      return t ? n / t : 0;
    };
    const cells: { x: number; y: number; dens: number; wide: number }[] = [];
    for (let y = Math.round(H * 0.3); y < H * 0.85; y += 30)
      for (let x = 60; x < W - 60; x += 30) cells.push({ x, y, dens: density(x, y, Math.round(R / 4)), wide: density(x, y, Math.round(R)) });
    const inner = cells.filter((q) => q.x > R && q.x < W - R);
    const P = [...inner].sort((a, b) => b.dens - a.dens)[0]!;
    const far = inner.filter((q) => Math.hypot(q.x - P.x, q.y - P.y) > 2 * R + R / 3 + R / 4 + 10).sort((a, b) => b.dens - a.dens)[0]!;
    // Q: the cleanest spot anywhere on the canvas (the healing mask now also covers spot halos, so the middle
    // band can be lesion-bearing everywhere; the backdrop near the top and bottom edges is clean)
    const all: typeof cells = [];
    for (let y = Math.round(R / 2); y < H - R / 2; y += 30) for (let x = 60; x < W - 60; x += 30) all.push({ x, y, dens: 0, wide: density(x, y, Math.round(R)) });
    const Q = all.filter((q) => Math.hypot(q.x - P.x, q.y - P.y) > 2 * R).sort((a, b) => a.wide - b.wide)[0]!;
    const abs = (p: { x: number; y: number }) => ({ x: p.x + sec.left, y: p.y + sec.top });
    return { P: abs(P), F: abs(far), Q: abs(Q), dens: [P.dens, far.dens, Q.wide] };
  }, R);
  const half = Math.round(R / 4);
  const before = await diffs(page, [spots.P, spots.F], half);
  const m0 = await meter(page);

  // short stroke through P
  const steps = 8;
  for (let i = 0; i <= steps; i++) {
    await page.mouse.move(spots.P.x - R / 3 + ((2 * R) / 3) * (i / steps), spots.P.y);
    await page.waitForTimeout(30);
  }
  await page.waitForTimeout(700);
  const mid = await diffs(page, [spots.P, spots.F], half);
  await page.screenshot({ path: 'test-results/shots/heal-mid.png' });
  const m1 = await meter(page);

  expect(before[0]!.toHealed).toBeGreaterThan(2); // the spot really differs between diseased and healed
  expect(mid[0]!.toHealed).toBeLessThan(0.5 * before[0]!.toHealed); // healed by > 50 %
  expect(mid[1]!.toDiseased).toBeLessThan(1.0); // far region unchanged
  expect(m1).toBeLessThan(m0);

  // park the pointer on a lesion-free spot (small moves keep the idle ghost away) and wait 4 s
  await page.mouse.move(spots.Q.x, spots.Q.y, { steps: 10 });
  for (let t = 0; t < 4; t++) {
    await page.waitForTimeout(1000);
    await page.mouse.move(spots.Q.x + (t % 2 ? 2 : -2), spots.Q.y);
  }
  const after = await diffs(page, [spots.P], half);
  const m2 = await meter(page);
  expect(after[0]!.toDiseased).toBeLessThan(0.5 * mid[0]!.toDiseased); // decayed back toward diseased
  expect(m2).toBeGreaterThan(m1 + 0.5 * (m0 - m1)); // meter recovered most of the way
  console.log(JSON.stringify({ R, spots, before, mid, after, meter: [m0, m1, m2] }));
});
