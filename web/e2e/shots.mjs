// Section screenshots: node e2e/shots.mjs [baseURL] [sections,comma] [viewports: desktop,mobile,laptop] [positions: 0.5 or 0,0.5,0.95]
import { chromium } from '@playwright/test';
import { mkdirSync } from 'node:fs';

const base = process.argv[2] ?? 'http://localhost:5180';
const only = (process.argv[3] ?? 'field,problem,crops,pipeline,cnn,results,system,run').split(',');
const vps = { desktop: [1440, 900], laptop: [1280, 800], mobile: [390, 844] };
const which = (process.argv[4] ?? 'desktop,mobile').split(',');
const positions = (process.argv[5] ?? '0.5').split(',').map(Number);
const out = 'test-results/shots';
mkdirSync(out, { recursive: true });

const browser = await chromium.launch();
for (const v of which) {
  const [w, h] = vps[v];
  const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1, isMobile: v === 'mobile', hasTouch: v === 'mobile' });
  await page.goto(base + '/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1500);
  for (const id of only) {
    for (const pos of positions) {
      const y = await page.evaluate(([id, pos]) => {
        const el = document.getElementById(id);
        if (!el) return null;
        const top = el.getBoundingClientRect().top + window.scrollY;
        const span = Math.max(0, el.offsetHeight - window.innerHeight);
        return Math.round(top + span * pos);
      }, [id, pos]);
      if (y == null) continue;
      await page.evaluate((y) => window.scrollTo(0, y), y);
      await page.waitForTimeout(1300);
      await page.screenshot({ path: `${out}/${v}-${id}-${String(pos).replace('.', '')}.png` });
    }
  }
  await page.close();
}
await browser.close();
console.log('shots in', out);
