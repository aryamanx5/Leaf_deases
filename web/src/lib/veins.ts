/**
 * Generative leaf venation (space colonisation, Runions et al. 2005) for the timeline section: a midrib is laid
 * down, then side veins grow toward scattered "auxin" points inside a lance-shaped blade. Growth is computed in
 * small time slices so no frame is blocked, and the result is an ordered list of segments that can be drawn up to
 * any fraction, which is what ties the drawing to the scroll position.
 *
 * Units: x runs 0..1 from the base of the blade to its tip, y is in the same unit (0 on the midrib).
 */
const mulberry32 = (seed: number) => () => {
  let t = (seed += 0x6d2b79f5);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

/** Half-width of the blade at x: a long lance shape, widest a little before the middle. */
export const halfWidth = (x: number) => 0.155 * Math.pow(Math.sin(Math.PI * Math.pow(Math.min(1, Math.max(0, x)), 0.82)), 0.8);

const STEP = 0.011;
const INFLUENCE = 0.085;
const KILL = 0.021;
const MAX_NODES = 1900;

export type Veins = {
  /** x0, y0, x1, y1, width (relative, 0..1) per segment, in drawing order */
  segs: Float32Array;
  count: number;
};

export function createGrower(seed = 7) {
  const rnd = mulberry32(seed);
  const nx: number[] = [0];
  const ny: number[] = [0];
  const parent: number[] = [-1];
  // the direction each node last grew in, so it does not sprout the same twig twice
  const lastX: number[] = [0];
  const lastY: number[] = [0];
  // midrib first: a straight spine with a slight natural wander
  for (let x = STEP; x < 0.975; x += STEP) {
    nx.push(x);
    ny.push(0.006 * Math.sin(x * 9) * x);
    parent.push(nx.length - 2);
    lastX.push(0);
    lastY.push(0);
  }
  // auxin points inside the blade
  let ax: number[] = [];
  let ay: number[] = [];
  while (ax.length < 620) {
    const x = 0.03 + rnd() * 0.94;
    const y = (rnd() * 2 - 1) * halfWidth(x) * 0.93;
    if (Math.abs(y) > 0.012) {
      ax.push(x);
      ay.push(y);
    }
  }
  let iterations = 0;
  let finished = false;
  let result: Veins | null = null;

  const iterate = () => {
    const n = nx.length;
    const dx = new Float32Array(n);
    const dy = new Float32Array(n);
    const hits = new Uint16Array(n);
    const keepX: number[] = [];
    const keepY: number[] = [];
    for (let a = 0; a < ax.length; a++) {
      const px = ax[a]!;
      const py = ay[a]!;
      let best = -1;
      let bd = INFLUENCE * INFLUENCE;
      for (let i = 0; i < n; i++) {
        const ex = px - nx[i]!;
        const ey = py - ny[i]!;
        const d = ex * ex + ey * ey;
        if (d < bd) {
          bd = d;
          best = i;
        }
      }
      if (best >= 0 && bd < KILL * KILL) continue; // reached: this point is used up
      keepX.push(px);
      keepY.push(py);
      if (best < 0) continue;
      const d = Math.sqrt(bd) || 1;
      dx[best] = dx[best]! + (px - nx[best]!) / d;
      dy[best] = dy[best]! + (py - ny[best]!) / d;
      hits[best] = hits[best]! + 1;
    }
    ax = keepX;
    ay = keepY;
    let grew = 0;
    for (let i = 0; i < n && nx.length < MAX_NODES; i++) {
      if (!hits[i]) continue;
      // side veins sweep toward the tip, as they do in a real blade
      let vx = dx[i]! / hits[i]! + 0.42;
      let vy = dy[i]! / hits[i]!;
      const len = Math.hypot(vx, vy) || 1;
      vx /= len;
      vy /= len;
      const x = nx[i]! + vx * STEP;
      const y = ny[i]! + vy * STEP;
      if (x < 0 || x > 1 || Math.abs(y) > halfWidth(x)) continue;
      if (vx * lastX[i]! + vy * lastY[i]! > 0.96) continue;
      lastX[i] = vx;
      lastY[i] = vy;
      nx.push(x);
      ny.push(y);
      parent.push(i);
      lastX.push(0);
      lastY.push(0);
      grew++;
    }
    iterations++;
    if (!grew || !ax.length || iterations > 240 || nx.length >= MAX_NODES) finish();
  };

  const finish = () => {
    finished = true;
    const n = nx.length;
    // vein weight: how much of the tree hangs off each node
    const load = new Float32Array(n).fill(1);
    for (let i = n - 1; i > 0; i--) load[parent[i]!] = load[parent[i]!]! + load[i]!;
    const maxLoad = load[0] || 1;
    // drawing order: base to tip, a parent always before its children
    const key = new Float32Array(n);
    for (let i = 1; i < n; i++) key[i] = Math.max(key[parent[i]!]! + 1e-5, nx[i]!);
    const order = Array.from({ length: n - 1 }, (_, k) => k + 1).sort((a, b) => key[a]! - key[b]!);
    const segs = new Float32Array(order.length * 5);
    order.forEach((i, k) => {
      const p = parent[i]!;
      segs.set([nx[p]!, ny[p]!, nx[i]!, ny[i]!, Math.pow(load[i]! / maxLoad, 0.42)], k * 5);
    });
    result = { segs, count: order.length };
  };

  return {
    /** Work for at most `budgetMs`; returns the finished venation, or null while there is more to do. */
    step(budgetMs: number): Veins | null {
      const t0 = performance.now();
      while (!finished && performance.now() - t0 < budgetMs) iterate();
      return result;
    },
  };
}

/** Draw the blade outline and segments [0, upTo) into a canvas of w x h CSS pixels. */
export function drawVeins(ctx: CanvasRenderingContext2D, v: Veins | null, w: number, h: number, from: number, upTo: number, colour: string) {
  const pad = w * 0.02;
  const L = w - pad * 2;
  const cy = h / 2;
  if (from === 0) {
    ctx.clearRect(0, 0, w, h);
    // the blade itself, barely there
    ctx.beginPath();
    for (let s = 0; s <= 80; s++) {
      const x = s / 80;
      const X = pad + x * L;
      const Y = cy - halfWidth(x) * L;
      if (s === 0) ctx.moveTo(X, Y);
      else ctx.lineTo(X, Y);
    }
    for (let s = 80; s >= 0; s--) ctx.lineTo(pad + (s / 80) * L, cy + halfWidth(s / 80) * L);
    ctx.closePath();
    ctx.fillStyle = 'rgba(141, 184, 107, 0.035)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(232, 240, 228, 0.14)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  if (!v) return;
  ctx.strokeStyle = colour;
  ctx.lineCap = 'round';
  const s = v.segs;
  const scale = Math.max(0.6, w / 1200);
  for (let k = from; k < Math.min(upTo, v.count); k++) {
    const o = k * 5;
    ctx.lineWidth = (0.55 + 3.1 * s[o + 4]!) * scale;
    ctx.globalAlpha = 0.35 + 0.6 * s[o + 4]!;
    ctx.beginPath();
    ctx.moveTo(pad + s[o]! * L, cy + s[o + 1]! * L);
    ctx.lineTo(pad + s[o + 2]! * L, cy + s[o + 3]! * L);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}
