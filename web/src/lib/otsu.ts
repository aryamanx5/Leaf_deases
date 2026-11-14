/**
 * Otsu's threshold, computed in the browser for the threshold lab. Input: the showcase leaf's photo and its
 * colour-distance map (the picture of step 3 of the pipeline), both drawn at n x n. Everything the lab shows
 * (histogram, between-class variance, the best cut, the mask) is derived here from those pixels.
 */
export type Lab = {
  n: number;
  photo: Uint8ClampedArray; // RGBA
  dist: Uint8Array; // grey level per pixel, 0..255
  hist: Uint32Array; // 256 bins
  /** between-class variance for a cut after level t (class 0 = levels <= t), scaled to its maximum */
  sigma: Float32Array;
  best: number;
  /** share of pixels brighter than t, for every t */
  above: Float32Array;
};

export function analyse(photo: ImageData, dist: ImageData): Lab {
  const n = photo.width;
  const total = n * photo.height;
  const d = new Uint8Array(total);
  const hist = new Uint32Array(256);
  for (let i = 0, j = 0; i < total; i++, j += 4) {
    const v = Math.round(0.299 * dist.data[j]! + 0.587 * dist.data[j + 1]! + 0.114 * dist.data[j + 2]!);
    d[i] = v;
    hist[v]!++;
  }
  let sumAll = 0;
  for (let v = 0; v < 256; v++) sumAll += v * hist[v]!;
  const sigma = new Float32Array(256);
  const above = new Float32Array(256);
  let w0 = 0;
  let sum0 = 0;
  let max = 0;
  let best = 0;
  for (let t = 0; t < 256; t++) {
    w0 += hist[t]!;
    sum0 += t * hist[t]!;
    const w1 = total - w0;
    above[t] = w1 / total;
    if (w0 === 0 || w1 === 0) continue;
    const m0 = sum0 / w0;
    const m1 = (sumAll - sum0) / w1;
    const s = (w0 / total) * (w1 / total) * (m0 - m1) * (m0 - m1);
    sigma[t] = s;
    if (s > max) {
      max = s;
      best = t;
    }
  }
  if (max > 0) for (let t = 0; t < 256; t++) sigma[t] = sigma[t]! / max;
  return { n, photo: photo.data, dist: d, hist, sigma, best, above };
}

export type LabView = 'cutout' | 'mask' | 'map';

/** Paint the chosen view of the leaf for threshold t into a same-size ImageData. */
export function paint(out: ImageData, lab: Lab, t: number, view: LabView) {
  const o = out.data;
  const { photo, dist } = lab;
  for (let i = 0, j = 0; i < dist.length; i++, j += 4) {
    const v = dist[i]!;
    const leaf = v > t;
    if (view === 'mask') {
      o[j] = leaf ? 236 : 14;
      o[j + 1] = leaf ? 232 : 20;
      o[j + 2] = leaf ? 223 : 16;
    } else if (view === 'map') {
      // the distance map itself, tinted leaf-green above the cut
      o[j] = leaf ? v * 0.55 + 30 : v * 0.5;
      o[j + 1] = leaf ? v * 0.72 + 50 : v * 0.5;
      o[j + 2] = leaf ? v * 0.42 + 20 : v * 0.5;
    } else if (leaf) {
      o[j] = photo[j]!;
      o[j + 1] = photo[j + 1]!;
      o[j + 2] = photo[j + 2]!;
    } else {
      // dropped as background: sunk into the ink
      o[j] = photo[j]! * 0.1 + 9;
      o[j + 1] = photo[j + 1]! * 0.1 + 13;
      o[j + 2] = photo[j + 2]! * 0.1 + 10;
    }
    o[j + 3] = 255;
  }
}

/** Histogram as an SVG area path in a 256 x h box (square-root scale, so the small leaf peak stays visible). */
export function histPath(hist: Uint32Array, h: number): string {
  let max = 0;
  for (let v = 0; v < 256; v++) max = Math.max(max, Math.sqrt(hist[v]!));
  let d = `M0 ${h}`;
  for (let v = 0; v < 256; v++) {
    const y = h - (max > 0 ? (Math.sqrt(hist[v]!) / max) * h : 0);
    d += ` L${v} ${y.toFixed(2)} L${v + 1} ${y.toFixed(2)}`;
  }
  return `${d} L256 ${h} Z`;
}

export function sigmaPath(sigma: Float32Array, h: number): string {
  let d = '';
  for (let t = 0; t < 256; t++) d += `${t === 0 ? 'M' : ' L'}${t + 0.5} ${(h - sigma[t]! * h).toFixed(2)}`;
  return d;
}
