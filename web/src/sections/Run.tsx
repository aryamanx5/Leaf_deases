import { ArrowUpRight } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { DEPLOYED_APP_URL } from '@/content/brand';
import { pctRaw } from '@/lib/format';
import { useReducedMotion } from '@/lib/motion';
import type { Heal, Site } from '@/lib/site';

// Appendix A spotlight gradient stops
const STOPS: [number, number][] = [[0, 1], [0.4, 1], [0.6, 0.75], [0.75, 0.4], [0.88, 0.12], [1, 0]];
const DECAY_PER_60HZ_FRAME = 0.01;

type Fit = { s: number; dx: number; dy: number };
const coverFit = (cw: number, ch: number, iw: number, ih: number): Fit => {
  const s = Math.max(cw / iw, ch / ih);
  return { s, dx: (cw - iw * s) / 2, dy: (ch - ih * s) / 2 };
};
const loadImg = (src: string) =>
  new Promise<HTMLImageElement>((res, rej) => {
    const im = new Image();
    im.decoding = 'async';
    im.onload = () => res(im);
    im.onerror = rej;
    im.src = src;
  });

function pickFile(files: string[], wantW: number) {
  const webp = files.filter((f) => f.endsWith('.webp'));
  const ws = webp.map((f) => Number(f.match(/-(\d+)\.webp$/)?.[1] ?? 0));
  const i = ws.findIndex((w) => w >= wantW);
  return `/media/heal/${webp[i >= 0 ? i : webp.length - 1]}`;
}

/** Lesion pixel sample (mask space, 0..1 coords) for the live meter. */
async function lesionSample(src: string, max = 6000): Promise<Float32Array> {
  const im = await loadImg(src);
  const c = document.createElement('canvas');
  c.width = im.naturalWidth;
  c.height = im.naturalHeight;
  const g = c.getContext('2d', { willReadFrequently: true })!;
  g.drawImage(im, 0, 0);
  const d = g.getImageData(0, 0, c.width, c.height).data;
  const pts: number[] = [];
  for (let i = 0; i < d.length; i += 4) if (d[i]! > 127) pts.push(((i / 4) % c.width) / c.width, Math.floor(i / 4 / c.width) / c.height);
  const n = pts.length / 2;
  const step = Math.max(1, Math.floor(n / max));
  const out: number[] = [];
  for (let k = 0; k < n; k += step) out.push(pts[2 * k]!, pts[2 * k + 1]!);
  return new Float32Array(out);
}

export default function Run({ site }: { site: Site }) {
  const reduced = useReducedMotion();
  const sectionRef = useRef<HTMLElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);
  const [entered, setEntered] = useState(false);
  const [variant, setVariant] = useState<'21x9' | '9x16'>('21x9');
  const heal: Heal = site.heal[variant];
  const initial = heal.affected_pct_initial;
  const [current, setCurrent] = useState(initial);
  const [slider, setSlider] = useState(0.5);
  const [assets, setAssets] = useState<{ dis: HTMLImageElement; hea: HTMLImageElement; lesion: Float32Array } | null>(null);

  // choose the variant by aspect ratio
  useEffect(() => {
    const on = () => setVariant(window.innerWidth / window.innerHeight < 0.9 ? '9x16' : '21x9');
    on();
    window.addEventListener('resize', on);
    return () => window.removeEventListener('resize', on);
  }, []);

  // entry: run Appendix A's animations when the section scrolls into view
  useEffect(() => {
    const el = sectionRef.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => e?.isIntersecting && setEntered(true), { threshold: 0.35 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  // load images + lesion sample once the section is near
  useEffect(() => {
    const el = sectionRef.current;
    if (!el) return;
    let alive = true;
    const io = new IntersectionObserver(
      async ([e]) => {
        if (!e?.isIntersecting) return;
        io.disconnect();
        const want = Math.min(2520, window.innerWidth * Math.min(2, window.devicePixelRatio || 1));
        const [dis, hea, lesion] = await Promise.all([
          loadImg(pickFile(heal.files.diseased, want)),
          loadImg(pickFile(heal.files.healed, want)),
          lesionSample(`/media/heal/lesion_mask_${variant}.png`),
        ]);
        if (alive) setAssets({ dis, hea, lesion });
      },
      { rootMargin: '600px 0px' },
    );
    io.observe(el);
    return () => {
      alive = false;
      io.disconnect();
    };
  }, [variant, heal.files.diseased, heal.files.healed]);

  // the healing loop
  useEffect(() => {
    const sec = sectionRef.current;
    const canvas = canvasRef.current;
    if (!sec || !canvas || !assets || reduced) return;
    const ctx = canvas.getContext('2d')!;
    canvas.dataset.healed = assets.hea.currentSrc || assets.hea.src;
    const mask = document.createElement('canvas');
    const mctx = mask.getContext('2d', { willReadFrequently: true })!;
    const layer = document.createElement('canvas');
    const lctx = layer.getContext('2d')!;
    let W = 0, H = 0, dpr = 1, R = 260, fit: Fit = { s: 1, dx: 0, dy: 0 };
    const resize = () => {
      dpr = Math.min(2, window.devicePixelRatio || 1);
      W = sec.clientWidth;
      H = sec.clientHeight;
      canvas.width = layer.width = Math.round(W * dpr);
      canvas.height = layer.height = Math.round(H * dpr);
      canvas.style.width = `${W}px`;
      canvas.style.height = `${H}px`;
      mask.width = Math.round(W * 0.5);
      mask.height = Math.round(H * 0.5);
      R = Math.max(140, Math.min(0.17 * Math.min(W, H) * 1.6, 260));
      fit = coverFit(W, H, assets.dis.naturalWidth, assets.dis.naturalHeight);
      canvas.dataset.r = String(R); // read by the e2e healing test
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(sec);

    const mouse = { x: -999, y: -999 };
    const smooth = { x: -999, y: -999 };
    let prev: { x: number; y: number } | null = null;
    let lastMove = performance.now();
    let ghost: { t0: number } | null = null;
    let visible = false;
    let raf = 0;
    let last = performance.now();
    let lastMeter = 0;
    const touchOnly = window.matchMedia('(hover: none)').matches;

    const stamp = (x: number, y: number) => {
      const r = R * 0.5;
      const g = mctx.createRadialGradient(x * 0.5, y * 0.5, 0, x * 0.5, y * 0.5, r);
      STOPS.forEach(([o, a]) => g.addColorStop(o, `rgba(255,255,255,${a})`));
      mctx.fillStyle = g;
      mctx.beginPath();
      mctx.arc(x * 0.5, y * 0.5, r, 0, Math.PI * 2);
      mctx.fill();
    };

    const frame = (now: number) => {
      const dt = Math.min(64, now - last);
      last = now;
      const f60 = dt / (1000 / 60);
      // idle hint: after 4 s without input a ghost cursor glides on a Lissajous path for 3 s
      if (!ghost && now - lastMove > 4000) ghost = { t0: now };
      if (ghost) {
        const t = (now - ghost.t0) / 1000;
        if (t < 3) {
          mouse.x = W * (0.5 + 0.36 * Math.sin(1.3 * t + 0.4));
          mouse.y = H * (0.55 + 0.22 * Math.sin(2.1 * t));
        } else {
          ghost = null;
          lastMove = now;
          mouse.x = mouse.y = -999;
          prev = null;
        }
      }
      // smoothing (Appendix A lerp 0.1 per 60 Hz frame, time-corrected)
      const k = 1 - Math.pow(0.9, f60);
      if (smooth.x < -900 && mouse.x > -900) Object.assign(smooth, mouse);
      smooth.x += (mouse.x - smooth.x) * k;
      smooth.y += (mouse.y - smooth.y) * k;
      // decay: healed areas slowly return
      mctx.globalCompositeOperation = 'destination-out';
      mctx.fillStyle = `rgba(0,0,0,${1 - Math.pow(1 - DECAY_PER_60HZ_FRAME, f60)})`;
      mctx.fillRect(0, 0, mask.width, mask.height);
      // stamp along the path since the last frame (spacing R/4), saturating with 'lighter'
      if (mouse.x > -900) {
        if (smooth.x < -900) Object.assign(smooth, mouse);
        mctx.globalCompositeOperation = 'lighter';
        const from = prev ?? smooth;
        const dist = Math.hypot(smooth.x - from.x, smooth.y - from.y);
        const n = Math.max(1, Math.ceil(dist / (R / 4)));
        for (let i = 1; i <= n; i++) stamp(from.x + ((smooth.x - from.x) * i) / n, from.y + ((smooth.y - from.y) * i) / n);
        prev = { x: smooth.x, y: smooth.y };
      }
      // render: diseased, then healed through the mask
      const iw = assets.dis.naturalWidth, ih = assets.dis.naturalHeight;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.drawImage(assets.dis, fit.dx, fit.dy, iw * fit.s, ih * fit.s);
      lctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      lctx.globalCompositeOperation = 'source-over';
      lctx.clearRect(0, 0, W, H);
      lctx.drawImage(assets.hea, fit.dx, fit.dy, iw * fit.s, ih * fit.s);
      lctx.globalCompositeOperation = 'destination-in';
      lctx.drawImage(mask, 0, 0, W, H);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.drawImage(layer, 0, 0);
      // live meter ~6x per second: lesion pixels not yet healed / leaf pixels
      if (now - lastMeter > 166) {
        lastMeter = now;
        const md = mctx.getImageData(0, 0, mask.width, mask.height).data;
        const L = assets.lesion;
        let healed = 0;
        for (let i = 0; i < L.length; i += 2) {
          const x = fit.dx + L[i]! * iw * fit.s, y = fit.dy + L[i + 1]! * ih * fit.s;
          if (x < 0 || y < 0 || x >= W || y >= H) continue;
          healed += md[(Math.floor(y * 0.5) * mask.width + Math.floor(x * 0.5)) * 4 + 3]! / 255;
        }
        setCurrent(initial * (1 - healed / (L.length / 2)));
      }
      raf = requestAnimationFrame(frame);
    };
    const start = () => {
      if (!raf && visible && !document.hidden) {
        last = performance.now();
        raf = requestAnimationFrame(frame);
      }
    };
    const stop = () => {
      cancelAnimationFrame(raf);
      raf = 0;
    };
    const io = new IntersectionObserver(([e]) => {
      visible = !!e?.isIntersecting;
      if (visible) {
        lastMove = touchOnly ? performance.now() - 4001 : performance.now(); // touch: ghost runs on enter
        start();
      } else stop();
    });
    io.observe(sec);
    const onVis = () => (document.hidden ? stop() : start());
    document.addEventListener('visibilitychange', onVis);
    const toLocal = (cx: number, cy: number) => {
      const r = sec.getBoundingClientRect();
      return { x: cx - r.left, y: cy - r.top };
    };
    const onMove = (e: PointerEvent) => {
      Object.assign(mouse, toLocal(e.clientX, e.clientY));
      lastMove = performance.now();
      ghost = null; // real input cancels the ghost
      const c = cursorRef.current;
      if (c) {
        c.style.transform = `translate3d(${mouse.x - 14}px, ${mouse.y - 14}px, 0)`;
        c.dataset.seen = '1'; // hidden until the pointer first enters
      }
    };
    const onTouch = (e: TouchEvent) => {
      const t = e.touches[0];
      if (!t) return;
      Object.assign(mouse, toLocal(t.clientX, t.clientY));
      lastMove = performance.now();
      ghost = null;
    };
    sec.addEventListener('pointermove', onMove);
    sec.addEventListener('touchmove', onTouch, { passive: true });
    return () => {
      stop();
      ro.disconnect();
      io.disconnect();
      document.removeEventListener('visibilitychange', onVis);
      sec.removeEventListener('pointermove', onMove);
      sec.removeEventListener('touchmove', onTouch);
    };
  }, [assets, reduced, initial]);

  // reduced motion: before/after slider drives the meter
  useEffect(() => {
    if (!reduced || !assets) return;
    const L = assets.lesion;
    let left = 0;
    for (let i = 0; i < L.length; i += 2) if (L[i]! < slider) left++;
    setCurrent(initial * (1 - left / (L.length / 2)));
  }, [reduced, assets, slider, initial]);

  const [down, setDown] = useState(false);
  const [overUI, setOverUI] = useState(false);
  const healedSrc = pickFile(heal.files.healed, 1280);
  const disSrc = pickFile(heal.files.diseased, 1280);

  return (
    <section
      id="run"
      ref={sectionRef}
      aria-labelledby="run-title"
      className={`relative h-screen w-full overflow-hidden bg-black ${entered ? '' : 'anim-paused'} ${reduced ? '' : 'cursor-none'}`}
      style={{ height: '100dvh' }}
      onPointerDown={() => setDown(true)}
      onPointerUp={() => setDown(false)}
      onPointerOver={(e) => setOverUI(!!(e.target as HTMLElement).closest('a,button,input'))}
    >
      <div className="hero-zoom absolute inset-0 z-10">
        {reduced ? (
          <div className="absolute inset-0">
            <img src={disSrc} alt="Real diseased turmeric and black gram leaves on a dark background." className="absolute inset-0 h-full w-full object-cover" />
            <img src={healedSrc} alt="The same leaves with the lesions our pipeline detected removed."
                 className="absolute inset-0 h-full w-full object-cover" style={{ clipPath: `inset(0 ${100 - slider * 100}% 0 0)` }} />
          </div>
        ) : (
          <canvas ref={canvasRef} className="absolute inset-0" role="img"
                  aria-label="Real diseased leaves; moving the pointer lifts away the lesions our pipeline detected." />
        )}
      </div>

      <div aria-hidden="true" className="pointer-events-none absolute inset-x-0 bottom-0 z-40 h-1/3 bg-gradient-to-t from-black/85 to-transparent" />
      <div className="pointer-events-none absolute left-0 right-0 top-[14%] z-50 flex flex-col items-center px-5 text-center">
        <h2 id="run-title" className="leading-[0.95] text-white">
          <span className="hero-anim hero-reveal block font-playfair text-5xl font-normal italic sm:text-7xl md:text-8xl"
                style={{ letterSpacing: '-0.05em', animationDelay: '0.25s' }}>
            Every spot
          </span>
          <span className="hero-anim hero-reveal -mt-1 block text-5xl font-normal sm:text-7xl md:text-8xl"
                style={{ letterSpacing: '-0.08em', animationDelay: '0.42s' }}>
            has a signature
          </span>
        </h2>
      </div>

      <div className="hero-anim hero-fade absolute bottom-14 left-10 z-50 hidden max-w-[260px] sm:block md:left-14" style={{ animationDelay: '0.7s' }}>
        <p className="text-sm leading-relaxed text-white/80">
          Move across the leaves. Where you pass, the lesions our pipeline detected lift away — the same mask we measure.
        </p>
        <p className="mono mt-4 text-xs text-white/80" aria-live="polite">
          Affected area&nbsp;&nbsp;{pctRaw(initial)}&nbsp;&nbsp;→&nbsp;&nbsp;<span data-testid="meter">{pctRaw(current)}</span>
        </p>
        <div className="mt-2 h-px w-full bg-white/15">
          <div className="h-full origin-left bg-leaf" style={{ transform: `scaleX(${initial > 0 ? 1 - current / initial : 0})` }} />
        </div>
        <p className="mt-3 text-[11px] leading-snug text-white/55">
          Visualisation: lesions inpainted from our own segmentation mask — not a medical prediction.
        </p>
      </div>

      <div className="hero-anim hero-fade absolute bottom-10 left-5 right-5 z-50 flex max-w-full flex-col items-start gap-4 sm:bottom-24 sm:left-auto sm:right-10 sm:max-w-[260px] sm:gap-5 md:right-14"
           style={{ animationDelay: '0.85s' }}>
        <p className="text-xs leading-relaxed text-white/80 sm:text-sm">
          Image processing measures it. A CNN learns it. Compare both on your own leaf.
        </p>
        {reduced && (
          <label className="w-full">
            <span className="caption text-[10px] text-white/70">Before / after</span>
            <input type="range" min={0} max={1} step={0.01} value={slider} onChange={(e) => setSlider(Number(e.target.value))}
                   className="mt-2 w-full accent-[var(--leaf)]" aria-label="Reveal the healed leaves" />
          </label>
        )}
        <div className="flex flex-wrap gap-3">
          <Link to="/analyze"
                className="cursor-pointer rounded-full bg-[#e8702a] px-7 py-3 text-sm font-medium text-white transition-all hover:scale-[1.03] hover:bg-[#d2611f] hover:shadow-lg hover:shadow-[#e8702a]/30 active:scale-95">
            Run the model
          </Link>
          {DEPLOYED_APP_URL && (
            <a href={DEPLOYED_APP_URL} target="_blank" rel="noreferrer"
               className="inline-flex cursor-pointer items-center gap-1.5 rounded-full border border-white/40 px-7 py-3 text-sm font-medium text-white transition-colors hover:bg-white/10">
              Open deployed app <ArrowUpRight size={16} aria-hidden="true" />
            </a>
          )}
        </div>
      </div>

      {!reduced && (
        <div ref={cursorRef} aria-hidden="true"
             className="pointer-events-none absolute left-0 top-0 z-[60] h-7 w-7 rounded-full border border-white/70 opacity-0 transition-opacity duration-200 data-[seen]:opacity-100 [@media(pointer:coarse)]:hidden"
             style={{ opacity: overUI ? 0 : undefined }}>
          <span className="absolute left-1/2 top-1/2 h-1 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white transition-transform duration-150"
                style={{ transform: `translate(-50%,-50%) scale(${down ? 0.8 : 1})` }} />
        </div>
      )}
    </section>
  );
}
