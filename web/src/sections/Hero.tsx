import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import Picture from '@/components/Picture';
import { BRAND_NAME } from '@/content/brand';
import type { Sources } from '@/lib/site';
import { ScrollTrigger } from '@/lib/smooth';
import { useReducedMotion } from '@/lib/motion';

/** The grower in the greenhouse: a real photo (poster, LCP, reduced-motion) and a 10 s loop of the same scene. */
const POSTER: Sources = {
  webp: { 960: '/media/hero/farmer-960.webp', 1356: '/media/hero/farmer-1356.webp' },
  avif: { 960: '/media/hero/farmer-960.avif', 1356: '/media/hero/farmer-1356.avif' },
  w: 1356,
  h: 1528,
};
// re-encoded from farmer.mp4 with a keyframe every 6 frames, so seeking for the scroll-scrub stays smooth;
// 1530 is a 1.5x Lanczos + unsharp-mask upscale for large / high-density frames
const VIDEO = { sharp: '/media/hero/farmer-1530.mp4', full: '/media/hero/farmer-1020.mp4', light: '/media/hero/farmer-480.mp4' };
const AR = 1020 / 1150; // width / height of the media
const SIDE_AR = 0.96; // the side-layout frame is a little wider than the media: a bigger frame, cover-cropped
// only by a few percent of foreground and glass (object-position keeps face, hands and leaves in)
const SCRUB_END = 0.85; // scroll progress at which the forward half of the clip (she turns to the lens) is reached
const BLEND = 0.15; // scroll distance over which a scrub that starts mid-loop eases into the position mapping

type Frame = { left: number; top: number; width: number; height: number; side: boolean };

/**
 * Where the framed media sits. Side layout (wide screens): a tall frame, slightly right of centre, from under the
 * nav to near the bottom, the headline overlapping its left edge. Stacked layout (portrait): the largest frame
 * that fits under the nav with room left for the copy below it. The whole subject is always in view.
 */
function frameFor(w: number, h: number): Frame {
  if ((w >= 1024 && w / h >= 1) || w / h >= 1.2) {
    const top = 80;
    const height = h - top - 16;
    const width = height * SIDE_AR;
    return { left: Math.min(w - width - 24, w * 0.58 - width / 2), top, width, height, side: true };
  }
  const top = 72;
  const copy = w < 640 ? 290 : w < 768 ? 400 : 480; // eyebrow + three headline lines (5xl / 7xl / 8xl) + paragraph
  let height = Math.max(h * 0.4, h - top - copy);
  let width = height * AR;
  if (width > w - 32) {
    width = w - 32;
    height = width / AR;
  }
  return { left: (w - width) / 2, top, width, height, side: false };
}

/** The lightest clip that is still sharp for this frame on this screen (Save-Data / 2g: the light one). */
function videoFor(frameWidth: number): string {
  const c = (navigator as Navigator & { connection?: { saveData?: boolean; effectiveType?: string } }).connection;
  if (c?.saveData || /2g$/.test(c?.effectiveType ?? '')) return VIDEO.light;
  return frameWidth * (window.devicePixelRatio || 1) > 1100 ? VIDEO.sharp : VIDEO.full;
}

const clampExpr = (from: number, span: number) => `clamp(0, calc((var(--p) - ${from}) / ${span}), 1)`;

export default function Hero() {
  const reduced = useReducedMotion();
  const sectionRef = useRef<HTMLElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [ready, setReady] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const onProgress = useRef<(p: number) => void>(() => {});
  const hasFrame = frame !== null; // the <video> lives in the frame, which exists after the first measurement

  useLayoutEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const fit = () => setFrame(frameFor(el.clientWidth, el.clientHeight));
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const sec = sectionRef.current;
    if (!sec) return;
    if (reduced) {
      sec.style.setProperty('--p', '0.18');
      return;
    }
    const st = ScrollTrigger.create({
      trigger: sec,
      start: 'top top',
      end: 'bottom bottom',
      scrub: true,
      onUpdate: (self) => {
        sec.style.setProperty('--p', self.progress.toFixed(4));
        if (self.progress > 0.01) setScrolled(true);
        onProgress.current(self.progress);
      },
    });
    return () => st.kill();
  }, [reduced]);

  // The old hero's large eager image held the window load event until the page had settled, so ScrollTrigger's
  // automatic refresh measured the final layout. The new hero loads light and early, and late reflows (web fonts,
  // site data) would leave the scroll-linked sections below a few pixels off. Re-measure once the page height has
  // been stable for 200 ms, during the first few seconds only.
  useEffect(() => {
    let t = 0;
    const ro = new ResizeObserver(() => {
      window.clearTimeout(t);
      t = window.setTimeout(() => ScrollTrigger.refresh(), 200);
    });
    ro.observe(document.body);
    const stop = window.setTimeout(() => ro.disconnect(), 4000);
    return () => {
      ro.disconnect();
      window.clearTimeout(t);
      window.clearTimeout(stop);
    };
  }, []);

  // The clip: loaded after first paint (the poster is the LCP), idle-loops, is scrubbed by scroll, pauses off screen.
  useEffect(() => {
    const v = videoRef.current;
    const sec = sectionRef.current;
    if (reduced || !v || !sec) return;
    let isReady = false;
    let inView = true;
    let scrubbing = false;
    let cur = 0;
    let target = 0;
    let p0 = 0;
    let off = 0;
    let raf = 0;
    let idle = 0;
    const half = () => v.duration / 2; // authored forward-then-reversed: [0, D/2] is the forward pass
    const mapped = (p: number) => (Math.min(p, SCRUB_END) / SCRUB_END) * half();
    const sync = () => {
      const play = isReady && inView && !scrubbing && document.visibilityState === 'visible';
      if (play) void v.play().catch(() => {});
      else v.pause();
    };
    const tick = () => {
      cur += (target - cur) * 0.22;
      if (Math.abs(target - cur) < 0.004) cur = target;
      if (!v.seeking && Math.abs(v.currentTime - cur) > 1 / 48) v.currentTime = cur;
      raf = cur !== target || v.seeking ? requestAnimationFrame(tick) : 0;
    };
    onProgress.current = (p) => {
      if (!isReady || !inView || !Number.isFinite(v.duration)) return;
      if (!scrubbing) {
        scrubbing = true;
        v.pause();
        let t = v.currentTime;
        if (t > half()) {
          t = v.duration - t; // the same picture on the forward pass (palindrome)
          v.currentTime = t;
        }
        cur = t;
        p0 = p;
        off = t - mapped(p);
      }
      target = Math.min(half(), Math.max(0, mapped(p) + off * Math.max(0, 1 - Math.abs(p - p0) / BLEND)));
      if (!raf) raf = requestAnimationFrame(tick);
      window.clearTimeout(idle);
      idle = window.setTimeout(() => {
        scrubbing = false;
        sync();
      }, 600);
    };
    const onData = () => {
      isReady = true;
      setReady(true);
      sync();
    };
    v.addEventListener('loadeddata', onData);
    // -1px: a hero that only touches the viewport edge (next section exactly at the top) counts as off screen
    const io = new IntersectionObserver(([e]) => {
      inView = !!e?.isIntersecting;
      sync();
    }, { rootMargin: '-1px 0px' });
    io.observe(sec);
    document.addEventListener('visibilitychange', sync);
    const start = () => {
      v.src = videoFor(v.getBoundingClientRect().width);
      v.load();
    };
    const later = () => {
      if (typeof window.requestIdleCallback === 'function') window.requestIdleCallback(start, { timeout: 1500 });
      else setTimeout(start, 300); // Safari has no requestIdleCallback
    };
    if (document.readyState === 'complete') later();
    else window.addEventListener('load', later, { once: true });
    return () => {
      onProgress.current = () => {};
      window.removeEventListener('load', later);
      document.removeEventListener('visibilitychange', sync);
      v.removeEventListener('loadeddata', onData);
      io.disconnect();
      cancelAnimationFrame(raf);
      window.clearTimeout(idle);
      v.pause();
      v.removeAttribute('src');
      v.load();
    };
  }, [reduced, hasFrame]);

  const late = clampExpr(0.7, 0.2);
  const exitFilter = `saturate(calc(1 - 0.85 * ${late})) brightness(calc(1 - 0.35 * ${late}))`;
  const side = frame?.side ?? true;

  return (
    <section
      id="field"
      ref={sectionRef}
      aria-label="Field: a grower inspecting plants in a greenhouse"
      className="relative"
      style={{ height: reduced ? '100dvh' : '250vh', ['--p' as string]: 0 }}
    >
      <div ref={stageRef} className="sticky top-0 h-[100dvh] w-full overflow-hidden bg-ink">
        {/* ambient: the same photo, tiny, pre-blurred and darkened, so the stage still reads full-bleed */}
        <div className="absolute inset-0" style={{ filter: exitFilter }} aria-hidden="true">
          <img src="/media/hero/farmer-ambient.webp" alt="" className="absolute inset-0 h-full w-full scale-110 object-cover opacity-80" />
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_58%_45%,rgba(10,14,11,0.45)_0%,rgba(10,14,11,0.82)_70%)]" />
        </div>
        <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,rgba(10,14,11,0.7)_0%,rgba(10,14,11,0)_14%,rgba(10,14,11,0)_72%,rgba(10,14,11,0.9)_100%)]" />

        {frame && (
          <div
            className="absolute overflow-hidden rounded-[24px] shadow-[0_30px_80px_rgba(0,0,0,0.45)] will-change-transform"
            data-testid="hero-frame"
            style={{
              left: frame.left,
              top: frame.top,
              width: frame.width,
              height: frame.height,
              transform: `scale(calc(1 + 0.04 * ${clampExpr(0, SCRUB_END)}))`,
              filter: exitFilter,
            }}
          >
            <Picture
              src={POSTER}
              alt="A grower in a greenhouse parting the leaves of a plant to look at them closely, hanging plants all around."
              eager
              sizes={`${Math.round(frame.width)}px`}
              className="absolute inset-0 h-full w-full object-cover object-[50%_22%]"
            />
            {!reduced && (
              <video
                ref={videoRef}
                muted
                playsInline
                loop
                preload="auto"
                disablePictureInPicture
                aria-hidden="true"
                tabIndex={-1}
                data-testid="hero-video"
                className={`absolute inset-0 h-full w-full object-cover object-[50%_22%] transition-opacity duration-[800ms] ease-out ${ready ? 'opacity-100' : 'opacity-0'}`}
              />
            )}
            {/* keeps the overlapping headline legible over the bright glasshouse without touching her face or hands */}
            <div className={`pointer-events-none absolute inset-0 ${side
              ? 'bg-[linear-gradient(90deg,rgba(10,14,11,0.55)_0%,rgba(10,14,11,0)_32%)]'
              : 'bg-[linear-gradient(0deg,rgba(10,14,11,0.75)_0%,rgba(10,14,11,0)_22%)]'}`} />
          </div>
        )}

        {/* exit: the lower stage sinks into the ink of the next section, as the old hero's ending did */}
        <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,rgba(10,14,11,0)_45%,rgba(10,14,11,0.85)_100%)]"
             style={{ opacity: late }} aria-hidden="true" />

        <div
          className="pointer-events-none absolute inset-x-0"
          style={{
            top: side || !frame ? '22%' : frame.top + frame.height - 40,
            opacity: `calc(1 - 0.7 * ${clampExpr(0.22, 0.12)} - 0.3 * ${clampExpr(0.7, 0.1)})`,
          }}
        >
          <div className={`container-grid flex flex-col ${side ? 'items-start text-left' : 'items-center text-center'}`}>
            <p className="caption hero-anim hero-fade mb-6 text-[11px] text-white/70 sm:text-[12px]" style={{ animationDelay: '0.1s' }}>
              Turmeric · Black gram · Image processing + CNN
            </p>
            <h1 className="leading-[0.95] text-white [text-shadow:0_2px_24px_rgba(10,14,11,0.35)]">
              <span className="hero-anim hero-reveal block font-playfair text-5xl font-normal italic sm:text-7xl md:text-8xl"
                    style={{ letterSpacing: '-0.05em', animationDelay: '0.25s' }}>
                Read the leaf
              </span>
              <span className="hero-anim hero-reveal -mt-1 block text-5xl font-normal sm:text-7xl md:text-8xl"
                    style={{ letterSpacing: '-0.06em', animationDelay: '0.42s' }}>
                <span className="block">before the</span>
                <span className="block">harvest does</span>
              </span>
            </h1>
            {!side && (
              <p className="hero-anim hero-fade mt-5 max-w-[340px] text-sm leading-relaxed text-white/80" style={{ animationDelay: '0.7s' }}>
                {BRAND_NAME} reads turmeric and black gram leaves two ways: explicit image processing that measures every
                lesion, and a CNN that learns its own features. Then it compares them.
              </p>
            )}
          </div>
        </div>

        {side && (
          <div className="hero-anim hero-fade absolute bottom-10 left-5 max-w-[300px] sm:bottom-14 sm:left-10 md:left-14"
               style={{ animationDelay: '0.7s' }}>
            <p className="text-sm leading-relaxed text-white/80">
              {BRAND_NAME} reads turmeric and black gram leaves two ways: explicit image processing that measures every
              lesion, and a CNN that learns its own features. Then it compares them.
            </p>
          </div>
        )}
        <div className="hero-anim hero-fade absolute bottom-10 right-5 hidden flex-col items-center gap-3 sm:flex md:right-14"
             style={{ animationDelay: '0.85s' }} aria-hidden="true">
          <span className="caption text-[11px] text-white/70">Scroll</span>
          <span className="relative block h-10 w-px overflow-hidden bg-white/15">
            <span className={`absolute inset-0 bg-white/80 ${scrolled || reduced ? '' : 'scroll-line'}`}
                  style={{ transform: scrolled ? 'scaleY(0)' : undefined }} />
          </span>
        </div>
      </div>
    </section>
  );
}
