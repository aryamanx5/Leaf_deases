import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import Picture from '@/components/Picture';
import { BRAND_NAME } from '@/content/brand';
import type { Site } from '@/lib/site';
import { ScrollTrigger } from '@/lib/smooth';
import { useReducedMotion } from '@/lib/motion';

type Box = { left: number; top: number; width: number; height: number };

/** Cover-fit rectangle of an image inside a container (what object-fit: cover would do). */
function coverBox(cw: number, ch: number, iw: number, ih: number, focusX: number, focusY: number): Box {
  const s = Math.max(cw / iw, ch / ih);
  const width = iw * s;
  const height = ih * s;
  const left = Math.min(0, Math.max(cw - width, cw / 2 - focusX * width));
  const top = Math.min(0, Math.max(ch - height, ch / 2 - focusY * height));
  return { left, top, width, height };
}

const clampExpr = (from: number, span: number) => `clamp(0, calc((var(--p) - ${from}) / ${span}), 1)`;

export default function Hero({ site }: { site: Site }) {
  const h = site.hero;
  const reduced = useReducedMotion();
  const sectionRef = useRef<HTMLElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState<Box | null>(null);
  const [scrolled, setScrolled] = useState(false);
  const [mx, my] = h.anchors.midrib;

  useLayoutEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const fit = () => setBox(coverBox(el.clientWidth, el.clientHeight, h.image.w, h.image.h, mx, my));
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    return () => ro.disconnect();
  }, [h.image.w, h.image.h, mx, my]);

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
      },
    });
    return () => st.kill();
  }, [reduced]);

  const outline = h.outline.map(([x, y]) => `${x},${y}`).join(' ');
  const zoom = `calc(1 + 0.35 * ${clampExpr(0.25, 0.45)})`;
  const late = clampExpr(0.7, 0.2);
  const callout = (key: 'midrib' | 'venation' | 'outline', label: string, dx: number, dy: number, delay: number) => {
    const [x, y] = h.anchors[key];
    return (
      <div
        key={key}
        className="pointer-events-none absolute"
        style={{
          left: `${x * 100}%`,
          top: `${y * 100}%`,
          opacity: `calc(${clampExpr(0.25 + delay, 0.08)} * (1 - ${late}))`,
        }}
      >
        <svg width={Math.abs(dx) + 8} height={Math.abs(dy) + 8} className="absolute overflow-visible" style={{ left: 0, top: 0 }}>
          <circle cx="0" cy="0" r="3" fill="var(--paper)" />
          <line x1="0" y1="0" x2={dx} y2={dy} stroke="var(--paper)" strokeWidth="1" strokeOpacity="0.7" />
        </svg>
        <span
          className="caption absolute whitespace-nowrap text-[11px] text-paper"
          style={{ left: dx + (dx >= 0 ? 8 : -8), top: dy - 8, transform: dx < 0 ? 'translateX(-100%)' : undefined,
                   scale: `calc(1 / ${zoom})`, transformOrigin: dx < 0 ? 'right center' : 'left center' }}
        >
          {label}
        </span>
      </div>
    );
  };

  return (
    <section
      id="field"
      ref={sectionRef}
      aria-label="Field: reading a turmeric leaf"
      className="relative"
      style={{ height: reduced ? '100dvh' : '250vh', ['--p' as string]: 0 }}
    >
      <div ref={stageRef} className="sticky top-0 h-[100dvh] w-full overflow-hidden bg-ink">
        {box && (
          <div
            className="absolute will-change-transform"
            style={{
              left: box.left,
              top: box.top,
              width: box.width,
              height: box.height,
              transform: `scale(${zoom})`,
              transformOrigin: `${mx * 100}% ${my * 100}%`,
            }}
          >
            <div className={reduced ? '' : 'hero-zoom'} style={{ position: 'absolute', inset: 0 }}>
              <Picture
                src={h.image}
                alt="Close-up of a turmeric leaf with morning dew: a long blade, a strong midrib and parallel veins."
                eager
                sizes="(max-width: 768px) 120vw, 100vw"
                className="absolute inset-0 h-full w-full"
                style={{ filter: `saturate(calc(1 - 0.85 * ${late})) brightness(calc(1 - 0.35 * ${late}))` }}
              />
              {/* the leaf itself keeps its colour inside the Stage 1 outline */}
              <div
                className="absolute inset-0"
                style={{ clipPath: `polygon(${h.outline.map(([x, y]) => `${x * 100}% ${y * 100}%`).join(',')})`, opacity: late }}
              >
                <Picture src={h.image} alt="" sizes="100vw" className="absolute inset-0 h-full w-full" />
              </div>
              <svg viewBox="0 0 1 1" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
                <polygon
                  points={outline}
                  fill="none"
                  stroke="var(--leaf)"
                  strokeWidth="2.5"
                  vectorEffect="non-scaling-stroke"
                  pathLength={1}
                  style={{ strokeDasharray: 1, strokeDashoffset: `calc(1 - ${late})` }}
                />
              </svg>
              {callout('midrib', 'midrib', 90, -70, 0)}
              {callout('venation', 'parallel venation', -110, 60, 0.08)}
              {callout('outline', 'leaf outline (Stage 1)', 80, 90, 0.16)}
            </div>
          </div>
        )}
        <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,rgba(10,14,11,0.82)_0%,rgba(10,14,11,0.25)_38%,rgba(10,14,11,0.15)_60%,rgba(10,14,11,0.9)_100%)]" />

        <div
          className="pointer-events-none absolute left-0 right-0 top-[16%] flex flex-col items-center px-5 text-center"
          style={{ opacity: `calc(1 - ${clampExpr(0.62, 0.14)})` }}
        >
          <p className="caption hero-anim hero-fade mb-6 text-[11px] text-white/70 sm:text-[12px]" style={{ animationDelay: '0.1s' }}>
            Turmeric · Black gram · Image processing + CNN
          </p>
          <h1 className="leading-[0.95] text-white">
            <span className="hero-anim hero-reveal block font-playfair text-5xl font-normal italic sm:text-7xl md:text-8xl"
                  style={{ letterSpacing: '-0.05em', animationDelay: '0.25s' }}>
              Read the leaf
            </span>
            <span className="hero-anim hero-reveal -mt-1 block text-5xl font-normal sm:text-7xl md:text-8xl"
                  style={{ letterSpacing: '-0.06em', animationDelay: '0.42s' }}>
              before the harvest does
            </span>
          </h1>
        </div>

        <div className="hero-anim hero-fade absolute bottom-10 left-5 max-w-[300px] sm:bottom-14 sm:left-10 md:left-14"
             style={{ animationDelay: '0.7s' }}>
          <p className="text-sm leading-relaxed text-white/80">
            {BRAND_NAME} reads turmeric and black gram leaves two ways: explicit image processing that measures every
            lesion, and a CNN that learns its own features. Then it compares them.
          </p>
          <p className="mono mt-3 text-[10px] text-white/45">
            Photo: {h.credit.author}, {h.credit.licence} (Wikimedia Commons)
          </p>
        </div>
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
