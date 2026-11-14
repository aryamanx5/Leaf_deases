import { useEffect, useRef } from 'react';
import LazyVideo from '@/components/fx/LazyVideo';
import Marquee from '@/components/fx/Marquee';
import { useReducedMotion } from '@/lib/motion';
import { ScrollTrigger } from '@/lib/smooth';

const LINES = ['One field.', 'Thousands of leaves.', 'One pair of eyes.'];
const WORDS = ['Leaf blotch', 'Yellow mosaic', 'Anthracnose', 'Powdery mildew', 'Leaf crinkle', 'Dry leaf', 'Healthy'];
const step = (from: number, span: number) => `clamp(0, calc((var(--p) - ${from}) / ${span}), 1)`;

/**
 * Film interlude between the problem and the crops: a framed clip of a field opens to full bleed as the page
 * scrolls, and three short lines land over it. Reduced motion: the still frame and the lines, nothing pinned.
 */
export default function Interlude() {
  const reduced = useReducedMotion();
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el || reduced) return;
    const st = ScrollTrigger.create({
      trigger: el,
      start: 'top top',
      end: 'bottom bottom',
      scrub: true,
      onUpdate: (s) => el.style.setProperty('--p', s.progress.toFixed(4)),
    });
    return () => st.kill();
  }, [reduced]);

  const open = step(0, 0.45); // 0 -> 1 while the frame opens
  const film = (
    <LazyVideo clip="field" className="absolute inset-0 h-full w-full" focus="center 60%" />
  );

  if (reduced) {
    return (
      <section id="interlude" aria-labelledby="interlude-title" className="bg-ink">
        <div className="container-grid section-pad">
          <p className="caption mb-6 text-muted">Interlude · The field</p>
          <div className="relative aspect-video w-full overflow-hidden rounded-[18px]">{film}</div>
          <h2 id="interlude-title" className="mt-10 max-w-[900px] text-4xl font-normal leading-[1.05] tracking-[-0.04em] sm:text-6xl">
            {LINES[0]} {LINES[1]} <span className="font-playfair italic">{LINES[2]}</span>
          </h2>
          <p className="mt-6 max-w-[520px] text-[17px] leading-relaxed text-muted">
            Nobody can turn over every leaf. A camera can look at each one, and say which ones deserve a second look.
          </p>
        </div>
        <Marquee words={WORDS} label="What the system reads" />
      </section>
    );
  }

  return (
    <section id="interlude" aria-labelledby="interlude-title" className="relative bg-ink">
      <div ref={ref} style={{ height: '260vh', ['--p' as string]: 0 }}>
        <div className="sticky top-0 h-[100dvh] w-full overflow-hidden">
          {/* the clip fills the stage; a clip-path frames it, then lets go */}
          <div className="absolute inset-0 will-change-[clip-path]"
               style={{
                 clipPath: `inset(calc((1 - ${open}) * 22vh) calc((1 - ${open}) * 18vw) calc((1 - ${open}) * 14vh) calc((1 - ${open}) * 18vw) round calc((1 - ${open}) * 28px))`,
               }}>
            <div className="absolute inset-0" style={{ transform: `scale(calc(1.18 - 0.18 * ${open}))` }}>{film}</div>
            <div className="pointer-events-none absolute inset-0 bg-ink" style={{ opacity: `calc(0.15 + 0.4 * ${step(0.4, 0.2)})` }} />
            <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,rgba(10,14,11,0.55)_0%,rgba(10,14,11,0)_30%,rgba(10,14,11,0)_60%,rgba(10,14,11,0.9)_100%)]" />
          </div>

          <p className="caption absolute left-1/2 top-[11vh] -translate-x-1/2 whitespace-nowrap text-[11px] text-muted"
             style={{ opacity: `calc(1 - ${step(0.05, 0.2)})` }}>
            Interlude · The field
          </p>

          <div className="pointer-events-none absolute inset-0 flex items-center">
            <div className="container-grid">
              <h2 id="interlude-title" className="text-[44px] font-normal leading-[0.98] tracking-[-0.05em] text-white [text-shadow:0_2px_30px_rgba(10,14,11,0.55)] sm:text-7xl lg:text-[112px]">
                {LINES.map((l, i) => (
                  <span key={l} className={`block ${i === 2 ? 'font-playfair italic' : ''}`}
                        style={{
                          opacity: step(0.42 + i * 0.12, 0.1),
                          transform: `translateY(calc((1 - ${step(0.42 + i * 0.12, 0.1)}) * 36px))`,
                        }}>
                    {l}
                  </span>
                ))}
              </h2>
              <p className="mt-8 max-w-[440px] text-[16px] leading-relaxed text-white/85 sm:text-[17px]"
                 style={{ opacity: step(0.8, 0.1) }}>
                Nobody can turn over every leaf. A camera can look at each one, and say which ones deserve a second look.
              </p>
            </div>
          </div>
        </div>
      </div>
      <Marquee words={WORDS} label="What the system reads" />
    </section>
  );
}
