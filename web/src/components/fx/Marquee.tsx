import { Fragment, useEffect, useRef } from 'react';
import { useReducedMotion } from '@/lib/motion';

/**
 * Two bands of words drifting in opposite directions (CSS only; still under reduced motion, paused on hover).
 * Decorative: the same words are listed once for screen readers.
 */
export default function Marquee({ words, label }: { words: string[]; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();
  // the bands lean with the scroll speed and settle when it stops
  useEffect(() => {
    const el = ref.current;
    if (!el || reduced) return;
    let last = window.scrollY, raf = 0, rest = 0, near = false;
    const io = new IntersectionObserver(([e]) => { near = !!e?.isIntersecting; }, { rootMargin: '200px 0px' });
    io.observe(el);
    const paint = () => {
      raf = 0;
      const v = window.scrollY - last;
      last = window.scrollY;
      el.style.setProperty('--lean', `${Math.max(-12, Math.min(12, -v * 0.35)).toFixed(2)}deg`);
      window.clearTimeout(rest);
      rest = window.setTimeout(() => el.style.setProperty('--lean', '0deg'), 120);
    };
    const on = () => {
      if (near && !raf) raf = requestAnimationFrame(paint);
      else if (!near) last = window.scrollY;
    };
    window.addEventListener('scroll', on, { passive: true });
    return () => {
      io.disconnect();
      cancelAnimationFrame(raf);
      window.clearTimeout(rest);
      window.removeEventListener('scroll', on);
    };
  }, [reduced]);
  const row = (rev: boolean) => (
    <div className={`marquee-track ${rev ? 'rev' : ''}`} aria-hidden="true">
      {[0, 1].map((copy) => (
        <div key={copy} className="flex shrink-0 items-center">
          {words.map((w, i) => (
            <Fragment key={`${copy}-${w}`}>
              <span className={`whitespace-nowrap px-6 text-[44px] leading-none tracking-[-0.04em] sm:text-[72px] lg:text-[96px] ${
                (i + (rev ? 1 : 0)) % 2 ? 'font-playfair italic text-text/90' : 'text-text/50'}`}>
                {w}
              </span>
              <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${i % 2 ? 'bg-turmeric' : 'bg-leaf'}`} />
            </Fragment>
          ))}
        </div>
      ))}
    </div>
  );
  return (
    <div ref={ref} className="marquee overflow-hidden border-y hairline bg-ink py-8 sm:py-10">
      <p className="sr-only">{label}: {words.join(', ')}.</p>
      <div className="flex flex-col gap-4 sm:gap-6">
        {row(false)}
        {row(true)}
      </div>
    </div>
  );
}
