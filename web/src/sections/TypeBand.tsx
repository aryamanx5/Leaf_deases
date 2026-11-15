import { useEffect, useRef } from 'react';
import LazyVideo from '@/components/fx/LazyVideo';
import { useReducedMotion } from '@/lib/motion';
import { ScrollTrigger } from '@/lib/smooth';

/**
 * A band of very large type with film playing inside the letters: the clip fills the band, and a dark sheet with
 * white letters is multiplied over it, so the picture survives only where the letters are. The two lines drift in
 * opposite directions as the band scrolls through (still for reduced motion, where the clip is a still frame).
 */
export default function TypeBand() {
  const reduced = useReducedMotion();
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el || reduced) return;
    const st = ScrollTrigger.create({
      trigger: el,
      start: 'top bottom',
      end: 'bottom top',
      scrub: true,
      onUpdate: (s) => el.style.setProperty('--p', s.progress.toFixed(4)),
    });
    return () => st.kill();
  }, [reduced]);

  return (
    <section id="band" ref={ref} aria-labelledby="band-title" className="relative overflow-hidden bg-black" style={{ ['--p' as string]: 0.5 }}>
      <div className="absolute inset-0">
        <LazyVideo clip="macro" className="h-full w-full" sizes="100vw" credit={false} />
      </div>
      <div className="type-sheet relative flex flex-col justify-center py-[9vw]">
        <h2 id="band-title" className="select-none whitespace-nowrap text-[23vw] font-bold uppercase leading-[0.84] tracking-[-0.055em] text-white">
          <span className="block will-change-transform" style={{ transform: 'translate3d(calc((0.5 - var(--p)) * 30vw - 3vw), 0, 0)' }}>Read the</span>
          <span className="block will-change-transform" style={{ transform: 'translate3d(calc((var(--p) - 0.5) * 30vw + 14vw), 0, 0)' }}>leaf.</span>
        </h2>
      </div>
      <div aria-hidden="true" className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,var(--ink)_0%,rgba(10,14,11,0)_14%,rgba(10,14,11,0)_86%,#000_100%)]" />
      <a href="https://www.pexels.com/video/extreme-close-up-of-a-green-leaf-15783159/" target="_blank" rel="noreferrer"
         className="mono absolute bottom-3 right-4 rounded-full bg-ink/80 px-2.5 py-1 text-[10px] text-white/80 underline-offset-2 hover:underline">
        Footage: CW Tan / Pexels
      </a>
    </section>
  );
}
