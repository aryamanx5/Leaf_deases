import { useEffect, useState } from 'react';
import { useInView, useReducedMotion } from '@/lib/motion';

/** Animates every number inside `value` (e.g. "37.6–52.9%") from 0 when it scrolls into view. */
export default function Counter({ value, duration = 1200, className }: { value: string; duration?: number; className?: string }) {
  const [ref, inView] = useInView<HTMLSpanElement>({ threshold: 0.6 });
  const reduced = useReducedMotion();
  const [t, setT] = useState(0);
  useEffect(() => {
    if (!inView) return;
    if (reduced) {
      setT(1);
      return;
    }
    let raf = 0;
    const t0 = performance.now();
    const step = (now: number) => {
      const k = Math.min(1, (now - t0) / duration);
      setT(1 - Math.pow(1 - k, 3));
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [inView, reduced, duration]);
  const text = value.replace(/\d+(?:\.\d+)?/g, (m) => {
    const dec = m.includes('.') ? m.split('.')[1]!.length : 0;
    return (Number(m) * t).toFixed(dec);
  });
  return (
    <span ref={ref} className={className}>
      <span className="sr-only">{value}</span>
      <span aria-hidden="true">{text}</span>
    </span>
  );
}
