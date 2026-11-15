import { useEffect, useRef } from 'react';
import { useReducedMotion } from '@/lib/motion';

/** A soft light that trails the pointer across the page (mouse only; off for reduced motion). Decorative. */
export default function Glow() {
  const ref = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();
  useEffect(() => {
    const el = ref.current;
    if (!el || reduced || !window.matchMedia('(hover: hover)').matches) return;
    let x = 0, y = 0, cx = 0, cy = 0, raf = 0, seen = false;
    const tick = () => {
      cx += (x - cx) * 0.14;
      cy += (y - cy) * 0.14;
      el.style.transform = `translate3d(${cx.toFixed(1)}px, ${cy.toFixed(1)}px, 0)`;
      raf = Math.abs(x - cx) + Math.abs(y - cy) > 0.5 ? requestAnimationFrame(tick) : 0;
    };
    const move = (e: PointerEvent) => {
      if (e.pointerType !== 'mouse') return;
      x = e.clientX;
      y = e.clientY;
      if (!seen) {
        seen = true;
        cx = x;
        cy = y;
        el.dataset.on = '1';
      }
      if (!raf) raf = requestAnimationFrame(tick);
    };
    const leave = () => {
      seen = false;
      delete el.dataset.on;
    };
    window.addEventListener('pointermove', move, { passive: true });
    document.documentElement.addEventListener('pointerleave', leave);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('pointermove', move);
      document.documentElement.removeEventListener('pointerleave', leave);
    };
  }, [reduced]);
  return <div ref={ref} className="glow" aria-hidden="true" />;
}
