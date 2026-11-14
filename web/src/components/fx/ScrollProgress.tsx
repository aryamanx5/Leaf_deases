import { useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';

/** A hairline along the top edge that fills as the page is read. Written straight to the element: no re-renders. */
export default function ScrollProgress() {
  const bar = useRef<HTMLDivElement>(null);
  const { pathname } = useLocation();
  useEffect(() => {
    const el = bar.current;
    if (!el) return;
    let raf = 0;
    const paint = () => {
      raf = 0;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      el.style.transform = `scaleX(${max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0})`;
    };
    const on = () => {
      if (!raf) raf = requestAnimationFrame(paint);
    };
    paint();
    window.addEventListener('scroll', on, { passive: true });
    window.addEventListener('resize', on);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('scroll', on);
      window.removeEventListener('resize', on);
    };
  }, [pathname]);
  return (
    <div aria-hidden="true" className="pointer-events-none fixed inset-x-0 top-0 z-[120] h-[2px]">
      <div ref={bar} className="h-full origin-left bg-gradient-to-r from-leaf via-leaf to-turmeric will-change-transform" style={{ transform: 'scaleX(0)' }} />
    </div>
  );
}
