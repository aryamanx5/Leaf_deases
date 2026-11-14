import { useEffect, type RefObject } from 'react';
import { ScrollTrigger } from '@/lib/smooth';

/**
 * Sections below a block whose height can change after load (a class with a longer description, an opened answer)
 * keep scroll-linked positions measured at load. Re-measure them shortly after the block's height settles.
 */
export function useRefreshOnResize(ref: RefObject<HTMLElement | null>) {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let t = 0;
    let h = el.offsetHeight;
    const ro = new ResizeObserver(() => {
      if (Math.abs(el.offsetHeight - h) < 2) return;
      h = el.offsetHeight;
      window.clearTimeout(t);
      t = window.setTimeout(() => ScrollTrigger.refresh(), 250);
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      window.clearTimeout(t);
    };
  }, [ref]);
}
