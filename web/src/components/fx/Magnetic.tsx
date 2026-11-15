import type { ReactNode } from 'react';

/** Pulls its child a few pixels toward the pointer while it hovers (mouse only). Decorative. */
export default function Magnetic({ children, strength = 0.28, className = '' }: { children: ReactNode; strength?: number; className?: string }) {
  const move = (e: React.PointerEvent<HTMLSpanElement>) => {
    if (e.pointerType !== 'mouse') return;
    const el = e.currentTarget;
    const r = el.getBoundingClientRect();
    el.style.transform = `translate3d(${((e.clientX - r.left - r.width / 2) * strength).toFixed(1)}px, ${((e.clientY - r.top - r.height / 2) * strength).toFixed(1)}px, 0)`;
  };
  return (
    <span className={`magnet inline-block ${className}`} onPointerMove={move} onPointerLeave={(e) => { e.currentTarget.style.transform = ''; }}>
      {children}
    </span>
  );
}
