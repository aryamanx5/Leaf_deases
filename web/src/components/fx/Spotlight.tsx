import { createElement, type ReactNode } from 'react';

/** A card with a soft light that follows the pointer (`.spot` in index.css). Purely decorative. */
export default function Spotlight({ as = 'div', className = '', paper = false, children, ...rest }: {
  as?: 'div' | 'article' | 'li' | 'figure'; className?: string; paper?: boolean; children: ReactNode;
} & { [k: `data-${string}` | `aria-${string}`]: string | undefined }) {
  const move = (e: React.PointerEvent<HTMLElement>) => {
    const el = e.currentTarget;
    const r = el.getBoundingClientRect();
    el.style.setProperty('--mx', `${e.clientX - r.left}px`);
    el.style.setProperty('--my', `${e.clientY - r.top}px`);
  };
  return createElement(as, { className: `spot ${paper ? 'spot-paper' : ''} ${className}`, onPointerMove: move, ...rest }, children);
}
