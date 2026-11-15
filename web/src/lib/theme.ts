/**
 * Colour themes: three night palettes that differ only in CSS custom properties (index.css). The choice is kept in
 * localStorage; switching opens the new palette as a growing circle from the control that was used (View
 * Transitions API), or instantly where that is not supported or reduced motion is on.
 */
import { useEffect, useState } from 'react';

export const THEMES = [
  { id: 'forest', name: 'Forest', swatch: ['#0a0e0b', '#8db86b', '#e8702a'] },
  { id: 'monsoon', name: 'Monsoon', swatch: ['#070c12', '#6fd0b4', '#f2a23a'] },
  { id: 'dusk', name: 'Dusk', swatch: ['#110b10', '#b5cf85', '#ff8257'] },
] as const;
export type ThemeId = (typeof THEMES)[number]['id'];

const KEY = 'patra:theme';
const EVENT = 'patra:theme-change';
const isTheme = (v: unknown): v is ThemeId => THEMES.some((t) => t.id === v);

export function currentTheme(): ThemeId {
  const v = document.documentElement.dataset.theme;
  return isTheme(v) ? v : 'forest';
}

function apply(id: ThemeId) {
  if (id === 'forest') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = id;
  const ink = THEMES.find((t) => t.id === id)!.swatch[0];
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', ink);
  window.dispatchEvent(new Event(EVENT));
}

/** Restore the saved theme before the first paint (called from main.tsx). */
export function initTheme() {
  try {
    const v = localStorage.getItem(KEY);
    if (isTheme(v) && v !== 'forest') apply(v);
  } catch {
    /* private mode: the default theme stays */
  }
}

type Doc = Document & { startViewTransition?: (cb: () => void) => { ready: Promise<void> } };

export function setTheme(id: ThemeId, origin?: { x: number; y: number }) {
  if (id === currentTheme()) return;
  try {
    localStorage.setItem(KEY, id);
  } catch {
    /* not persisted, still applied */
  }
  const doc = document as Doc;
  const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!doc.startViewTransition || still) {
    apply(id);
    return;
  }
  const x = origin?.x ?? window.innerWidth / 2;
  const y = origin?.y ?? 0;
  const r = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
  void doc.startViewTransition(() => apply(id)).ready.then(() => {
    document.documentElement.animate(
      { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${r}px at ${x}px ${y}px)`] },
      { duration: 750, easing: 'cubic-bezier(0.16, 1, 0.3, 1)', pseudoElement: '::view-transition-new(root)' },
    );
  }).catch(() => {});
}

export const nextTheme = (id: ThemeId): ThemeId => THEMES[(THEMES.findIndex((t) => t.id === id) + 1) % THEMES.length]!.id;

export function useTheme(): ThemeId {
  const [id, setId] = useState<ThemeId>(() => (typeof document === 'undefined' ? 'forest' : currentTheme()));
  useEffect(() => {
    const on = () => setId(currentTheme());
    window.addEventListener(EVENT, on);
    return () => window.removeEventListener(EVENT, on);
  }, []);
  return id;
}
