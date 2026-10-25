import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { GLOSSARY, type GlossaryKey } from '@/content/glossary';

/**
 * A small "?" that opens a plain-language explanation of a technical term. A disclosure button (keyboard and
 * screen-reader friendly), closed by Escape, a click outside or the button again; kept inside the viewport.
 * Only inline elements, so it can sit inside headings, paragraphs and table cells.
 */
export function Help({ term, extra }: { term: GlossaryKey; extra?: string }) {
  const g = GLOSSARY[term];
  const [open, setOpen] = useState(false);
  const [shift, setShift] = useState(0);
  const id = useId();
  const wrap = useRef<HTMLSpanElement>(null);
  const btn = useRef<HTMLButtonElement>(null);
  const box = useRef<HTMLSpanElement>(null);

  useLayoutEffect(() => {
    if (!open || !box.current) return;
    const r = box.current.getBoundingClientRect();
    const pad = 16;
    // clientWidth, not innerWidth: on phones an overflowing popover widens the layout viewport itself.
    // Moved with `left`, not a transform, so the unshifted box never counts as page overflow.
    const over = r.right - shift - (document.documentElement.clientWidth - pad);
    const under = pad - (r.left - shift);
    setShift(over > 0 ? -over : under > 0 ? under : 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- measure once per opening
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false);
        btn.current?.focus();
      }
    };
    const onDown = (e: PointerEvent) => {
      if (!wrap.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('pointerdown', onDown);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('pointerdown', onDown);
    };
  }, [open]);

  return (
    <span ref={wrap} className="relative inline-block align-middle normal-case tracking-normal">
      <button ref={btn} type="button" aria-expanded={open} aria-controls={id} data-help={term}
              aria-label={`What does “${g.term}” mean?`} onClick={() => { setShift(0); setOpen((o) => !o); }}
              className={`relative ml-1 inline-flex h-[18px] w-[18px] items-center justify-center rounded-full border font-sans text-[11px] leading-none transition-colors before:absolute before:-inset-[13px] before:content-[''] ${
                open ? 'border-leaf bg-leaf/15 text-text' : 'border-white/35 text-muted hover:border-white/70 hover:text-text'}`}>
        ?
      </button>
      {open && (
        <span ref={box} id={id} role="note" style={{ left: -8 + shift }}
              className="absolute top-[calc(100%+10px)] z-40 block w-[min(300px,calc(100vw-32px))] rounded-[6px] border border-white/20 bg-soil p-4 text-left font-sans text-[13px] font-normal leading-relaxed text-text/90 shadow-[0_12px_40px_rgba(0,0,0,0.5)]">
          <span className="mb-1 block text-[14px] text-text">{g.term}</span>
          {g.text}
          {extra && <span className="mt-2 block text-muted">{extra}</span>}
        </span>
      )}
    </span>
  );
}
