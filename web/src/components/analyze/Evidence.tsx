import { Minus, Plus, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Help } from '@/components/analyze/Help';
import type { GlossaryKey } from '@/content/glossary';
import type { AnalyzeResponse } from '@/lib/api';

// the exact overlay colours drawn by leafdx.ip.lesions.overlay_lesions (BGR there, RGB here)
const LEGEND: [string, string, 'fill' | 'line'][] = [
  ['rgb(255,255,0)', 'yellow = yellowing', 'fill'],
  ['rgb(200,80,0)', 'orange-red = brown dead tissue', 'fill'],
  ['rgb(255,0,255)', 'magenta = dark tissue', 'fill'],
  ['rgb(0,160,255)', 'blue = pale, mildew-like', 'fill'],
  ['rgb(0,255,0)', 'green line = edge of the leaf', 'line'],
];

type Key = 'original' | 'segmented_leaf' | 'abnormal_regions' | 'gradcam';
const ITEMS: { key: Key; title: string; tech?: string; help?: GlossaryKey; caption: string }[] = [
  { key: 'original', title: 'Your photo', caption: 'As it was analysed, resized.' },
  { key: 'segmented_leaf', title: 'Leaf cut-out', tech: 'segmentation', help: 'segmentation', caption: 'Only the leaf, with the background removed.' },
  { key: 'abnormal_regions', title: 'Damaged spots', tech: 'lesions', help: 'lesion', caption: 'Patches that differ from this leaf’s own healthy green, coloured by kind.' },
  { key: 'gradcam', title: 'Where the AI looked', tech: 'Grad-CAM', help: 'gradcam', caption: 'Red is where the neural network looked most; it is not a map of damage.' },
];

/**
 * Inline before/after: drag (or use the arrow keys on) the handle to wipe between the photo and its damaged-spots
 * overlay. A transparent range input covers the picture, so pointer, touch and keyboard all drive one control.
 */
function BeforeAfter({ before, after }: { before: string; after: string }) {
  const [split, setSplit] = useState(0.5);
  return (
    <div className="relative mx-auto w-full max-w-[560px] select-none overflow-hidden rounded-[6px] border hairline bg-soil has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-[3px] has-[:focus-visible]:outline-[var(--turmeric)]" data-testid="before-after">
      <img src={before} alt="" draggable={false} className="block aspect-square w-full object-contain" />
      <img src={after} alt="" draggable={false} className="absolute inset-0 h-full w-full object-contain"
           style={{ clipPath: `inset(0 0 0 ${split * 100}%)` }} />
      <span aria-hidden="true" className="pointer-events-none absolute inset-y-0 w-[2px] -translate-x-1/2 bg-paper shadow-[0_0_0_1px_rgba(0,0,0,0.35)]"
            style={{ left: `${split * 100}%` }}>
        <span className="absolute left-1/2 top-1/2 flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-paper text-[13px] text-ink shadow-lg">
          ‹›
        </span>
      </span>
      <span aria-hidden="true" className="mono pointer-events-none absolute left-3 top-3 rounded-full bg-ink/75 px-2.5 py-1 text-[11px] text-text">Your photo</span>
      <span aria-hidden="true" className="mono pointer-events-none absolute right-3 top-3 rounded-full bg-ink/75 px-2.5 py-1 text-[11px] text-text">Damaged spots</span>
      <input type="range" min={0} max={1} step={0.01} value={split} onChange={(e) => setSplit(Number(e.target.value))}
             aria-label="Compare your photo with its damaged spots" aria-valuetext={`${Math.round(split * 100)}% your photo`}
             className="absolute inset-0 h-full w-full cursor-ew-resize opacity-0" />
    </div>
  );
}

export function Evidence({ r }: { r: AnalyzeResponse }) {
  const [open, setOpen] = useState<Key | null>(null);
  return (
    <section aria-labelledby="ev-title">
      <h2 id="ev-title" className="text-[20px] tracking-[-0.02em]">What the model saw</h2>
      <p className="mb-5 mt-1 text-[14px] text-muted">Drag across the picture to compare your photo with the damaged spots it found.</p>
      <BeforeAfter before={r.images.original} after={r.images.abnormal_regions} />
      <ul className="mt-4 flex flex-wrap justify-center gap-x-5 gap-y-2" aria-label="Colours in the damaged spots picture">
        {LEGEND.map(([c, l, kind]) => (
          <li key={l} className="mono flex items-center gap-2 text-[12px] text-muted">
            <span aria-hidden="true" className={kind === 'fill' ? 'h-2.5 w-2.5 rounded-[2px]' : 'h-[2px] w-4'} style={{ background: c }} />
            {l}
          </li>
        ))}
      </ul>
      <ol className="mt-7 grid grid-cols-2 gap-4 sm:grid-cols-4" aria-label="The four pictures, in order">
        {ITEMS.map((it, i) => {
          const src = r.images[it.key];
          return (
            <li key={it.key} className="min-w-0">
              <figure>
                {src ? (
                  <button type="button" onClick={() => setOpen(it.key)} aria-label={`Open ${it.title} larger`}
                          className="group block w-full overflow-hidden rounded-[4px] border hairline bg-soil">
                    <img src={src} alt={`${it.title}: ${it.caption}`} className="aspect-square w-full object-contain transition-transform duration-500 ease-out group-hover:scale-[1.03] motion-reduce:transition-none" />
                  </button>
                ) : (
                  <div className="flex aspect-square items-center justify-center rounded-[4px] border hairline bg-soil text-[13px] text-muted">
                    Not available
                  </div>
                )}
                <figcaption className="mt-2">
                  <span className="mono text-[11px] text-muted">{String(i + 1).padStart(2, '0')}</span>
                  {i > 0 && <span aria-hidden="true" className="mono text-[11px] text-muted"> →</span>}{' '}
                  <span className="text-[14px] text-text">{it.title}</span>
                  {it.tech && <span className="text-[13px] text-muted"> ({it.tech})</span>}
                  {it.help && <Help term={it.help} />}
                  <span className="block text-[13px] leading-snug text-muted">{it.caption}</span>
                </figcaption>
              </figure>
            </li>
          );
        })}
      </ol>
      {/* portal: the route-transition wrapper is its own stacking context, below the fixed nav */}
      {open && createPortal(<Lightbox r={r} initial={open} onClose={() => setOpen(null)} />, document.body)}
    </section>
  );
}

/** Zoomable lightbox with a before/after slider between Original and Abnormal regions. */
function Lightbox({ r, initial, onClose }: { r: AnalyzeResponse; initial: Key; onClose: () => void }) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [split, setSplit] = useState(0.5);
  const [compare, setCompare] = useState(initial === 'original' || initial === 'abnormal_regions');
  const closeRef = useRef<HTMLButtonElement>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const item = ITEMS.find((i) => i.key === initial)!;

  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === '+' || e.key === '=') setZoom((z) => Math.min(4, z * 1.25));
      if (e.key === '-') setZoom((z) => Math.max(1, z / 1.25));
    };
    window.addEventListener('keydown', onKey);
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = '';
      prev?.focus();
    };
  }, [onClose]);
  useEffect(() => {
    if (zoom === 1) setPan({ x: 0, y: 0 });
  }, [zoom]);

  const before = r.images.original;
  const after = compare ? r.images.abnormal_regions : r.images[initial] ?? before;
  return (
    <div role="dialog" aria-modal="true" aria-label={compare ? 'Your photo and its damaged spots, before and after' : item.title}
         className="fixed inset-0 z-[200] flex flex-col bg-ink/95 backdrop-blur-sm">
      <div className="flex flex-wrap items-center justify-between gap-3 p-4 sm:p-5">
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" aria-pressed={compare} onClick={() => setCompare((c) => !c)}
                  className={`mono rounded-full border px-3 py-1.5 text-[11px] uppercase tracking-[0.08em] ${compare ? 'border-paper bg-paper text-ink' : 'border-white/25 text-muted'}`}>
            Before / after
          </button>
          <button type="button" aria-label="Zoom out" onClick={() => setZoom((z) => Math.max(1, z / 1.25))} className="rounded-full border border-white/25 p-2 text-text"><Minus size={14} /></button>
          <span className="mono w-12 text-center text-[12px] text-muted">{Math.round(zoom * 100)}%</span>
          <button type="button" aria-label="Zoom in" onClick={() => setZoom((z) => Math.min(4, z * 1.25))} className="rounded-full border border-white/25 p-2 text-text"><Plus size={14} /></button>
        </div>
        <button ref={closeRef} type="button" aria-label="Close" onClick={onClose} className="rounded-full border border-white/25 p-2 text-text">
          <X size={18} />
        </button>
      </div>
      <div className="relative flex-1 overflow-hidden"
           onWheel={(e) => setZoom((z) => Math.min(4, Math.max(1, z * (e.deltaY < 0 ? 1.1 : 1 / 1.1))))}
           onPointerDown={(e) => {
             if (zoom > 1) drag.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
           }}
           onPointerMove={(e) => drag.current && setPan({ x: e.clientX - drag.current.x, y: e.clientY - drag.current.y })}
           onPointerUp={() => (drag.current = null)}
           onPointerLeave={() => (drag.current = null)}>
        <div className="absolute inset-0 flex items-center justify-center p-4" style={{ cursor: zoom > 1 ? 'grab' : 'default' }}>
          <div className="relative max-h-full max-w-full" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transition: drag.current ? 'none' : 'transform 200ms var(--ease)' }}>
            <img src={compare ? before : after} alt={compare ? 'Original photo.' : `${item.title}: ${item.caption}`} draggable={false}
                 className="block max-h-[calc(100dvh-170px)] max-w-[calc(100vw-32px)] select-none object-contain" />
            {compare && (
              <img src={after} alt="Damaged spots." draggable={false}
                   className="absolute inset-0 h-full w-full select-none object-contain" style={{ clipPath: `inset(0 0 0 ${split * 100}%)` }} />
            )}
            {compare && <span aria-hidden="true" className="absolute inset-y-0 w-px bg-paper" style={{ left: `${split * 100}%` }} />}
          </div>
        </div>
      </div>
      {compare && (
        <label className="mx-auto flex w-full max-w-[520px] items-center gap-3 px-5 pb-6">
          <span className="mono text-[11px] text-muted">Your photo</span>
          <input type="range" min={0} max={1} step={0.01} value={split} onChange={(e) => setSplit(Number(e.target.value))}
                 className="flex-1 accent-[var(--leaf)]" aria-label="Before / after position" />
          <span className="mono text-[11px] text-muted">Damaged spots</span>
        </label>
      )}
    </div>
  );
}
