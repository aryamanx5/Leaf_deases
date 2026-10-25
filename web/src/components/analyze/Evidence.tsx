import { Minus, Plus, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { AnalyzeResponse } from '@/lib/api';

// the exact overlay colours drawn by leafdx.ip.lesions.overlay_lesions (BGR there, RGB here)
const LEGEND: [string, string, 'fill' | 'line'][] = [
  ['rgb(255,255,0)', 'yellow: chlorosis', 'fill'],
  ['rgb(200,80,0)', 'orange-red: brown necrosis', 'fill'],
  ['rgb(255,0,255)', 'magenta: dark tissue', 'fill'],
  ['rgb(0,160,255)', 'blue: pale / mildew', 'fill'],
  ['rgb(0,255,0)', 'green line: leaf outline', 'line'],
];

type Key = 'original' | 'segmented_leaf' | 'abnormal_regions' | 'gradcam';
const ITEMS: { key: Key; title: string; caption: string }[] = [
  { key: 'original', title: 'Original', caption: 'Your photo, resized for analysis.' },
  { key: 'segmented_leaf', title: 'Segmented leaf', caption: 'The leaf cut from its background by Stage 1.' },
  { key: 'abnormal_regions', title: 'Detected abnormal regions', caption: 'Tissue that differs from this leaf’s own green.' },
  { key: 'gradcam', title: 'Grad-CAM', caption: 'Where the disease CNN looked for its answer.' },
];

export function Evidence({ r }: { r: AnalyzeResponse }) {
  const [open, setOpen] = useState<Key | null>(null);
  return (
    <section aria-labelledby="ev-title">
      <h2 id="ev-title" className="mb-5 text-[20px] tracking-[-0.02em]">Evidence</h2>
      <ol className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {ITEMS.map((it, i) => {
          const src = r.images[it.key];
          return (
            <li key={it.key}>
              <figure>
                {src ? (
                  <button type="button" onClick={() => setOpen(it.key)} aria-label={`Open ${it.title} larger`}
                          className="block w-full overflow-hidden rounded-[4px] border hairline bg-soil">
                    <img src={src} alt={`${it.title}: ${it.caption}`} className="aspect-square w-full object-contain transition-transform duration-500 ease-out hover:scale-[1.03]" />
                  </button>
                ) : (
                  <div className="flex aspect-square items-center justify-center rounded-[4px] border hairline bg-soil text-[13px] text-muted">
                    Not available
                  </div>
                )}
                <figcaption className="mt-2">
                  <span className="mono text-[11px] text-muted">{String(i + 1).padStart(2, '0')}</span>{' '}
                  <span className="text-[14px] text-text">{it.title}</span>
                  <span className="block text-[13px] leading-snug text-muted">{it.caption}</span>
                </figcaption>
              </figure>
            </li>
          );
        })}
      </ol>
      <ul className="mt-5 flex flex-wrap gap-x-5 gap-y-2" aria-label="Overlay legend">
        {LEGEND.map(([c, l, kind]) => (
          <li key={l} className="mono flex items-center gap-2 text-[12px] text-muted">
            <span aria-hidden="true" className={kind === 'fill' ? 'h-2.5 w-2.5 rounded-[2px]' : 'h-[2px] w-4'} style={{ background: c }} />
            {l}
          </li>
        ))}
      </ul>
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
    <div role="dialog" aria-modal="true" aria-label={compare ? 'Original and abnormal regions, before and after' : item.title}
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
              <img src={after} alt="Detected abnormal regions." draggable={false}
                   className="absolute inset-0 h-full w-full select-none object-contain" style={{ clipPath: `inset(0 0 0 ${split * 100}%)` }} />
            )}
            {compare && <span aria-hidden="true" className="absolute inset-y-0 w-px bg-paper" style={{ left: `${split * 100}%` }} />}
          </div>
        </div>
      </div>
      {compare && (
        <label className="mx-auto flex w-full max-w-[520px] items-center gap-3 px-5 pb-6">
          <span className="mono text-[11px] text-muted">Original</span>
          <input type="range" min={0} max={1} step={0.01} value={split} onChange={(e) => setSplit(Number(e.target.value))}
                 className="flex-1 accent-[var(--leaf)]" aria-label="Before / after position" />
          <span className="mono text-[11px] text-muted">Abnormal regions</span>
        </label>
      )}
    </div>
  );
}
