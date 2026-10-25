import { Crosshair, ImageUp } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { CropChoice } from '@/lib/api';
import { CROP_NAME } from '@/lib/format';
import type { Sample } from '@/lib/site';

export type Picked = { blob: Blob; url: string; name: string };

/** Drop zone: drag/drop, paste, file picker, sample chips. Calls onPick with the chosen image. */
export function DropZone({ samples, onPick, compact }: { samples: Sample[]; onPick: (p: Picked) => void; compact: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const take = (f: File | null | undefined) => {
    if (f) onPick({ blob: f, url: URL.createObjectURL(f), name: f.name || 'pasted image' });
  };

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const f = [...(e.clipboardData?.files ?? [])].find((x) => x.type.startsWith('image/'));
      if (f) take(f);
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  });

  const sample = async (s: Sample) => {
    const blob = await fetch(s.src).then((r) => r.blob());
    onPick({ blob, url: s.src, name: `${s.id}.jpg` });
  };

  return (
    <div className="flex flex-col gap-4">
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          take(e.dataTransfer.files[0]);
        }}
        className={`group flex cursor-pointer flex-col items-center justify-center gap-3 rounded-[6px] border border-dashed text-center transition-colors focus-within:border-leaf ${
          compact ? 'px-4 py-5' : 'aspect-[4/3] px-6'
        } ${over ? 'border-leaf bg-leaf/5' : 'border-white/25 hover:border-white/50'}`}
      >
        <ImageUp size={compact ? 20 : 28} strokeWidth={1.4} className="text-muted group-hover:text-text" aria-hidden="true" />
        <span className="text-[15px] text-text">
          {compact ? 'Upload another photo' : 'Drop a leaf photo, paste one, or browse'}
        </span>
        {!compact && <span className="mono text-[12px] text-muted">JPEG, PNG or WebP · up to 10 MB · one leaf, close up</span>}
        <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" className="sr-only"
               data-testid="file-input" onChange={(e) => take(e.target.files?.[0])} />
      </label>
      <div>
        <p className="caption mb-2 text-[11px] text-muted">Or try a test-set photo</p>
        <ul className="flex flex-wrap gap-2">
          {samples.map((s) => (
            <li key={s.id}>
              <button type="button" onClick={() => sample(s)} data-testid={`sample-${s.id}`}
                      className="flex items-center gap-2 rounded-full border border-white/15 py-1 pl-1 pr-3 text-[13px] text-text/90 transition-colors hover:border-white/40">
                <img src={s.thumb} alt="" width={28} height={28} className="h-7 w-7 rounded-full object-cover" />
                {CROP_NAME[s.crop]} · {s.orig_class.replace(/_/g, ' ')}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export function CropControl({ value, onChange, disabled }: { value: CropChoice; onChange: (c: CropChoice) => void; disabled: boolean }) {
  const opts: [CropChoice, string][] = [['auto', 'Auto'], ['turmeric', CROP_NAME.turmeric], ['blackgram', CROP_NAME.blackgram]];
  return (
    <div role="radiogroup" aria-label="Crop" className="inline-flex rounded-full border border-white/15 p-1">
      {opts.map(([v, l]) => (
        <button key={v} type="button" role="radio" aria-checked={value === v} disabled={disabled} onClick={() => onChange(v)}
                className={`mono rounded-full px-3.5 py-1.5 text-[11px] uppercase tracking-[0.08em] transition-colors disabled:opacity-50 ${
                  value === v ? 'bg-paper text-ink' : 'text-muted hover:text-text'}`}>
          {l}
        </button>
      ))}
    </div>
  );
}

/**
 * The uploaded photo. While busy a scanning line passes over it; in "point at the leaf" mode a
 * click sets the ROI (0-1 coordinates of the displayed image).
 */
export function Preview({ url, busy, pointMode, roi, onPoint }: {
  url: string; busy: boolean; pointMode: boolean; roi: { x: number; y: number } | null; onPoint: (p: { x: number; y: number }) => void;
}) {
  const imgRef = useRef<HTMLImageElement>(null);
  const click = (e: React.MouseEvent) => {
    const r = imgRef.current?.getBoundingClientRect();
    if (!pointMode || !r) return;
    onPoint({ x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) });
  };
  const key = (e: React.KeyboardEvent) => {
    if (pointMode && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      onPoint({ x: 0.5, y: 0.5 });
    }
  };
  return (
    <div className="relative overflow-hidden rounded-[6px] border hairline bg-soil">
      <div className={`relative mx-auto w-fit ${pointMode ? 'cursor-crosshair' : ''}`} onClick={click} onKeyDown={key}
           role={pointMode ? 'button' : undefined} tabIndex={pointMode ? 0 : undefined}
           aria-label={pointMode ? 'Click the leaf to point the analysis at it' : undefined}>
        <img ref={imgRef} src={url} alt="Your uploaded photo." className="block max-h-[52vh] w-auto max-w-full" />
        {roi && (
          <span aria-hidden="true" className="pointer-events-none absolute h-6 w-6 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-leaf"
                style={{ left: `${roi.x * 100}%`, top: `${roi.y * 100}%` }}>
            <span className="absolute left-1/2 top-1/2 h-1 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-leaf" />
          </span>
        )}
        {busy && (
          <>
            <span aria-hidden="true" className="absolute inset-0 bg-ink/30" />
            <span aria-hidden="true" className="scan-line pointer-events-none absolute inset-x-0 h-[2px] bg-leaf" />
          </>
        )}
      </div>
      {pointMode && (
        <p className="mono flex items-center gap-2 border-t hairline px-4 py-2.5 text-[12px] text-muted">
          <Crosshair size={14} aria-hidden="true" /> Click the leaf you want analysed.
        </p>
      )}
    </div>
  );
}
