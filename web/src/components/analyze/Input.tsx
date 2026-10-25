import { Crosshair, ImageUp } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { CropChoice } from '@/lib/api';
import { SampleGallery } from '@/components/analyze/Samples';
import { CROP_NAME } from '@/lib/format';
import type { Sample } from '@/lib/site';

export type Picked = { blob: Blob; url: string; name: string; sample?: Sample };

/** Drop zone: drag/drop, paste, file picker, sample gallery. Calls onPick with the chosen image. */
export function DropZone({ samples, onPick, compact, revealed, before }: {
  samples: Sample[]; onPick: (p: Picked) => void; compact: boolean; revealed: ReadonlySet<string>; before?: React.ReactNode;
}) {
  // once a photo is chosen the gallery folds to two rows (grid layouts), so the input column stays short
  const [activeId, setActiveId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const take = (f: File | null | undefined) => {
    if (!f) return;
    setActiveId(null);
    onPick({ blob: f, url: URL.createObjectURL(f), name: f.name || 'pasted image' });
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
    setActiveId(s.id);
    const blob = await fetch(s.src).then((r) => r.blob());
    onPick({ blob, url: s.src, name: `${s.id}.jpg`, sample: s });
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
        className={`group flex cursor-pointer items-center justify-center rounded-[6px] border border-dashed text-center transition-colors focus-within:border-leaf ${
          compact ? 'min-h-[52px] flex-row gap-2.5 px-4 py-2.5' : 'flex-col gap-2 px-6 py-7'
        } ${over ? 'border-leaf bg-leaf/5' : 'border-white/25 hover:border-white/50'}`}
      >
        <ImageUp size={compact ? 20 : 24} strokeWidth={1.4} className="text-muted group-hover:text-text" aria-hidden="true" />
        <span className="text-[15px] text-text">
          {compact ? 'Upload another photo' : 'Drop a leaf photo here, paste one, or '}
          {!compact && <span className="text-leaf underline underline-offset-4">browse</span>}
        </span>
        {!compact && <span className="mono text-[12px] text-muted">a photo from your phone or camera · up to 10 megabytes · one leaf, close up</span>}
        <input ref={inputRef} id="leaf-file" type="file" accept="image/jpeg,image/png,image/webp" className="sr-only"
               data-testid="file-input" onChange={(e) => take(e.target.files?.[0])} />
      </label>
      {before}
      <SampleGallery samples={samples} onPick={sample} activeId={activeId} revealed={revealed} />
    </div>
  );
}

export function CropControl({ value, onChange, disabled }: { value: CropChoice; onChange: (c: CropChoice) => void; disabled: boolean }) {
  const opts: [CropChoice, string][] = [['auto', 'Auto'], ['turmeric', CROP_NAME.turmeric], ['blackgram', CROP_NAME.blackgram]];
  return (
    <div role="radiogroup" aria-label="Crop" className="inline-flex self-start rounded-full border border-white/15 p-1">
      {opts.map(([v, l]) => (
        <button key={v} type="button" role="radio" aria-checked={value === v} disabled={disabled} onClick={() => onChange(v)}
                className={`mono min-h-[44px] rounded-full px-3.5 text-[11px] uppercase tracking-[0.08em] disabled:opacity-50 ${
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
        <img ref={imgRef} src={url} alt="Your uploaded photo." className="block max-h-[52vh] w-auto max-w-full lg:max-h-[28vh]" />
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

/** "Point at the leaf": a switch, with one line saying when it helps. */
export function PointToggle({ on, onChange }: { on: boolean; onChange: (on: boolean) => void }) {
  return (
    <div className="flex flex-wrap items-start gap-x-3">
      <button type="button" role="switch" aria-checked={on} aria-describedby="point-help" onClick={() => onChange(!on)}
              data-testid="point-toggle"
              className="group flex min-h-[44px] shrink-0 items-center gap-3 text-left text-[14px] text-text">
        <span aria-hidden="true" className={`relative h-6 w-11 rounded-full border transition-colors motion-reduce:transition-none ${
          on ? 'border-leaf bg-leaf/30' : 'border-white/25 bg-white/5 group-hover:border-white/50'}`}>
          <span className={`absolute top-[3px] h-4 w-4 rounded-full transition-transform motion-reduce:transition-none ${
            on ? 'translate-x-[22px] bg-leaf' : 'translate-x-[3px] bg-white/70'}`} />
        </span>
        Point at the leaf
      </button>
      <p id="point-help" className="min-w-[180px] flex-1 pt-[13px] text-[12px] leading-snug text-muted">
        Several leaves in the photo? Turn this on and tap the one to check.
      </p>
    </div>
  );
}
