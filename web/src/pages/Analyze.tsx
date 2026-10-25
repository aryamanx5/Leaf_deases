import { RotateCcw } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Evidence } from '@/components/analyze/Evidence';
import { CropControl, DropZone, Preview, type Picked } from '@/components/analyze/Input';
import { Compare, Measurements, ResultCard, Steps } from '@/components/analyze/Result';
import { analyze, ApiError, type AnalyzeResponse, type CropChoice } from '@/lib/api';
import { useSite } from '@/lib/site';

type State =
  | { kind: 'idle' }
  | { kind: 'busy' }
  | { kind: 'done'; r: AnalyzeResponse }
  | { kind: 'error'; e: ApiError };

const ERROR_TITLE: Record<string, string> = {
  offline: 'The analysis service is offline',
  too_large: 'That photo is too large',
  not_a_plant: 'This does not look like a leaf',
  unsupported_type: 'Unsupported file type',
  invalid_image: 'The image could not be read',
};

export default function Analyze() {
  const site = useSite();
  const [picked, setPicked] = useState<Picked | null>(null);
  const [crop, setCrop] = useState<CropChoice>('auto');
  const [pointMode, setPointMode] = useState(false);
  const [roi, setRoi] = useState<{ x: number; y: number } | null>(null);
  const [state, setState] = useState<State>({ kind: 'idle' });
  const abort = useRef<AbortController | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    document.title = 'Run the model · Patra';
    window.scrollTo(0, 0);
  }, []);

  const run = useCallback(async (p: Picked, c: CropChoice, point: { x: number; y: number } | null) => {
    abort.current?.abort();
    const ac = new AbortController();
    abort.current = ac;
    setState({ kind: 'busy' });
    try {
      const r = await analyze(p.blob, c, point, ac.signal);
      if (ac.signal.aborted) return;
      setState({ kind: 'done', r });
      // stacked layout: the result sits below the inputs, bring it into view
      if (window.innerWidth < 1024) requestAnimationFrame(() => resultRef.current?.scrollIntoView({ block: 'start' }));
    } catch (e) {
      if ((e as Error).name === 'AbortError') return;
      setState({ kind: 'error', e: e instanceof ApiError ? e : new ApiError('internal', (e as Error).message) });
    }
  }, []);

  const pick = (p: Picked) => {
    setPicked(p);
    setRoi(null);
    void run(p, crop, null);
  };
  const chooseCrop = (c: CropChoice) => {
    setCrop(c);
    if (picked) void run(picked, c, roi);
  };
  const point = (p: { x: number; y: number }) => {
    setRoi(p);
    if (picked) void run(picked, crop, p);
  };

  const busy = state.kind === 'busy';
  const r = state.kind === 'done' ? state.r : null;

  return (
    <main className="min-h-dvh bg-ink pb-24 pt-28 sm:pt-32">
      <div className="container-grid">
        <p className="caption mb-4 text-muted">Run the model</p>
        <h1 className="max-w-[820px] text-4xl font-normal leading-[1.05] tracking-[-0.04em] sm:text-5xl">
          Upload a leaf. <span className="font-playfair italic">See both readings.</span>
        </h1>
        <p className="mt-4 max-w-[600px] text-[17px] leading-relaxed text-muted">
          One turmeric or black gram leaf, photographed close up. The image is processed on our server and not stored.
        </p>

        <div className="mt-12 grid gap-10 lg:grid-cols-12 lg:gap-6">
          <div className="flex flex-col gap-5 lg:col-span-5">
            {picked && (
              <Preview url={picked.url} busy={busy} pointMode={pointMode} roi={roi} onPoint={point} />
            )}
            <DropZone samples={site?.samples ?? []} onPick={pick} compact={!!picked} />
            <div className="flex flex-wrap items-center gap-4 border-t hairline pt-5">
              <CropControl value={crop} onChange={chooseCrop} disabled={busy} />
              <label className="flex cursor-pointer items-center gap-2 text-[14px] text-text/85">
                <input type="checkbox" checked={pointMode} onChange={(e) => {
                  setPointMode(e.target.checked);
                  if (!e.target.checked && roi) {
                    setRoi(null);
                    if (picked) void run(picked, crop, null);
                  }
                }} className="h-4 w-4 accent-[var(--leaf)]" />
                Point at the leaf
              </label>
            </div>
          </div>

          <div ref={resultRef} className="flex flex-col gap-6 lg:col-span-7">
            {state.kind === 'idle' && (
              <div className="flex min-h-[240px] items-center rounded-[6px] border hairline p-7 lg:min-h-[360px]">
                <p className="max-w-[420px] text-[17px] leading-relaxed text-muted">
                  Results appear here: the crop, the healthy/diseased call, how much of the leaf is affected, and the evidence
                  from both image processing and the CNN.
                </p>
              </div>
            )}
            {(busy || r) && <Steps busy={busy} done={!!r} totalMs={r?.timings_ms.total ?? null} />}
            {state.kind === 'error' && (
              <div role="alert" className="rounded-[6px] border border-turmeric/40 bg-turmeric/5 p-6" data-testid="error">
                <p className="text-[18px] text-text">{ERROR_TITLE[state.e.code] ?? 'Something went wrong'}</p>
                <p className="mt-2 text-[15px] leading-relaxed text-muted">
                  {state.e.code === 'offline'
                    ? 'Start the API (make dev) or try again in a moment.'
                    : state.e.message.charAt(0).toUpperCase() + state.e.message.slice(1)}
                </p>
                {picked && state.e.code !== 'too_large' && (
                  <button type="button" onClick={() => run(picked, crop, roi)}
                          className="mt-4 inline-flex items-center gap-2 rounded-full bg-turmeric px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-turmeric-hover">
                    <RotateCcw size={15} aria-hidden="true" /> Retry
                  </button>
                )}
              </div>
            )}
            {r && (
              <>
                <ResultCard r={r} onPickCrop={(c) => chooseCrop(c)} />
                <Compare r={r} />
              </>
            )}
          </div>
        </div>

        {r && (
          <div className="mt-16 grid gap-14">
            <Evidence r={r} />
            <div className="lg:w-7/12">
              <Measurements r={r} />
            </div>
            <p className="max-w-[640px] text-[13px] leading-relaxed text-muted">
              A research prototype trained on two public datasets. It is not a diagnosis; confirm with an agronomist before
              treating a crop.
            </p>
          </div>
        )}
      </div>
    </main>
  );
}
