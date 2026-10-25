import { Check } from 'lucide-react';
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Evidence } from '@/components/analyze/Evidence';
import { HowCalculated } from '@/components/analyze/HowCalculated';
import { CropControl, DropZone, PointToggle, Preview, type Picked } from '@/components/analyze/Input';
import { Compare, Measurements, Progress, ResultCard } from '@/components/analyze/Result';
import { SampleCheck } from '@/components/analyze/Samples';
import { EmptyState, ErrorState, ResultSkeleton } from '@/components/analyze/States';
import { Verdict } from '@/components/analyze/Verdict';
import { useViewMode, type ViewMode } from '@/lib/verdict';
import { analyze, ApiError, type AnalyzeResponse, type CropChoice } from '@/lib/api';
import { useSite } from '@/lib/site';

type State =
  | { kind: 'idle' }
  | { kind: 'busy' }
  | { kind: 'done'; r: AnalyzeResponse }
  | { kind: 'error'; e: ApiError };

/** True while `el` fits in the viewport below the fixed nav: only then may it be sticky (else its end is unreachable). */
function useFitsViewport(el: React.RefObject<HTMLElement>, top: number) {
  const [fits, setFits] = useState(false);
  useLayoutEffect(() => {
    const node = el.current;
    if (!node) return;
    const check = () => setFits(node.offsetHeight <= window.innerHeight - top - 16);
    const ro = new ResizeObserver(check);
    ro.observe(node);
    window.addEventListener('resize', check);
    check();
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', check);
    };
  }, [el, top]);
  return fits;
}

export default function Analyze() {
  const site = useSite();
  const [picked, setPicked] = useState<Picked | null>(null);
  const [crop, setCrop] = useState<CropChoice>('auto');
  const [pointMode, setPointMode] = useState(false);
  const [roi, setRoi] = useState<{ x: number; y: number } | null>(null);
  const [state, setState] = useState<State>({ kind: 'idle' });
  const [revealed, setRevealed] = useState<ReadonlySet<string>>(new Set());
  const [mode, setMode] = useViewMode();
  const [openHow, setOpenHow] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);
  const inputCol = useRef<HTMLDivElement>(null);
  const sticky = useFitsViewport(inputCol, 96);

  useEffect(() => {
    document.title = 'Run the model · Patra';
    window.scrollTo(0, 0);
  }, []);

  const run = useCallback(async (p: Picked, c: CropChoice, point: { x: number; y: number } | null) => {
    abort.current?.abort();
    const ac = new AbortController();
    abort.current = ac;
    setState({ kind: 'busy' });
    // bring step 2 ("Reading it") into view right away when its top is not on screen (always on phones, where
    // the result sits below the inputs; on desktop after choosing a photo from further down)
    requestAnimationFrame(() => {
      const top = resultRef.current?.getBoundingClientRect().top ?? 0;
      if (top < 64 || top > window.innerHeight * 0.6) resultRef.current?.scrollIntoView({ block: 'start' });
    });
    try {
      const r = await analyze(p.blob, c, point, ac.signal, p.name);
      if (ac.signal.aborted) return;
      setState({ kind: 'done', r });
      const id = p.sample?.id;
      if (id) setRevealed((v) => (v.has(id) ? v : new Set(v).add(id)));
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
  const detailed = mode === 'detailed';
  const step = state.kind === 'idle' ? 1 : busy ? 2 : 3;
  // Simple -> Detailed with the "How this was calculated" panel open, then bring it into view
  const showWorkings = () => {
    setOpenHow(true);
    setMode('detailed');
    requestAnimationFrame(() => requestAnimationFrame(() => document.getElementById('how')?.scrollIntoView({ block: 'start' })));
  };
  const chooseAnother = () => {
    document.getElementById('step-1')?.scrollIntoView({ block: 'start' });
    document.getElementById('leaf-file')?.click();
  };
  const togglePoint = (on: boolean) => {
    setPointMode(on);
    if (!on && roi) {
      setRoi(null);
      if (picked) void run(picked, crop, null);
    }
  };

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

        <Stepper step={step} />

        <div className="mt-8 grid gap-12 lg:grid-cols-12 lg:gap-10">
          <div ref={inputCol} id="step-1"
               className={`flex min-w-0 scroll-mt-24 flex-col gap-5 lg:col-span-5 lg:self-start lg:gap-4 ${sticky ? 'lg:sticky lg:top-24' : ''}`}
               data-sticky={sticky ? 'true' : 'false'} data-testid="input-column">
            <div className="flex min-h-[52px] items-center">
              <StepHeading n={1} title="Choose a leaf" />
            </div>
            {picked && <Preview url={picked.url} busy={busy} pointMode={pointMode} roi={roi} onPoint={point} />}
            {picked && (
              <div className="flex flex-col gap-2 border-b hairline pb-4">
                <CropControl value={crop} onChange={chooseCrop} disabled={busy} />
                <PointToggle on={pointMode} onChange={togglePoint} />
              </div>
            )}
            <DropZone samples={site?.samples ?? []} onPick={pick} compact={!!picked} revealed={revealed}
                      before={picked ? null : (
                        <details className="group rounded-[6px] border hairline px-4 [&_summary::-webkit-details-marker]:hidden">
                          <summary className="flex min-h-[44px] cursor-pointer list-none items-center justify-between text-[14px] text-text/85">
                            Options: crop and pointing at one leaf
                            <span aria-hidden="true" className="text-muted transition-transform group-open:rotate-45 motion-reduce:transition-none">+</span>
                          </summary>
                          <div className="flex flex-col gap-3 pb-4">
                            <CropControl value={crop} onChange={chooseCrop} disabled={busy} />
                            <PointToggle on={pointMode} onChange={togglePoint} />
                          </div>
                        </details>
                      )} />
          </div>

          <div ref={resultRef} className="flex min-w-0 scroll-mt-24 flex-col gap-8 lg:col-span-7" data-testid="result-column">
            <div className="flex min-h-[52px] flex-wrap items-center justify-between gap-3">
              <StepHeading n={step === 1 ? 3 : step} title={step === 2 ? 'Reading it' : 'Your result'} />
              <ModeToggle mode={mode} onChange={setMode} />
            </div>
            {state.kind === 'idle' && <EmptyState showcase={site?.showcase.turmeric} />}
            {busy && (
              <>
                <Progress busy r={null} detailed={detailed} />
                <ResultSkeleton />
              </>
            )}
            {state.kind === 'error' && (
              <ErrorState e={state.e} detailed={detailed} onChooseAnother={chooseAnother}
                          onRetry={picked && state.e.code !== 'too_large' ? () => void run(picked, crop, roi) : null} />
            )}
            {r && (
              <div className="grid gap-8" key={r.request_id}>
                <div className="reveal grid gap-4">
                  <Verdict r={r} levels={site?.confidence_levels} onPickCrop={(c) => chooseCrop(c)} />
                  {picked?.sample && <SampleCheck s={picked.sample} r={r} />}
                  <Progress busy={false} r={r} detailed={detailed} />
                </div>
                <div className="reveal [animation-delay:80ms]">
                  <Evidence r={r} />
                </div>
                <div className="reveal [animation-delay:160ms]">
                  <Compare r={r} detailed={detailed} />
                </div>
                {detailed ? (
                  <>
                    <ResultCard r={r} />
                    <Measurements r={r} />
                    <div id="how" className="scroll-mt-28">
                      <HowCalculated r={r} startOpen={openHow} />
                    </div>
                  </>
                ) : (
                  <div className="flex flex-wrap items-center gap-4 rounded-[6px] border hairline p-5 sm:p-6">
                    <p className="min-w-0 flex-1 basis-[260px] text-[15px] leading-relaxed text-muted">
                      Want the numbers? Detailed view shows every measurement, both models’ scores, the code that ran and the raw data.
                    </p>
                    <button type="button" onClick={showWorkings} data-testid="show-workings"
                            className="min-h-[44px] shrink-0 rounded-full border border-white/30 px-5 text-sm text-text transition-colors hover:bg-white/10">
                      See how this was calculated
                    </button>
                  </div>
                )}
                <p className="max-w-[640px] text-[13px] leading-relaxed text-muted">
                  A research prototype trained on two public datasets. It is not a diagnosis; confirm with an agronomist before
                  treating a crop.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

const STEPS = ['Choose a leaf', 'Reading it', 'Your result'] as const;

/** 1 Choose a leaf -> 2 Reading it -> 3 Your result, with the current step marked. */
function Stepper({ step }: { step: 1 | 2 | 3 }) {
  return (
    <ol aria-label="Steps" className="mt-10 flex flex-wrap items-center gap-x-4 gap-y-2 sm:gap-x-3" data-testid="stepper">
      {STEPS.map((label, i) => {
        const n = i + 1;
        const state = n < step ? 'done' : n === step ? 'current' : 'next';
        return (
          <li key={label} aria-current={state === 'current' ? 'step' : undefined} className="flex items-center gap-2 sm:gap-3">
            <span className={`flex h-7 w-7 items-center justify-center rounded-full border text-[12px] transition-colors duration-300 motion-reduce:transition-none ${
              state === 'current' ? 'border-paper bg-paper text-ink' : state === 'done' ? 'border-leaf/70 text-leaf' : 'border-white/20 text-muted'}`}>
              {state === 'done' ? <Check size={13} aria-hidden="true" /> : n}
            </span>
            <span className={`text-[13px] sm:text-[14px] ${state === 'next' ? 'text-muted' : 'text-text'}`}>
              {label}
              <span className="sr-only">{state === 'done' ? ' (done)' : state === 'current' ? ' (current step)' : ''}</span>
            </span>
            {n < 3 && <span aria-hidden="true" className={`hidden h-px w-12 sm:block ${n < step ? 'bg-leaf/60' : 'bg-white/15'}`} />}
          </li>
        );
      })}
    </ol>
  );
}

function StepHeading({ n, title }: { n: number; title: string }) {
  return (
    <h2 className="flex items-baseline gap-3 text-[20px] tracking-[-0.02em] text-text">
      <span className="mono text-[13px] text-muted">{String(n).padStart(2, '0')}</span>
      {title}
    </h2>
  );
}

/** Simple (plain words, no tables) or Detailed (every number that exists), remembered per browser. */
function ModeToggle({ mode, onChange }: { mode: ViewMode; onChange: (m: ViewMode) => void }) {
  const opts: [ViewMode, string][] = [['simple', 'Simple'], ['detailed', 'Detailed']];
  return (
    <div role="radiogroup" aria-label="How much detail" className="inline-flex rounded-full border border-white/15 p-1" data-testid="mode-toggle">
      {opts.map(([v, l]) => (
        <button key={v} type="button" role="radio" aria-checked={mode === v} onClick={() => onChange(v)}
                className={`mono min-h-[44px] rounded-full px-4 text-[11px] uppercase tracking-[0.08em] ${
                  mode === v ? 'bg-paper text-ink' : 'text-muted hover:text-text'}`}>
          {l}
        </button>
      ))}
    </div>
  );
}
