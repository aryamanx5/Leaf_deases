import { Play, Target } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Reveal, SplitHeading } from '@/components/fx/Reveal';
import { CROP_NAME, fix, pct } from '@/lib/format';
import { useInView, useReducedMotion } from '@/lib/motion';
import { analyse, histPath, paint, sigmaPath, type Lab, type LabView } from '@/lib/otsu';
import type { Site, Sources } from '@/lib/site';

const N = 400; // the lab works on a 400 px copy of the leaf
const H = 100; // histogram box height (SVG units)
const VIEWS: [LabView, string][] = [['cutout', 'Leaf cut-out'], ['mask', 'Mask'], ['map', 'Distance map']];

const pick = (s: Sources) => s.webp['640'] ?? Object.values(s.webp)[0]!;
async function pixels(src: string): Promise<ImageData> {
  const im = new Image();
  im.decoding = 'async';
  im.src = src;
  await im.decode(); // decode off the main thread, so the draw below does not block a frame
  const c = document.createElement('canvas');
  c.width = c.height = N;
  const g = c.getContext('2d', { willReadFrequently: true })!;
  g.drawImage(im, 0, 0, N, N);
  return g.getImageData(0, 0, N, N);
}

/**
 * Threshold lab: Otsu's method, live. The visitor drags the cut across the histogram of the showcase leaf's
 * colour-distance map and watches the mask change; the curve shows how well each cut separates leaf from
 * background, and its peak is the cut Otsu's method picks. All numbers are computed in the browser from the picture.
 */
export default function OtsuLab({ site }: { site: Site }) {
  const reduced = useReducedMotion();
  const [crop, setCrop] = useState<'turmeric' | 'blackgram'>('turmeric');
  const [view, setView] = useState<LabView>('cutout');
  const [lab, setLab] = useState<Lab | null>(null);
  const [t, setT] = useState(128);
  const [sweeping, setSweeping] = useState(false);
  const [nearRef, near] = useInView<HTMLDivElement>({ margin: '500px 0px', threshold: 0 });
  const [stageRef, seen] = useInView<HTMLDivElement>({ threshold: 0.45 });
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const out = useRef<ImageData | null>(null);
  const raf = useRef(0);
  const s = site.showcase[crop];

  // load the two pictures once the section is near, and again when the leaf changes
  useEffect(() => {
    if (!near) return;
    let alive = true;
    setLab(null);
    void Promise.all([pixels(pick(s.media.original)), pixels(pick(s.media.segmentation))]).then(([photo, dist]) => {
      if (alive) setLab(analyse(photo, dist));
    }).catch(() => {});
    return () => {
      alive = false;
    };
  }, [near, s]);

  // repaint whenever the cut, the view or the leaf changes
  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !lab) return;
    const g = c.getContext('2d')!;
    out.current ??= g.createImageData(N, N);
    paint(out.current, lab, t, view);
    g.putImageData(out.current, 0, 0);
  }, [lab, t, view]);

  // the search, replayed: sweep the cut across every level, then settle on the best one
  const sweep = (to: number) => {
    cancelAnimationFrame(raf.current);
    if (reduced) {
      setT(to);
      return;
    }
    setSweeping(true);
    const t0 = performance.now();
    const step = (now: number) => {
      const k = (now - t0) / 2600;
      if (k < 0.7) setT(Math.round((k / 0.7) * 255));
      else if (k < 1) {
        const e = 1 - Math.pow(1 - (k - 0.7) / 0.3, 3);
        setT(Math.round(255 + (to - 255) * e));
      } else {
        setT(to);
        setSweeping(false);
        return;
      }
      raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
  };
  const played = useRef('');
  useEffect(() => {
    if (!lab || !seen || played.current === crop) return;
    played.current = crop;
    sweep(lab.best);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lab, seen, crop]);
  useEffect(() => () => cancelAnimationFrame(raf.current), []);

  const paths = useMemo(() => (lab ? { hist: histPath(lab.hist, H), sigma: sigmaPath(lab.sigma, H) } : null), [lab]);
  const grab = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.type === 'pointermove' && e.buttons === 0) return;
    cancelAnimationFrame(raf.current);
    setSweeping(false);
    const r = e.currentTarget.getBoundingClientRect();
    setT(Math.round(Math.min(255, Math.max(0, ((e.clientX - r.left) / r.width) * 255))));
  };
  const quality = lab ? lab.sigma[t]! : 0;
  const atBest = !!lab && Math.abs(t - lab.best) <= 1;
  const p = s.params;

  return (
    <section id="lab" aria-labelledby="lab-title" className="section-pad bg-ink">
      <div ref={nearRef} className="container-grid">
        <div className="mb-12 flex flex-wrap items-end justify-between gap-6">
          <div>
            <Reveal as="p" className="caption mb-6 text-muted">Try it · Otsu’s method</Reveal>
            <SplitHeading id="lab-title" text="Where does leaf end and background begin?" italic="Drag the line."
                          className="max-w-[820px] text-4xl font-normal leading-[1.05] tracking-[-0.04em] sm:text-5xl lg:text-[52px]" />
          </div>
          <div role="group" aria-label="Lab leaf" className="inline-flex rounded-full border hairline p-1">
            {(['turmeric', 'blackgram'] as const).map((c) => (
              <button key={c} type="button" aria-pressed={crop === c} onClick={() => setCrop(c)}
                      className={`mono min-h-[36px] rounded-full px-3 text-[11px] uppercase tracking-[0.08em] transition-colors ${
                        crop === c ? 'bg-paper text-ink' : 'text-muted hover:text-text'}`}>
                {CROP_NAME[c]}
              </button>
            ))}
          </div>
        </div>

        <div ref={stageRef} className="grid gap-10 lg:grid-cols-12 lg:gap-12">
          <Reveal className="lg:col-span-5">
            <figure>
              <div className="relative aspect-square overflow-hidden rounded-[6px] border hairline bg-soil">
                <canvas ref={canvasRef} width={N} height={N} role="img" data-testid="lab-canvas"
                        aria-label={`The ${CROP_NAME[crop].toLowerCase()} test leaf with every pixel below the chosen cut removed as background.`}
                        className={`h-full w-full transition-opacity duration-500 ${lab ? 'opacity-100' : 'opacity-0'}`} />
                {!lab && <div className="shimmer absolute inset-0" aria-hidden="true" />}
                <span className={`caption absolute left-3 top-3 rounded-full px-2.5 py-1 text-[10px] transition-colors duration-300 ${
                  atBest ? 'bg-leaf text-ink' : 'bg-ink/80 text-white/80'}`}>
                  {atBest ? 'Otsu’s pick' : sweeping ? 'Searching' : 'Your cut'}
                </span>
              </div>
              <figcaption className="mt-4 flex flex-wrap items-center justify-between gap-3">
                <div role="group" aria-label="What to show" className="inline-flex rounded-full border hairline p-1">
                  {VIEWS.map(([v, l]) => (
                    <button key={v} type="button" aria-pressed={view === v} onClick={() => setView(v)}
                            className={`mono min-h-[36px] rounded-full px-3 text-[11px] uppercase tracking-[0.08em] transition-colors ${
                              view === v ? 'bg-paper text-ink' : 'text-muted hover:text-text'}`}>
                      {l}
                    </button>
                  ))}
                </div>
                <span className="mono text-[11px] text-muted">{CROP_NAME[crop]} · test image</span>
              </figcaption>
            </figure>
          </Reveal>

          <Reveal delay={120} className="lg:col-span-7">
            <div className="rounded-[6px] border hairline p-5 sm:p-7">
              <div className="flex items-baseline justify-between gap-4">
                <p className="caption text-[11px] text-muted">Brightness of the distance map</p>
                <p className="mono flex items-center gap-4 text-[11px] text-muted" aria-hidden="true">
                  <span className="flex items-center gap-1.5"><span className="h-2 w-2 bg-white/35" /> pixels</span>
                  <span className="flex items-center gap-1.5"><span className="h-px w-4 bg-leaf" /> separation</span>
                </p>
              </div>
              <svg viewBox={`0 0 256 ${H}`} preserveAspectRatio="none" role="img" data-testid="lab-hist"
                   aria-label="Histogram of the distance map with the separation curve; drag to move the cut."
                   className="mt-4 h-[180px] w-full cursor-ew-resize touch-none select-none sm:h-[220px]"
                   onPointerDown={(e) => {
                     e.currentTarget.setPointerCapture(e.pointerId);
                     grab(e);
                   }}
                   onPointerMove={grab}>
                <rect x={0} y={0} width={t + 0.5} height={H} fill="rgba(232,240,228,0.03)" />
                {paths && (
                  <>
                    <path d={paths.hist} fill="rgba(232,240,228,0.32)" />
                    <path d={paths.sigma} fill="none" stroke="var(--leaf)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
                  </>
                )}
                {lab && (
                  <line x1={lab.best + 0.5} x2={lab.best + 0.5} y1={0} y2={H} stroke="var(--leaf)" strokeWidth="1" strokeDasharray="2 3"
                        vectorEffect="non-scaling-stroke" />
                )}
                <line x1={t + 0.5} x2={t + 0.5} y1={0} y2={H} stroke="var(--paper)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
              </svg>
              <div className="mono flex justify-between text-[10px] text-muted" aria-hidden="true">
                <span>dark · background</span>
                <span>bright · leaf</span>
              </div>
              <input type="range" min={0} max={255} step={1} value={t} data-testid="lab-range"
                     onChange={(e) => {
                       cancelAnimationFrame(raf.current);
                       setSweeping(false);
                       setT(Number(e.target.value));
                     }}
                     aria-label="Threshold" aria-valuetext={`level ${t} of 255`} className="lab-range w-full" />

              <dl className="mt-2 grid grid-cols-3 gap-4 border-t hairline pt-5">
                <div>
                  <dt className="caption text-[10px] text-muted">Cut at level</dt>
                  <dd className="mono mt-1 text-[26px] leading-none text-text sm:text-[32px]" data-testid="lab-t">{t}</dd>
                </div>
                <div>
                  <dt className="caption text-[10px] text-muted">Kept as leaf</dt>
                  <dd className="mono mt-1 text-[26px] leading-none text-text sm:text-[32px]">{lab ? pct(lab.above[t], 0) : '–'}</dd>
                </div>
                <div>
                  <dt className="caption text-[10px] text-muted">Separation</dt>
                  <dd className={`mono mt-1 text-[26px] leading-none sm:text-[32px] ${atBest ? 'text-leaf' : 'text-text'}`}>{lab ? pct(quality, 0) : '–'}</dd>
                </div>
              </dl>
              <div className="mt-6 flex flex-wrap gap-3">
                <button type="button" disabled={!lab} onClick={() => lab && sweep(lab.best)} data-testid="lab-sweep"
                        className="inline-flex min-h-[44px] items-center gap-2 rounded-full bg-cta px-5 text-sm font-medium text-white transition-colors hover:bg-cta-hover disabled:opacity-50">
                  <Play size={14} aria-hidden="true" /> Watch Otsu search
                </button>
                <button type="button" disabled={!lab} data-testid="lab-snap"
                        onClick={() => {
                          cancelAnimationFrame(raf.current);
                          setSweeping(false);
                          if (lab) setT(lab.best);
                        }}
                        className="inline-flex min-h-[44px] items-center gap-2 rounded-full border border-white/30 px-5 text-sm text-text transition-colors hover:bg-white/10 disabled:opacity-50">
                  <Target size={14} aria-hidden="true" /> Snap to the best cut
                </button>
              </div>
            </div>

            <ol className="mt-8 grid gap-5 sm:grid-cols-3">
              {[
                ['Count', 'Every pixel of the distance map goes into a histogram: dark background on the left, bright leaf on the right.'],
                ['Try every cut', 'For each possible level the method asks how far apart the two groups it creates are. That is the green curve.'],
                ['Keep the best', 'The peak of the curve is the cut that separates leaf from background most cleanly. No setting is tuned by hand.'],
              ].map(([h, body], i) => (
                <li key={h} className="border-t hairline pt-4">
                  <p className="mono text-[12px] text-leaf">{String(i + 1).padStart(2, '0')}</p>
                  <h3 className="mt-1 text-[17px] tracking-[-0.02em]">{h}</h3>
                  <p className="mt-1.5 text-[14px] leading-relaxed text-muted">{body}</p>
                </li>
              ))}
            </ol>
            <p className="mono mt-6 text-[12px] leading-relaxed text-muted">
              Computed in your browser on a {N}-pixel copy of the picture from step 3.
              {p.threshold.otsu_distance != null && (
                <> In the real pipeline, at full size, Otsu cut this leaf at a colour distance of {fix(p.threshold.otsu_distance, 1)} and
                  kept {pct(p.morphology.leaf_coverage, 0)} of the frame as leaf.</>
              )}
            </p>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
