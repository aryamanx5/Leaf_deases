import { Check, ChevronDown, Code2, Copy, Download } from 'lucide-react';
import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { fetchCode, type AnalyzeResponse, type CodeItem } from '@/lib/api';
import { CROP_NAME, fix, int, pct, pctRaw, segMethod } from '@/lib/format';
import { highlightPython, type Tok } from '@/lib/highlight';

type Values = Record<string, number | string | boolean | null>;
type Row = [string, string];

const RULE_PART: Record<string, string> = {
  affected_pct: 'damaged area', log_lesion_count: 'number of spots (log)', abn_colour_pct: 'abnormal colour', glcm_contrast: 'texture contrast',
};

/** Plain-language description of every stage (ids match GET /api/code and response.trace). */
const STAGES: Record<string, { title: string; short: string; text: string; noTime?: string; rows?: (v: Values) => Row[] }> = {
  denoise: {
    title: 'Clean up the photo',
    short: 'Clean up',
    text: 'The photo is shrunk and smoothed so camera noise and glare are not mistaken for spots.',
    rows: (v) => [['longest side', `${v.max_side_px} px`], ['median filter', `${v.median_ksize}×${v.median_ksize}`],
                  ['Gaussian blur', `${v.gaussian_ksize}×${v.gaussian_ksize}`]],
  },
  colour: {
    title: 'Look at the colours differently (HSV / Lab)',
    short: 'Colours',
    noTime: 'timed with step 1',
    text: 'Colours are re-expressed as hue, saturation and brightness (HSV) and as Lab, where green-versus-red is its own channel. Brightness is evened out for texture.',
    rows: (v) => [['colour spaces', String(v.spaces)], ['contrast limit (CLAHE)', String(v.clahe_clip)], ['tiles', `${v.clahe_tile}×${v.clahe_tile}`]],
  },
  segment: {
    title: 'Cut the leaf out (segmentation)',
    short: 'Cut out the leaf',
    text: 'The leaf is separated from what is behind it: on a plain backdrop by how far each pixel’s colour is from the backdrop, in a busy field photo with GrabCut. The first two code blocks are the Otsu threshold, which picks the cut-off between leaf and backdrop automatically.',
    rows: (v) => [['method', segMethod(String(v.method))],
                  ...(v.border_background_share != null ? [['border that is backdrop', pct(Number(v.border_background_share), 0)] as Row] : []),
                  ...(v.distance_threshold != null ? [['colour-distance threshold (Otsu)', fix(Number(v.distance_threshold), 1)] as Row] : []),
                  ['leaf share of the photo', pctRaw(Number(v.leaf_coverage_pct))],
                  ...(v.fallback_whole_image ? [['fallback', 'whole image used'] as Row] : [])],
  },
  lesions: {
    title: 'Find damaged spots (lesions)',
    short: 'Damaged spots',
    text: 'Pixels whose colour differs from this leaf’s own healthy green are grouped into spots. Spots smaller than the minimum size are ignored.',
    rows: (v) => [['leaf area', `${int(Number(v.leaf_area_px))} px`], ['smallest spot counted', `${int(Number(v.min_lesion_px))} px`],
                  ['spots found', int(Number(v.lesion_count))], ['damaged share of the leaf', pctRaw(Number(v.affected_pct))]],
  },
  measure: {
    title: 'Measure the leaf',
    short: 'Measure',
    text: 'The leaf is described by a list of numbers: colour statistics, damaged area, texture and the shape of the spots.',
    rows: (v) => [['numbers measured', String(v.n_features)], ['abnormal colour', pctRaw(Number(v.abn_colour_pct))],
                  ['texture contrast (GLCM)', fix(Number(v.glcm_contrast), 2)], ['texture evenness (GLCM)', fix(Number(v.glcm_homogeneity), 3)]],
  },
  ip_decision: {
    title: 'Rule-based score and machine-learning reading (SVM)',
    short: 'Rule score + SVM',
    text: 'A hand-tuned rule adds up four measurements and compares the total with a threshold. Separately, an SVM, a classic machine-learning model, reads all the measurements.',
    rows: (v) => [['rule score', `${fix(Number(v.rule_score), 2)} vs threshold ${fix(Number(v.rule_threshold), 2)} → ${v.rule_label}`],
                  ['SVM margin', fix(Number(v.svm_margin), 2)], ['SVM chance of disease', `${pct(Number(v.svm_prob_diseased), 0)} → ${v.svm_label}`]],
  },
  cnn: {
    title: 'Ask the neural network (CNN)',
    short: 'Neural network',
    text: 'A neural network looks at the photo itself rather than the measurements: it decides the crop, checks whether the photo is unusual, then gives a chance of disease.',
    rows: (v) => [['chance it is turmeric', pct(Number(v.p_turmeric), 0)], ['crop decision', String(v.crop_status).replace(/_/g, ' ')],
                  ['chance of disease', `${pct(Number(v.p_diseased), 0)} → ${v.cnn_label}`],
                  ['unusual-photo distance (OOD)', `${fix(Number(v.ood_score), 3)} (limit ${fix(Number(v.ood_threshold), 3)})`]],
  },
  compare: {
    title: 'Compare both answers',
    short: 'Compare',
    noTime: 'instant',
    text: 'The answer from the measurements (SVM) is compared with the answer from the pixels (CNN). When they agree the result is stronger.',
    rows: (v) => [['SVM says', String(v.svm_label)], ['CNN says', String(v.cnn_label)], ['agree', v.agreement ? 'yes' : 'no']],
  },
  card: { title: 'Write the result card', short: 'Result card', text: 'The four lines at the top of the result.' },
};

const OPEN_KEY = 'patra:how-open';
const readOpen = () => {
  try {
    return window.localStorage.getItem(OPEN_KEY) === '1';
  } catch {
    return false;
  }
};

/** The response without the embedded images (they are large base64 strings), for copy and download. */
const withoutImages = (r: AnalyzeResponse) => {
  const { images: _images, ...rest } = r;
  void _images;
  return rest;
};

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

function CopyButton({ text, label, testId }: { text: string; label: string; testId?: string }) {
  const [state, setState] = useState<'idle' | 'ok' | 'fail'>('idle');
  useEffect(() => {
    if (state === 'idle') return;
    const id = window.setTimeout(() => setState('idle'), 1800);
    return () => window.clearTimeout(id);
  }, [state]);
  return (
    <button type="button" data-testid={testId} onClick={async () => setState((await copyText(text)) ? 'ok' : 'fail')}
            className="inline-flex min-h-[44px] shrink-0 items-center gap-2 rounded-full border border-white/25 px-4 text-[13px] text-text transition-colors hover:bg-white/10">
      {state === 'ok' ? <Check size={14} aria-hidden="true" className="text-leaf" /> : <Copy size={14} aria-hidden="true" />}
      <span aria-live="polite">{state === 'ok' ? 'Copied' : state === 'fail' ? 'Copy failed' : label}</span>
    </button>
  );
}

/** "How this was calculated": the real stages, their code and this run's data. Collapsed by default. */
export function HowCalculated({ r, startOpen = false }: { r: AnalyzeResponse; startOpen?: boolean }) {
  const [open, setOpen] = useState(() => startOpen || readOpen());
  const [tab, setTab] = useState<'steps' | 'code' | 'data'>('steps');
  const [codeStage, setCodeStage] = useState('denoise');
  const uid = useId();
  const tabs = [['steps', 'Steps'], ['code', 'Code'], ['data', 'Data']] as const;
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const toggle = () => {
    setOpen((o) => {
      try {
        window.localStorage.setItem(OPEN_KEY, o ? '0' : '1');
      } catch {
        /* private mode: the panel just forgets */
      }
      return !o;
    });
  };
  const viewCode = (stage: string) => {
    setCodeStage(stage);
    setTab('code');
    requestAnimationFrame(() => tabRefs.current[1]?.focus());
  };
  const onTabKey = (e: React.KeyboardEvent, i: number) => {
    const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    const n = (i + d + tabs.length) % tabs.length;
    setTab(tabs[n]![0]);
    tabRefs.current[n]?.focus();
  };

  return (
    <section aria-labelledby={`${uid}-h`} className="rounded-[6px] border hairline" data-testid="how-panel">
      <h2 id={`${uid}-h`} className="m-0">
        <button type="button" onClick={toggle} aria-expanded={open} aria-controls={`${uid}-body`} data-testid="how-toggle"
                className="flex min-h-[64px] w-full items-center justify-between gap-4 px-5 py-4 text-left sm:px-6">
          <span>
            <span className="block text-[20px] tracking-[-0.02em] text-text">How this was calculated</span>
            <span className="mt-1 block text-[14px] text-muted">The steps, the real code and the data from this photo.</span>
          </span>
          <ChevronDown size={20} aria-hidden="true" className={`shrink-0 text-muted transition-transform duration-300 motion-reduce:transition-none ${open ? 'rotate-180' : ''}`} />
        </button>
      </h2>
      {open && (
        <div id={`${uid}-body`} className="border-t hairline px-5 pb-6 pt-4 sm:px-6">
          <div role="tablist" aria-label="How this was calculated" className="-mx-1 mb-6 flex gap-1.5 overflow-x-auto px-1 pb-1">
            {tabs.map(([v, l], i) => (
              <button key={v} ref={(el) => { tabRefs.current[i] = el; }} type="button" role="tab" id={`${uid}-tab-${v}`}
                      aria-selected={tab === v} aria-controls={`${uid}-panel-${v}`} tabIndex={tab === v ? 0 : -1}
                      data-testid={`how-tab-${v}`} onClick={() => setTab(v)} onKeyDown={(e) => onTabKey(e, i)}
                      className={`mono min-h-[44px] shrink-0 rounded-full border px-4 text-[12px] uppercase tracking-[0.08em] transition-colors ${
                        tab === v ? 'border-paper bg-paper text-ink' : 'border-white/15 text-muted hover:border-white/40 hover:text-text'}`}>
                {l}
              </button>
            ))}
          </div>
          <div role="tabpanel" id={`${uid}-panel-${tab}`} aria-labelledby={`${uid}-tab-${tab}`} tabIndex={0} className="min-w-0 focus-visible:outline-offset-8">
            {tab === 'steps' && <StepsTab r={r} onViewCode={viewCode} />}
            {tab === 'code' && <CodeTab stage={codeStage} onStage={setCodeStage} />}
            {tab === 'data' && <DataTab r={r} />}
          </div>
        </div>
      )}
    </section>
  );
}

function StepsTab({ r, onViewCode }: { r: AnalyzeResponse; onViewCode: (stage: string) => void }) {
  const stages = r.trace?.stages ?? [];
  if (!stages.length) return <p className="text-[15px] text-muted">This server did not send the step-by-step trace. Update the API and run the photo again.</p>;
  return (
    <ol className="relative grid gap-0" data-testid="how-steps">
      {stages.map((s, i) => {
        const info = STAGES[s.id];
        const rows = info?.rows?.(s.values) ?? Object.entries(s.values).map(([k, v]) => [k, String(v)] as Row);
        return (
          <li key={s.id} className="relative grid grid-cols-[28px_1fr] gap-x-4 pb-7 last:pb-0">
            {i < stages.length - 1 && <span aria-hidden="true" className="absolute bottom-0 left-[13px] top-7 w-px bg-white/15" />}
            <span aria-hidden="true" className="mono z-[1] flex h-7 w-7 items-center justify-center rounded-full border border-leaf/60 bg-ink text-[11px] text-leaf">
              {i + 1}
            </span>
            <div className="min-w-0">
              <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <h3 className="text-[16px] font-normal text-text">{info?.title ?? s.id}</h3>
                <span className="mono text-[12px] text-muted">{s.ms != null ? `${fix(s.ms, 0)} ms` : info?.noTime ?? ''}</span>
              </div>
              <p className="mt-1 max-w-[68ch] text-[14px] leading-relaxed text-muted">{info?.text}</p>
              <dl className="mt-3 flex flex-wrap gap-2">
                {rows.map(([k, v]) => (
                  <div key={k} className="rounded-[4px] border hairline bg-soil px-2.5 py-1.5 text-[12px]">
                    <dt className="inline text-muted">{k}: </dt>
                    <dd className="mono inline text-text">{v}</dd>
                  </div>
                ))}
              </dl>
              <button type="button" onClick={() => onViewCode(s.id)}
                      className="mt-2 inline-flex min-h-[44px] items-center gap-1.5 text-[13px] text-leaf underline-offset-4 hover:underline">
                <Code2 size={14} aria-hidden="true" /> View code<span className="sr-only"> for {info?.title ?? s.id}</span>
              </button>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

const TOK_CLASS: Record<Tok['k'], string> = {
  kw: 'text-turmeric', str: 'text-leaf', com: 'italic text-muted', num: 'text-chlorosis', deco: 'text-turmeric',
  fn: 'font-semibold text-paper', const: 'text-mildew', txt: '',
};

function CodeBlock({ item }: { item: CodeItem }) {
  const lines = useMemo(() => highlightPython(item.source), [item.source]);
  return (
    <figure className="min-w-0 overflow-hidden rounded-[6px] border hairline bg-soil" data-testid="code-block">
      <figcaption className="flex flex-wrap items-center justify-between gap-3 border-b hairline px-4 py-2">
        <span className="mono min-w-0 text-[12px] leading-relaxed text-muted">
          <span className="block text-text">
            {item.name}
            <span className="ml-2 inline-block rounded-full border border-leaf/40 px-2 py-px align-middle text-[10px] uppercase tracking-[0.08em] text-leaf">Python</span>
          </span>
          <span className="block break-all">{item.file}:{item.start_line}–{item.end_line}</span>
        </span>
        <CopyButton text={item.source} label="Copy" />
      </figcaption>
      <pre className="max-h-[480px] overflow-auto py-3 text-[12.5px] leading-[1.6]" data-lenis-prevent tabIndex={0}
           aria-label={`Source of ${item.name}, lines ${item.start_line} to ${item.end_line} of ${item.file}`}>
        <code className="mono block w-max min-w-full">
          {lines.map((ln, i) => (
            <span key={i} className="flex">
              <span aria-hidden="true" className="sticky left-0 w-[4.5ch] shrink-0 select-none bg-soil pr-3 text-right text-muted/70">
                {item.start_line + i}
              </span>
              <span className="whitespace-pre pr-4">
                {ln.map((t, j) => (t.k === 'txt' ? t.v : <span key={j} className={TOK_CLASS[t.k]}>{t.v}</span>))}
                {'\n'}
              </span>
            </span>
          ))}
        </code>
      </pre>
    </figure>
  );
}

function CodeTab({ stage, onStage }: { stage: string; onStage: (s: string) => void }) {
  const [items, setItems] = useState<CodeItem[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const rowRef = useRef<HTMLDivElement>(null);
  // keep the chosen stage's chip in view in the (horizontally scrolling) row, without moving the page
  useEffect(() => {
    const row = rowRef.current;
    const chip = row?.querySelector<HTMLElement>('[aria-pressed="true"]');
    if (row && chip) row.scrollLeft = chip.offsetLeft - (row.clientWidth - chip.offsetWidth) / 2;
  }, [stage, items]);
  useEffect(() => {
    let live = true;
    setFailed(false);
    fetchCode().then((x) => live && setItems(x)).catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, [attempt]);
  if (failed) {
    return (
      <div role="alert" className="text-[15px] text-muted">
        The code could not be loaded from the server.{' '}
        <button type="button" onClick={() => setAttempt((a) => a + 1)} className="min-h-[44px] text-leaf underline underline-offset-4">Try again</button>
      </div>
    );
  }
  if (!items) return <div className="h-[320px] animate-pulse rounded-[6px] bg-soil motion-reduce:animate-none" aria-label="Loading the code" />;
  const ids = [...new Set(items.map((i) => i.stage))];
  const shown = items.filter((i) => i.stage === stage);
  return (
    <div className="grid min-w-0 gap-4">
      <div ref={rowRef} role="group" aria-label="Stage" className="relative -mx-1 flex gap-1.5 overflow-x-auto px-1 pb-1">
        {ids.map((id, n) => (
          <button key={id} type="button" aria-pressed={id === stage} onClick={() => onStage(id)} data-testid={`code-stage-${id}`}
                  className={`min-h-[44px] shrink-0 rounded-full border px-3.5 text-[13px] transition-colors ${
                    id === stage ? 'border-leaf bg-leaf/10 text-text' : 'border-white/15 text-muted hover:border-white/40 hover:text-text'}`}>
            <span className="mono mr-1.5 text-[11px] text-muted">{n + 1}</span>
            {STAGES[id]?.short ?? id}
          </button>
        ))}
      </div>
      <p className="max-w-[68ch] text-[14px] leading-relaxed text-muted">
        <span className="text-text">{STAGES[stage]?.title}.</span> {STAGES[stage]?.text} This is the code the server ran, read from the
        repository when you opened this tab.
      </p>
      {shown.map((it) => <CodeBlock key={it.name} item={it} />)}
    </div>
  );
}

function KV({ title, rows }: { title: string; rows: Row[] }) {
  return (
    <div className="min-w-0">
      <h3 className="caption mb-2 text-[11px] text-muted">{title}</h3>
      <table className="w-full border-collapse text-left">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k} className="border-b hairline">
              <th scope="row" className={`py-2 pr-4 text-[13px] font-normal ${k.startsWith('  ') ? 'pl-4 text-muted' : 'text-text/85'}`}>{k.trim()}</th>
              <td className="mono break-all py-2 text-right text-[13px] text-text">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DataTab({ r }: { r: AnalyzeResponse }) {
  const json = useMemo(() => JSON.stringify(withoutImages(r), null, 2), [r]);
  const inp = r.trace?.input;
  const { rule, svm, cnn } = r.disease.by;
  const card = r.result_card ?? [];
  const download = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
    a.download = `patra-run-${r.request_id}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const crop = r.crop.status === 'uncertain' ? `unsure (${pct(r.crop.probs.turmeric, 0)} turmeric)`
    : `${CROP_NAME[r.crop.label]} ${pct(r.crop.confidence, 0)}${r.crop.status === 'user_selected' ? ', chosen by you' : ''}`;
  const roi = inp?.roi ? Object.entries(inp.roi).map(([k, v]) => `${k} ${Array.isArray(v) ? v.map((x) => fix(x, 2)).join(',') : fix(v, 2)}`).join(', ') : 'none';
  return (
    <div className="grid min-w-0 gap-8" data-testid="how-data">
      <div className="grid gap-8 md:grid-cols-2">
        <KV title="Input" rows={[
          ['file', inp?.filename ?? 'n/a'],
          ['size', inp ? `${inp.width} × ${inp.height} px` : 'n/a'],
          ['analysed at', inp ? `${inp.analysed_width} × ${inp.analysed_height} px` : 'n/a'],
          ['crop you gave', inp ? (inp.crop_given === 'auto' ? 'auto' : CROP_NAME[inp.crop_given]) : 'n/a'],
          ['crop used', crop],
          ['point on the leaf', roi],
          ['leaf-coloured share', inp ? pct(inp.plant_colour_fraction, 0) : 'n/a'],
        ]} />
        <KV title="Measurements" rows={[
          ['affected region', pctRaw(r.measurements.affected_region_pct)],
          ['spots (lesions)', int(r.measurements.lesion_count)],
          ['mean spot area', `${int(r.measurements.mean_lesion_area_px)} px`],
          ['abnormal colour', pctRaw(r.measurements.abn_colour_pct)],
          ...Object.entries(r.abnormal_colour_pct).map(([k, v]) => [`  ${k}`, pctRaw(v)] as Row),
          ['GLCM contrast / energy / homogeneity', `${fix(r.measurements.glcm_contrast, 2)} / ${fix(r.measurements.glcm_energy, 3)} / ${fix(r.measurements.glcm_homogeneity, 3)}`],
          ['segmentation', segMethod(r.measurements.segmentation_mode)],
        ]} />
        <KV title="Rule-based score" rows={[
          ['score', `${fix(rule.score, 3)} (threshold ${fix(rule.threshold, 3)}) → ${rule.label}`],
          ...Object.entries(rule.components).map(([k, v]) => [`  ${RULE_PART[k] ?? k.replace(/_/g, ' ')}`, fix(v, 3)] as Row),
        ]} />
        <KV title="Probabilities and checks" rows={[
          ['SVM, chance of disease', `${fix(svm.prob, 3)} → ${svm.label}`],
          ['CNN, chance of disease', `${fix(cnn.prob, 3)} → ${cnn.label}`],
          ['crop, chance of turmeric', fix(r.crop.probs.turmeric, 3)],
          ['unusual-photo score (OOD)', `${fix(r.ood.ood_score, 3)}${r.ood.flagged ? ' · flagged' : ''}`],
          ['both methods agree', r.agreement ? 'yes' : 'no'],
        ]} />
        <KV title="Time on the server" rows={Object.entries(r.timings_ms).map(([k, v]) => [k.replace(/_/g, ' '), `${fix(v, 0)} ms`] as Row)} />
        {card.length > 0 && (
          <div className="min-w-0">
            <h3 className="caption mb-2 text-[11px] text-muted">Result card (plain text)</h3>
            <pre className="mono overflow-x-auto rounded-[6px] border hairline bg-soil p-4 text-[13px] leading-relaxed text-text" data-testid="result-card-text"
                 tabIndex={0} data-lenis-prevent>{card.join('\n')}</pre>
            <div className="mt-2"><CopyButton text={card.join('\n')} label="Copy card" /></div>
          </div>
        )}
      </div>
      <div className="min-w-0">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <h3 className="caption text-[11px] text-muted">Raw response (JSON)</h3>
          <div className="flex flex-wrap gap-2">
            <CopyButton text={json} label="Copy JSON" testId="copy-json" />
            <button type="button" onClick={download} data-testid="download-json"
                    className="inline-flex min-h-[44px] items-center gap-2 rounded-full border border-white/25 px-4 text-[13px] text-text transition-colors hover:bg-white/10">
              <Download size={14} aria-hidden="true" /> Download JSON
            </button>
          </div>
        </div>
        <div className="mono max-h-[520px] overflow-auto rounded-[6px] border hairline bg-soil p-4 text-[12.5px] leading-[1.7]" data-lenis-prevent
             data-testid="json-tree">
          <JsonNode name="response" value={r} depth={0} />
        </div>
        <p className="mt-2 text-[12px] text-muted">The four images are shown shortened here and left out of Copy and Download.</p>
      </div>
    </div>
  );
}

function JsonNode({ name, value, depth }: { name: string; value: unknown; depth: number }) {
  if (value !== null && typeof value === 'object') {
    const entries = Array.isArray(value) ? value.map((v, i) => [String(i), v] as const) : Object.entries(value);
    const brackets = Array.isArray(value) ? ['[', ']'] : ['{', '}'];
    return (
      <details open={depth < 1 || (depth < 2 && name !== 'images' && name !== 'trace')}>
        <summary className="cursor-pointer select-none rounded-[3px] text-text/85 hover:text-text">
          <span className="text-chlorosis">{name}</span> {brackets[0]}<span className="text-muted">{entries.length} {Array.isArray(value) ? 'items' : 'keys'}</span>{brackets[1]}
        </summary>
        <div className="ml-[0.6ch] border-l border-white/10 pl-[2ch]">
          {entries.map(([k, v]) => <JsonNode key={k} name={k} value={v} depth={depth + 1} />)}
        </div>
      </details>
    );
  }
  let shown: string;
  let cls = 'text-text';
  if (typeof value === 'string') {
    shown = value.startsWith('data:') ? `"${value.slice(0, 32)}…" (${int(value.length / 1024)} kB image)` : JSON.stringify(value);
    cls = 'text-leaf';
  } else if (typeof value === 'number') {
    shown = Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(6)));
    cls = 'text-mildew';
  } else {
    shown = String(value);
    cls = 'text-turmeric';
  }
  return (
    <div className="break-all pl-[2ch] -indent-[2ch]">
      <span className="text-chlorosis">{name}</span>: <span className={cls}>{shown}</span>
    </div>
  );
}
