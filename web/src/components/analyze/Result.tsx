import { Check } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Help } from '@/components/analyze/Help';
import type { GlossaryKey } from '@/content/glossary';
import type { AnalyzeResponse } from '@/lib/api';
import { CROP_NAME, fix, int, pct, pctRaw, segMethod } from '@/lib/format';

/** The four phases the server runs, in plain words, with the trace stages (and older timing keys) behind each. */
const PHASES = [
  { label: 'Finding the leaf', tech: 'denoise · colour spaces · segmentation', ids: ['denoise', 'colour', 'segment'], key: 'stage1' },
  { label: 'Spotting damaged areas', tech: 'lesions · measurements', ids: ['lesions', 'measure'], key: null },
  { label: 'Asking the neural network', tech: 'crop model · unusual-photo check · disease CNN', ids: ['cnn'], key: 'cnn' },
  { label: 'Comparing both answers', tech: 'rule score · SVM · agreement', ids: ['ip_decision', 'compare'], key: 'ip_decision' },
] as const;

/**
 * What the server is doing. The request is a single call, so while it runs there is no per-step progress to
 * report: every phase shows as "working" next to a real elapsed-time counter. When the answer arrives each phase
 * gets its real time from the response (trace / timings_ms). Nothing is marked done before it is.
 */
export function Progress({ busy, r, detailed }: { busy: boolean; r: AnalyzeResponse | null; detailed: boolean }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!busy) return;
    const t0 = performance.now();
    setElapsed(0);
    const id = window.setInterval(() => setElapsed((performance.now() - t0) / 1000), 100);
    return () => window.clearInterval(id);
  }, [busy]);
  const phaseMs = (ph: (typeof PHASES)[number]): number | null => {
    const st = r?.trace?.stages.filter((x) => (ph.ids as readonly string[]).includes(x.id) && x.ms != null);
    if (st?.length) return st.reduce((a, x) => a + (x.ms ?? 0), 0);
    return ph.key && r ? r.timings_ms[ph.key] ?? null : null;
  };
  const total = r?.timings_ms.total ?? null;
  return (
    <div data-testid="progress">
      <p className="sr-only" aria-live="polite">{busy ? 'Reading the leaf on the server.' : r && total != null ? `Done in ${fix(total / 1000, 1)} seconds.` : ''}</p>
      <p className="mono text-[12px] text-muted" aria-hidden="true">
        {busy ? `Reading the leaf on the server… ${fix(elapsed, 1)} s` : r && total != null ? `Read in ${fix(total / 1000, 1)} s on the server` : ''}
      </p>
      <ol className="mt-3 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
        {PHASES.map((ph) => {
          const ms = phaseMs(ph);
          const done = !busy && !!r;
          return (
            <li key={ph.label} className={`border-t pt-2.5 transition-colors duration-300 ${done ? 'border-leaf' : busy ? 'border-white/40' : 'border-white/15'}`}>
              <span className={`flex items-start gap-1.5 text-[13px] leading-snug ${done ? 'text-text' : 'text-text/80'}`}>
                {done ? (
                  <Check size={13} aria-hidden="true" className="mt-[3px] shrink-0 text-leaf" />
                ) : (
                  <span aria-hidden="true" className="mt-[6px] h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-white/70 motion-reduce:animate-none" />
                )}
                {ph.label}
              </span>
              <span className="mono mt-0.5 block pl-[19px] text-[11px] text-muted">
                {done ? (ms != null ? `${fix(ms, 0)} ms` : 'done') : busy ? 'working' : ''}
              </span>
              {detailed && <span className="mono block pl-[19px] text-[10px] leading-snug text-muted/80">{ph.tech}</span>}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

/** The four-line card (same format as leafdx.service.result_card). */
export function ResultCard({ r }: { r: AnalyzeResponse }) {
  const c = r.crop;
  const unsure = c.status === 'uncertain';
  const rows: [string, React.ReactNode][] = [
    [
      'Crop',
      unsure ? (
        <>Unsure — Turmeric {pct(c.probs.turmeric, 0)} / Black gram {pct(c.probs.blackgram, 0)}</>
      ) : (
        <>
          {CROP_NAME[c.label]} ({pct(c.confidence, 0)}){c.status === 'user_selected' && <span className="text-muted"> · selected by you</span>}
        </>
      ),
    ],
    ['Prediction', r.disease.label === 'diseased' ? 'Diseased' : 'Healthy'],
    ['Affected Region', pctRaw(r.measurements.affected_region_pct)],
    ['CNN Confidence', pct(r.disease.confidence, 0)],
  ];
  return (
    <div className="rounded-[6px] border hairline bg-soil p-6 sm:p-7" data-testid="result-card">
      <dl className="mono grid gap-2 text-[15px] leading-relaxed sm:text-[17px]">
        {rows.map(([k, v]) => (
          <div key={k} className="flex flex-wrap gap-x-2" data-testid={`card-${k.toLowerCase().replace(' ', '-')}`}>
            <dt className="text-text/70">{k}:</dt>
            <dd className={k === 'Prediction' ? (r.disease.label === 'diseased' ? 'text-turmeric' : 'text-leaf') : 'text-text'}>{v}{k === 'CNN Confidence' && <Help term="cnn" />}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function Meter({ label, value, max = 1, marker, tone = 'bg-text' }: { label: React.ReactNode; value: number; max?: number; marker?: number; tone?: string }) {
  return (
    <div>
      <div className="flex justify-between gap-4">
        <span className="text-[13px] text-muted">{label}</span>
        <span className="mono text-[13px] text-text">{fix(value, 2)}</span>
      </div>
      <div className="relative mt-1.5 h-[3px] bg-white/10">
        <div className={`h-full ${tone}`} style={{ width: `${Math.min(100, (value / max) * 100)}%` }} />
        {marker != null && (
          <span className="absolute -top-1.5 h-[15px] w-px bg-paper" style={{ left: `${(marker / max) * 100}%` }} aria-hidden="true" />
        )}
      </div>
    </div>
  );
}

const COMPONENT_NAME: Record<string, string> = {
  affected_pct: 'damaged area',
  log_lesion_count: 'number of spots (log)',
  abn_colour_pct: 'abnormal colour',
  glcm_contrast: 'texture contrast',
};

function Answer({ label, prob }: { label: string; prob: number }) {
  return (
    <div className="grid gap-2">
      <p className={`font-playfair text-[26px] italic leading-none ${label === 'diseased' ? 'text-turmeric' : 'text-leaf'}`}>{label}</p>
      <div className="flex items-center gap-3">
        <span className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-white/10" aria-hidden="true">
          <span className={`absolute inset-y-0 left-0 rounded-full ${prob >= 0.5 ? 'bg-turmeric' : 'bg-leaf'}`} style={{ width: `${Math.round(prob * 100)}%` }} />
        </span>
        <span className="mono shrink-0 text-[12px] text-text">{pct(prob, 0)} chance of disease</span>
      </div>
    </div>
  );
}

function Facts({ title, rows }: { title: string; rows: React.ReactNode[] }) {
  return (
    <div>
      <p className="caption mb-1.5 text-[10px] text-muted">{title}</p>
      <ul className="grid gap-1.5 text-[13px] leading-snug text-text/85">
        {rows.map((r, i) => (
          <li key={i} className="flex gap-2"><span aria-hidden="true" className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-white/40" />{r}</li>
        ))}
      </ul>
    </div>
  );
}

const COLOUR_WORD: Record<string, string> = { yellow: 'yellowing', brown: 'brown dead tissue', dark: 'dark tissue', pale: 'pale, mildew-like' };

/** Two readings compared, side by side: image processing (measurements) and the neural network (the photo).
 *  Both modes: each side's answer, its chance of disease and what it based that on, in words.
 *  Detailed adds the rule score against its threshold with its parts, as meters. */
export function Compare({ r, detailed }: { r: AnalyzeResponse; detailed: boolean }) {
  const { rule, svm, cnn } = r.disease.by;
  const m = r.measurements;
  const colours = Object.entries(r.abnormal_colour_pct).filter(([, v]) => v >= 0.5).sort((x, y) => y[1] - x[1])
    .map(([k, v]) => `${COLOUR_WORD[k] ?? k} ${pctRaw(v, 1)}`);
  const crop = r.crop.status === 'uncertain'
    ? `Crop: not sure (turmeric ${pct(r.crop.probs.turmeric, 0)} / black gram ${pct(r.crop.probs.blackgram, 0)})`
    : `Crop: ${CROP_NAME[r.crop.label]} (${r.crop.status === 'user_selected' ? 'chosen by you' : `${pct(r.crop.confidence, 0)} sure`})`;
  return (
    <section aria-labelledby="cmp-title" className="rounded-[6px] border hairline p-5 sm:p-6" data-testid="compare">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 id="cmp-title" className="text-[20px] tracking-[-0.02em]">Two readings compared<Help term="methods" /></h2>
        <span data-testid="agreement"
              className={`mono rounded-full px-3 py-1 text-[11px] uppercase tracking-[0.08em] ${r.agreement ? 'bg-leaf/15 text-leaf' : 'bg-turmeric/15 text-turmeric'}`}>
          {r.agreement ? 'Agree' : 'Disagree'}
        </span>
      </div>
      <p className="mt-2 text-[14px] leading-relaxed text-muted">
        {r.agreement
          ? `Both readings say ${r.disease.label}, one from measuring the leaf and one from looking at the photo, so the answer is stronger.`
          : `The measurements say ${svm.label}; the neural network says ${cnn.label}. When they disagree, check the pictures above and the leaf itself.`}
      </p>
      <div className="mt-5 grid gap-px overflow-hidden rounded-[4px] border hairline bg-[var(--line)] sm:grid-cols-2">
        <div className="grid content-start gap-4 bg-ink p-4 sm:p-5" data-testid="reading-ip">
          <div>
            <p className="caption text-[11px] text-muted">Image processing</p>
            <p className="mt-1 text-[13px] leading-snug text-muted">Measures colour, damaged area and texture, then a machine-learning model (SVM) judges the numbers.<Help term="svm" /></p>
          </div>
          <Answer label={svm.label} prob={svm.prob} />
          <Facts title="What it measured" rows={[
            m.lesion_count > 0 ? `Damaged area: ${pctRaw(m.affected_region_pct, 1)} of the leaf, in ${int(m.lesion_count)} ${m.lesion_count === 1 ? 'spot' : 'spots'}` : 'No damaged spots found',
            colours.length ? `Unusual colour: ${pctRaw(m.abn_colour_pct, 1)} of the leaf (${colours.join(', ')})` : `Unusual colour: ${pctRaw(m.abn_colour_pct, 1)} of the leaf`,
            <>Hand-written rule: {rule.label} (score {fix(rule.score, 2)}, diseased from {fix(rule.threshold, 2)})<Help term="rule" /></>,
          ]} />
          {detailed && (
            <div className="grid gap-3 border-t hairline pt-4">
              <p className="caption text-[11px] text-muted">Rule-based score · parts</p>
              <Meter label={`score (threshold ${fix(rule.threshold, 2)})`} value={rule.score} marker={rule.threshold} tone={rule.label === 'diseased' ? 'bg-turmeric' : 'bg-leaf'} />
              {Object.entries(rule.components).map(([k, v]) => (
                <Meter key={k} label={COMPONENT_NAME[k] ?? k.replace(/_/g, ' ')} value={v} tone="bg-white/50" />
              ))}
            </div>
          )}
        </div>
        <div className="grid content-start gap-4 bg-ink p-4 sm:p-5" data-testid="reading-cnn">
          <div>
            <p className="caption text-[11px] text-muted">Neural network</p>
            <p className="mt-1 text-[13px] leading-snug text-muted">Looks at the photo itself, without any measurements.<Help term="cnn" /></p>
          </div>
          <Answer label={cnn.label} prob={cnn.prob} />
          <Facts title="What it saw" rows={[
            crop,
            r.ood.flagged
              ? <>Photo check: unusual, unlike the photos it learned from, so less reliable<Help term="ood" /></>
              : <>Photo check: looks like the photos it learned from<Help term="ood" /></>,
            'Where it looked: the red areas in “Where the AI looked” above',
          ]} />
          {detailed && (
            <div className="grid gap-3 border-t hairline pt-4">
              <Meter label="crop model: chance it is turmeric" value={r.crop.probs.turmeric} tone="bg-white/50" />
              <Meter label="unusual-photo distance (OOD)" value={r.ood.ood_score} max={Math.max(1, r.ood.ood_score)} tone="bg-white/50" />
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

export function Measurements({ r }: { r: AnalyzeResponse }) {
  const m = r.measurements;
  const ac = r.abnormal_colour_pct;
  // [plain label, technical term, glossary entry, value]; sub-rows start with two spaces
  const rows: [string, string, GlossaryKey | null, string][] = [
    ['Damaged area', 'affected region', 'lesion', pctRaw(m.affected_region_pct)],
    ['Damaged spots', 'lesion count', 'lesion', int(m.lesion_count)],
    ['Average spot size', 'mean lesion area', null, `${int(m.mean_lesion_area_px)} px`],
    ['Unusual colour, whole leaf', 'abnormal colour', null, pctRaw(m.abn_colour_pct)],
    ['  yellowing', 'chlorosis', null, pctRaw(ac.yellow)],
    ['  brown dead tissue', 'necrosis', null, pctRaw(ac.brown)],
    ['  dark tissue', '', null, pctRaw(ac.dark)],
    ['  pale, mildew-like', '', null, pctRaw(ac.pale)],
    ['Texture contrast', 'GLCM', 'glcm', fix(m.glcm_contrast, 2)],
    ['Texture uniformity', 'GLCM energy', 'glcm', fix(m.glcm_energy, 3)],
    ['Texture smoothness', 'GLCM homogeneity', 'glcm', fix(m.glcm_homogeneity, 3)],
    ['Leaf cut-out method', 'segmentation', 'segmentation', segMethod(m.segmentation_mode)],
    ['Unusual photo score', 'OOD distance', 'ood', `${fix(r.ood.ood_score, 3)}${r.ood.flagged ? ' (unusual photo)' : ''}`],
  ];
  return (
    <section aria-labelledby="meas-title">
      <h2 id="meas-title" className="mb-4 text-[20px] tracking-[-0.02em]">Measurements</h2>
      <table className="w-full border-collapse text-left">
        <caption className="sr-only">Stage 1 measurements for this photo</caption>
        <tbody>
          {rows.map(([k, tech, g, v]) => (
            <tr key={k} className="border-b hairline">
              <th scope="row" className={`py-2.5 pr-4 text-[14px] font-normal ${k.startsWith('  ') ? 'pl-4 text-muted' : 'text-text/85'}`}>
                {k.trim()}
                {tech && <span className="text-muted"> ({tech})</span>}
                {g && <Help term={g} />}
              </th>
              <td className="mono py-2.5 text-right text-[14px] text-text">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mono mt-3 text-[11px] text-muted">request {r.request_id} · model {r.model_version} · schema v{r.schema_version}</p>
    </section>
  );
}
