import { Check, Download } from 'lucide-react';
import { useEffect, useState } from 'react';
import type { AnalyzeResponse, Crop } from '@/lib/api';
import { CROP_NAME, fix, int, pct, pctRaw } from '@/lib/format';

const STEPS = ['Denoise', 'Colour space', 'Segment', 'Threshold', 'Morphology', 'Lesions', 'Features', 'CNN'] as const;

/**
 * Pipeline steps. While waiting, the highlight moves through the steps and holds on the last one;
 * nothing is marked finished until the response has arrived (no invented progress).
 */
export function Steps({ busy, done, totalMs }: { busy: boolean; done: boolean; totalMs: number | null }) {
  const [at, setAt] = useState(0);
  useEffect(() => {
    if (!busy) return;
    setAt(0);
    const id = window.setInterval(() => setAt((a) => Math.min(STEPS.length - 1, a + 1)), 320);
    return () => window.clearInterval(id);
  }, [busy]);
  return (
    <div aria-live="polite">
      <ol className="grid grid-cols-4 gap-x-4 gap-y-2 sm:grid-cols-8">
        {STEPS.map((s, i) => {
          const active = busy && i === at;
          return (
            <li key={s} className={`border-t pt-2 transition-colors duration-300 ${done ? 'border-leaf' : active ? 'border-text' : 'border-white/15'}`}>
              <span className={`mono flex items-center gap-1 text-[11px] ${done ? 'text-leaf' : active ? 'text-text' : 'text-muted'}`}>
                {done && <Check size={11} aria-hidden="true" />}
                {s}
              </span>
            </li>
          );
        })}
      </ol>
      <p className="mono mt-3 text-[12px] text-muted">
        {busy ? 'Running on the server…' : done && totalMs != null ? `Done in ${fix(totalMs / 1000, 1)} s on the server.` : ''}
      </p>
    </div>
  );
}

/** The four-line card (same format as leafdx.service.result_card). */
export function ResultCard({ r, onPickCrop }: { r: AnalyzeResponse; onPickCrop: (c: Crop) => void }) {
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
            <dd className={k === 'Prediction' ? (r.disease.label === 'diseased' ? 'text-turmeric' : 'text-leaf') : 'text-text'}>{v}</dd>
          </div>
        ))}
      </dl>
      {unsure && (
        <div className="mt-5 flex flex-wrap items-center gap-3 border-t hairline pt-5">
          <p className="w-full text-[14px] text-muted">The crop model is not sure. Tell it which crop this is and it will re-run.</p>
          {(['turmeric', 'blackgram'] as const).map((k) => (
            <button key={k} type="button" onClick={() => onPickCrop(k)}
                    className="rounded-full border border-white/30 px-4 py-2 text-sm text-text transition-colors hover:bg-white/10">
              It’s {CROP_NAME[k].toLowerCase()}
            </button>
          ))}
        </div>
      )}
      {r.ood.flagged && !unsure && (
        <p className="mt-4 text-[13px] text-chlorosis">This photo looks unlike our training images; treat the result with care.</p>
      )}
    </div>
  );
}

function Meter({ label, value, max = 1, marker, tone = 'bg-text' }: { label: string; value: number; max?: number; marker?: number; tone?: string }) {
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
  affected_pct: 'affected area',
  log_lesion_count: 'lesion count (log)',
  abn_colour_pct: 'abnormal colour',
  glcm_contrast: 'texture contrast',
};

/** Image processing vs CNN: agreement badge, rule score against its threshold, both probabilities. */
export function Compare({ r }: { r: AnalyzeResponse }) {
  const { rule, svm, cnn } = r.disease.by;
  return (
    <section aria-labelledby="cmp-title" className="rounded-[6px] border hairline p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 id="cmp-title" className="text-[20px] tracking-[-0.02em]">Image processing vs CNN</h2>
        <span data-testid="agreement"
              className={`mono rounded-full px-3 py-1 text-[11px] uppercase tracking-[0.08em] ${r.agreement ? 'bg-leaf/15 text-leaf' : 'bg-turmeric/15 text-turmeric'}`}>
          {r.agreement ? 'Agree' : 'Disagree'}
        </span>
      </div>
      <p className="mt-2 text-[14px] leading-relaxed text-muted">
        {r.agreement
          ? `Both readings say ${r.disease.label}.`
          : `The SVM on image-processing features says ${svm.label}; the CNN says ${cnn.label}. Check the evidence below.`}
      </p>
      <div className="mt-6 grid gap-6 md:grid-cols-2">
        <div className="grid content-start gap-3">
          <p className="caption text-[11px] text-muted">Rule score · {rule.label}</p>
          <Meter label={`score (threshold ${fix(rule.threshold, 2)})`} value={rule.score} marker={rule.threshold} tone={rule.label === 'diseased' ? 'bg-turmeric' : 'bg-leaf'} />
          {Object.entries(rule.components).map(([k, v]) => (
            <Meter key={k} label={COMPONENT_NAME[k] ?? k.replace(/_/g, ' ')} value={v} tone="bg-white/50" />
          ))}
        </div>
        <div className="grid content-start gap-3">
          <p className="caption text-[11px] text-muted">Probability of disease</p>
          <Meter label={`SVM on IP features · ${svm.label}`} value={svm.prob} tone="bg-text" />
          <Meter label={`CNN · ${cnn.label}`} value={cnn.prob} tone="bg-text" />
        </div>
      </div>
    </section>
  );
}

export function Measurements({ r }: { r: AnalyzeResponse }) {
  const m = r.measurements;
  const ac = r.abnormal_colour_pct;
  const rows: [string, string][] = [
    ['Affected region', pctRaw(m.affected_region_pct)],
    ['Lesion count', int(m.lesion_count)],
    ['Mean lesion area', `${int(m.mean_lesion_area_px)} px`],
    ['Abnormal colour (whole leaf)', pctRaw(m.abn_colour_pct)],
    ['  yellow (chlorosis)', pctRaw(ac.yellow)],
    ['  brown (necrosis)', pctRaw(ac.brown)],
    ['  dark tissue', pctRaw(ac.dark)],
    ['  pale (mildew)', pctRaw(ac.pale)],
    ['GLCM contrast', fix(m.glcm_contrast, 2)],
    ['GLCM energy', fix(m.glcm_energy, 3)],
    ['GLCM homogeneity', fix(m.glcm_homogeneity, 3)],
    ['Segmentation', m.segmentation_mode.replace(/_/g, ' ')],
    ['OOD distance', `${fix(r.ood.ood_score, 3)}${r.ood.flagged ? ' (unusual photo)' : ''}`],
  ];
  const download = () => {
    const { images: _images, ...rest } = r; // the JSON report omits the embedded images
    void _images;
    const blob = new Blob([JSON.stringify(rest, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `patra-report-${r.request_id}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  return (
    <section aria-labelledby="meas-title">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h2 id="meas-title" className="text-[20px] tracking-[-0.02em]">Measurements</h2>
        <button type="button" onClick={download}
                className="inline-flex items-center gap-2 rounded-full border border-white/30 px-4 py-2 text-sm text-text transition-colors hover:bg-white/10">
          <Download size={15} aria-hidden="true" /> Download report (JSON)
        </button>
      </div>
      <table className="w-full border-collapse text-left">
        <caption className="sr-only">Stage 1 measurements for this photo</caption>
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k} className="border-b hairline">
              <th scope="row" className={`py-2.5 pr-4 text-[14px] font-normal ${k.startsWith('  ') ? 'pl-4 text-muted' : 'text-text/85'}`}>{k.trim()}</th>
              <td className="mono py-2.5 text-right text-[14px] text-text">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mono mt-3 text-[11px] text-muted">request {r.request_id} · model {r.model_version} · schema v{r.schema_version}</p>
    </section>
  );
}
