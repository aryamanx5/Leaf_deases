import { Help } from '@/components/analyze/Help';
import type { AnalyzeResponse } from '@/lib/api';
import { fix, pct } from '@/lib/format';

/** Every disease model's answer on this photo (Phase 3 arms), the production model marked. Probabilities are
 *  temperature-scaled on validation photos; which arm is production was decided on validation, not on this photo. */
export function Arms({ r }: { r: AnalyzeResponse }) {
  if (!r.arms) return null;
  const entries = Object.entries(r.arms);
  const nDis = entries.filter(([, a]) => a.label === 'diseased').length;
  return (
    <section aria-labelledby="arms-title" className="rounded-[6px] border hairline p-5 sm:p-6" data-testid="arms">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 id="arms-title" className="text-[20px] tracking-[-0.02em]">Every model’s answer<Help term="methods" /></h2>
        <span className="mono text-[12px] text-muted">{nDis} of {entries.length} say diseased</span>
      </div>
      <p className="mt-2 text-[14px] leading-relaxed text-muted">
        The same photo, read by each model we compared. The answer above comes from{' '}
        <span className="text-text">{r.production_model_name ?? r.production_model}</span>, picked on validation photos.
      </p>
      <table className="mt-4 w-full border-collapse text-left">
        <caption className="sr-only">Disease prediction of each model for this photo</caption>
        <thead>
          <tr className="border-b hairline text-[11px] uppercase tracking-[0.08em] text-muted">
            <th scope="col" className="py-2 pr-3 font-normal">Model</th>
            <th scope="col" className="py-2 pr-3 font-normal">Answer</th>
            <th scope="col" className="py-2 font-normal">Chance of disease</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([k, a]) => {
            const prod = k === r.production_model;
            return (
              <tr key={k} className={`border-b hairline ${prod ? 'bg-white/[0.04]' : ''}`} data-testid={`arm-${k}`}>
                <th scope="row" className="py-2.5 pr-3 text-[14px] font-normal text-text/85">
                  {a.name}
                  {prod && <span className="mono ml-2 rounded-full bg-leaf/15 px-2 py-0.5 text-[10px] uppercase tracking-[0.08em] text-leaf">production</span>}
                </th>
                <td className={`py-2.5 pr-3 font-playfair text-[16px] italic ${a.label === 'diseased' ? 'text-turmeric' : 'text-leaf'}`}>{a.label}</td>
                <td className="py-2.5">
                  <div className="flex items-center gap-3">
                    <span className="relative h-1.5 w-full max-w-[160px] overflow-hidden rounded-full bg-white/10" aria-hidden="true">
                      <span className={`absolute inset-y-0 left-0 rounded-full ${a.prob >= 0.5 ? 'bg-turmeric' : 'bg-leaf'}`}
                            style={{ width: `${Math.round(a.prob * 100)}%` }} />
                    </span>
                    <span className="mono shrink-0 text-[12px] text-text">{pct(a.prob, 0)}</span>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {r.disease_set && (
        <p className="mono mt-3 text-[11px] text-muted">
          conformal set: {r.disease_set.join(' / ') || 'empty'} · crop set: {r.crop_set?.join(' / ') || 'empty'} · segmenter {r.segmenter}
          {r.timings_ms.total != null && ` · ${fix(r.timings_ms.total / 1000, 1)} s`}
        </p>
      )}
    </section>
  );
}
