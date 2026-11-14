import { useEffect, useState } from 'react';
import type { AnalyzeResponse } from '@/lib/api';
import { pctRaw } from '@/lib/format';
import { useReducedMotion } from '@/lib/motion';

const TISSUE: [keyof AnalyzeResponse['abnormal_colour_pct'], string, string][] = [
  ['yellow', 'yellowing', 'var(--chlorosis)'],
  ['brown', 'brown dead tissue', 'var(--necrosis)'],
  ['dark', 'dark tissue', 'var(--dark-tissue)'],
  ['pale', 'pale, mildew-like', 'var(--mildew)'],
];
const RING = 2 * Math.PI * 42;

/** 0 -> 1 once, eased, when the result appears (jumps to 1 for reduced motion). */
function useRise(key: string, ms = 1100) {
  const reduced = useReducedMotion();
  const [k, setK] = useState(reduced ? 1 : 0);
  useEffect(() => {
    if (reduced) {
      setK(1);
      return;
    }
    let raf = 0;
    const t0 = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - t0) / ms);
      setK(1 - Math.pow(1 - t, 3));
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [key, ms, reduced]);
  return k;
}

/**
 * The measurements at a glance, drawn: how much of the leaf is damaged (a ring), in how many spots, and which
 * kinds of abnormal tissue make it up. The same numbers the verdict sentence and the Measurements table use.
 */
export function Glance({ r }: { r: AnalyzeResponse }) {
  const k = useRise(r.request_id);
  const m = r.measurements;
  const diseased = r.disease.label === 'diseased';
  const tone = diseased ? 'var(--turmeric)' : 'var(--leaf)';
  const area = Math.max(0, Math.min(100, m.affected_region_pct));
  const mix = TISSUE.map(([key, label, colour]) => ({ label, colour, v: Math.max(0, r.abnormal_colour_pct[key] ?? 0) }));
  const sum = mix.reduce((a, x) => a + x.v, 0);
  return (
    <section aria-label="Measurements at a glance" data-testid="glance" className="grid gap-3 sm:grid-cols-[auto_auto_1fr]">
      <div className="flex items-center gap-4 rounded-[6px] border hairline p-4">
        <svg viewBox="0 0 100 100" className="h-[76px] w-[76px] shrink-0 -rotate-90" role="img"
             aria-label={`${pctRaw(area, 1)} of the leaf area is marked as ${diseased ? 'damaged' : 'unusual in colour'}.`}>
          <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(232,240,228,0.12)" strokeWidth="7" />
          <circle cx="50" cy="50" r="42" fill="none" stroke={tone} strokeWidth="7" strokeLinecap="round"
                  strokeDasharray={RING} strokeDashoffset={RING * (1 - (area / 100) * k)} />
        </svg>
        <p>
          <span className="mono block text-[26px] leading-none text-text" aria-hidden="true">{pctRaw(area * k, 1)}</span>
          <span className="mt-1.5 block text-[13px] leading-snug text-muted">{diseased ? 'of the leaf damaged' : 'of the leaf off-colour'}</span>
        </p>
      </div>
      <div className="flex flex-col justify-center rounded-[6px] border hairline p-4 sm:min-w-[130px]">
        <span className="mono block text-[26px] leading-none text-text">{Math.round(m.lesion_count * k)}</span>
        <span className="mt-1.5 block text-[13px] leading-snug text-muted">separate {m.lesion_count === 1 ? 'spot' : 'spots'}</span>
      </div>
      <div className="flex min-w-0 flex-col justify-center rounded-[6px] border hairline p-4">
        <p className="text-[13px] text-muted">What the marked tissue looks like</p>
        {sum > 0 ? (
          <>
            <div className="mt-2.5 flex h-2 w-full overflow-hidden rounded-full bg-white/10" role="img"
                 aria-label={`Abnormal tissue by kind: ${mix.filter((x) => x.v > 0).map((x) => `${x.label} ${pctRaw((x.v / sum) * 100, 0)}`).join(', ')}.`}>
              {mix.map((x) => x.v > 0 && (
                <span key={x.label} className="h-full origin-left transition-none" style={{ width: `${(x.v / sum) * 100 * k}%`, background: x.colour }} />
              ))}
            </div>
            <ul className="mt-2.5 flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-text/85">
              {mix.map((x) => x.v > 0 && (
                <li key={x.label} className="flex items-center gap-1.5">
                  <span aria-hidden="true" className="h-2 w-2 rounded-full" style={{ background: x.colour }} />
                  {x.label}
                </li>
              ))}
            </ul>
          </>
        ) : (
          <p className="mt-2 text-[14px] text-text/85">No yellow, brown, dark or pale tissue stood out.</p>
        )}
      </div>
    </section>
  );
}
