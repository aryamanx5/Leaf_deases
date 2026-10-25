import { AlertTriangle } from 'lucide-react';
import { Help } from '@/components/analyze/Help';
import type { AnalyzeResponse, Crop } from '@/lib/api';
import { CROP_NAME, pctRaw } from '@/lib/format';
import type { Site } from '@/lib/site';
import { confidenceLevel, LEVEL_CUTOFFS, type Level } from '@/lib/verdict';

const LEVEL_WORD: Record<Level, string> = { high: 'High', medium: 'Medium', low: 'Low' };
const LEVEL_FILL: Record<Level, number> = { high: 3, medium: 2, low: 1 };

/** The plain-language answer, first in the result: verdict, how much is damaged, agreement, confidence, next step. */
export function Verdict({ r, levels, onPickCrop }: { r: AnalyzeResponse; levels: Site['confidence_levels'] | undefined; onPickCrop: (c: Crop) => void }) {
  const diseased = r.disease.label === 'diseased';
  const level = confidenceLevel(r, levels ?? LEVEL_CUTOFFS);
  const unsureCrop = r.crop.status === 'uncertain';
  const crop = CROP_NAME[r.crop.label];
  const m = r.measurements;
  const n = m.lesion_count;
  const area = m.affected_region_pct < 1 && m.affected_region_pct > 0 ? 'under 1%' : `about ${pctRaw(m.affected_region_pct, 0)}`;
  const spots = `${n} separate ${n === 1 ? 'spot' : 'spots'}`;
  const damage = n === 0
    ? 'No damaged spots were found on the leaf.'
    : diseased
      ? `${area[0]!.toUpperCase()}${area.slice(1)} of the leaf shows damage, in ${spots}.`
      : `${area[0]!.toUpperCase()}${area.slice(1)} of the leaf has unusual colour, in ${spots}; on a healthy call these are usually glare, veins or small marks.`;
  const ev = levels?.[level];
  const evidence = ev && levels
    ? `On ${levels.n} held-back lab photos (${levels.set}), “${LEVEL_WORD[level]}” answers were right ${ev.right} of ${ev.n} times.`
    : undefined;
  const levelRule = levels ?? LEVEL_CUTOFFS;

  return (
    <section aria-labelledby="verdict-title" data-testid="verdict"
             className={`rounded-[6px] border p-6 sm:p-7 ${diseased ? 'border-turmeric/35 bg-turmeric/[0.06]' : 'border-leaf/35 bg-leaf/[0.06]'}`}>
      <p className="caption text-[11px] text-muted" data-testid="verdict-crop">
        {unsureCrop ? 'Crop: not sure' : `${crop} leaf${r.crop.status === 'user_selected' ? ' · you chose the crop' : ''}`}
      </p>
      <h2 id="verdict-title" data-testid="verdict-title" className="mt-2 text-[28px] leading-[1.15] tracking-[-0.03em] sm:text-[34px]">
        This leaf looks{' '}
        <span className={`font-playfair italic ${diseased ? 'text-turmeric' : 'text-leaf'}`}>{diseased ? 'diseased' : 'healthy'}</span>.
      </h2>
      <p className="mt-3 max-w-[60ch] text-[16px] leading-relaxed text-text/90">{damage}</p>
      <p className="mt-2 max-w-[60ch] text-[16px] leading-relaxed text-text/90" data-testid="verdict-agreement">
        {r.agreement
          ? 'Both methods agree, so this is a stronger answer.'
          : 'The two methods disagree, so treat this as a maybe and check the leaf yourself.'}
        <Help term="methods" />
      </p>

      <div className="mt-5 flex flex-wrap items-center gap-x-4 gap-y-2" data-testid="confidence">
        <span className="text-[14px] text-muted">How sure:</span>
        <span className="flex gap-1" role="img" aria-label={`${LEVEL_WORD[level]} confidence: ${LEVEL_FILL[level]} of 3`}>
          {[1, 2, 3].map((i) => (
            <span key={i} className={`h-2 w-8 rounded-full ${i <= LEVEL_FILL[level] ? (level === 'low' ? 'bg-turmeric' : level === 'medium' ? 'bg-chlorosis' : 'bg-leaf') : 'bg-white/15'}`} />
          ))}
        </span>
        <span className="text-[15px] text-text" data-testid="confidence-level">
          {LEVEL_WORD[level]}
          <HelpConfidence level={level} evidence={evidence} high={levelRule.high_min_conf} low={levelRule.low_below_conf} />
        </span>
      </div>

      {r.ood.flagged && (
        <p className="mt-4 flex gap-2 text-[14px] leading-relaxed text-chlorosis">
          <AlertTriangle size={16} aria-hidden="true" className="mt-0.5 shrink-0" />
          <span>
            This photo looks unlike the photos the model learned from, so the answer is less reliable.
            <Help term="ood" />
          </span>
        </p>
      )}
      {unsureCrop && (
        <div className="mt-5 flex flex-wrap items-center gap-3 border-t hairline pt-5">
          <p className="w-full text-[14px] text-text/85">We could not tell if this is turmeric or black gram. Tell us and it will run again.</p>
          {(['turmeric', 'blackgram'] as const).map((k) => (
            <button key={k} type="button" onClick={() => onPickCrop(k)}
                    className="min-h-[44px] rounded-full border border-white/30 px-4 text-sm text-text transition-colors hover:bg-white/10">
              It’s {CROP_NAME[k].toLowerCase()}
            </button>
          ))}
        </div>
      )}

      <div className="mt-6 border-t hairline pt-5" data-testid="what-to-do">
        <h3 className="text-[15px] font-medium text-text">What should I do?</h3>
        <p className="mt-1.5 max-w-[62ch] text-[15px] leading-relaxed text-text/85">
          {level === 'low'
            ? 'Take another photo before deciding anything: one leaf, close up, in daylight, filling most of the frame. '
            : diseased
              ? 'Check the other leaves on the plant. If several look like this, show the plant to an agronomist or your local agricultural extension officer before treating it. '
              : 'Nothing to act on from this photo. Keep checking other leaves now and then; early or unusual symptoms can be missed. '}
          <span className="text-muted">Patra is a research prototype, not a diagnosis.</span>
        </p>
      </div>
    </section>
  );
}

function HelpConfidence({ level, evidence, high, low }: { level: Level; evidence?: string; high: number; low: number }) {
  const rule = level === 'high'
    ? `High: the neural network is at least ${Math.round(high * 100)}% sure, both methods agree and the photo looks normal.`
    : level === 'medium'
      ? `Medium: the neural network is fairly sure (${Math.round(low * 100)}% or more) but something holds it back: the methods disagree or the photo looks unusual, or it is under ${Math.round(high * 100)}% sure.`
      : `Low: the neural network is under ${Math.round(low * 100)}% sure, or the two methods disagree, or the crop is unclear. Treat it as a guess.`;
  return <Help term="confidence" extra={[rule, evidence].filter(Boolean).join(' ')} />;
}
