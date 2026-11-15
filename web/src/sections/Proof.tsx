import Counter from '@/components/Counter';
import { Reveal, SplitHeading } from '@/components/fx/Reveal';
import Spotlight from '@/components/fx/Spotlight';
import { fix, int, pct } from '@/lib/format';
import { useInView } from '@/lib/motion';
import type { FieldSet, Site } from '@/lib/site';

const PITCH = 11;
const R = 3.6;

/** One block of the unit chart: `n` dots, the wrong answers last and marked. */
function Dots({ n, right, cols, label, word }: { n: number; right: number; cols: number; label: string; word: string }) {
  const rows = Math.ceil(n / cols);
  const w = cols * PITCH;
  const h = rows * PITCH;
  return (
    <figure className="shrink-0">
      <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} role="img" aria-label={label} className="block h-auto max-w-full">
        {Array.from({ length: n }, (_, i) => {
          const wrong = i >= right;
          return (
            <circle key={i} className="dot-pop" cx={(i % cols) * PITCH + PITCH / 2} cy={Math.floor(i / cols) * PITCH + PITCH / 2} r={R}
                    fill={wrong ? 'var(--cta)' : 'var(--ink)'} fillOpacity={wrong ? 1 : 0.82}
                    style={{ ['--dot-d' as string]: `${Math.floor(i / cols) * 22 + (i % cols) * 5}ms` }} />
          );
        })}
      </svg>
      <figcaption className="mt-3">
        <span className="block font-playfair text-[26px] italic leading-none">{word}</span>
        <span className="mono mt-1.5 block text-[12px] text-ink/70">right {int(right)} of {int(n)}</span>
      </figcaption>
    </figure>
  );
}

const SETS: { key: FieldSet; name: string }[] = [
  { key: 'p4_test', name: 'Held-out field photos' },
  { key: 'wild_ext', name: 'Extended field set' },
  { key: 'wild', name: 'Original field set' },
  { key: 'test', name: 'Clean test' },
];

/**
 * The evidence, drawn (paper band, straight after the results tables): what "High / Medium / Low" were worth on
 * held-out photos, what the safety checks do to a hundred field leaves, and the deployed model against the one
 * it replaced. Every number is read from site.json.
 */
export default function Proof({ site }: { site: Site }) {
  const lv = site.confidence_levels;
  const f = site.field_model;
  const cm = site.confidence_model; // the newest step, when there is one: its numbers are the deployed model's
  const trust = cm?.trust.new ?? f.trust.new;
  const before = cm?.trust.old ?? f.trust.old;
  const [dotsRef, dotsIn] = useInView<HTMLDivElement>({ threshold: 0.25 });
  const [waffleRef, waffleIn] = useInView<HTMLDivElement>({ threshold: 0.4 });
  const [bellRef, bellIn] = useInView<HTMLDivElement>({ threshold: 0.4 });

  // a hundred held-out field leaves, rounded to whole leaves
  const answered = Math.round(trust.answered_share_labelled * 100);
  const rightOnes = Math.round(answered * trust.end_to_end_accuracy_answered);

  // shared scale of the before/after chart
  const all = SETS.flatMap(({ key }) => [f.sets[key].old.macro_f1, f.sets[key].new.macro_f1, (cm ?? f).sets[key].new.ci95[0],
    ...(cm ? [cm.sets[key].new.macro_f1] : [])]);
  const lo = Math.floor((Math.min(...all) - 0.005) * 20) / 20;
  const x = (v: number) => ((Math.min(1, Math.max(lo, v)) - lo) / (1 - lo)) * 100;
  const ticks = Array.from({ length: Math.round((1 - lo) / 0.05) + 1 }, (_, i) => lo + i * 0.05);

  return (
    <section id="proof" aria-labelledby="proof-title" className="bg-paper pb-20 text-ink lg:pb-[140px]">
      <div className="container-grid">
        <div className="border-t border-ink/15 pt-12 lg:pt-16">
          <Reveal as="p" className="caption mb-6 text-ink/70">The evidence, drawn</Reveal>
          <SplitHeading id="proof-title" text="How sure is sure?" italic="Every dot is a photo the model never trained on."
                        className="max-w-[980px] text-[34px] font-normal leading-[1.08] tracking-[-0.04em] sm:text-5xl lg:text-[60px]" />

          <div className="mt-12 grid gap-10 lg:grid-cols-12">
            <Reveal className="lg:col-span-4">
              <p className="text-[17px] leading-relaxed">
                The result page answers in words, not percentages: High, Medium or Low. These are {int(lv.n)} held-back
                photos ({lv.set}), sorted by the word the system used and marked where it was wrong.
              </p>
              <p className="mono mt-6 flex flex-wrap gap-x-5 gap-y-2 text-[12px] text-ink/70">
                <span className="flex items-center gap-2"><span aria-hidden="true" className="h-2.5 w-2.5 rounded-full bg-ink/80" /> right</span>
                <span className="flex items-center gap-2"><span aria-hidden="true" className="h-2.5 w-2.5 rounded-full bg-cta" /> wrong</span>
              </p>
              <p className="mt-6 text-[15px] leading-relaxed text-ink/70">
                A “High” answer needs the neural network to be at least {pct(lv.high_min_conf, 0)} sure, both methods to agree and
                the photo to look ordinary. Below {pct(lv.low_below_conf, 0)}, the page says “Low” and asks for another photo.
              </p>
            </Reveal>
            <div ref={dotsRef} className={`flex flex-wrap items-end gap-x-10 gap-y-8 lg:col-span-8 lg:justify-end ${dotsIn ? 'dots-in' : ''}`}
                 data-testid="proof-dots">
              <Dots n={lv.high.n} right={lv.high.right} cols={31} word="High"
                    label={`High confidence: right ${lv.high.right} of ${lv.high.n} times.`} />
              <Dots n={lv.medium.n} right={lv.medium.right} cols={6} word="Medium"
                    label={`Medium confidence: right ${lv.medium.right} of ${lv.medium.n} times.`} />
              <Dots n={lv.low.n} right={lv.low.right} cols={Math.max(1, Math.min(5, lv.low.n))} word="Low"
                    label={`Low confidence: right ${lv.low.right} of ${lv.low.n} times.`} />
            </div>
          </div>

          <div className="mt-16 grid gap-6 lg:grid-cols-12">
            {/* a hundred field leaves through the safety checks */}
            <Spotlight paper className="rounded-[10px] border border-ink/15 p-6 sm:p-8 lg:col-span-5">
              <div ref={waffleRef} className={waffleIn ? 'dots-in' : ''}>
                <p className="caption text-[11px] text-ink/70">Of every hundred field leaves</p>
                <div className="mt-5 grid w-full max-w-[300px] grid-cols-10 gap-[5px]" role="img"
                     aria-label={`Of 100 held-out field leaves, about ${answered} get an answer and ${100 - answered} are declined.`}>
                  {Array.from({ length: 100 }, (_, i) => (
                    <span key={i} className={`dot-pop aspect-square rounded-[2px] ${
                      i < rightOnes ? 'bg-ink/85' : i < answered ? 'bg-cta' : 'border border-ink/40'}`}
                          style={{ ['--dot-d' as string]: `${Math.floor(i / 10) * 40 + (i % 10) * 12}ms` }} />
                  ))}
                </div>
                <p className="mt-6 text-[17px] leading-relaxed">
                  <span className="mono">{pct(trust.answered_share_labelled, 0)}</span> get an answer, and{' '}
                  <span className="mono">{pct(trust.end_to_end_accuracy_answered, 0)}</span> of those answers are right. The rest
                  are declined with a reason instead of a guess.
                  {cm && <> One step earlier it answered <span className="mono">{pct(before.answered_share_labelled, 0)}</span>.</>}
                </p>
                <p className="mono mt-4 flex flex-wrap gap-x-5 gap-y-2 text-[12px] text-ink/70">
                  <span className="flex items-center gap-2"><span aria-hidden="true" className="h-2.5 w-2.5 rounded-[2px] bg-ink/85" /> answered, right</span>
                  {rightOnes < answered && (
                    <span className="flex items-center gap-2"><span aria-hidden="true" className="h-2.5 w-2.5 rounded-[2px] bg-cta" /> answered, wrong</span>
                  )}
                  <span className="flex items-center gap-2"><span aria-hidden="true" className="h-2.5 w-2.5 rounded-[2px] border border-ink/50" /> declined</span>
                </p>
                <p className="mt-4 text-[13px] leading-relaxed text-ink/70">
                  Held-out field photos ({int(f.sets.p4_test.n)} labelled leaves), rounded to whole leaves. Leaves of other plants are a
                  separate test: {pct(trust.other_species_refused, 0)} of {int(trust.n_other)} were turned away.
                </p>
              </div>
            </Spotlight>

            {/* previous model -> deployed model */}
            <Spotlight paper className="rounded-[10px] border border-ink/15 p-6 sm:p-8 lg:col-span-7">
              <div ref={bellRef} data-testid="proof-bells">
                <div className="flex flex-wrap items-baseline justify-between gap-3">
                  <p className="caption text-[11px] text-ink/70">{cm ? 'Three versions of the model · macro-F1' : 'Before and after field training · macro-F1'}</p>
                  <p className="mono flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-ink/70" aria-hidden="true">
                    <span className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full border border-ink" /> {cm ? 'lab-trained' : 'before'}</span>
                    {cm && <span className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full border-2 border-ink bg-ink/40" /> field-trained</span>}
                    <span className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-ink" /> now</span>
                  </p>
                </div>
                <ul className="mt-6 grid gap-5">
                  {SETS.map(({ key, name }, i) => {
                    const r = f.sets[key];
                    const now = cm ? cm.sets[key].new : r.new;
                    const a = x(r.old.macro_f1);
                    const mid = cm ? x(r.new.macro_f1) : null;
                    const b = x(now.macro_f1);
                    const up = b >= a;
                    return (
                      <li key={key}>
                        <div className="flex items-baseline justify-between gap-3">
                          <span className="text-[15px]">{name}</span>
                          <span className="mono text-[13px] text-ink/70">
                            {fix(r.old.macro_f1)} → {cm && <>{fix(r.new.macro_f1)} → </>}<span className="text-ink">{fix(now.macro_f1)}</span>
                          </span>
                        </div>
                        <div className="relative mt-2 h-5" aria-hidden="true">
                          <span className="absolute inset-x-0 top-1/2 h-px bg-ink/15" />
                          {/* 95% interval of the deployed model */}
                          <span className="absolute top-1/2 h-[7px] -translate-y-1/2 rounded-full bg-ink/10"
                                style={{ left: `${x(now.ci95[0])}%`, width: `${x(now.ci95[1]) - x(now.ci95[0])}%` }} />
                          <span className={`absolute top-1/2 h-[2px] -translate-y-1/2 transition-transform duration-[1200ms] ease-out ${up ? 'origin-left bg-ink' : 'origin-right bg-cta'}`}
                                style={{ left: `${Math.min(a, b)}%`, width: `${Math.abs(b - a)}%`, transform: `scaleX(${bellIn ? 1 : 0})`, transitionDelay: `${i * 120}ms` }} />
                          <span className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-ink bg-paper"
                                style={{ left: `${a}%` }} />
                          {mid != null && (
                            <span className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-ink bg-[#8f8d88] transition-opacity duration-700"
                                  style={{ left: `${mid}%`, opacity: bellIn ? 1 : 0, transitionDelay: `${i * 120 + 500}ms` }} />
                          )}
                          <span className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-ink transition-[left] duration-[1200ms] ease-out"
                                style={{ left: `${bellIn ? b : a}%`, transitionDelay: `${i * 120}ms` }} />
                        </div>
                      </li>
                    );
                  })}
                </ul>
                <div className="relative mt-3 h-5" aria-hidden="true">
                  {ticks.map((v) => (
                    <span key={v} className="mono absolute -translate-x-1/2 text-[10px] text-ink/70" style={{ left: `${x(v)}%` }}>{fix(v, 2)}</span>
                  ))}
                </div>
                <p className="mt-4 text-[13px] leading-relaxed text-ink/70">
                  The pale band is the deployed model’s interval. {cm
                    ? <>First came {int(f.train_field_leaves)} more field leaves in training, then an average of {cm.members} equally good models in place of one.</>
                    : <>What separates the two models is {int(f.train_field_leaves)} more field leaves in training.</>}{' '}
                  The table for the first step, with every interval, is above.
                </p>
              </div>
            </Spotlight>
          </div>
          {cm && <LastStep cm={cm} />}
        </div>
      </div>
    </section>
  );
}

/** Phase 5 in three figures: more leaves answered, how often those answers are right, and the refusal line that moved. */
function LastStep({ cm }: { cm: NonNullable<Site['confidence_model']> }) {
  const o = cm.trust.old;
  const n = cm.trust.new;
  const held = cm.sets.p4_test;
  return (
    <div className="mt-16 border-t border-ink/15 pt-12" data-testid="proof-last-step">
      <Reveal as="p" className="caption mb-4 text-[11px] text-ink/70">The newest step · fewer needless refusals</Reveal>
      <SplitHeading as="h3" text="It used to say “not sure” about leaves it had right." italic="Now it answers them."
                    className="max-w-[900px] text-[28px] font-normal leading-[1.12] tracking-[-0.03em] sm:text-[40px]" />
      <dl className="mt-10 grid gap-8 sm:grid-cols-3">
        <Reveal>
          <dt className="caption text-[11px] text-ink/70">Field leaves answered</dt>
          <dd className="mono mt-2 text-[44px] leading-none tracking-[-0.02em] lg:text-[60px]">
            <Counter value={pct(n.answered_share_labelled, 0)} />
          </dd>
          <dd className="mt-3 max-w-[300px] text-[15px] leading-relaxed text-ink/70">
            up from {pct(o.answered_share_labelled, 0)}: {int(n.answered_labelled)} of {int(n.n_labelled)} held-out leaves, where it was{' '}
            {int(o.answered_labelled)}.
          </dd>
        </Reveal>
        <Reveal delay={100}>
          <dt className="caption text-[11px] text-ink/70">Right when it answers</dt>
          <dd className="mono mt-2 text-[44px] leading-none tracking-[-0.02em] lg:text-[60px]">
            <Counter value={pct(n.end_to_end_accuracy_answered, 0)} />
          </dd>
          <dd className="mt-3 max-w-[300px] text-[15px] leading-relaxed text-ink/70">
            {int(n.errors_answered)} wrong among those {int(n.answered_labelled)} answers. Before, {int(o.errors_answered)} wrong among{' '}
            {int(o.answered_labelled)}: answering more leaves costs a few mistakes, and that trade was set in advance.
          </dd>
        </Reveal>
        <Reveal delay={200}>
          <dt className="caption text-[11px] text-ink/70">Declines below</dt>
          <dd className="mono mt-2 text-[44px] leading-none tracking-[-0.02em] lg:text-[60px]">
            <Counter value={pct(n.disease_min_confidence, 0)} />
          </dd>
          <dd className="mt-3 max-w-[300px] text-[15px] leading-relaxed text-ink/70">
            confidence, where the line used to sit at {pct(o.disease_min_confidence, 0)}. It was moved as far as validation
            leaves allowed while keeping wrong answers under {pct(cm.max_error_target, 0)}.
          </dd>
        </Reveal>
      </dl>
      <Reveal as="p" delay={120} className="mt-10 max-w-[920px] border-l-2 border-ink/20 pl-4 text-[15px] leading-relaxed text-ink/70">
        How: the answer is now the average of {cm.members} models built on different vision backbones, chosen among{' '}
        {cm.candidates_compared} options on validation leaves only, with no retraining. On the held-out field leaves macro-F1 went
        from {fix(held.old.macro_f1)} to {fix(held.new.macro_f1)}, and the confidence it puts on right answers from{' '}
        {pct(cm.confidence_p4_test.old.mean_confidence_right, 1)} to {pct(cm.confidence_p4_test.new.mean_confidence_right, 1)}.
        {cm.second_read_of_p4_test && ' These held-out leaves had been scored once before, for the previous step, so this second look is weaker evidence than a first one.'}
      </Reveal>
    </div>
  );
}
