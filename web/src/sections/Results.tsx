import { useState } from 'react';
import { fix, int, METHOD_NAME, pct } from '@/lib/format';
import { useInView } from '@/lib/motion';
import type { Method, Site } from '@/lib/site';

// the last flag marks columns kept on narrow screens
const COLS = [
  ['accuracy', 'Accuracy', true],
  ['precision', 'Precision', false],
  ['recall', 'Recall', false],
  ['f1', 'F1', false],
  ['macro_f1', 'Macro-F1', true],
] as const;
const cell = (keep: boolean) => (keep ? '' : 'hidden md:table-cell');
const METHODS: Method[] = ['rule_based', 'svm', 'cnn'];

/** Short, honest takeaway composed from the exported numbers only. */
function takeaway(site: Site, view: 'clean' | 'wild'): string[] {
  const m = site.comparison[view];
  const order = [...METHODS].sort((a, b) => m[b].macro_f1 - m[a].macro_f1);
  const [best, second] = order as [Method, Method, Method];
  const out = [
    `${METHOD_NAME[best]} leads on ${view === 'clean' ? 'the clean test set' : 'field photos from other sources'} ` +
      `(macro-F1 ${fix(m[best].macro_f1)}), ahead of ${METHOD_NAME[second]} (${fix(m[second].macro_f1)}).`,
  ];
  if (view === 'clean') {
    const p = site.comparison.mcnemar_p;
    out.push(
      p > 0.05
        ? `The SVM–CNN difference is not statistically significant (McNemar p = ${fix(p)}).`
        : `The SVM–CNN difference is statistically significant (McNemar p = ${fix(p)}).`,
    );
  } else {
    const drops = METHODS.map((k) => site.comparison.clean[k].macro_f1 - m[k].macro_f1);
    if (drops.every((d) => d > 0)) out.push('Every method loses accuracy on photos it was not trained on.');
  }
  out.push(
    `Image processing gives the measurement behind each call; on CPU it takes ${fix(site.comparison.clean_ms.svm, 0)} ms per image ` +
      `against ${fix(site.comparison.clean_ms.cnn, 0)} ms for the calibrated CNN pipeline.`,
  );
  return out;
}

export default function Results({ site }: { site: Site }) {
  const [view, setView] = useState<'clean' | 'wild'>('clean');
  const [ref, inView] = useInView<HTMLDivElement>({ threshold: 0.25 });
  const m = site.comparison[view];
  const lk = site.leakage;
  const n = view === 'clean' ? site.comparison.test_n : site.dataset.wild_test_labelled;

  return (
    <section id="results" aria-labelledby="results-title" className="section-pad bg-paper text-ink">
      <div className="container-grid">
        <p className="caption mb-6 text-ink/55">06 / Results</p>
        <h2 id="results-title" className="max-w-[1080px] font-playfair text-[34px] italic leading-[1.12] tracking-[-0.03em] sm:text-5xl lg:text-[60px]">
          Can conventional image processing identify disease in turmeric and black gram leaves, and how does it compare with a CNN?
        </h2>

        <div className="mt-16 flex flex-wrap items-center justify-between gap-4">
          <div role="group" aria-label="Test set" className="inline-flex rounded-full border border-ink/20 p-1">
            {(['clean', 'wild'] as const).map((v) => (
              <button key={v} type="button" aria-pressed={view === v} onClick={() => setView(v)}
                      className={`mono rounded-full px-4 py-1.5 text-[11px] uppercase tracking-[0.08em] ${view === v ? 'bg-ink text-paper' : 'text-ink/65 hover:text-ink'}`}>
                {v === 'clean' ? 'Clean test' : 'Field photos (wild)'}
              </button>
            ))}
          </div>
          <p className="mono text-[12px] text-ink/60">n = {int(n)} images · leakage-free split</p>
        </div>

        <div ref={ref} className="mt-6 overflow-x-auto">
          <table className="w-full border-collapse text-left md:min-w-[720px]">
            <caption className="sr-only">Healthy vs diseased performance per method</caption>
            <thead>
              <tr className="border-b border-ink/20">
                <th scope="col" className="caption py-3 pr-4 text-[11px] font-normal text-ink/55">Method</th>
                {COLS.map(([, l, keep]) => (
                  <th key={l} scope="col" className={`caption py-3 pr-4 text-[11px] font-normal text-ink/55 ${cell(keep)}`}>{l}</th>
                ))}
                <th scope="col" className="caption py-3 text-[11px] font-normal text-ink/55">ms / image<span className="hidden md:inline"> (CPU)</span></th>
              </tr>
            </thead>
            <tbody>
              {METHODS.map((k, r) => (
                <tr key={k} className="border-b border-ink/10">
                  <th scope="row" className="py-5 pr-4 text-[15px] font-normal md:text-[17px]">{METHOD_NAME[k]}</th>
                  {COLS.map(([c, , keep], j) => (
                    <td key={c} className={`py-5 pr-4 align-top ${cell(keep)}`}>
                      <span className="mono text-[17px]">{fix(m[k][c])}</span>
                      <span className="mt-2 block h-[3px] w-full max-w-[120px] bg-ink/10">
                        <span className="block h-full origin-left bg-ink transition-transform duration-1000 ease-out"
                              style={{ transform: `scaleX(${inView ? m[k][c] : 0})`, transitionDelay: `${(r * 5 + j) * 40}ms` }} />
                      </span>
                    </td>
                  ))}
                  <td className="mono py-5 align-top text-[17px]">{fix(site.comparison.clean_ms[k], 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-16 grid gap-10 lg:grid-cols-12">
          <div className="lg:col-span-7">
            <p className="caption mb-4 text-[11px] text-ink/55">What the numbers say</p>
            <ul className="grid gap-3 text-[17px] leading-relaxed">
              {takeaway(site, view).map((t) => (
                <li key={t} className="border-l-2 border-ink/20 pl-4">{t}</li>
              ))}
            </ul>
          </div>
          <aside className="rounded-[6px] border border-ink/15 p-6 lg:col-span-5" aria-labelledby="leak-title">
            <p id="leak-title" className="caption text-[11px] text-ink/55">Leakage ablation · SVM accuracy</p>
            <p className="mt-4 text-[17px] leading-relaxed">
              Augmented copies split at random: <span className="mono">{pct(lk.augment_then_split.svm.accuracy)}</span>
              {' → '}leakage-free: <span className="mono">{pct(lk.clean.svm.accuracy)}</span>.
            </p>
            <p className="mt-2 text-[15px] leading-relaxed text-ink/70">
              In the naive split <span className="mono">{pct(lk.augment_then_split.svm.dup_share, 0)}</span> of test images have a
              near-copy in training. Original photos split at random: <span className="mono">{pct(lk.naive_random.svm.accuracy)}</span>.
            </p>
          </aside>
        </div>
      </div>
    </section>
  );
}
