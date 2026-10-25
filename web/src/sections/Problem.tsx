import { useEffect, useRef } from 'react';
import Counter from '@/components/Counter';
import { citations } from '@/content/citations';
import { useReducedMotion } from '@/lib/motion';
import { ScrollTrigger } from '@/lib/smooth';

const SENTENCE =
  'Diagnosis still happens by eye — subjective, slow, and dependent on experts who are rarely in the field.';

export default function Problem() {
  const ref = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();
  const words = SENTENCE.split(' ');
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (reduced) {
      el.style.setProperty('--p', '1');
      return;
    }
    const st = ScrollTrigger.create({
      trigger: el,
      start: 'top top',
      end: 'bottom bottom',
      scrub: true,
      onUpdate: (s) => el.style.setProperty('--p', s.progress.toFixed(4)),
    });
    return () => st.kill();
  }, [reduced]);

  return (
    <section id="problem" aria-labelledby="problem-title" className="relative bg-ink">
      <div ref={ref} style={{ height: reduced ? 'auto' : '150vh', ['--p' as string]: 0 }}>
        <div className={`${reduced ? '' : 'sticky top-0'} flex min-h-[100dvh] items-center`}>
          <div className="container-grid">
            <p className="caption mb-8 text-muted">02 / Problem</p>
            <h2 id="problem-title" className="max-w-[1100px] text-[34px] font-normal leading-[1.08] tracking-[-0.04em] sm:text-5xl lg:text-[64px]">
              {words.map((w, i) => (
                <span key={i} style={{ opacity: `clamp(0.15, calc(var(--p) * ${words.length + 4} - ${i}), 1)` }}>
                  {w}{' '}
                </span>
              ))}
            </h2>
          </div>
        </div>
      </div>
      <div className="container-grid pb-[120px]">
        <dl className="grid grid-cols-1 gap-x-6 gap-y-10 border-t hairline pt-10 sm:grid-cols-2 lg:grid-cols-4">
          {citations.map((c, i) => (
            <div key={c.id} className="flex flex-col gap-3">
              <dt className="order-2 max-w-[260px] text-[15px] leading-snug text-muted">
                {c.text}
                <a href={`#cite-${i + 1}`} className="mono ml-1 align-super text-[11px] text-turmeric hover:underline"
                   aria-label={`Source ${i + 1}`}>
                  [{i + 1}]
                </a>
              </dt>
              <dd className="order-1 mono text-[40px] font-medium leading-none tracking-[-0.02em] text-text lg:text-[48px]">
                <Counter value={c.value} />
              </dd>
            </div>
          ))}
        </dl>
        <ol className="mt-14 grid gap-2 text-[12px] leading-relaxed text-muted lg:grid-cols-2 lg:gap-x-10">
          {citations.map((c, i) => (
            <li key={c.id} id={`cite-${i + 1}`} className="flex gap-2">
              <span className="mono text-turmeric">[{i + 1}]</span>
              <span>
                {c.source_url ? (
                  <a href={c.source_url} target="_blank" rel="noreferrer" className="underline decoration-white/20 underline-offset-2 hover:decoration-white/60">
                    {c.source_title}
                  </a>
                ) : (
                  c.source_title
                )}
                . {c.note}
              </span>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
