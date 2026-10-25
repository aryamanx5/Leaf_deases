import { useEffect, useRef, useState } from 'react';
import Picture from '@/components/Picture';
import { CROP_NAME, int } from '@/lib/format';
import type { Site } from '@/lib/site';
import { useReducedMotion } from '@/lib/motion';
import { ScrollTrigger } from '@/lib/smooth';

const INFO = {
  turmeric: {
    botanical: 'Curcuma longa',
    traits: ['long lanceolate blade', 'strong midrib', 'parallel venation'],
    diseases: ['leaf blotch (Taphrina maculans)', 'leaf spot (Colletotrichum)', 'drying / scorch'],
  },
  blackgram: {
    botanical: 'Vigna mungo',
    traits: ['trifoliate leaves', 'ovate leaflets', 'net venation, hairy surface'],
    diseases: ['yellow mosaic (MYMV)', 'anthracnose', 'powdery mildew', 'leaf crinkle'],
  },
} as const;

function Panel({ site, crop, index }: { site: Site; crop: 'turmeric' | 'blackgram'; index: number }) {
  const [state, setState] = useState<'healthy' | 'diseased'>('diseased');
  const ref = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();
  const info = INFO[crop];
  const imgs = site.crops[crop];
  const n = site.dataset.by_crop_label;

  useEffect(() => {
    const el = ref.current;
    if (!el || reduced) return;
    const st = ScrollTrigger.create({
      trigger: el,
      start: 'top 85%',
      end: 'top 25%',
      scrub: true,
      onUpdate: (s) => el.style.setProperty('--r', s.progress.toFixed(4)),
    });
    return () => st.kill();
  }, [reduced]);

  return (
    <article className="flex flex-col gap-6" aria-labelledby={`crop-${crop}`}>
      <div
        ref={ref}
        className="group relative aspect-square max-h-[78vh] overflow-hidden rounded-[6px] bg-soil"
        style={{ ['--r' as string]: reduced ? 1 : 0, clipPath: `inset(calc((1 - var(--r)) * 100%) 0 0 0)` }}
      >
        {(['healthy', 'diseased'] as const).map((k) => (
          <Picture
            key={k}
            src={imgs[k].img}
            alt={`${CROP_NAME[crop]} leaf, ${k === 'healthy' ? 'healthy' : imgs[k].orig_class.replace('_', ' ')} (dataset test image)`}
            sizes="(max-width: 1024px) 92vw, 46vw"
            className="absolute inset-0 h-full w-full object-cover transition-[opacity,transform] duration-700 ease-out group-hover:scale-[1.04] group-focus-within:scale-[1.04]"
            style={{ opacity: state === k ? 1 : 0 }}
          />
        ))}
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-1/3 bg-gradient-to-t from-ink/80 to-transparent" />
        <div role="group" aria-label="Show leaf state" className="absolute bottom-4 left-4 flex rounded-full border border-white/20 bg-ink/90 p-1">
          {(['healthy', 'diseased'] as const).map((k) => (
            <button
              key={k}
              type="button"
              aria-pressed={state === k}
              onClick={() => setState(k)}
              className={`mono rounded-full px-3 py-1 text-[11px] uppercase tracking-[0.08em] transition-colors ${
                state === k ? 'bg-paper text-ink' : 'text-white/75 hover:text-white'
              }`}
            >
              {k}
            </button>
          ))}
        </div>
        <span className="caption absolute right-4 top-4 rounded-full bg-ink/80 px-2.5 py-1 text-[11px] text-white/80">0{index + 1}</span>
      </div>
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-[1fr_1fr]">
        <div>
          <h3 id={`crop-${crop}`} className="text-[28px] font-normal tracking-[-0.03em]">
            {CROP_NAME[crop]}
          </h3>
          <p className="font-playfair text-lg italic text-muted">{info.botanical}</p>
          <p className="mono mt-3 text-[12px] text-muted">
            {int(n[`${crop}_healthy`])} healthy · {int(n[`${crop}_diseased`])} diseased photos
          </p>
        </div>
        <dl className="mono grid gap-3 text-[12px] leading-relaxed">
          <div>
            <dt className="caption text-[11px] text-leaf">Leaf</dt>
            <dd className="text-text/85">{info.traits.join(' · ')}</dd>
          </div>
          <div>
            <dt className="caption text-[11px] text-turmeric">Main diseases</dt>
            <dd className="text-text/85">{info.diseases.join(' · ')}</dd>
          </div>
        </dl>
      </div>
    </article>
  );
}

export default function Crops({ site }: { site: Site }) {
  return (
    <section id="crops" aria-labelledby="crops-title" className="section-pad bg-ink">
      <div className="container-grid">
        <p className="caption mb-6 text-muted">03 / Two crops</p>
        <h2 id="crops-title" className="mb-14 max-w-[820px] text-4xl font-normal leading-[1.05] tracking-[-0.04em] sm:text-5xl lg:text-[56px]">
          Two leaves that look nothing alike, <span className="font-playfair italic">read by one system.</span>
        </h2>
        <div className="grid grid-cols-1 gap-12 lg:grid-cols-2 lg:gap-6">
          <Panel site={site} crop="turmeric" index={0} />
          <Panel site={site} crop="blackgram" index={1} />
        </div>
      </div>
    </section>
  );
}
