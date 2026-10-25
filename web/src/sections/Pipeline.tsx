import { useEffect, useRef, useState } from 'react';
import Counter from '@/components/Counter';
import Picture from '@/components/Picture';
import { CROP_NAME, fix, int, pctRaw } from '@/lib/format';
import { useReducedMotion } from '@/lib/motion';
import type { Showcase, Site, StepKey } from '@/lib/site';
import { ScrollTrigger } from '@/lib/smooth';

type Step = { key: StepKey; title: string; body: string; param: (s: Showcase) => string; alt: string };

const STEPS: Step[] = [
  {
    key: 'denoised',
    title: 'Noise reduction',
    body: 'A median filter removes sensor speckle, a light Gaussian smooths what is left. Nothing is measured on raw pixels.',
    param: (s) => `median ${s.params.denoise.median_ksize}×${s.params.denoise.median_ksize} · Gaussian ${s.params.denoise.gaussian_ksize}×${s.params.denoise.gaussian_ksize}`,
    alt: 'The leaf after median and Gaussian filtering.',
  },
  {
    key: 'colour_space',
    title: 'RGB → HSV / Lab',
    body: 'Colour is separated from brightness. On the a* axis green tissue stands apart from paper, soil and lesions.',
    param: (s) => `a* from ${fix(s.params.colour_space.a_star_min, 0)} to ${fix(s.params.colour_space.a_star_max, 0)}`,
    alt: 'The Lab a* channel: green tissue bright, background and lesions dark.',
  },
  {
    key: 'segmentation',
    title: 'Leaf segmentation',
    body: 'A colour model of the background is built from the photo’s border; every pixel gets its distance from it.',
    param: (s) => `border background share ${fix(s.params.segmentation.border_background_share, 2)}`,
    alt: 'Colour distance to the background model: the leaf glows, the paper is dark.',
  },
  {
    key: 'threshold',
    title: 'Thresholding',
    body: 'Otsu’s method picks the single cut that best separates the two populations of distances.',
    param: (s) => (s.params.threshold.otsu_distance == null ? 'GrabCut (field scene)' : `Otsu t = ${fix(s.params.threshold.otsu_distance, 1)}`),
    alt: 'Binary mask after thresholding the colour distance.',
  },
  {
    key: 'morphology',
    title: 'Morphology',
    body: 'Opening removes specks, closing seals the outline; the dominant component is kept and its holes filled.',
    param: (s) => `open ${s.params.morphology.open_ksize} · close ${s.params.morphology.close_ksize} · leaf ${pctRaw(s.params.morphology.leaf_coverage * 100, 0)} of frame`,
    alt: 'The cleaned leaf mask.',
  },
  {
    key: 'lesions',
    title: 'Disease region detection',
    body: 'Yellow, brown, dark and pale tissue is found relative to this leaf’s own green, then split into connected regions.',
    param: (s) => `${int(s.params.lesions.count)} regions · min area ${int(s.params.lesions.min_area_px)} px`,
    alt: 'Detected abnormal regions coloured by type, leaf outline in green.',
  },
  {
    key: 'lesions',
    title: 'Features',
    body: 'Colour, texture (GLCM) and shape are measured on the regions. These numbers, not the picture, feed the decision.',
    param: () => 'colour · texture · shape',
    alt: 'Detected abnormal regions with the measured features.',
  },
];

function Readouts({ s }: { s: Showcase }) {
  const r = s.readouts;
  const items: [string, string][] = [
    ['affected area', pctRaw(r.affected_pct)],
    ['lesion count', int(r.lesion_count)],
    ['abnormal colour', pctRaw(r.abn_colour_pct)],
    ['GLCM contrast', fix(r.glcm_contrast, 2)],
    ['GLCM energy', fix(r.glcm_energy, 3)],
    ['GLCM homogeneity', fix(r.glcm_homogeneity, 3)],
    ['circularity (mean)', fix(r.lesion_circularity_mean, 2)],
  ];
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-3">
      {items.map(([k, v]) => (
        <div key={k}>
          <dt className="caption text-[10px] text-muted">{k}</dt>
          <dd className="mono text-xl text-text">
            <Counter value={v} duration={900} />
          </dd>
        </div>
      ))}
    </dl>
  );
}

function CropToggle({ crop, setCrop }: { crop: 'turmeric' | 'blackgram'; setCrop: (c: 'turmeric' | 'blackgram') => void }) {
  return (
    <div role="group" aria-label="Showcase leaf" className="inline-flex rounded-full border hairline p-1">
      {(['turmeric', 'blackgram'] as const).map((c) => (
        <button key={c} type="button" aria-pressed={crop === c} onClick={() => setCrop(c)}
                className={`mono rounded-full px-3 py-1 text-[11px] uppercase tracking-[0.08em] transition-colors ${
                  crop === c ? 'bg-paper text-ink' : 'text-muted hover:text-text'}`}>
          {CROP_NAME[c]}
        </button>
      ))}
    </div>
  );
}

export default function Pipeline({ site }: { site: Site }) {
  const reduced = useReducedMotion();
  const [crop, setCrop] = useState<'turmeric' | 'blackgram'>('turmeric');
  const [active, setActive] = useState(0);
  const [wide, setWide] = useState(() => typeof window !== 'undefined' && window.innerWidth >= 1024);
  const ref = useRef<HTMLDivElement>(null);
  const s = site.showcase[crop];
  const pinned = wide && !reduced;

  useEffect(() => {
    const on = () => setWide(window.innerWidth >= 1024);
    window.addEventListener('resize', on);
    return () => window.removeEventListener('resize', on);
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!el || !pinned) return;
    const st = ScrollTrigger.create({
      trigger: el,
      start: 'top top',
      end: 'bottom bottom',
      onUpdate: (self) => setActive(Math.min(STEPS.length - 1, Math.floor(self.progress * STEPS.length))),
    });
    return () => st.kill();
  }, [pinned]);

  const header = (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-6">
      <div>
        <p className="caption mb-4 text-muted">04 / Stage 1 · Image processing</p>
        <h2 className="max-w-[720px] text-4xl font-normal leading-[1.05] tracking-[-0.04em] lg:text-[44px]">
          We don’t compare a healthy photo to a diseased photo. <span className="font-playfair italic">We measure this leaf.</span>
        </h2>
      </div>
      <CropToggle crop={crop} setCrop={setCrop} />
    </div>
  );

  if (!pinned) {
    return (
      <section id="pipeline" aria-label="Stage 1: image processing" className="section-pad bg-ink">
        <div className="container-grid">
          {header}
          <ol className="grid gap-10">
            {STEPS.map((st, i) => (
              <li key={i} className="grid gap-4 border-t hairline pt-6 md:grid-cols-2">
                <div>
                  <p className="mono text-[12px] text-muted">{String(i + 1).padStart(2, '0')}</p>
                  <h3 className="mt-1 text-[28px] tracking-[-0.03em]">{st.title}</h3>
                  <p className="mt-2 max-w-[460px] text-[17px] leading-relaxed text-muted">{st.body}</p>
                  <p className="mono mt-3 text-[13px] text-leaf">{st.param(s)}</p>
                  {i === STEPS.length - 1 && <div className="mt-6"><Readouts s={s} /></div>}
                </div>
                <Picture src={s.media[st.key]} alt={st.alt} sizes="(max-width: 768px) 92vw, 46vw"
                         className="aspect-square w-full rounded-[6px] object-cover" />
              </li>
            ))}
          </ol>
        </div>
      </section>
    );
  }

  return (
    <section id="pipeline" aria-label="Stage 1: image processing" className="relative bg-ink">
      <div ref={ref} style={{ height: '500vh' }}>
        <div className="sticky top-0 flex h-[100dvh] items-center pt-16">
          <div className="container-grid">
            {header}
            <div className="grid grid-cols-12 gap-6">
              <ol className="col-span-5 flex flex-col">
                {STEPS.map((st, i) => {
                  const on = i === active;
                  return (
                    <li key={i} className={`border-t hairline py-2.5 transition-colors duration-500 ${on ? '' : 'text-muted'}`}>
                      <div className="flex items-baseline gap-4">
                        <span className={`mono text-[12px] ${on ? 'text-leaf' : ''}`}>{String(i + 1).padStart(2, '0')}</span>
                        <h3 className={`text-[19px] tracking-[-0.02em] ${on ? 'text-text' : ''}`}>{st.title}</h3>
                      </div>
                      <div className="grid transition-[grid-template-rows,opacity] duration-700 ease-out"
                           style={{ gridTemplateRows: on ? '1fr' : '0fr', opacity: on ? 1 : 0 }}>
                        <div className="overflow-hidden pl-9">
                          <p className="pt-2 text-[15px] leading-relaxed text-muted">{st.body}</p>
                          <p className="mono pt-2 text-[13px] text-leaf">{st.param(s)}</p>
                          {on && i === STEPS.length - 1 && <div className="pt-4"><Readouts s={s} /></div>}
                        </div>
                      </div>
                    </li>
                  );
                })}
              </ol>
              <div className="col-span-7">
                <figure className="relative mx-auto aspect-square max-h-[60vh] overflow-hidden rounded-[6px] border hairline bg-soil">
                  <Picture src={s.media.original} alt="The showcase leaf as photographed." sizes="40vw"
                           className="absolute inset-0 h-full w-full object-cover" />
                  {STEPS.map((st, i) => (
                    <Picture key={i} src={s.media[st.key]} alt={i === active ? st.alt : ''} sizes="40vw"
                             className="absolute inset-0 h-full w-full object-cover transition-opacity duration-700 ease-out"
                             style={{ opacity: i <= active && (i === active || STEPS[active]?.key === st.key) ? 1 : 0 }} />
                  ))}
                  <figcaption className="caption absolute bottom-3 left-3 rounded-full bg-ink/70 px-3 py-1 text-[10px] text-white/80">
                    {CROP_NAME[crop]} · test image · step {active + 1}/{STEPS.length}
                  </figcaption>
                </figure>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
