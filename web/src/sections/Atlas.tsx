import { AnimatePresence, motion } from 'motion/react';
import { useMemo, useState } from 'react';
import { Reveal, SplitHeading } from '@/components/fx/Reveal';
import { ATLAS, TISSUE_COLOUR, TISSUE_LABEL, type AtlasEntry } from '@/content/atlas';
import { CROP_NAME, int } from '@/lib/format';
import { EASE, useReducedMotion } from '@/lib/motion';
import type { Sample, Site } from '@/lib/site';

// where the three photos of a class rest: a loose fan, the first one on top
const FAN = [
  { x: 0, y: 0, r: -2.5, z: 3 },
  { x: 17, y: 6, r: 6, z: 2 },
  { x: -16, y: 8, r: -9, z: 1 },
];

function Photos({ entry, photos }: { entry: AtlasEntry; photos: Sample[] }) {
  const reduced = useReducedMotion();
  const [front, setFront] = useState(0);
  const order = photos.map((_, i) => (i - front + photos.length) % photos.length);
  const label = `${CROP_NAME[entry.crop]} leaf, ${entry.healthy ? 'healthy' : entry.name.toLowerCase()}`;
  return (
    <div className="relative mx-auto aspect-square w-[min(70vw,430px)]">
      {photos.map((s, i) => {
        const slot = FAN[order[i]!] ?? FAN[2]!;
        const top = order[i] === 0;
        return (
          <motion.button
            key={s.id}
            type="button"
            onClick={() => setFront(top ? (i + 1) % photos.length : i)}
            aria-label={top ? `${label}, photo ${i + 1} of ${photos.length}. Show the next photo` : `Bring photo ${i + 1} to the front`}
            className="absolute inset-[13%] overflow-hidden rounded-[10px] border border-white/15 bg-soil shadow-[0_24px_60px_rgba(0,0,0,0.5)]"
            style={{ zIndex: slot.z }}
            initial={reduced ? false : { opacity: 0, x: 0, y: 40, rotate: 0, scale: 0.92 }}
            animate={{ opacity: 1, x: `${slot.x}%`, y: `${slot.y}%`, rotate: slot.r, scale: top ? 1 : 0.94 }}
            exit={reduced ? { opacity: 0 } : { opacity: 0, y: -30, scale: 0.94 }}
            transition={reduced ? { duration: 0 } : { type: 'spring', stiffness: 170, damping: 22, delay: order[i]! * 0.05 }}
            whileHover={reduced ? undefined : { scale: top ? 1.02 : 0.97 }}
          >
            <img src={s.src} alt={top ? `${label} (test photo)` : ''} width={512} height={512} loading="lazy" decoding="async"
                 className="h-full w-full object-cover" />
          </motion.button>
        );
      })}
    </div>
  );
}

/**
 * Field guide: the eight classes of the two datasets, each with real test photos, its cause, what it looks like and
 * how the pipeline's measurements pick it up. Counts come from site.json; the descriptions are general pathology.
 */
export default function Atlas({ site }: { site: Site }) {
  const [active, setActive] = useState(1);
  const entry = ATLAS[active]!;
  const photos = useMemo(() => {
    const by: Record<string, Sample[]> = {};
    for (const e of ATLAS) {
      by[e.key] = site.samples.filter((s) => s.crop === e.crop && e.classes.includes(s.orig_class)).slice(0, 3);
    }
    return by;
  }, [site.samples]);
  const count = site.dataset.by_class[entry.key];

  return (
    <section id="atlas" aria-labelledby="atlas-title" className="section-pad bg-ink">
      <div className="container-grid">
        <Reveal as="p" className="caption mb-6 text-muted">Field guide · Eight classes</Reveal>
        <SplitHeading id="atlas-title" text="What the camera is asked to tell apart," italic="leaf by leaf."
                      className="mb-12 max-w-[900px] text-4xl font-normal leading-[1.05] tracking-[-0.04em] sm:text-5xl lg:mb-16 lg:text-[56px]" />

        <div className="grid gap-10 lg:grid-cols-12 lg:gap-8">
          {/* the index: every class, grouped by crop */}
          <Reveal className="min-w-0 lg:col-span-3">
            <div className="-mx-6 flex gap-6 overflow-x-auto px-6 pb-2 md:-mx-10 md:px-10 lg:mx-0 lg:block lg:overflow-visible lg:px-0 lg:pb-0">
              {(['turmeric', 'blackgram'] as const).map((crop) => (
                <div key={crop} role="group" aria-label={`${CROP_NAME[crop]} classes`} className="shrink-0 lg:mb-8">
                  <p className="caption mb-2 text-[11px] text-muted">{CROP_NAME[crop]}</p>
                  <ul className="flex gap-2 lg:block">
                    {ATLAS.map((e, i) => e.crop === crop && (
                      <li key={e.key} className="lg:border-t lg:hairline">
                        <button type="button" aria-pressed={i === active} onClick={() => setActive(i)}
                                className={`group flex min-h-[44px] w-full items-center gap-3 whitespace-nowrap rounded-full border px-4 text-left transition-colors duration-300 lg:rounded-none lg:border-0 lg:px-0 lg:py-3 ${
                                  i === active ? 'border-paper text-text' : 'border-white/15 text-muted hover:text-text'}`}>
                          <span aria-hidden="true" className={`hidden h-px bg-leaf transition-all duration-500 ease-out lg:block ${i === active ? 'w-8' : 'w-0 group-hover:w-3'}`} />
                          <span className={`text-[15px] tracking-[-0.02em] lg:text-[26px] ${i === active ? 'lg:font-playfair lg:italic' : ''}`}>{e.name}</span>
                          {e.healthy && <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-leaf" />}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </Reveal>

          {/* the photos of the chosen class */}
          <Reveal delay={80} className="min-w-0 lg:col-span-5">
            <div className="lg:min-h-[460px]">
              <AnimatePresence mode="popLayout" initial={false}>
                <motion.div key={entry.key} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>
                  {photos[entry.key]?.length ? <Photos entry={entry} photos={photos[entry.key]!} /> : null}
                </motion.div>
              </AnimatePresence>
            </div>
            <p className="mono mt-4 text-center text-[11px] text-muted">Real test photos · tap a photo to shuffle</p>
          </Reveal>

          {/* what it is and how it reads */}
          <Reveal delay={160} className="min-w-0 lg:col-span-4">
            <div className="lg:min-h-[460px]" aria-live="polite">
              <AnimatePresence mode="wait" initial={false}>
                <motion.div key={entry.key} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
                            transition={{ duration: 0.35, ease: EASE }}>
                  <p className="caption text-[11px]" style={{ color: entry.healthy ? 'var(--leaf)' : 'var(--turmeric)' }}>
                    {CROP_NAME[entry.crop]} · {entry.healthy ? 'healthy' : 'diseased'}
                  </p>
                  <h3 className="mt-2 font-playfair text-[40px] italic leading-none tracking-[-0.03em]">{entry.name}</h3>
                  <p className="mono mt-3 text-[12px] leading-relaxed text-muted">{entry.cause}</p>
                  <dl className="mt-6 grid gap-5">
                    <div>
                      <dt className="caption text-[11px] text-muted">What you see</dt>
                      <dd className="mt-1.5 text-[16px] leading-relaxed text-text/90">{entry.looks}</dd>
                    </div>
                    <div>
                      <dt className="caption text-[11px] text-muted">How it reads</dt>
                      <dd className="mt-1.5 text-[16px] leading-relaxed text-text/90">{entry.reads}</dd>
                    </div>
                    <div>
                      <dt className="caption text-[11px] text-muted">Colour signature</dt>
                      <dd className="mt-2 flex flex-wrap gap-2">
                        {entry.tissue.map((t) => (
                          <span key={t} className="mono inline-flex items-center gap-2 rounded-full border border-white/15 px-3 py-1 text-[11px] text-text/85">
                            <span aria-hidden="true" className="h-2 w-2 rounded-full" style={{ background: TISSUE_COLOUR[t] }} />
                            {TISSUE_LABEL[t]}
                          </span>
                        ))}
                      </dd>
                    </div>
                  </dl>
                  {count != null && (
                    <p className="mono mt-6 border-t hairline pt-4 text-[12px] text-muted">
                      {int(count)} photos of this class across the two datasets
                    </p>
                  )}
                </motion.div>
              </AnimatePresence>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
