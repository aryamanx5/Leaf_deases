import { useEffect, useRef, useState } from 'react';
import Picture from '@/components/Picture';
import { CROP_NAME, pct } from '@/lib/format';
import { useInView, useReducedMotion } from '@/lib/motion';
import type { Showcase, Site } from '@/lib/site';
import { ScrollTrigger } from '@/lib/smooth';

function Tiles({ urls, cols, show, label }: { urls: string[]; cols: number; show: boolean; label: string }) {
  return (
    <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }} role="img" aria-label={label}>
      {urls.map((u, i) => (
        <img key={u} src={u} alt="" width={160} height={160} loading="lazy" decoding="async"
             className="aspect-square w-full rounded-[2px] transition-[opacity,transform] duration-500 ease-out"
             style={{ opacity: show ? 1 : 0, transform: show ? 'none' : 'scale(0.9)', transitionDelay: `${show ? i * 40 : 0}ms` }} />
      ))}
    </div>
  );
}

function Bar({ label, value, show }: { label: string; value: number; show: boolean }) {
  return (
    <div>
      <div className="flex justify-between">
        <span className="caption text-[11px] text-muted">{label}</span>
        <span className="mono text-[15px] text-text">{pct(value)}</span>
      </div>
      <div className="mt-2 h-[3px] w-full bg-white/10">
        <div className="h-full origin-left bg-leaf transition-transform duration-1000 ease-out"
             style={{ transform: `scaleX(${show ? value : 0})` }} />
      </div>
    </div>
  );
}

function Panels({ s, stage }: { s: Showcase; stage: number }) {
  const c = s.cnn;
  const [e0, eh, ew] = c.tiles.early_shape;
  const [l0, lh, lw] = c.tiles.late_shape;
  const crop = c.crop_pred;
  return [
    <div key="in" className="flex flex-col gap-4">
      <p className="caption text-[11px] text-muted">input</p>
      <Picture src={s.media.cnn_input} alt="The test leaf resized for the network." sizes="320px" className="aspect-square w-full rounded-[4px]" />
      <p className="mono text-[12px] text-muted">resized and normalised</p>
    </div>,
    <div key="early" className="flex flex-col gap-4">
      <p className="caption text-[11px] text-muted">early layer · {e0} maps at {eh}×{ew}</p>
      <Tiles urls={c.tiles.early} cols={4} show={stage >= 1} label="Early-layer feature maps: edges, veins and spot outlines." />
      <p className="mono text-[12px] text-muted">edges, veins, spot borders</p>
    </div>,
    <div key="late" className="flex flex-col gap-4">
      <p className="caption text-[11px] text-muted">late layer · {l0} maps at {lh}×{lw}</p>
      <Tiles urls={c.tiles.late} cols={3} show={stage >= 2} label="Late-layer feature maps: coarse, abstract patterns." />
      <p className="mono text-[12px] text-muted">coarse, abstract patterns</p>
    </div>,
    <div key="cam" className="flex flex-col gap-4">
      <p className="caption text-[11px] text-muted">Grad-CAM · disease head</p>
      <div className="relative aspect-square w-full overflow-hidden rounded-[4px]">
        <Picture src={s.media.cnn_input} alt="" sizes="320px" className="absolute inset-0 h-full w-full" />
        <Picture src={s.media.gradcam} alt="Grad-CAM heat map over the leaf: where the disease network looked." sizes="320px"
                 className="absolute inset-0 h-full w-full transition-opacity duration-1000" style={{ opacity: stage >= 3 ? 1 : 0 }} />
      </div>
      <p className="mono text-[12px] text-muted">where it looked</p>
    </div>,
    <div key="p" className="flex flex-col justify-center gap-7">
      <p className="caption text-[11px] text-muted">calibrated output</p>
      <Bar label={`P(crop = ${CROP_NAME[crop].toLowerCase()})`} value={c.p_crop[crop]} show={stage >= 4} />
      <Bar label={`P(diseased | ${CROP_NAME[crop].toLowerCase()})`} value={c.p_diseased_given_crop} show={stage >= 4} />
      <p className="mono text-[12px] text-muted">crop status: {c.crop_status}</p>
    </div>,
  ];
}

export default function Cnn({ site }: { site: Site }) {
  const reduced = useReducedMotion();
  const [crop, setCrop] = useState<'turmeric' | 'blackgram'>('blackgram');
  const [stage, setStage] = useState(0);
  const [wide, setWide] = useState(() => typeof window !== 'undefined' && window.innerWidth >= 1024);
  const pinRef = useRef<HTMLDivElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const [stackRef, stackIn] = useInView<HTMLDivElement>({ threshold: 0.15 });
  const s = site.showcase[crop];
  const pinned = wide && !reduced;

  useEffect(() => {
    const on = () => setWide(window.innerWidth >= 1024);
    window.addEventListener('resize', on);
    return () => window.removeEventListener('resize', on);
  }, []);
  useEffect(() => {
    if (!pinned) {
      if (stackIn) setStage(4);
      return;
    }
    const el = pinRef.current;
    const track = trackRef.current;
    if (!el || !track) return;
    const st = ScrollTrigger.create({
      trigger: el,
      start: 'top top',
      end: 'bottom bottom',
      onUpdate: (self) => {
        const maxX = Math.max(0, track.scrollWidth - window.innerWidth + 80);
        track.style.transform = `translate3d(${-self.progress * maxX}px,0,0)`;
        setStage(Math.min(4, Math.floor(self.progress * 5.2)));
      },
    });
    return () => st.kill();
  }, [pinned, stackIn]);

  const head = (
    <div className="mb-10 flex flex-wrap items-end justify-between gap-6">
      <div>
        <p className="caption mb-5 text-muted">05 / Stage 2 · CNN</p>
        <h2 className="max-w-[640px] text-4xl font-normal leading-[1.05] tracking-[-0.04em] lg:text-[52px]">
          The CNN learns its own features. <span className="font-playfair italic">We show where it looked.</span>
        </h2>
      </div>
      <div role="group" aria-label="Showcase leaf" className="inline-flex rounded-full border hairline p-1">
        {(['blackgram', 'turmeric'] as const).map((c) => (
          <button key={c} type="button" aria-pressed={crop === c} onClick={() => setCrop(c)}
                  className={`mono rounded-full px-3 py-1 text-[11px] uppercase tracking-[0.08em] ${crop === c ? 'bg-paper text-ink' : 'text-muted hover:text-text'}`}>
            {CROP_NAME[c]}
          </button>
        ))}
      </div>
    </div>
  );
  const panels = Panels({ s, stage });

  if (!pinned) {
    return (
      <section id="cnn" aria-label="Stage 2: CNN" className="section-pad bg-ink">
        <div ref={stackRef} className="container-grid">
          {head}
          <div className="grid gap-12 sm:grid-cols-2">{panels}</div>
        </div>
      </section>
    );
  }
  return (
    <section id="cnn" aria-label="Stage 2: CNN" className="relative bg-ink">
      <div ref={pinRef} style={{ height: '320vh' }}>
        <div className="sticky top-0 flex h-[100dvh] flex-col justify-center overflow-hidden">
          <div className="container-grid">{head}</div>
          <div ref={trackRef} className="flex items-stretch gap-16 pl-[max(24px,calc((100vw-1360px)/2+40px))] will-change-transform">
            {panels.map((p, i) => (
              <div key={i} className="relative w-[min(30vw,360px)] shrink-0">
                {p}
                {i < panels.length - 1 && (
                  <span aria-hidden="true" className="absolute -right-12 top-1/2 h-px w-8 bg-white/25" />
                )}
              </div>
            ))}
            <div className="w-[20vw] shrink-0" aria-hidden="true" />
          </div>
        </div>
      </div>
    </section>
  );
}
