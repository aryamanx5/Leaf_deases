import { useEffect, useRef, useState } from 'react';
import { Reveal, SplitHeading } from '@/components/fx/Reveal';
import { fix, int, pct } from '@/lib/format';
import { useInView, useReducedMotion } from '@/lib/motion';
import type { Site } from '@/lib/site';
import { ScrollTrigger } from '@/lib/smooth';
import { createGrower, drawVeins, type Veins } from '@/lib/veins';

type Phase = { n: string; title: string; broke: string; changed: (site: Site) => string; when?: (site: Site) => boolean };

const PHASES: Phase[] = [
  {
    n: '01',
    title: 'Measure, then learn',
    broke: 'Nothing yet. That was the problem: on lab photos everything looked excellent.',
    changed: (s) =>
      `An image-processing pipeline that measures every lesion, a neural network beside it, and a split with no near-copies across training and test (${int(s.dataset.total)} photos, ${int(s.dataset.clusters)} groups).`,
  },
  {
    n: '02',
    title: 'Leave the lab',
    broke: 'A turmeric photo from the web came back “black gram”, confidently. Every training turmeric leaf lay on white paper, so the network had partly learned the paper.',
    changed: () =>
      'Field photos kept for testing only, backgrounds swapped during training, a crop model that may answer “unsure”, and a check for photos unlike anything it has seen.',
  },
  {
    n: '03',
    title: 'Bigger eyes, more doubt',
    broke: 'Small networks trained on two datasets did not carry over to new farms well enough.',
    changed: () =>
      'Large pretrained vision models used as frozen feature extractors, joined with the measured features, plus a trust layer that declines with a reason instead of guessing.',
  },
  {
    n: '04',
    title: 'More field leaves',
    broke: 'Black gram in real fields was still the weak spot.',
    changed: (s) => {
      const h = s.field_model.sets.p4_test;
      return `${int(s.field_model.train_field_leaves)} field leaves added to training, under a switch rule written down before any result was seen. Held-out field macro-F1 went from ${fix(h.old.macro_f1)} to ${fix(h.new.macro_f1)}.`;
    },
  },
  {
    n: '05',
    title: 'Fewer needless refusals',
    when: (s) => !!s.confidence_model,
    broke: 'It said “not sure” about leaves it had right. The refusal line had been set stricter than the models needed.',
    changed: (s) => {
      const c = s.confidence_model!;
      return `The answer became the average of ${c.members} equally good models, and the refusal line was reset against a stated error target. It now answers ${pct(c.trust.new.answered_share_labelled, 0)} of held-out field leaves, up from ${pct(c.trust.old.answered_share_labelled, 0)}.`;
    },
  },
];

/**
 * How the project got here, in four phases, each of which began with something that broke. A leaf's veins grow
 * across the band above the cards as the section scrolls by (lib/veins.ts), one stretch per phase.
 */
export default function Journey({ site }: { site: Site }) {
  const reduced = useReducedMotion();
  const [nearRef, near] = useInView<HTMLElement>({ margin: '600px 0px', threshold: 0 });
  const bandRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const phases = PHASES.filter((p) => !p.when || p.when(site));
  const count = phases.length;
  const [phase, setPhase] = useState(reduced ? count : 0);

  useEffect(() => {
    const band = bandRef.current;
    const canvas = canvasRef.current;
    if (!band || !canvas || !near) return;
    const ctx = canvas.getContext('2d')!;
    const grower = createGrower();
    let veins: Veins | null = null;
    let w = 0, h = 0;
    let drawn = 0; // segments on the canvas
    let progress = reduced ? 1 : 0;
    let raf = 0;
    let dirty = true;

    const size = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      w = band.clientWidth;
      h = band.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawn = 0;
      dirty = true;
      kick();
    };
    const frame = () => {
      raf = 0;
      if (!veins) {
        veins = grower.step(4); // a few milliseconds of growth per frame until the venation is complete
        if (!veins) {
          if (dirty) {
            drawVeins(ctx, null, w, h, 0, 0, '');
            dirty = false;
          }
          kick();
          return;
        }
        drawn = 0;
        dirty = true;
      }
      const want = Math.round(progress * veins.count);
      if (dirty || want < drawn) {
        drawVeins(ctx, veins, w, h, 0, want, 'rgb(141, 184, 107)');
      } else if (want > drawn) {
        // at most a few hundred new segments per frame: cheap, and the growth stays visible on a fast scroll
        const to = Math.min(want, drawn + 260);
        drawVeins(ctx, veins, w, h, drawn, to, 'rgb(141, 184, 107)');
        drawn = to;
        if (to < want) kick();
        dirty = false;
        return;
      }
      drawn = want;
      dirty = false;
    };
    const kick = () => {
      if (!raf) raf = requestAnimationFrame(frame);
    };
    size();
    const ro = new ResizeObserver(size);
    ro.observe(band);
    const st = reduced ? null : ScrollTrigger.create({
      trigger: band,
      start: 'top 85%',
      end: 'bottom 35%',
      scrub: true,
      onUpdate: (self) => {
        progress = self.progress;
        setPhase(Math.min(count, Math.floor(self.progress * count * 1.02) + (self.progress > 0.02 ? 1 : 0)));
        kick();
      },
    });
    return () => {
      st?.kill();
      ro.disconnect();
      cancelAnimationFrame(raf);
    };
  }, [near, reduced, count]);

  return (
    <section id="journey" ref={nearRef} aria-labelledby="journey-title" className="section-pad overflow-hidden bg-ink">
      <div className="container-grid">
        <Reveal as="p" className="caption mb-6 text-muted">How it was built · {count > 4 ? 'Five' : 'Four'} phases</Reveal>
        <SplitHeading id="journey-title" text="It grew the way a leaf does:" italic="one vein at a time."
                      className="max-w-[860px] text-4xl font-normal leading-[1.05] tracking-[-0.04em] sm:text-5xl lg:text-[56px]" />
        <Reveal as="p" delay={120} className="mt-6 max-w-[560px] text-[17px] leading-relaxed text-muted">
          Each phase began with something that broke. The fix became the next layer of the system.
        </Reveal>
      </div>

      <div ref={bandRef} className="relative mx-auto mt-10 h-[38vw] max-h-[460px] min-h-[150px] w-full max-w-[1500px]">
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" role="img"
                aria-label="A leaf’s veins growing from the base to the tip as the page scrolls." data-testid="veins" />
        {/* one marker per phase along the midrib */}
        <ol className="pointer-events-none absolute inset-x-[2%] top-1/2 hidden sm:block" aria-hidden="true">
          {phases.map((p, i) => (
            <li key={p.n} className="absolute -translate-x-1/2 -translate-y-1/2" style={{ left: `calc(${(i + 0.5) / count} * 100%)` }}>
              <span className={`mono flex h-9 w-9 items-center justify-center rounded-full border text-[11px] transition-all duration-700 ease-out ${
                phase > i ? 'scale-100 border-leaf bg-ink text-leaf' : 'scale-75 border-white/20 bg-ink text-muted'}`}>
                {p.n}
              </span>
            </li>
          ))}
        </ol>
      </div>

      <div className="container-grid">
        <ol className={`mt-10 grid gap-6 sm:grid-cols-2 ${count > 4 ? 'lg:grid-cols-5' : 'lg:grid-cols-4'}`} data-testid="journey-phases">
          {phases.map((p, i) => (
            <li key={p.n} className={`border-t pt-5 transition-[border-color,transform] duration-700 ease-out ${phase > i ? 'border-leaf/70' : 'translate-y-2 border-white/15'}`}>
              <p className="mono text-[12px] text-leaf">Phase {p.n}</p>
              <h3 className="mt-2 font-playfair text-[26px] italic leading-tight tracking-[-0.02em]">{p.title}</h3>
              <p className="caption mt-5 text-[10px] text-turmeric">What broke</p>
              <p className="mt-1.5 text-[15px] leading-relaxed text-text/85">{p.broke}</p>
              <p className="caption mt-4 text-[10px] text-leaf">What changed</p>
              <p className="mt-1.5 text-[15px] leading-relaxed text-text/85">{p.changed(site)}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
