import { useEffect, useRef, useState } from 'react';
import { CROP_NAME, pct, pctRaw } from '@/lib/format';
import { useInView, useReducedMotion } from '@/lib/motion';
import type { Site } from '@/lib/site';
import { ScrollTrigger } from '@/lib/smooth';

const NODES = [
  { id: 'leaf', x: 70, y: 210, label: 'Leaf image', t: 0 },
  { id: 'ip', x: 290, y: 100, label: 'Image processing', t: 0.18 },
  { id: 'feat', x: 510, y: 100, label: 'Regions + features', t: 0.36 },
  { id: 'svm', x: 720, y: 100, label: 'SVM', t: 0.54 },
  { id: 'cnn', x: 290, y: 320, label: 'CNN', t: 0.18 },
  { id: 'learned', x: 510, y: 320, label: 'Learned features', t: 0.36 },
  { id: 'clf', x: 720, y: 320, label: 'Classifier', t: 0.54 },
  { id: 'out', x: 920, y: 210, label: 'Healthy / Diseased', t: 0.74 },
] as const;
const PATHS = [
  'M110 210 C 170 210, 190 100, 240 100 L 450 100 L 670 100 C 800 100, 820 210, 870 210',
  'M110 210 C 170 210, 190 320, 240 320 L 450 320 L 670 320 C 800 320, 820 210, 870 210',
];

function cardFrom(site: Site): string[] {
  const s = site.showcase.blackgram;
  const c = s.cnn;
  const dis = c.p_diseased_given_crop >= 0.5;
  return [
    `Crop: ${CROP_NAME[c.crop_pred]} (${pct(c.p_crop[c.crop_pred], 0)})`,
    `Prediction: ${dis ? 'Diseased' : 'Healthy'}`,
    `Affected Region: ${pctRaw(s.readouts.affected_pct)}`,
    `CNN Confidence: ${pct(dis ? c.p_diseased_given_crop : 1 - c.p_diseased_given_crop, 0)}`,
  ];
}

function TypedCard({ lines, start }: { lines: string[]; start: boolean }) {
  const full = lines.join('\n');
  const [n, setN] = useState(0);
  const reduced = useReducedMotion();
  useEffect(() => {
    if (!start) return;
    if (reduced) {
      setN(full.length);
      return;
    }
    let i = 0;
    const id = window.setInterval(() => {
      i += 2;
      setN(Math.min(full.length, i));
      if (i >= full.length) window.clearInterval(id);
    }, 28);
    return () => window.clearInterval(id);
  }, [start, full, reduced]);
  return (
    <figure className="rounded-[6px] border hairline bg-soil p-6" aria-label="Example result card">
      <pre className="mono whitespace-pre-wrap text-[15px] leading-[1.9] text-text" aria-hidden="true">
        {full.slice(0, n)}
        <span className="inline-block w-2 animate-pulse text-leaf">{n < full.length ? '▌' : ''}</span>
      </pre>
      <span className="sr-only">{full}</span>
      <figcaption className="caption mt-3 text-[10px] text-muted">real prediction · black gram test leaf</figcaption>
    </figure>
  );
}

export default function System({ site }: { site: Site }) {
  const reduced = useReducedMotion();
  const ref = useRef<HTMLDivElement>(null);
  const pathRefs = useRef<(SVGPathElement | null)[]>([]);
  const dotRefs = useRef<(SVGCircleElement | null)[]>([]);
  const [p, setP] = useState(reduced ? 1 : 0);
  const [cardRef, cardIn] = useInView<HTMLDivElement>({ threshold: 0.5 });

  useEffect(() => {
    const el = ref.current;
    if (!el || reduced) return;
    const st = ScrollTrigger.create({
      trigger: el,
      start: 'top 70%',
      end: 'bottom 60%',
      scrub: true,
      onUpdate: (self) => {
        const k = self.progress;
        setP(k);
        pathRefs.current.forEach((path, i) => {
          const dot = dotRefs.current[i];
          if (!path || !dot) return;
          const L = path.getTotalLength();
          const pt = path.getPointAtLength(Math.min(1, k * 1.15) * L);
          dot.setAttribute('cx', String(pt.x));
          dot.setAttribute('cy', String(pt.y));
          dot.style.opacity = k > 0.02 && k < 0.98 ? '1' : '0';
        });
      },
    });
    return () => st.kill();
  }, [reduced]);

  return (
    <section id="system" aria-labelledby="system-title" className="section-pad bg-ink">
      <div className="container-grid">
        <p className="caption mb-6 text-muted">07 / The system</p>
        <h2 id="system-title" className="mb-14 max-w-[760px] text-4xl font-normal leading-[1.05] tracking-[-0.04em] lg:text-[52px]">
          One leaf, two readings, <span className="font-playfair italic">one answer you can check.</span>
        </h2>
        <div ref={ref} className="overflow-x-auto">
          <svg viewBox="0 0 1000 420" className="w-full min-w-[640px]" role="img"
               aria-label="Final system: the leaf image goes through image processing to regions and features and an SVM, and through a CNN to learned features and a classifier; both give healthy or diseased.">
            {PATHS.map((d, i) => (
              <g key={i}>
                <path d={d} fill="none" stroke="var(--line)" strokeWidth="1.5" />
                <path ref={(el) => { pathRefs.current[i] = el; }} d={d} fill="none" stroke={i === 0 ? 'var(--leaf)' : 'var(--turmeric)'}
                      strokeWidth="1.5" pathLength={1} style={{ strokeDasharray: 1, strokeDashoffset: 1 - Math.min(1, p * 1.15) }} />
                <circle ref={(el) => { dotRefs.current[i] = el; }} r="5" fill={i === 0 ? 'var(--leaf)' : 'var(--turmeric)'} opacity="0" />
              </g>
            ))}
            {NODES.map((n) => {
              const on = p * 1.15 >= n.t;
              const w = Math.max(96, n.label.length * 9 + 28);
              return (
                <g key={n.id} transform={`translate(${n.x} ${n.y})`} style={{ transition: 'opacity 600ms', opacity: on ? 1 : 0.35 }}>
                  <rect x={-w / 2} y={-22} width={w} height={44} rx={22} fill="var(--soil)"
                        stroke={on ? (n.y < 210 ? 'var(--leaf)' : n.y > 210 ? 'var(--turmeric)' : 'var(--paper)') : 'var(--line)'} />
                  <text textAnchor="middle" dy="5" fill="var(--text)" fontSize="15" fontFamily="Inter">{n.label}</text>
                </g>
              );
            })}
          </svg>
        </div>
        <div ref={cardRef} className="mt-14 grid gap-8 lg:grid-cols-12">
          <p className="max-w-[440px] text-[17px] leading-relaxed text-muted lg:col-span-5">
            Both readings run on every photo. When they agree the answer stands; when they disagree the app says so and shows
            the evidence from each side.
          </p>
          <div className="lg:col-span-5 lg:col-start-8">
            <TypedCard lines={cardFrom(site)} start={cardIn} />
          </div>
        </div>
      </div>
    </section>
  );
}
