import { useState } from 'react';
import LazyVideo from '@/components/fx/LazyVideo';
import { Reveal, SplitHeading } from '@/components/fx/Reveal';
import { FILM, type ClipName } from '@/lib/film';

const FRAMES: { clip: ClipName; title: string; line: string; focus?: string }[] = [
  {
    clip: 'veins',
    title: 'Held to the light',
    line: 'Living tissue glows; dead tissue stays dark. Colour is the first thing the pipeline measures, before any shape or texture.',
  },
  {
    clip: 'rain',
    title: 'Monsoon',
    line: 'A wet leaf shines, and shine reads as pale tissue. It is why the page asks for daylight, a dry leaf and no flash.',
  },
  {
    clip: 'soil',
    title: 'By hand',
    line: 'These are crops of small fields, checked plant by plant. The tool had to be something a grower already carries: a phone.',
  },
  {
    clip: 'farmer',
    title: 'Walking the rows',
    line: 'Scouting means walking the whole field. Reading one leaf takes the system a few seconds, and it can say “not sure”.',
  },
  {
    clip: 'macro',
    title: 'Venation',
    line: 'Veins are lines of paler tissue too. The lesion rules have to tell a vein from a streak of disease, on every leaf.',
  },
];

/**
 * Film reel: five short clips side by side. The open panel plays and shows its note; the others rest on their
 * still frame, so only one clip is ever decoding. Panels open on hover, focus or tap (they are real buttons).
 */
export default function Reel() {
  const [open, setOpen] = useState(0);
  return (
    <section id="reel" aria-labelledby="reel-title" className="section-pad relative overflow-hidden bg-ink">
      <div aria-hidden="true" className="aurora aurora-a" />
      <div className="container-grid relative">
        <Reveal as="p" className="caption mb-6 text-muted">Reel · From the field</Reveal>
        <SplitHeading id="reel-title" text="Five things a leaf does" italic="that a model has to live with."
                      className="mb-12 max-w-[900px] text-4xl font-normal leading-[1.05] tracking-[-0.04em] sm:text-5xl lg:text-[56px]" />
      </div>
      <Reveal className="container-grid relative">
        <ul className="flex flex-col gap-3 lg:h-[min(70vh,620px)] lg:flex-row" data-testid="reel">
          {FRAMES.map((f, i) => {
            const on = i === open;
            return (
              <li key={f.clip}
                  className={`reel-panel relative min-w-0 overflow-hidden rounded-[14px] border transition-[flex-grow,height,border-color] duration-700 ease-out motion-reduce:transition-none ${
                    on ? 'h-[62vw] max-h-[420px] border-white/25 lg:h-auto lg:max-h-none lg:grow-[6]' : 'h-[72px] border-white/10 lg:h-auto lg:grow'} lg:basis-0`}>
                <LazyVideo clip={f.clip} active={on} credit={false} focus={f.focus} sizes="(max-width: 1024px) 100vw, 60vw"
                           className="absolute inset-0 h-full w-full" />
                <div aria-hidden="true" className={`pointer-events-none absolute inset-0 transition-colors duration-700 ${
                  on ? 'bg-[linear-gradient(180deg,rgba(10,14,11,0.05)_35%,rgba(10,14,11,0.88)_100%)]' : 'bg-ink/65'}`} />
                <button type="button" aria-pressed={on} onClick={() => setOpen(i)} onPointerEnter={(e) => e.pointerType === 'mouse' && setOpen(i)}
                        onFocus={() => setOpen(i)}
                        className="absolute inset-0 flex w-full items-end p-5 text-left sm:p-6">
                  <span className="flex w-full items-end justify-between gap-4">
                    <span className="min-w-0">
                      <span className="mono block text-[11px] text-white/80">{String(i + 1).padStart(2, '0')}</span>
                      <span className={`mt-1 block whitespace-nowrap font-playfair italic leading-none tracking-[-0.02em] text-white transition-[font-size] duration-500 motion-reduce:transition-none ${
                        on ? 'text-[34px] sm:text-[44px]' : 'text-[22px] lg:[writing-mode:vertical-rl] lg:rotate-180'}`}>
                        {f.title}
                      </span>
                    </span>
                  </span>
                </button>
                {on && (
                  <a href={FILM[f.clip].url} target="_blank" rel="noreferrer"
                     className="mono reveal absolute right-3 top-3 rounded-full bg-ink/80 px-2.5 py-1 text-[10px] text-white/80 underline-offset-2 hover:underline">
                    Footage: {FILM[f.clip].by} / Pexels
                  </a>
                )}
                {on && (
                  <p className="reveal pointer-events-none absolute bottom-5 right-5 hidden max-w-[300px] text-[14px] leading-relaxed text-white/90 sm:bottom-6 sm:right-6 md:block lg:bottom-14">
                    {f.line}
                  </p>
                )}
              </li>
            );
          })}
        </ul>
        <p className="mt-6 max-w-[620px] text-[15px] leading-relaxed text-muted md:hidden" aria-live="polite">{FRAMES[open]!.line}</p>
        <p className="mono mt-6 text-[11px] text-muted">
          Stock footage for atmosphere, not dataset images. Hover, tab or tap a frame to open it.
        </p>
      </Reveal>
    </section>
  );
}
