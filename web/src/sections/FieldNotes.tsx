import { Focus, Leaf, Plus, Square, Sun } from 'lucide-react';
import { useRef, type ReactNode } from 'react';
import LazyVideo from '@/components/fx/LazyVideo';
import Magnetic from '@/components/fx/Magnetic';
import { Reveal, SplitHeading } from '@/components/fx/Reveal';
import RunModelLink from '@/components/RunModelLink';
import Spotlight from '@/components/fx/Spotlight';
import { fix, int, pct } from '@/lib/format';
import { useRefreshOnResize } from '@/lib/refresh';
import type { Site } from '@/lib/site';

const HABITS: { Icon: typeof Leaf; title: string; why: string }[] = [
  { Icon: Leaf, title: 'One leaf', why: 'The pipeline keeps the largest leaf-coloured shape. Two overlapping leaves become one odd one.' },
  { Icon: Focus, title: 'Close up', why: 'Let the leaf fill most of the frame. Small spots need pixels to be measured at all.' },
  { Icon: Sun, title: 'Daylight, no flash', why: 'Flash glare reads as pale tissue, and deep shade hides the colours the rules rely on.' },
  { Icon: Square, title: 'A plain backdrop helps', why: 'Paper, a wall or the ground behind the leaf makes its outline easy to find. Not required.' },
];

function faq(site: Site): { q: string; a: ReactNode }[] {
  const t = site.field_model.trust.new;
  const held = site.field_model.sets.p4_test;
  const clean = site.field_model.sets.test;
  return [
    {
      q: 'Is this a diagnosis?',
      a: 'No. It is a research prototype trained on public datasets. It can tell you a leaf looks diseased and show you why it thinks so; it cannot tell you what to spray. Show the plant to an agronomist or your local extension officer before treating a crop.',
    },
    {
      q: 'Which plants does it know?',
      a: <>Turmeric and black gram, nothing else. A leaf from another plant should be declined: in a test with {int(t.n_other)} leaves of other species, {pct(t.other_species_refused, 0)} were turned away.</>,
    },
    {
      q: 'Why two methods instead of one?',
      a: 'Image processing gives measurements you can check: how many spots, how much area, what colour. The neural network is usually more accurate but cannot explain itself. When the two agree the answer is stronger; when they disagree the page says so and shows both.',
    },
    {
      q: 'What does “Not sure” mean?',
      a: 'The system declined to answer and tells you why: the photo is blurry, too dark or over-exposed, too little of it is leaf, the crop is unclear, or the leaf looks unlike anything it learned from. A refusal is a result, not an error.',
    },
    {
      q: 'How well does it work outside the lab?',
      a: <>On {int(held.n)} held-out field photos the deployed model scores a macro-F1 of {fix(held.new.macro_f1)}; on {int(clean.n)} lab-style test photos, {fix(clean.new.macro_f1)}. Those field photos come from sources it has seen other photos from. Farms it has never seen will be harder, and the results section says by how much.</>,
    },
    {
      q: 'What is macro-F1?',
      a: 'F1 balances two questions: of the leaves called diseased, how many were, and of the diseased leaves, how many were caught. Macro-F1 is that score averaged over the healthy and the diseased class, so the smaller class counts as much as the larger one.',
    },
    {
      q: 'What happens to my photo?',
      a: 'It is sent to the analysis server, processed, and not stored. The result page can show you every intermediate picture and number that was made from it.',
    },
  ];
}

/**
 * Field notes: four habits for a photo the model can read (beside a short clip), then the questions people ask.
 * Answers that quote a number read it from site.json.
 */
export default function FieldNotes({ site }: { site: Site }) {
  const ref = useRef<HTMLElement>(null);
  useRefreshOnResize(ref);
  return (
    <section id="notes" ref={ref} aria-labelledby="notes-title" className="section-pad relative overflow-hidden bg-ink">
      <div aria-hidden="true" className="aurora aurora-b" />
      <div className="container-grid relative">
        <Reveal as="p" className="caption mb-6 text-muted">Field notes · Before you take the photo</Reveal>
        <SplitHeading id="notes-title" text="A photo the model can read" italic="takes four habits."
                      className="mb-12 max-w-[820px] text-4xl font-normal leading-[1.05] tracking-[-0.04em] sm:text-5xl lg:mb-16 lg:text-[56px]" />

        <div className="grid gap-8 lg:grid-cols-12">
          <Reveal className="lg:col-span-5">
            <div className="relative">
              <LazyVideo clip="hand" softLoop className="aspect-[4/5] w-full rounded-[18px] sm:aspect-[16/11] lg:aspect-[4/5]" focus="left center" />
              {/* a viewfinder drawn over the clip: where the leaf should sit */}
              <div aria-hidden="true" className="pointer-events-none absolute inset-[12%]">
                {['left-0 top-0 border-l border-t', 'right-0 top-0 border-r border-t', 'bottom-0 left-0 border-b border-l', 'bottom-0 right-0 border-b border-r'].map((c) => (
                  <span key={c} className={`absolute h-7 w-7 border-white/80 ${c}`} />
                ))}
                <span className="absolute left-1/2 top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2">
                  <span className="pulse-ring absolute inset-0 rounded-full border border-leaf" />
                  <span className="absolute inset-[3px] rounded-full bg-leaf" />
                </span>
              </div>
              <p className="caption absolute left-4 top-4 rounded-full bg-ink/80 px-3 py-1 text-[10px] text-white/85">Get close · fill the frame</p>
            </div>
          </Reveal>

          <ul className="grid gap-4 sm:grid-cols-2 lg:col-span-7">
            {HABITS.map(({ Icon, title, why }, i) => (
              <Reveal as="li" key={title} delay={i * 90}>
                <Spotlight className="flex h-full flex-col rounded-[10px] border hairline bg-soil/50 p-6">
                  <span className="float-y flex h-11 w-11 items-center justify-center rounded-full border border-leaf/50 text-leaf"
                        style={{ animationDelay: `${i * 600}ms` }}>
                    <Icon size={19} strokeWidth={1.5} aria-hidden="true" />
                  </span>
                  <p className="mono mt-6 text-[12px] text-muted">{String(i + 1).padStart(2, '0')}</p>
                  <h3 className="mt-1 text-[22px] tracking-[-0.02em]">{title}</h3>
                  <p className="mt-2 text-[15px] leading-relaxed text-muted">{why}</p>
                </Spotlight>
              </Reveal>
            ))}
          </ul>
        </div>

        <div className="mt-20 grid gap-10 lg:mt-28 lg:grid-cols-12">
          <div className="lg:col-span-4">
            <Reveal as="p" className="caption mb-4 text-muted">Questions</Reveal>
            <Reveal as="p" delay={80} className="max-w-[340px] font-playfair text-[30px] italic leading-[1.15] tracking-[-0.02em]">
              What people ask before they trust a camera with a crop.
            </Reveal>
            <Reveal delay={160} className="mt-8">
              <Magnetic>
                <RunModelLink className="inline-flex min-h-[44px] items-center rounded-full bg-cta px-6 text-sm font-medium text-white transition-colors hover:bg-cta-hover">
                  Run the model
                </RunModelLink>
              </Magnetic>
            </Reveal>
            <Reveal delay={220} className="mt-10 hidden lg:block">
              <LazyVideo clip="dew" className="aspect-[4/3] w-full max-w-[340px] rounded-[14px]" />
            </Reveal>
          </div>
          <div className="lg:col-span-8" data-testid="faq">
            {faq(site).map(({ q, a }, i) => (
              <Reveal key={q} delay={i * 50}>
                <details className="group border-t hairline last:border-b [&_summary::-webkit-details-marker]:hidden">
                  <summary className="flex min-h-[64px] cursor-pointer list-none items-center justify-between gap-6 py-4 text-[19px] tracking-[-0.02em] text-text transition-colors hover:text-leaf sm:text-[22px]">
                    {q}
                    <span aria-hidden="true" className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-white/20 text-muted transition-transform duration-500 ease-out group-open:rotate-45 group-open:border-leaf group-open:text-leaf motion-reduce:transition-none">
                      <Plus size={15} />
                    </span>
                  </summary>
                  <p className="reveal max-w-[62ch] pb-6 text-[16px] leading-relaxed text-text/85">{a}</p>
                </details>
              </Reveal>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
