import { Fragment } from 'react';

/**
 * Two bands of words drifting in opposite directions (CSS only; still under reduced motion, paused on hover).
 * Decorative: the same words are listed once for screen readers.
 */
export default function Marquee({ words, label }: { words: string[]; label: string }) {
  const row = (rev: boolean) => (
    <div className={`marquee-track ${rev ? 'rev' : ''}`} aria-hidden="true">
      {[0, 1].map((copy) => (
        <div key={copy} className="flex shrink-0 items-center">
          {words.map((w, i) => (
            <Fragment key={`${copy}-${w}`}>
              <span className={`whitespace-nowrap px-6 text-[44px] leading-none tracking-[-0.04em] sm:text-[72px] lg:text-[96px] ${
                (i + (rev ? 1 : 0)) % 2 ? 'font-playfair italic text-text/90' : 'text-text/50'}`}>
                {w}
              </span>
              <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${i % 2 ? 'bg-turmeric' : 'bg-leaf'}`} />
            </Fragment>
          ))}
        </div>
      ))}
    </div>
  );
  return (
    <div className="marquee overflow-hidden border-y hairline bg-ink py-8 sm:py-10">
      <p className="sr-only">{label}: {words.join(', ')}.</p>
      <div className="flex flex-col gap-4 sm:gap-6">
        {row(false)}
        {row(true)}
      </div>
    </div>
  );
}
