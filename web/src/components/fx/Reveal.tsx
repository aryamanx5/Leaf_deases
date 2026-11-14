import { createElement, Fragment, type CSSProperties, type ReactNode } from 'react';
import { useInView } from '@/lib/motion';

type Tag = 'div' | 'p' | 'li' | 'section' | 'article' | 'figure' | 'span' | 'ul' | 'ol' | 'dl';

/**
 * Fades and lifts its children in once, when they scroll into view (styles: `.rv` in index.css; reduced motion
 * shows them at rest). `delay` staggers siblings.
 */
export function Reveal({ as = 'div', delay = 0, className = '', style, children, ...rest }: {
  as?: Tag; delay?: number; className?: string; style?: CSSProperties; children: ReactNode;
} & { id?: string; role?: string } & { [k: `data-${string}` | `aria-${string}`]: string | undefined }) {
  const [ref, inView] = useInView<HTMLElement>({ threshold: 0.15, margin: '0px 0px -8% 0px' });
  return createElement(as, {
    ref,
    className: `rv ${inView ? 'rv-in' : ''} ${className}`,
    style: { ...style, ['--rv-d' as string]: `${delay}ms` },
    ...rest,
  }, children);
}

/**
 * A heading whose words rise out of a mask, one after another, when it scrolls into view. The words stay real text
 * (screen readers read the sentence as written); `italic` marks a trailing phrase set in the display italic.
 */
export function SplitHeading({ as = 'h2', id, text, italic, className = '' }: {
  as?: 'h2' | 'h3'; id?: string; text: string; italic?: string; className?: string;
}) {
  const [ref, inView] = useInView<HTMLHeadingElement>({ threshold: 0.3 });
  const plain = text.split(' ').filter(Boolean);
  const tail = italic ? italic.split(' ').filter(Boolean) : [];
  const word = (w: string, i: number, it: boolean) => (
    <Fragment key={`${i}-${w}`}>
      <span className="sh-mask">
        <span className={`sh-word ${it ? 'font-playfair italic' : ''}`} style={{ ['--sh-i' as string]: i }}>{w}</span>
      </span>
      {' '}
    </Fragment>
  );
  return createElement(as, { ref, id, className: `sh ${inView ? 'sh-in' : ''} ${className}` },
    plain.map((w, i) => word(w, i, false)),
    tail.map((w, i) => word(w, plain.length + i, true)),
  );
}
