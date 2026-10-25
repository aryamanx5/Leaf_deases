import type { Sources } from '@/lib/site';

type Props = {
  src: Sources;
  alt: string;
  sizes?: string;
  className?: string;
  eager?: boolean;
  style?: React.CSSProperties;
};

/** <picture> with AVIF + WebP srcsets; width/height set to avoid layout shift. */
export default function Picture({ src, alt, sizes = '100vw', className, eager, style }: Props) {
  const set = (m: Record<string, string>) =>
    Object.entries(m)
      .map(([w, u]) => `${u} ${w}w`)
      .join(', ');
  const widths = Object.keys(src.webp).map(Number);
  const fallback = src.webp[String(Math.max(...widths))] ?? '';
  return (
    <picture>
      <source type="image/avif" srcSet={set(src.avif)} sizes={sizes} />
      <source type="image/webp" srcSet={set(src.webp)} sizes={sizes} />
      <img
        src={fallback}
        alt={alt}
        width={src.w}
        height={src.h}
        loading={eager ? 'eager' : 'lazy'}
        decoding="async"
        {...{ fetchpriority: eager ? 'high' : 'auto' }} /* React 18 only forwards the lowercase attribute */
        className={className}
        style={style}
      />
    </picture>
  );
}
