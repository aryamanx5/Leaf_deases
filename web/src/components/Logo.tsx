/** Geometric leaf mark: two arcs + midrib, 26x26, currentColor. */
export default function Logo({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 26 26" fill="none" aria-hidden="true">
      <path d="M4 22C4 11.5 10 4.5 22 4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M4 22C14.5 22 21.5 16 22 4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M4 22L18 8" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  );
}
