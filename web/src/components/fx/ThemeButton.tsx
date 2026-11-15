import { nextTheme, setTheme, THEMES, useTheme } from '@/lib/theme';

/** Cycles the colour theme; the new palette opens as a circle from this button. */
export default function ThemeButton({ className = '' }: { className?: string }) {
  const id = useTheme();
  const theme = THEMES.find((t) => t.id === id)!;
  const next = THEMES.find((t) => t.id === nextTheme(id))!;
  return (
    <button type="button" data-testid="theme-button" title={`Theme: ${theme.name}`}
            onClick={(e) => {
              const r = e.currentTarget.getBoundingClientRect();
              setTheme(next.id, { x: r.left + r.width / 2, y: r.top + r.height / 2 });
            }}
            className={`group flex h-10 w-10 items-center justify-center rounded-full border border-white/20 backdrop-blur-md transition-colors duration-500 ${className}`}>
      <span className="sr-only">Colour theme: {theme.name}. Switch to {next.name}</span>
      <span aria-hidden="true" className="relative h-4 w-4 overflow-hidden rounded-full ring-1 ring-white/40 transition-transform duration-500 ease-out group-hover:rotate-180">
        <span className="absolute inset-y-0 left-0 w-1/2" style={{ background: theme.swatch[1] }} />
        <span className="absolute inset-y-0 right-0 w-1/2" style={{ background: theme.swatch[2] }} />
      </span>
    </button>
  );
}
