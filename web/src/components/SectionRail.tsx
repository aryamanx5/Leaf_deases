import { SECTIONS, useActiveSection } from '@/lib/sections';
import { scrollToId } from '@/lib/smooth';

/** Right-edge rail: thin ticks with mono labels; the active tick lengthens. */
export default function SectionRail() {
  const active = useActiveSection();
  const light = active === 'results';
  return (
    <nav aria-label="Sections" className="fixed right-5 top-1/2 z-[90] hidden -translate-y-1/2 lg:block">
      <ol className="flex flex-col items-end gap-3">
        {SECTIONS.map((s, i) => {
          const on = s.id === active;
          return (
            <li key={s.id}>
              <a
                href={`#${s.id}`}
                onClick={(e) => {
                  e.preventDefault();
                  scrollToId(s.id);
                }}
                aria-current={on ? 'true' : undefined}
                className="group flex items-center gap-3 py-0.5"
              >
                <span
                  className={`mono text-[11px] uppercase tracking-[0.08em] transition-opacity duration-500 ${
                    on ? `${light ? 'text-ink' : 'text-text'} opacity-100` : 'text-muted opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100'
                  }`}
                >
                  {String(i + 1).padStart(2, '0')} {s.label}
                </span>
                <span
                  className="block h-px origin-right bg-current transition-all duration-700 ease-out"
                  style={{ width: on ? 28 : 12, color: on ? 'var(--leaf)' : light ? 'rgba(10,14,11,0.35)' : 'rgba(232,240,228,0.35)' }}
                />
              </a>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
