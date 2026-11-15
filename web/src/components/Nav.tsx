import { Menu, Search, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import Logo from '@/components/Logo';
import RunModelLink from '@/components/RunModelLink';
import { BRAND_NAME } from '@/content/brand';
import { openPalette } from '@/lib/palette';
import { useActiveSection, type SectionId } from '@/lib/sections';
import { scrollToId } from '@/lib/smooth';

const ITEMS: { id: SectionId; label: string }[] = [
  { id: 'problem', label: 'Problem' },
  { id: 'crops', label: 'Crops' },
  { id: 'pipeline', label: 'Pipeline' },
  { id: 'results', label: 'Results' },
  { id: 'run', label: 'Run model' },
];
const GROUP: Partial<Record<SectionId, SectionId>> = { cnn: 'pipeline', system: 'results', field: 'problem' };

export default function Nav() {
  const { pathname } = useLocation();
  const onLanding = pathname === '/';
  const active = useActiveSection();
  const current = onLanding ? (GROUP[active] ?? active) : null;
  const light = onLanding && active === 'results';
  const [open, setOpen] = useState(false);
  useEffect(() => setOpen(false), [pathname]);

  const go = (id: SectionId) => (e: React.MouseEvent) => {
    if (!onLanding) return;
    e.preventDefault();
    setOpen(false);
    scrollToId(id);
    history.replaceState(null, '', `#${id}`);
  };

  return (
    <nav aria-label="Primary" className={`fixed left-0 right-0 top-0 z-[100] flex items-center justify-between p-4 sm:p-5 max-md:bg-gradient-to-b max-md:to-transparent ${light ? 'max-md:from-paper/95' : 'max-md:from-ink/90'}`}>
      <Link to="/" className={`flex items-center gap-2.5 rounded-full pr-2 transition-colors duration-500 ${light ? 'text-ink' : 'text-white'}`} aria-label={`${BRAND_NAME}, home`}>
        <Logo />
        <span className="font-playfair text-2xl italic">{BRAND_NAME}</span>
      </Link>
      <div className={`absolute left-1/2 hidden -translate-x-1/2 items-center gap-1 rounded-full border border-white/20 px-2 py-2 backdrop-blur-md transition-colors duration-500 md:flex ${light ? 'bg-ink/85' : 'bg-ink/45'}`}>
        {ITEMS.map((it) => (
          <a
            key={it.id}
            href={onLanding ? `#${it.id}` : `/#${it.id}`}
            onClick={go(it.id)}
            aria-current={current === it.id ? 'true' : undefined}
            className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
              current === it.id ? 'bg-white/15 text-white' : 'text-white/80 hover:bg-white/20 hover:text-white'
            }`}
          >
            {it.label}
          </a>
        ))}
      </div>
      <div className="hidden items-center gap-2 md:flex">
        <button type="button" onClick={openPalette} aria-keyshortcuts="Meta+K Control+K"
                className={`hidden h-10 items-center gap-2 rounded-full border border-white/20 pl-3 pr-2.5 text-white/85 backdrop-blur-md transition-colors duration-500 hover:text-white lg:flex ${light ? 'bg-ink/85' : 'bg-ink/45'}`}>
          <Search size={14} aria-hidden="true" />
          <span className="sr-only">Jump to a part of the site, </span>
          <kbd className="mono rounded border border-white/20 px-1.5 py-0.5 text-[10px]">⌘K</kbd>
        </button>
        <RunModelLink className="rounded-full bg-white px-6 py-2.5 text-sm font-semibold text-gray-900 transition-colors hover:bg-gray-100">
          Run the model
        </RunModelLink>
      </div>
      <button
        type="button"
        className={`rounded-full p-2 md:hidden ${light ? 'text-ink' : 'text-white'}`}
        aria-label={open ? 'Close menu' : 'Open menu'}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        {open ? <X size={22} /> : <Menu size={22} />}
      </button>
      {open && (
        <div className="fixed inset-x-0 top-16 mx-4 rounded-2xl border border-line bg-soil p-2 md:hidden">
          {ITEMS.map((it) => (
            <a key={it.id} href={onLanding ? `#${it.id}` : `/#${it.id}`} onClick={go(it.id)}
               className="block rounded-xl px-4 py-3 text-base text-text hover:bg-white/5">
              {it.label}
            </a>
          ))}
          <RunModelLink className="mt-1 block rounded-xl bg-cta px-4 py-3 text-center font-medium text-white">
            Run the model
          </RunModelLink>
        </div>
      )}
    </nav>
  );
}
