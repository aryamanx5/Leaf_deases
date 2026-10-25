import { Menu, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import Logo from '@/components/Logo';
import { BRAND_NAME } from '@/content/brand';
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
    <nav aria-label="Primary" className="fixed left-0 right-0 top-0 z-[100] flex items-center justify-between p-4 sm:p-5">
      <Link to="/" className="flex items-center gap-2.5 rounded-full pr-2" aria-label={`${BRAND_NAME}, home`}>
        <Logo />
        <span className="font-playfair text-2xl italic text-white">{BRAND_NAME}</span>
      </Link>
      <div className="absolute left-1/2 hidden -translate-x-1/2 items-center gap-1 rounded-full border border-white/30 bg-white/20 px-2 py-2 backdrop-blur-md md:flex">
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
      <Link
        to="/analyze"
        className="hidden rounded-full bg-white px-6 py-2.5 text-sm font-semibold text-gray-900 transition-colors hover:bg-gray-100 md:block"
      >
        Run the model
      </Link>
      <button
        type="button"
        className="rounded-full p-2 text-white md:hidden"
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
          <Link to="/analyze" className="mt-1 block rounded-xl bg-turmeric px-4 py-3 text-center font-medium text-white">
            Run the model
          </Link>
        </div>
      )}
    </nav>
  );
}
