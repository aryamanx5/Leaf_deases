import { ArrowRight, CornerDownLeft, Search } from 'lucide-react';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { DEPLOYED_APP_URL } from '@/content/brand';
import { PALETTE_EVENT } from '@/lib/palette';
import { getLenis, scrollToId } from '@/lib/smooth';
import { setTheme, THEMES } from '@/lib/theme';

type Item = { id: string; label: string; hint: string; group: 'Go to' | 'Explore' | 'Do' | 'Theme'; run: () => void };


const SECTIONS: [string, string, string][] = [
  ['top', 'Top of the page', 'the grower in the greenhouse'],
  ['problem', 'The problem', 'why diagnosis by eye falls short'],
  ['interlude', 'The field', 'film: one field, thousands of leaves'],
  ['crops', 'Two crops', 'turmeric and black gram'],
  ['atlas', 'Field guide', 'what each disease looks like'],
  ['pipeline', 'Image processing', 'stage one, step by step'],
  ['lab', 'Threshold lab', 'try Otsu’s method yourself'],
  ['cnn', 'The neural network', 'stage two and where it looked'],
  ['results', 'Results', 'the comparison table'],
  ['proof', 'How sure is sure', 'every held-out photo as a dot'],
  ['system', 'The system', 'two readings, one answer'],
  ['reel', 'Film reel', 'five clips from the field'],
  ['journey', 'How it was built', 'four phases'],
  ['notes', 'Photo tips and questions', 'take a photo the model can read'],
  ['run', 'Heal the leaves', 'the interactive finale'],
];

/**
 * Command palette: press ⌘K / Ctrl K (or "/" outside a text field) to jump to any part of the site.
 * A modal dialog with a filter box and a listbox; arrow keys move, Enter runs, Escape closes and returns focus.
 */
export default function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState('');
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const opener = useRef<HTMLElement | null>(null);
  const navigate = useNavigate();
  const { pathname } = useLocation();

  const items = useMemo<Item[]>(() => {
    const go = (id: string) => () => {
      if (pathname !== '/') {
        navigate(`/#${id}`);
        return;
      }
      // after the dialog has closed and the page scroll is unlocked
      window.setTimeout(() => {
        scrollToId(id);
        history.replaceState(null, '', id === 'top' ? window.location.pathname : `#${id}`);
      }, 60);
    };
    const runModel = () => {
      if (DEPLOYED_APP_URL) window.open(DEPLOYED_APP_URL, '_blank', 'noreferrer');
      else navigate('/analyze');
    };
    return [
      { id: 'do-run', label: 'Run the model on a leaf', hint: 'upload or pick a test photo', group: 'Do', run: runModel },
      ...SECTIONS.map(([id, label, hint]): Item => ({
        id: `go-${id}`, label, hint, group: ['atlas', 'lab', 'proof', 'interlude', 'reel', 'journey', 'notes'].includes(id) ? 'Explore' : 'Go to', run: go(id),
      })),
      { id: 'do-home', label: 'Back to the landing page', hint: 'home', group: 'Do', run: () => navigate('/') },
      ...THEMES.map((t): Item => ({
        id: `theme-${t.id}`, label: `${t.name} theme`, hint: 'change the colours', group: 'Theme',
        run: () => window.setTimeout(() => setTheme(t.id), 60),
      })),
    ];
  }, [navigate, pathname]);

  const shown = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    return items.filter((it) => words.every((w) => `${it.label} ${it.hint} ${it.group}`.toLowerCase().includes(w)));
  }, [items, q]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = /^(INPUT|TEXTAREA|SELECT)$/.test((e.target as HTMLElement | null)?.tagName ?? '') || (e.target as HTMLElement | null)?.isContentEditable;
      if ((e.key.toLowerCase() === 'k' && (e.metaKey || e.ctrlKey)) || (e.key === '/' && !typing && !e.metaKey && !e.ctrlKey && !e.altKey)) {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    const onOpen = () => setOpen(true);
    window.addEventListener('keydown', onKey);
    window.addEventListener(PALETTE_EVENT, onOpen);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener(PALETTE_EVENT, onOpen);
    };
  }, []);

  useLayoutEffect(() => {
    if (!open) return;
    opener.current = document.activeElement as HTMLElement | null;
    setQ('');
    setCursor(0);
    getLenis()?.stop();
    inputRef.current?.focus(); // at once, so keys typed straight after the shortcut land in the box
    return () => {
      getLenis()?.start();
      opener.current?.focus?.();
    };
  }, [open]);

  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' });
  }, [cursor, open]);

  if (!open) return null;
  const choose = (it: Item | undefined) => {
    if (!it) return;
    setOpen(false);
    it.run();
  };
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      setOpen(false);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setCursor((c) => (shown.length ? (c + 1) % shown.length : 0));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setCursor((c) => (shown.length ? (c - 1 + shown.length) % shown.length : 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      choose(shown[cursor]);
    } else if (e.key === 'Tab') {
      e.preventDefault(); // one focus stop: the filter box drives the list
    }
  };
  let lastGroup = '';

  return (
    <div className="fixed inset-0 z-[400] flex items-start justify-center px-4 pt-[12vh]" data-testid="palette" onKeyDown={onKeyDown}>
      <div className="veil-in absolute inset-0 bg-ink/70 backdrop-blur-sm" onClick={() => setOpen(false)} aria-hidden="true" />
      <div role="dialog" aria-modal="true" aria-label="Jump to a part of the site" data-lenis-prevent
           className="palette-in relative w-full max-w-[560px] overflow-hidden rounded-[14px] border border-white/15 bg-soil shadow-[0_30px_80px_rgba(0,0,0,0.55)]">
        <div className="flex items-center gap-3 border-b hairline px-4">
          <Search size={17} aria-hidden="true" className="shrink-0 text-muted" />
          <input ref={inputRef} value={q} role="combobox" aria-expanded="true" aria-controls="palette-list" aria-autocomplete="list"
                 aria-activedescendant={shown[cursor] ? `palette-${shown[cursor].id}` : undefined}
                 onChange={(e) => {
                   setQ(e.target.value);
                   setCursor(0);
                 }}
                 placeholder="Where to?" aria-label="Filter destinations" autoComplete="off" spellCheck={false}
                 className="h-14 w-full bg-transparent text-[16px] text-text outline-none placeholder:text-muted" />
          <kbd className="mono hidden shrink-0 rounded border border-white/15 px-1.5 py-0.5 text-[10px] text-muted sm:block">esc</kbd>
        </div>
        <ul ref={listRef} id="palette-list" role="listbox" aria-label="Destinations" className="max-h-[min(52vh,420px)] overflow-y-auto p-2">
          {shown.length === 0 && <li role="presentation" className="px-3 py-6 text-center text-[14px] text-muted">Nothing matches “{q}”.</li>}
          {shown.map((it, i) => {
            const head = it.group !== lastGroup;
            lastGroup = it.group;
            const on = i === cursor;
            return (
              <li key={it.id} role="presentation">
                {head && <p className="caption px-3 pb-1 pt-3 text-[10px] text-muted">{it.group}</p>}
                <div id={`palette-${it.id}`} role="option" aria-selected={on} onClick={() => choose(it)} onPointerMove={() => setCursor(i)}
                     className={`flex min-h-[44px] cursor-pointer items-center gap-3 rounded-[8px] px-3 text-[15px] ${on ? 'bg-white/10 text-text' : 'text-text/85'}`}>
                  <ArrowRight size={14} aria-hidden="true" className={on ? 'text-leaf' : 'text-muted'} />
                  <span className="shrink-0">{it.label}</span>
                  <span className="min-w-0 flex-1 truncate text-[13px] text-muted">{it.hint}</span>
                  {on && <CornerDownLeft size={14} aria-hidden="true" className="shrink-0 text-muted" />}
                </div>
              </li>
            );
          })}
        </ul>
        <p className="mono flex items-center justify-between gap-3 border-t hairline px-4 py-2.5 text-[11px] text-muted">
          <span>↑ ↓ to move · enter to go</span>
          <span>⌘K or / to open</span>
        </p>
      </div>
    </div>
  );
}
