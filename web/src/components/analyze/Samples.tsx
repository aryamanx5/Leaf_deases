import { Check, X } from 'lucide-react';
import { useMemo, useState } from 'react';
import type { AnalyzeResponse } from '@/lib/api';
import { className, CROP_NAME } from '@/lib/format';
import type { Sample } from '@/lib/site';

type Filter = 'all' | 'turmeric' | 'blackgram' | 'healthy' | 'diseased';
const FILTERS: [Filter, string][] = [['all', 'All'], ['turmeric', 'Turmeric'], ['blackgram', 'Black gram'], ['healthy', 'Healthy'], ['diseased', 'Diseased']];

/** Stable shuffle, so neighbouring tiles do not give away the class (site.json lists them class by class). */
const hash = (s: string) => {
  let h = [...s].reduce((a, c) => Math.imul(a ^ c.charCodeAt(0), 0x01000193), 0x811c9dc5); // FNV-1a
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b); // murmur3 finaliser, so similar ids land far apart
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return (h ^ (h >>> 16)) >>> 0;
};

/**
 * Test-set photos to try without your own leaf. A tile shows the crop and the kind of photo; its class stays
 * hidden until it has been run (`revealed`), so the model's answer is not primed. The Healthy / Diseased
 * filters are an explicit choice to see the labels.
 */
const FIRST = 6; // photos shown before "More options"

export function SampleGallery({ samples, onPick, activeId, revealed }: {
  samples: Sample[]; onPick: (s: Sample) => void; activeId: string | null; revealed: ReadonlySet<string>;
}) {
  const [filter, setFilter] = useState<Filter>('all');
  const [all, setAll] = useState(false);
  const ordered = useMemo(() => {
    const seen: Record<string, number> = {};
    return [...samples]
      .sort((a, b) => hash(a.id) - hash(b.id))
      .map((s) => ({ s, n: (seen[s.crop] = (seen[s.crop] ?? 0) + 1) }));
  }, [samples]);
  const shown = ordered.filter(({ s }) => filter === 'all' || s.crop === filter || s.label === filter);
  const visible = all ? shown : shown.slice(0, FIRST);
  const more = shown.length - visible.length;
  if (!samples.length) return null;

  return (
    <section aria-labelledby="samples-title" className="min-w-0">
      <h2 id="samples-title" className="caption mb-3 text-[11px] text-muted">
        Or try one of {samples.length} test photos
      </h2>
      <div role="group" aria-label="Show test photos" className="-mx-1 mb-3 flex gap-1.5 overflow-x-auto px-1 pb-1">
        {FILTERS.map(([v, l]) => (
          <button key={v} type="button" aria-pressed={filter === v} onClick={() => setFilter(v)}
                  className={`mono min-h-[44px] shrink-0 rounded-full border px-3.5 text-[11px] uppercase tracking-[0.08em] transition-colors ${
                    filter === v ? 'border-paper bg-paper text-ink' : 'border-white/15 text-muted hover:border-white/40 hover:text-text'}`}>
            {l}
          </button>
        ))}
      </div>
      <ul className="-mx-1 flex snap-x gap-2 overflow-x-auto px-1 pb-2 sm:mx-0 sm:grid sm:grid-cols-6 sm:overflow-visible sm:px-0 lg:grid-cols-3"
          data-testid="sample-gallery">
        {visible.map(({ s, n }) => {
          const active = s.id === activeId;
          // the accessible name is the visible text plus sr-only words, e.g. "Turmeric, photo 3: Plain background"
          const [kind, kindRest] = s.set === 'wild_test' ? ['Field photo', ''] : s.bg === 'field' ? ['Outdoor', ' background'] : ['Plain', ' background'];
          const shown = revealed.has(s.id);
          return (
            <li key={s.id} className="w-[104px] shrink-0 snap-start sm:w-auto">
              <button type="button" onClick={() => onPick(s)} data-testid={`sample-${s.id}`} aria-pressed={active}
                      className={`group relative block w-full rounded-[6px] border p-1 text-left transition-colors ${
                        active ? 'border-leaf bg-leaf/10' : 'border-white/10 hover:border-white/40'}`}>
                <span className="relative block aspect-square overflow-hidden rounded-[4px] bg-soil">
                  <img src={s.thumb} alt="" width={160} height={160} loading="lazy" decoding="async"
                       className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.04] motion-reduce:transition-none" />
                  {s.set === 'wild_test' && (
                    <span className="mono absolute left-1 top-1 rounded-full bg-ink/80 px-1.5 py-0.5 text-[9px] uppercase tracking-[0.06em] text-text">
                      Field
                    </span>
                  )}
                </span>
                <span className="block px-1 pb-0.5 pt-1.5 text-[12px] leading-tight text-text">
                  {CROP_NAME[s.crop]}<span className="sr-only">, photo {n}: </span>
                </span>
                <span className="mono block truncate px-1 pb-1 text-[10px] leading-tight text-muted">
                  {shown ? className(s.orig_class) : kind}
                  <span className="sr-only">{shown ? `, ${kind.toLowerCase()}${kindRest}` : kindRest}</span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
      {(more > 0 || all) && (
        <button type="button" onClick={() => setAll((a) => !a)} aria-expanded={all} data-testid="samples-show-all"
                className="mt-2 inline-flex min-h-[44px] items-center gap-2 rounded-full border border-white/20 px-4 text-[13px] text-text transition-colors hover:border-white/50">
          {all ? 'Fewer options' : `More options (${more} more photos)`}
        </button>
      )}
    </section>
  );
}

/** After a sample has been run: its real label next to what the model said, with a tick or a cross. */
export function SampleCheck({ s, r }: { s: Sample; r: AnalyzeResponse }) {
  // a crop chosen by the user is not the model's answer, so only the health call is judged then
  const byUser = r.crop.status === 'user_selected';
  const right = r.disease.label === s.label && (byUser || r.crop.label === s.crop);
  const Icon = right ? Check : X;
  return (
    <div data-testid="label-check"
         className={`flex items-start gap-3 rounded-[6px] border px-5 py-4 text-[14px] leading-relaxed ${
           right ? 'border-leaf/40 bg-leaf/5' : 'border-turmeric/40 bg-turmeric/5'}`}>
      <Icon size={18} aria-hidden="true" className={`mt-0.5 shrink-0 ${right ? 'text-leaf' : 'text-turmeric'}`} />
      <p className="text-text/90">
        <span className="sr-only">{right ? 'Model was right. ' : 'Model was wrong. '}</span>
        Photo’s real label: <strong className="font-medium text-text">{CROP_NAME[s.crop]} · {className(s.orig_class)}</strong>
        {s.label === 'diseased' && ' (diseased)'}
        <span className="text-muted"> · </span>
        Model said: <strong className="font-medium text-text">{CROP_NAME[r.crop.label]}{byUser && ' (your choice)'} · {r.disease.label}</strong>
        {s.credit && <span className="mt-1 block text-[12px] text-muted">Field photo: {s.credit}</span>}
      </p>
    </div>
  );
}
