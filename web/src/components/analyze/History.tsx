import type { Picked } from '@/components/analyze/Input';
import type { AnalyzeResponse } from '@/lib/api';
import { CROP_NAME } from '@/lib/format';

export type Run = { picked: Picked; r: AnalyzeResponse };
export const HISTORY_MAX = 8;

/**
 * Photos read earlier in this visit, newest first. Choosing one shows its result again from memory (no new
 * request). Kept in the page only: nothing is written to disk, and a reload clears it.
 */
export function History({ runs, currentId, onOpen }: { runs: Run[]; currentId: string | null; onOpen: (run: Run) => void }) {
  if (runs.length < 2) return null;
  return (
    <section aria-labelledby="history-title" data-testid="history" className="reveal min-w-0">
      <h2 id="history-title" className="caption mb-3 text-[11px] text-muted">Read in this visit</h2>
      <ul className="-mx-1 flex gap-2.5 overflow-x-auto px-1 pb-2">
        {runs.map((run) => {
          const { r, picked } = run;
          const abstained = !!r.abstain?.flag;
          const diseased = r.disease.label === 'diseased';
          const word = abstained ? 'not sure' : diseased ? 'diseased' : 'healthy';
          const on = r.request_id === currentId;
          return (
            <li key={r.request_id} className="shrink-0">
              <button type="button" onClick={() => onOpen(run)} aria-pressed={on}
                      className={`group relative block w-[84px] rounded-[6px] border p-1 text-left transition-colors ${
                        on ? 'border-paper' : 'border-white/15 hover:border-white/50'}`}>
                <img src={picked.url} alt="" width={76} height={76} className="aspect-square w-full rounded-[3px] object-cover" />
                <span className="mt-1.5 flex items-center gap-1.5 px-0.5 pb-0.5 text-[11px] leading-tight text-text/90">
                  <span aria-hidden="true" className={`h-1.5 w-1.5 shrink-0 rounded-full ${abstained ? 'bg-chlorosis' : diseased ? 'bg-turmeric' : 'bg-leaf'}`} />
                  <span className="truncate">{word}</span>
                </span>
                <span className="sr-only">, {CROP_NAME[r.crop.label]} leaf. Show this result again</span>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
