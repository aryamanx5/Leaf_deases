import { fix, int, pct } from '@/lib/format';
import type { FieldSet, Site } from '@/lib/site';

const ROWS: { key: FieldSet; name: string; note: string }[] = [
  { key: 'p4_test', name: 'Held-out field photos', note: 'kept aside before training, scored once' },
  { key: 'wild_ext', name: 'Extended field set', note: 'more photos from the same field datasets' },
  { key: 'wild', name: 'Original field set', note: 'the first field photos' },
  { key: 'test', name: 'Clean test', note: 'lab-style photos' },
];

const signed = (x: number) => `${x >= 0 ? '+' : '−'}${fix(Math.abs(x))}`;

/** The deployed disease model against the one it replaced, every number from site.field_model. */
export default function FieldModel({ site }: { site: Site }) {
  const f = site.field_model;
  const held = f.sets.p4_test;
  const bg = held.by_crop.blackgram;
  const tu = held.by_crop.turmeric;
  const t = f.trust;
  return (
    <div className="mt-20 border-t border-ink/15 pt-12" aria-labelledby="field-model-title">
      <p className="caption mb-4 text-[11px] text-ink/70">The deployed model on field photos</p>
      <h3 id="field-model-title" className="max-w-[860px] font-playfair text-[26px] italic leading-[1.2] tracking-[-0.02em] sm:text-[34px]">
        Training on {int(f.train_field_leaves)} more field leaves lifted macro-F1 on held-out field photos from{' '}
        <span className="mono not-italic">{fix(held.old.macro_f1)}</span> to <span className="mono not-italic">{fix(held.new.macro_f1)}</span>.
      </h3>

      <div className="mt-8 overflow-x-auto">
        <table className="w-full border-collapse text-left md:min-w-[720px]">
          <caption className="sr-only">Macro-F1 of the previous and the deployed disease model per test set</caption>
          <thead>
            <tr className="border-b border-ink/20">
              <th scope="col" className="caption py-3 pr-4 text-[11px] font-normal text-ink/70">Test set</th>
              <th scope="col" className="caption hidden py-3 pr-4 text-[11px] font-normal text-ink/70 md:table-cell">Photos</th>
              <th scope="col" className="caption py-3 pr-4 text-[11px] font-normal text-ink/70">Before</th>
              <th scope="col" className="caption py-3 pr-4 text-[11px] font-normal text-ink/70">Now</th>
              <th scope="col" className="caption py-3 text-[11px] font-normal text-ink/70">Change<span className="hidden md:inline"> [95% CI]</span></th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map(({ key, name, note }) => {
              const r = f.sets[key];
              const sig = r.diff_ci95[0] > 0 || r.diff_ci95[1] < 0;
              return (
                <tr key={key} className="border-b border-ink/10">
                  <th scope="row" className="py-4 pr-4 align-top font-normal">
                    <span className="block text-[15px] md:text-[17px]">{name}</span>
                    <span className="mono mt-1 block text-[12px] text-ink/70 md:hidden">n = {int(r.n)}</span>
                    <span className="mt-1 hidden text-[13px] text-ink/70 md:block">{note}</span>
                  </th>
                  <td className="mono hidden py-4 pr-4 align-top text-[15px] md:table-cell">{int(r.n)}</td>
                  <td className="mono py-4 pr-4 align-top text-[15px] text-ink/70">{fix(r.old.macro_f1)}</td>
                  <td className="mono py-4 pr-4 align-top text-[17px]">{fix(r.new.macro_f1)}</td>
                  <td className="mono py-4 align-top text-[15px]">
                    {signed(r.diff)}
                    <span className="hidden text-ink/70 md:inline"> [{signed(r.diff_ci95[0])}, {signed(r.diff_ci95[1])}]</span>
                    <span className="block text-[11px] uppercase tracking-[0.08em] text-ink/70">{sig ? 'significant' : 'not significant'}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <ul className="mt-10 grid max-w-[920px] gap-3 text-[17px] leading-relaxed">
        {bg && (
          <li className="border-l-2 border-ink/20 pl-4">
            The gain is in black gram: <span className="mono">{fix(bg.old)}</span> → <span className="mono">{fix(bg.new)}</span> on{' '}
            {int(bg.n)} held-out leaves.
            {tu && (
              <>
                {' '}Turmeric{' '}
                {fix(tu.old) === fix(tu.new) ? (
                  <>stayed at <span className="mono">{fix(tu.new)}</span></>
                ) : (
                  <>went <span className="mono">{fix(tu.old)}</span> → <span className="mono">{fix(tu.new)}</span></>
                )}{' '}
                on only{' '}
                {int(tu.n)} leaves: disease-labelled turmeric field photos are scarce.
              </>
            )}
          </li>
        )}
        <li className="border-l-2 border-ink/20 pl-4">
          With its safety checks on, the system now answers <span className="mono">{pct(t.new.answered_share_labelled, 0)}</span> of
          held-out field leaves (was <span className="mono">{pct(t.old.answered_share_labelled, 0)}</span>) and is right on{' '}
          <span className="mono">{pct(t.new.end_to_end_accuracy_answered, 0)}</span> of those it answers. It declines{' '}
          <span className="mono">{pct(t.new.other_species_refused, 0)}</span> of {int(t.new.n_other)} leaves from other plants.
        </li>
        {f.ms.new != null && f.ms.old != null && (
          <li className="border-l-2 border-ink/20 pl-4">
            The new disease model takes <span className="mono">{fix(f.ms.new, 0)}</span> ms per image on CPU, against{' '}
            <span className="mono">{fix(f.ms.old, 0)}</span> ms for the one it replaced.
          </li>
        )}
        <li className="border-l-2 border-ink/20 pl-4 text-ink/70">
          The field sets share sources (not photos, scenes or capture minutes) with the new training data, so they measure new
          photos from known farms. The result on sources the model has never seen is the earlier one in the project report.
        </li>
      </ul>
    </div>
  );
}
