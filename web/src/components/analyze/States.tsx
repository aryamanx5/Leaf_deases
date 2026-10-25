import { FileWarning, ImageOff, Leaf, RotateCcw, ScanSearch, WifiOff } from 'lucide-react';
import Picture from '@/components/Picture';
import type { ApiError } from '@/lib/api';
import type { Showcase } from '@/lib/site';

/**
 * Before any photo: a preview of what the result will hold, drawn from the site's own showcase leaf (a real
 * test photo and the real Stage 1 outputs for it), not a grey box.
 */
export function EmptyState({ showcase }: { showcase: Showcase | undefined }) {
  const steps: [keyof Showcase['media'], string][] = [['original', 'Your photo'], ['segmented', 'Leaf cut-out'], ['lesions', 'Damaged spots'], ['gradcam', 'Where the AI looked']];
  return (
    <div className="rounded-[6px] border hairline p-6 sm:p-7" data-testid="empty-state">
      <p className="text-[20px] leading-snug tracking-[-0.02em] text-text">Pick a test photo or upload your own leaf.</p>
      <p className="mt-2 max-w-[52ch] text-[15px] leading-relaxed text-muted">
        In a second or two you get a plain answer (healthy or diseased), how much of the leaf is damaged, how sure the
        answer is, and the pictures behind it, like this example:
      </p>
      {/* the squares are drawn before site.json arrives, so the pictures land without moving anything */}
      <ol className="mt-6 grid grid-cols-4 gap-2 sm:gap-3" aria-label="Example result for a turmeric leaf from the test set">
          {steps.map(([k, label], i) => (
            <li key={k} className="min-w-0">
              <div className="aspect-square overflow-hidden rounded-[4px] border hairline bg-soil">
                {showcase && <Picture src={showcase.media[k]} alt="" sizes="(min-width: 1024px) 160px, 22vw" className="h-full w-full object-cover opacity-90" />}
              </div>
              <p className="mono mt-1.5 truncate text-[10px] text-muted sm:text-[11px]">
                <span aria-hidden="true">{i + 1} </span>{label}
              </p>
            </li>
          ))}
      </ol>
      <ul className="mt-6 grid gap-2 text-[14px] text-text/85 sm:grid-cols-3">
        <li className="flex items-center gap-2"><Leaf size={15} aria-hidden="true" className="text-leaf" /> One leaf, close up</li>
        <li className="flex items-center gap-2"><ScanSearch size={15} aria-hidden="true" className="text-leaf" /> Daylight, no flash</li>
        <li className="flex items-center gap-2"><ImageOff size={15} aria-hidden="true" className="text-leaf" /> Nothing is stored</li>
      </ul>
    </div>
  );
}

/** Placeholder with the same footprint as the verdict and the before/after picture, so nothing jumps. */
export function ResultSkeleton() {
  return (
    <div aria-hidden="true" className="grid gap-6" data-testid="result-skeleton">
      <div className="rounded-[6px] border hairline p-6 sm:p-7">
        <div className="shimmer h-3 w-28 rounded" />
        <div className="shimmer mt-4 h-8 w-3/4 rounded" />
        <div className="shimmer mt-4 h-4 w-full max-w-[460px] rounded" />
        <div className="shimmer mt-2 h-4 w-2/3 max-w-[380px] rounded" />
        <div className="mt-6 flex gap-1">
          <div className="shimmer h-2 w-8 rounded-full" /><div className="shimmer h-2 w-8 rounded-full" /><div className="shimmer h-2 w-8 rounded-full" />
        </div>
      </div>
      <div className="shimmer mx-auto aspect-square w-full max-w-[560px] rounded-[6px]" />
    </div>
  );
}

const ERROR: Record<string, { title: string; hint: string; Icon: typeof WifiOff }> = {
  offline: { title: 'We could not reach the analysis server', hint: 'Check your connection and try again in a moment.', Icon: WifiOff },
  too_large: { title: 'That photo is too large', hint: 'Choose a photo under 10 megabytes, or take a new one with your phone camera.', Icon: FileWarning },
  not_a_plant: { title: 'This does not look like a leaf', hint: 'Use a close-up of one turmeric or black gram leaf that fills most of the photo.', Icon: Leaf },
  unsupported_type: { title: 'That file is not a photo we can read', hint: 'Use a normal photo from your phone or camera.', Icon: FileWarning },
  invalid_image: { title: 'The image could not be read', hint: 'The file may be damaged. Try taking or choosing the photo again.', Icon: ImageOff },
};

/** Friendly error with the server's own message underneath; Retry (or choose another photo when retrying cannot help). */
export function ErrorState({ e, detailed, onRetry, onChooseAnother }: {
  e: ApiError; detailed: boolean; onRetry: (() => void) | null; onChooseAnother: () => void;
}) {
  const m = ERROR[e.code] ?? { title: 'Something went wrong', hint: 'Try again in a moment.', Icon: FileWarning };
  const detail = e.code === 'offline' ? (detailed ? 'Developers: start the API with make dev.' : '') : e.message.charAt(0).toUpperCase() + e.message.slice(1);
  return (
    <div role="alert" className="reveal rounded-[6px] border border-turmeric/40 bg-turmeric/5 p-6 sm:p-7" data-testid="error" data-code={e.code}>
      <m.Icon size={22} aria-hidden="true" className="text-turmeric" />
      <p className="mt-3 text-[20px] leading-snug tracking-[-0.02em] text-text">{m.title}</p>
      <p className="mt-2 max-w-[56ch] text-[15px] leading-relaxed text-text/85">{m.hint}</p>
      {detail && <p className="mt-2 max-w-[56ch] text-[13px] leading-relaxed text-muted">{detail}</p>}
      <div className="mt-5 flex flex-wrap gap-3">
        {onRetry && (
          <button type="button" onClick={onRetry}
                  className="inline-flex min-h-[44px] items-center gap-2 rounded-full bg-cta px-5 text-sm font-medium text-white transition-colors hover:bg-cta-hover">
            <RotateCcw size={15} aria-hidden="true" /> Retry
          </button>
        )}
        <button type="button" onClick={onChooseAnother}
                className="inline-flex min-h-[44px] items-center rounded-full border border-white/30 px-5 text-sm text-text transition-colors hover:bg-white/10">
          Choose another photo
        </button>
      </div>
    </div>
  );
}
