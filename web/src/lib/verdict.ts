import { useCallback, useState } from 'react';
import type { AnalyzeResponse } from '@/lib/api';
import type { Site } from '@/lib/site';

export type Level = 'high' | 'medium' | 'low';

/**
 * How sure the verdict is, in words. The rule is the one scripts/export_site_data.py `confidence_levels()` scores
 * on held-out photos (site.json "confidence_levels"), with its cut-offs:
 *  low    the CNN is under `low_below_conf` sure, or the two methods disagree and it is under `high_min_conf`,
 *         or the crop itself is unsure, or the trust layer abstained;
 *  high   the CNN is at least `high_min_conf` sure, both methods agree and the photo is not flagged as unusual;
 *  medium everything else.
 */
export function confidenceLevel(r: AnalyzeResponse, c: Pick<Site['confidence_levels'], 'high_min_conf' | 'low_below_conf'>): Level {
  const conf = r.disease.confidence;
  if (r.abstain?.flag || r.crop.status === 'uncertain' || conf < c.low_below_conf || (!r.agreement && conf < c.high_min_conf)) return 'low';
  if (conf >= c.high_min_conf && r.agreement && !r.ood.flagged) return 'high';
  return 'medium';
}

/** Defaults used until site.json has loaded (same values as the export script). */
export const LEVEL_CUTOFFS = { high_min_conf: 0.9, low_below_conf: 0.7 };

export type ViewMode = 'simple' | 'detailed';
const MODE_KEY = 'patra:view';

/** Simple / Detailed, remembered per browser (Simple by default; storage failures are ignored). */
export function useViewMode(): [ViewMode, (m: ViewMode) => void] {
  const [mode, setModeState] = useState<ViewMode>(() => {
    try {
      return window.localStorage.getItem(MODE_KEY) === 'detailed' ? 'detailed' : 'simple';
    } catch {
      return 'simple';
    }
  });
  const setMode = useCallback((m: ViewMode) => {
    setModeState(m);
    try {
      window.localStorage.setItem(MODE_KEY, m);
    } catch {
      /* private mode: not remembered */
    }
  }, []);
  return [mode, setMode];
}
