export const pct = (x: number | null | undefined, d = 1) => (x == null ? 'n/a' : `${(x * 100).toFixed(d)}%`);
export const pctRaw = (x: number | null | undefined, d = 1) => (x == null ? 'n/a' : `${x.toFixed(d)}%`);
export const fix = (x: number | null | undefined, d = 3) => (x == null ? 'n/a' : x.toFixed(d));
export const int = (x: number | null | undefined) => (x == null ? 'n/a' : Math.round(x).toLocaleString('en-IN'));
export const CROP_NAME = { turmeric: 'Turmeric', blackgram: 'Black gram' } as const;
export const METHOD_NAME = { rule_based: 'Rule-based score', svm: 'SVM on IP features', cnn: 'CNN' } as const;

/** Dataset class names in plain words (the black gram dataset spells crinkle "crinckle"). */
const CLASS_WORDS: Record<string, string> = { healthy_leaf: 'healthy', leaf_crinckle: 'leaf crinkle', blotch: 'leaf blotch' };
export const className = (c: string) => CLASS_WORDS[c] ?? c.replace(/_/g, ' ');

const SEG_METHOD: Record<string, string> = {
  plain_bg_threshold: 'plain backdrop, colour threshold',
  field_grabcut: 'busy background, GrabCut',
  field_grabcut_core: 'busy background, GrabCut on the core colour',
  user_roi_watershed: 'from your point, watershed',
  foliage_watershed: 'foliage watershed',
  colour_fallback: 'leaf-colour fallback',
};
/** "field_grabcut+colour_fallback" -> "busy background, GrabCut, then leaf-colour fallback" */
export const segMethod = (m: string) => m.split('+').map((x) => SEG_METHOD[x] ?? x.replace(/_/g, ' ')).join(', then ');
