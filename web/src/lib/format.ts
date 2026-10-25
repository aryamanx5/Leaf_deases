export const pct = (x: number | null | undefined, d = 1) => (x == null ? 'n/a' : `${(x * 100).toFixed(d)}%`);
export const pctRaw = (x: number | null | undefined, d = 1) => (x == null ? 'n/a' : `${x.toFixed(d)}%`);
export const fix = (x: number | null | undefined, d = 3) => (x == null ? 'n/a' : x.toFixed(d));
export const int = (x: number | null | undefined) => (x == null ? 'n/a' : Math.round(x).toLocaleString('en-IN'));
export const CROP_NAME = { turmeric: 'Turmeric', blackgram: 'Black gram' } as const;
export const METHOD_NAME = { rule_based: 'Rule-based score', svm: 'SVM on IP features', cnn: 'CNN' } as const;
