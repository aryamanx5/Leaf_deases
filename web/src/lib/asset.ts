/** Prefix a root-relative public path ("/media/...") with the Vite base, so the site also works under a sub-path
 * (GitHub Pages project sites are served from /<repo>/). */
export const asset = (p: string): string => (p.startsWith('/') ? import.meta.env.BASE_URL + p.slice(1) : p);

/** Same, for every "/..." string inside a JSON value (site.json lists its images as "/media/..."). */
export function withBase<T>(v: T): T {
  if (typeof v === 'string') return asset(v) as T;
  if (Array.isArray(v)) return v.map(withBase) as T;
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, withBase(x)])) as T;
  return v;
}
