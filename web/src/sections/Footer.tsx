import Logo from '@/components/Logo';
import { BRAND_NAME, GITHUB_URL, TAGLINE } from '@/content/brand';
import { datasetCredits } from '@/content/citations';

export default function Footer() {
  return (
    <footer className="border-t hairline bg-ink">
      <div className="container-grid grid gap-12 py-16 lg:grid-cols-12">
        <div className="lg:col-span-5">
          <div className="flex items-center gap-2.5">
            <Logo />
            <span className="font-playfair text-2xl italic">{BRAND_NAME}</span>
          </div>
          <p className="mt-4 max-w-[420px] text-[15px] leading-relaxed text-muted">
            Image Processing + CNN for Disease Detection in Turmeric and Black Gram. {TAGLINE}
          </p>
          <p className="mt-6 text-[13px] text-muted">Built for research and education; not a substitute for an agronomist.</p>
        </div>
        <dl className="text-[14px] lg:col-span-7">
          <dt className="caption text-[11px] text-muted">Data</dt>
          {datasetCredits.map((d) => (
            <dd key={d.name} className="mt-3 leading-snug">
              {d.url ? <a className="underline decoration-white/25 underline-offset-2 hover:decoration-white/70" href={d.url} target="_blank" rel="noreferrer">{d.name}</a> : d.name}
              <span className="block text-[12px] text-muted">{d.by}</span>
              {d.links && (
                <span className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[12px]">
                  {d.links.map((l) => (
                    <a key={l.url} className="underline decoration-white/25 underline-offset-2 hover:decoration-white/70" href={l.url} target="_blank" rel="noreferrer">{l.label}</a>
                  ))}
                </span>
              )}
            </dd>
          ))}
          {GITHUB_URL && (
            <dd className="mt-5">
              <a className="underline decoration-white/25 underline-offset-2" href={GITHUB_URL} target="_blank" rel="noreferrer">Source on GitHub</a>
            </dd>
          )}
        </dl>
      </div>
    </footer>
  );
}
