import Logo from '@/components/Logo';
import { BRAND_NAME, GITHUB_URL, INSTITUTION, MENTOR, TAGLINE, TEAM } from '@/content/brand';
import { datasetCredits } from '@/content/citations';
import type { Site } from '@/lib/site';

export default function Footer({ site }: { site: Site | null }) {
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
        <dl className="grid gap-6 text-[14px] sm:grid-cols-2 lg:col-span-7">
          <div>
            <dt className="caption text-[11px] text-muted">Team</dt>
            <dd className="mt-2">{TEAM.join(' · ')}</dd>
            <dt className="caption mt-5 text-[11px] text-muted">Faculty mentor</dt>
            <dd className="mt-2">{MENTOR}</dd>
            <dt className="caption mt-5 text-[11px] text-muted">Institution</dt>
            <dd className="mt-2">{INSTITUTION}</dd>
          </div>
          <div>
            <dt className="caption text-[11px] text-muted">Data</dt>
            {datasetCredits.map((d) => (
              <dd key={d.name} className="mt-2 leading-snug">
                {d.url ? <a className="underline decoration-white/25 underline-offset-2 hover:decoration-white/70" href={d.url} target="_blank" rel="noreferrer">{d.name}</a> : d.name}
                <span className="block text-[12px] text-muted">{d.by}</span>
              </dd>
            ))}
            {site && (
              <dd className="mt-2 text-[12px] text-muted">
                Hero photo: <a className="underline decoration-white/25 underline-offset-2" href={site.hero.credit.url} target="_blank" rel="noreferrer">{site.hero.credit.author}</a>, {site.hero.credit.licence}
              </dd>
            )}
            {GITHUB_URL && (
              <dd className="mt-5">
                <a className="underline decoration-white/25 underline-offset-2" href={GITHUB_URL} target="_blank" rel="noreferrer">Source on GitHub</a>
              </dd>
            )}
          </div>
        </dl>
      </div>
    </footer>
  );
}
