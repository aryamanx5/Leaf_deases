import { useEffect } from 'react';
import SectionRail from '@/components/SectionRail';
import { useReducedMotion } from '@/lib/motion';
import { useSite } from '@/lib/site';
import { ScrollTrigger, scrollToId, useSmoothScroll } from '@/lib/smooth';
import Cnn from '@/sections/Cnn';
import Crops from '@/sections/Crops';
import Footer from '@/sections/Footer';
import Hero from '@/sections/Hero';
import Pipeline from '@/sections/Pipeline';
import Problem from '@/sections/Problem';
import Results from '@/sections/Results';
import Run from '@/sections/Run';
import System from '@/sections/System';

export default function Landing() {
  const site = useSite();
  const reduced = useReducedMotion();
  useSmoothScroll(!reduced);
  useEffect(() => {
    if (!site) return;
    const id = window.setTimeout(() => {
      ScrollTrigger.refresh();
      const hash = window.location.hash.slice(1);
      if (hash) scrollToId(hash);
    }, 50);
    return () => window.clearTimeout(id);
  }, [site]);

  if (!site) return <main className="min-h-dvh bg-ink" aria-busy="true" />;
  return (
    <>
      <SectionRail />
      <main id="top">
        <Hero />
        <Problem />
        <Crops site={site} />
        <Pipeline site={site} />
        <Cnn site={site} />
        <Results site={site} />
        <System site={site} />
        <Run site={site} />
      </main>
      <Footer />
    </>
  );
}
