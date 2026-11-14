import { useEffect } from 'react';
import SectionRail from '@/components/SectionRail';
import { useReducedMotion } from '@/lib/motion';
import { useSite } from '@/lib/site';
import { ScrollTrigger, scrollToId, useSmoothScroll } from '@/lib/smooth';
import Atlas from '@/sections/Atlas';
import Cnn from '@/sections/Cnn';
import Crops from '@/sections/Crops';
import FieldNotes from '@/sections/FieldNotes';
import Footer from '@/sections/Footer';
import Hero from '@/sections/Hero';
import Interlude from '@/sections/Interlude';
import Journey from '@/sections/Journey';
import OtsuLab from '@/sections/OtsuLab';
import Pipeline from '@/sections/Pipeline';
import Problem from '@/sections/Problem';
import Proof from '@/sections/Proof';
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
        <Interlude />
        <Crops site={site} />
        <Atlas site={site} />
        <Pipeline site={site} />
        <OtsuLab site={site} />
        <Cnn site={site} />
        <Results site={site} />
        <Proof site={site} />
        <System site={site} />
        <Journey site={site} />
        <FieldNotes site={site} />
        <Run site={site} />
      </main>
      <Footer />
    </>
  );
}
