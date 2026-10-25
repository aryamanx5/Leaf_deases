import { useEffect, useState } from 'react';

export const SECTIONS = [
  { id: 'field', label: 'Field' },
  { id: 'problem', label: 'Problem' },
  { id: 'crops', label: 'Crops' },
  { id: 'pipeline', label: 'Image processing' },
  { id: 'cnn', label: 'CNN' },
  { id: 'results', label: 'Results' },
  { id: 'system', label: 'System' },
  { id: 'run', label: 'Run' },
] as const;
export type SectionId = (typeof SECTIONS)[number]['id'];

/** The section whose box crosses the viewport's middle line. */
export function useActiveSection(): SectionId {
  const [active, setActive] = useState<SectionId>('field');
  useEffect(() => {
    const els = SECTIONS.map((s) => document.getElementById(s.id)).filter(Boolean) as HTMLElement[];
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) if (e.isIntersecting) setActive(e.target.id as SectionId);
      },
      { rootMargin: '-50% 0px -50% 0px', threshold: 0 },
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);
  return active;
}
