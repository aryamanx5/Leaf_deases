/**
 * Atmosphere footage (web/public/media/film). Stock clips under the Pexels licence, credited where they play and
 * in docs/LICENSES.md. None of them is a dataset photo, and none shows a diagnosis: they set the scene only.
 */
import { asset } from '@/lib/asset';

export type Clip = {
  hd: string;
  sd: string;
  poster: { small: string; large: string };
  /** What the footage really shows (used as the poster's alt text). */
  alt: string;
  by: string;
  url: string;
};

const clip = (name: string, alt: string, by: string, url: string): Clip => ({
  hd: asset(`/media/film/${name}-720.mp4`),
  sd: asset(`/media/film/${name}-540.mp4`),
  poster: { small: asset(`/media/film/${name}-poster-640.webp`), large: asset(`/media/film/${name}-poster-1280.webp`) },
  alt,
  by,
  url,
});

export const FILM = {
  field: clip('field', 'A wide field of ginger, a close relative of turmeric, moving in the wind.', 'vijay kumar',
    'https://www.pexels.com/video/lush-ginger-field-in-rural-setting-39172117/'),
  hand: clip('hand', 'A hand brushing along tall green leaves in a field.', 'Feyza Yıldırım',
    'https://www.pexels.com/video/woman-touching-leaves-14035105/'),
  dew: clip('dew', 'Green leaves beaded with water drops, swaying.', 'Nothing Ahead',
    'https://www.pexels.com/video/close-up-of-dewy-green-leaves-in-garden-setting-34796496/'),
  veins: clip('veins', 'A leaf held against the sun: its net of veins glows, and patches of dead tissue stay dark.', 'Bil Hinton',
    'https://www.pexels.com/video/close-up-of-sunlit-leaf-macro-view-40382375/'),
  macro: clip('macro', 'The midrib and fine veins of a bright green leaf, very close up.', 'CW Tan',
    'https://www.pexels.com/video/extreme-close-up-of-a-green-leaf-15783159/'),
  rain: clip('rain', 'Rain falling on broad banana leaves.', 'Srinivas Kashyap',
    'https://www.pexels.com/video/tropical-rainfall-on-banana-leaves-in-slow-motion-32089669/'),
  soil: clip('soil', 'Two hands holding soil with a seedling in it.', 'Sergei Skrynnik',
    'https://www.pexels.com/video/close-up-view-of-land-in-hands-11960802/'),
  farmer: clip('farmer', 'A farmer walking through a wide field below hills.', 'Hữu Thịnh 79',
    'https://www.pexels.com/video/tranquil-scene-of-a-farmer-in-lush-paddy-field-39364382/'),
} as const;
export type ClipName = keyof typeof FILM;

/** Save-Data or a 2g connection: keep the poster, never fetch the clip. */
export function clipAllowed(): boolean {
  const c = (navigator as Navigator & { connection?: { saveData?: boolean; effectiveType?: string } }).connection;
  return !(c?.saveData || /2g$/.test(c?.effectiveType ?? ''));
}
