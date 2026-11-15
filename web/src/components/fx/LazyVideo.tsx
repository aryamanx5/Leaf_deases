import { useEffect, useRef, useState } from 'react';
import { clipAllowed, FILM, type ClipName } from '@/lib/film';
import { useReducedMotion } from '@/lib/motion';

type Props = {
  clip: ClipName;
  className?: string;
  /** object-position of poster and clip (any CSS position) */
  focus?: string;
  /** short clips with a hard cut at the loop point dip to the poster before restarting */
  softLoop?: boolean;
  credit?: boolean;
  /** false holds the clip on its poster even while it is on screen (e.g. the closed panels of the reel) */
  active?: boolean;
  /** how the poster and clip fill the frame */
  sizes?: string;
};

/**
 * A muted, looping atmosphere clip that costs nothing until it is needed: the poster is a lazy image, the clip is
 * fetched only when the frame comes near the viewport, plays only while on screen and in a visible tab, and is
 * never fetched for reduced motion, Save-Data or 2g (the poster stays).
 */
export default function LazyVideo({ clip, className = '', focus = 'center', softLoop = false, credit = true, active = true, sizes = '(max-width: 768px) 100vw, 60vw' }: Props) {
  const reduced = useReducedMotion();
  const film = FILM[clip];
  const boxRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [shown, setShown] = useState(false);
  const activeRef = useRef(active);
  const resync = useRef<() => void>(() => {});
  useEffect(() => {
    activeRef.current = active;
    resync.current();
  }, [active]);

  useEffect(() => {
    const box = boxRef.current;
    const v = videoRef.current;
    if (reduced || !box || !v || !clipAllowed()) return;
    let loaded = false;
    let nearby = false;
    let inView = false;
    let dip = 0;
    const load = () => {
      if (loaded || !nearby || !activeRef.current) return;
      loaded = true;
      v.src = box.clientWidth * Math.min(2, window.devicePixelRatio || 1) > 1000 ? film.hd : film.sd;
      v.load();
    };
    const sync = () => {
      load(); // a clip that is held on its poster is not fetched until it is first wanted
      if (loaded && inView && activeRef.current && document.visibilityState === 'visible') void v.play().catch(() => {});
      else v.pause();
    };
    resync.current = sync;
    const near = new IntersectionObserver(([e]) => {
      if (!e?.isIntersecting) return;
      nearby = true;
      near.disconnect();
      sync();
    }, { rootMargin: '500px 0px' });
    const visible = new IntersectionObserver(([e]) => {
      inView = !!e?.isIntersecting;
      sync();
    }, { threshold: 0.05 });
    near.observe(box);
    visible.observe(box);
    const onPlaying = () => setShown(true);
    const onTime = () => {
      if (!softLoop || !Number.isFinite(v.duration) || dip) return;
      if (v.duration - v.currentTime < 0.45) {
        setShown(false);
        dip = window.setTimeout(() => {
          v.currentTime = 0;
          dip = 0;
          sync();
        }, 500);
      }
    };
    v.addEventListener('playing', onPlaying);
    v.addEventListener('timeupdate', onTime);
    document.addEventListener('visibilitychange', sync);
    return () => {
      resync.current = () => {};
      near.disconnect();
      visible.disconnect();
      window.clearTimeout(dip);
      v.removeEventListener('playing', onPlaying);
      v.removeEventListener('timeupdate', onTime);
      document.removeEventListener('visibilitychange', sync);
      v.pause();
      v.removeAttribute('src');
      v.load();
    };
  }, [reduced, film, softLoop]);

  return (
    <div ref={boxRef} className={`relative overflow-hidden bg-soil ${className}`}>
      <img src={film.poster.large} srcSet={`${film.poster.small} 640w, ${film.poster.large} 1280w`} sizes={sizes}
           alt={film.alt} width={1280} height={720} loading="lazy" decoding="async"
           className="absolute inset-0 h-full w-full object-cover" style={{ objectPosition: focus }} />
      {!reduced && (
        <video ref={videoRef} muted playsInline loop={!softLoop} preload="none" disablePictureInPicture aria-hidden="true" tabIndex={-1}
               className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-500 ease-out ${shown && active ? 'opacity-100' : 'opacity-0'}`}
               style={{ objectPosition: focus }} />
      )}
      {credit && (
        <a href={film.url} target="_blank" rel="noreferrer"
           className="mono absolute bottom-2 right-2 z-10 rounded-full bg-ink/80 px-2.5 py-1 text-[10px] text-white/80 underline-offset-2 hover:underline">
          Footage: {film.by} / Pexels
        </a>
      )}
    </div>
  );
}
