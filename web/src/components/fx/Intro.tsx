import { useEffect, useState } from 'react';
import { BRAND_NAME } from '@/content/brand';

const KEY = 'patra:intro';

/**
 * Intro curtain for the first page of a visit: the leaf mark draws itself, then the sheet lifts off the page
 * (about two seconds, CSS only, never blocks input, hidden for reduced motion). It sits over a page that is
 * already rendering underneath, so nothing waits for it.
 */
export default function Intro() {
  const [show, setShow] = useState(() => {
    try {
      return !sessionStorage.getItem(KEY);
    } catch {
      return false;
    }
  });
  useEffect(() => {
    try {
      sessionStorage.setItem(KEY, '1');
    } catch {
      /* private mode: the curtain simply shows again next time */
    }
  }, []);
  if (!show) return null;
  return (
    <div className="curtain" aria-hidden="true" data-testid="intro"
         onAnimationEnd={(e) => e.animationName === 'curtainLift' && setShow(false)}>
      <svg width="54" height="54" viewBox="0 0 26 26" fill="none" className="text-leaf">
        <path d="M4 22C4 11.5 10 4.5 22 4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" pathLength={1} />
        <path d="M4 22C14.5 22 21.5 16 22 4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" pathLength={1} />
        <path d="M4 22L18 8" stroke="currentColor" strokeWidth="0.9" strokeLinecap="round" pathLength={1} />
      </svg>
      <span className="curtain-word font-playfair text-[30px] italic text-white">{BRAND_NAME}</span>
      <span className="curtain-bar" />
    </div>
  );
}
