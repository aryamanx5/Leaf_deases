import { AnimatePresence, motion } from 'motion/react';
import { lazy, Suspense } from 'react';
import { useEffect } from 'react';
import { Route, Routes, useLocation } from 'react-router-dom';
import Nav from '@/components/Nav';
import { DEPLOYED_APP_URL } from '@/content/brand';
import { useReducedMotion } from '@/lib/motion';
import Landing from '@/pages/Landing';

const Analyze = lazy(() => import('@/pages/Analyze'));

/** With a hosted analysis app configured, this site's own /analyze page is not used: send visitors there. */
function ToDeployedApp() {
  useEffect(() => {
    window.location.replace(DEPLOYED_APP_URL!);
  }, []);
  return <div className="min-h-dvh bg-ink" />;
}

export default function App() {
  const location = useLocation();
  const reduced = useReducedMotion();
  return (
    <div className="grain min-h-dvh bg-ink text-text">
      <a href="#main-content" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[300] focus:rounded-full focus:bg-paper focus:px-4 focus:py-2 focus:text-ink">
        Skip to content
      </a>
      <Nav />
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={location.pathname}
          id="main-content"
          initial={reduced ? { opacity: 0 } : { clipPath: 'inset(0 0 100% 0)' }}
          animate={reduced ? { opacity: 1 } : { clipPath: 'inset(0 0 0% 0)' }}
          exit={reduced ? { opacity: 0 } : { clipPath: 'inset(100% 0 0 0)' }}
          transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
        >
          <Suspense fallback={<div className="min-h-dvh bg-ink" />}>
            <Routes location={location}>
              <Route path="/" element={<Landing />} />
              <Route path="/analyze" element={DEPLOYED_APP_URL ? <ToDeployedApp /> : <Analyze />} />
            </Routes>
          </Suspense>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
