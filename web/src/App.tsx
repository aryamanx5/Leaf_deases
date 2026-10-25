import { lazy, Suspense } from 'react';
import { Route, Routes } from 'react-router-dom';
import Landing from '@/pages/Landing';

const Analyze = lazy(() => import('@/pages/Analyze'));

export default function App() {
  return (
    <div className="grain min-h-dvh bg-ink text-text">
      <Suspense fallback={<div className="min-h-dvh bg-ink" />}>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/analyze" element={<Analyze />} />
        </Routes>
      </Suspense>
    </div>
  );
}
