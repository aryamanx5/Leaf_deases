import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { lazy, Suspense } from 'react';
import { Route, Routes } from 'react-router-dom';
import Landing from '@/pages/Landing';
const Analyze = lazy(() => import('@/pages/Analyze'));
export default function App() {
    return (_jsx("div", { className: "grain min-h-dvh bg-ink text-text", children: _jsx(Suspense, { fallback: _jsx("div", { className: "min-h-dvh bg-ink" }), children: _jsxs(Routes, { children: [_jsx(Route, { path: "/", element: _jsx(Landing, {}) }), _jsx(Route, { path: "/analyze", element: _jsx(Analyze, {}) })] }) }) }));
}
