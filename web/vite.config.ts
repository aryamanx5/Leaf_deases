import { fileURLToPath, URL } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  server: {
    port: 5180,
    proxy: { '/api': { target: process.env.VITE_API_PROXY ?? 'http://127.0.0.1:8010', changeOrigin: true } },
  },
  preview: {
    port: 4180,
    proxy: { '/api': { target: process.env.VITE_API_PROXY ?? 'http://127.0.0.1:8010', changeOrigin: true } },
  },
  build: { target: 'es2022', sourcemap: false, chunkSizeWarningLimit: 600 },
});
