import { defineConfig } from 'vite';
export default defineConfig({
  base: './',
  build: { chunkSizeWarningLimit: 1200, target: 'es2020' },
  server: { host: true },
});
