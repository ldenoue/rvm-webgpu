import { defineConfig } from 'vite';
import typegpu from 'unplugin-typegpu/vite';

const isolationHeaders = {
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Opener-Policy': 'same-origin',
};

export default defineConfig({
  base: '/rvm-webgpu/',
  plugins: [typegpu()],
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
});
