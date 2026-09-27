import { defineConfig, Plugin } from 'vite';
import fs from 'fs';
import path from 'path';

// three's Basis Universal transcoder (KTX2Loader) served at ./basis/ in dev and emitted into the build
const BASIS = path.resolve(__dirname, 'node_modules/three/examples/jsm/libs/basis');
const basisTranscoder = (): Plugin => ({
  name: 'basis-transcoder',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const m = /\/basis\/(basis_transcoder\.(js|wasm))$/.exec(req.url?.split('?')[0] ?? '');
      if (!m) return next();
      res.setHeader('Content-Type', m[2] === 'wasm' ? 'application/wasm' : 'text/javascript');
      fs.createReadStream(path.join(BASIS, m[1])).pipe(res);
    });
  },
  generateBundle() {
    for (const f of ['basis_transcoder.js', 'basis_transcoder.wasm']) this.emitFile({ type: 'asset', fileName: `basis/${f}`, source: fs.readFileSync(path.join(BASIS, f)) });
  },
});

export default defineConfig({
  base: './',
  plugins: [basisTranscoder()],
  build: { chunkSizeWarningLimit: 1200, target: 'es2020' },
  server: { host: true },
});
