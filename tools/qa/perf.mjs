// Performance probe: fixed fly-through path, per quality preset.
// usage: node tools/qa/perf.mjs URL [--presets low,medium,high,ultra] [--samples 24] [--out perf.json] [--gpu]
//   URL = a static build (npx vite build --outDir /tmp/qa && npx vite preview --outDir /tmp/qa --port 4173)
//   --gpu: real Chrome with the GPU (M-series Mac: frame times are real); default = SwiftShader, where
//   frame times are meaningless and the PROXIES matter: draw calls / triangles (main vs shadow),
//   shader programs, GPU upload bytes (textures + buffers), download bytes, load time, JS cull ms.
import { chromium } from 'playwright';
import fs from 'fs';
const a = process.argv.slice(2);
const url = a[0];
const get = (k, d) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
const presets = get('--presets', 'low,medium,high,ultra').split(',');
const samples = +get('--samples', 24);
const gpu = a.includes('--gpu');
const launch = gpu
  ? { channel: 'chrome', headless: false, args: ['--use-angle=metal', '--ignore-gpu-blocklist'] }
  : { executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome', args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] };
const browser = await chromium.launch(launch);
const page = await browser.newPage({ viewport: { width: +get('--w', 1470), height: +get('--h', 956) } });
page.setDefaultTimeout(3600000);
const logs = [];
page.on('pageerror', (e) => logs.push('pageerror: ' + e.message));
// count GPU uploads (bytes) by wrapping WebGL2 upload calls
await page.addInitScript(() => {
  const g = (window.__gpu = { tex: 0, buf: 0 });
  const P = WebGL2RenderingContext.prototype;
  const bpp = (fmt, type) => (type === 0x1406 ? 16 : type === 0x140b || type === 0x8d61 ? 8 : 4);
  const ts2 = P.texStorage2D; P.texStorage2D = function (t, lv, f, w, h) { let s = 0, W = w, H = h; for (let i = 0; i < lv; i++) { s += W * H; W = Math.max(1, W >> 1); H = Math.max(1, H >> 1); } g.tex += s * (f === 0x881a || f === 0x822f ? 8 : f === 0x8814 ? 16 : 4); return ts2.apply(this, arguments); };
  const ti2 = P.texImage2D; P.texImage2D = function (...x) { if (x.length >= 9 && typeof x[3] === 'number') g.tex += x[3] * x[4] * bpp(x[6], x[7]); else if (x.length === 6 && x[5] && x[5].width) g.tex += x[5].width * x[5].height * 4 * 1.33; return ti2.apply(this, x); };
  const ci2 = P.compressedTexImage2D; P.compressedTexImage2D = function (...x) { const d = x[x.length - 1]; g.tex += d && d.byteLength ? d.byteLength : 0; return ci2.apply(this, x); };
  const bd = P.bufferData; P.bufferData = function (t, d) { g.buf += typeof d === 'number' ? d : d && d.byteLength ? d.byteLength : 0; return bd.apply(this, arguments); };
});
const t0 = Date.now();
await page.goto(url);
await page.waitForFunction(() => { const w = window.w3d; return w && w.vegReady && w.camsReady && w.infraReady !== false && (w.gcReady ?? true) && w.backdropReady; }, null, { timeout: 3600000 });
const loadS = (Date.now() - t0) / 1000;
await page.waitForTimeout(3000);

const result = await page.evaluate(async ({ presets, samples }) => {
  const w = window.w3d; w.stop();
  const T = w.w.terrain;
  const P = (x, y, agl) => [(x - 1000) * 2.5, T.at(x, y) + agl, (y - 333.5) * 2.5];
  // waypoints: [camera px, py, agl], [target px, py, agl]
  const WP = [
    [[1050, 580, 650], [1050, 330, 0]],     // hero valley view
    [[1030, 350, 30], [1085, 330, 5]],      // Hollow Ridge 30 m
    [[1000, 336, 3], [1060, 334, 3]],       // along US 19 at 3 m
    [[848, 380, 300], [700, 360, 0]],       // over Laurel Gap 300 m
    [[1760, 340, 60], [1864, 302, 10]],     // Tannersville riverfront
    [[1000, 800, 1500], [1000, 300, 0]],    // high overview
  ];
  const cr = (p0, p1, p2, p3, t) => p1.map((_, i) => 0.5 * ((2 * p1[i]) + (-p0[i] + p2[i]) * t + (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * t * t + (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * t * t * t));
  const cam = WP.map(([c]) => c), tgt = WP.map(([, t]) => t);
  const at = (arr, u) => { const n = arr.length - 1, s = Math.min(n - 1e-6, u * n), i = Math.floor(s), f = s - i; return cr(arr[Math.max(0, i - 1)], arr[i], arr[i + 1], arr[Math.min(n, i + 2)], f); };
  const out = {};
  const info = w.renderer.info;
  const gl = w.renderer.getContext();
  for (const q of presets) {
    w.setQuality(q); w.autoScale = false; w.renderScale = 1; w['applyScale']?.();
    const frames = [];
    for (let k = 0; k < samples; k++) {
      const u = k / (samples - 1);
      const c = at(cam, u), t = at(tgt, u);
      const p = P(c[0], c[1], c[2]), tp = P(t[0], t[1], t[2]);
      w.camera.position.set(p[0], p[1], p[2]); w.camera.lookAt(tp[0], tp[1], tp[2]); w.camera.updateMatrixWorld();
      w['syncFly']?.();
      const c0 = performance.now(); w['updateCulling'](); const cullMs = performance.now() - c0;
      w.cine.prepare(w.scene);
      info.reset(); const r0 = performance.now(); w.cine.render(); gl.finish(); const renderMs = performance.now() - r0;
      const total = { calls: info.render.calls, tris: info.render.triangles };
      // main pass only (shadow maps not re-rendered)
      const au = w.renderer.shadowMap.autoUpdate; w.renderer.shadowMap.autoUpdate = false;
      info.reset(); w.cine.render(); w.renderer.shadowMap.autoUpdate = au;
      frames.push({ u: +u.toFixed(3), cullMs: +cullMs.toFixed(2), renderMs: +renderMs.toFixed(1), calls: total.calls, tris: total.tris, mainCalls: info.render.calls, mainTris: info.render.triangles });
      await new Promise((r) => setTimeout(r, 0));
    }
    const pct = (arr, p) => { const s = [...arr].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
    const col = (k) => frames.map((f) => f[k]);
    out[q] = {
      frames,
      summary: {
        mainCalls_p50: pct(col('mainCalls'), 0.5), mainCalls_max: Math.max(...col('mainCalls')),
        shadowCalls_p50: pct(frames.map((f) => f.calls - f.mainCalls), 0.5),
        mainTris_p50: pct(col('mainTris'), 0.5), mainTris_max: Math.max(...col('mainTris')),
        allTris_max: Math.max(...col('tris')),
        cullMs_p50: +pct(col('cullMs'), 0.5).toFixed(2), cullMs_max: +Math.max(...col('cullMs')).toFixed(2),
        renderMs_p50: pct(col('renderMs'), 0.5), renderMs_p99: pct(col('renderMs'), 0.99),
        programs: info.programs.length, geometries: info.memory.geometries, textures: info.memory.textures,
      },
    };
  }
  const res = performance.getEntriesByType('resource');
  const dl = res.reduce((s, r) => s + (r.encodedBodySize || r.transferSize || 0), 0);
  return { presets: out, gpuUploadMB: { textures: +(window.__gpu.tex / 1e6).toFixed(1), buffers: +(window.__gpu.buf / 1e6).toFixed(1) }, downloadMB: +(dl / 1e6).toFixed(1), heapMB: performance.memory ? +(performance.memory.usedJSHeapSize / 1e6).toFixed(0) : null };
}, { presets, samples });
result.loadS = loadS; result.mode = gpu ? 'real GPU' : 'SwiftShader (frame times NOT meaningful; use proxies)';
result.errors = logs;
fs.writeFileSync(get('--out', 'perf.json'), JSON.stringify(result, null, 1));
for (const [q, r] of Object.entries(result.presets)) {
  const s = r.summary;
  console.log(`${q.padEnd(6)} main calls p50 ${s.mainCalls_p50} max ${s.mainCalls_max} | shadow calls p50 ${s.shadowCalls_p50} | main tris p50 ${(s.mainTris_p50 / 1e6).toFixed(2)}M max ${(s.mainTris_max / 1e6).toFixed(2)}M | all-pass tris max ${(s.allTris_max / 1e6).toFixed(2)}M | cull ${s.cullMs_p50}/${s.cullMs_max} ms | programs ${s.programs} | render ${s.renderMs_p50}/${s.renderMs_p99} ms`);
}
console.log(`load ${loadS.toFixed(0)} s | download ${result.downloadMB} MB | GPU uploads tex ${result.gpuUploadMB.textures} MB buf ${result.gpuUploadMB.buffers} MB | heap ${result.heapMB} MB | ${result.mode}`);
await browser.close();
