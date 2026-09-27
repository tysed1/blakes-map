// Multi-camera QA: load the viewer once, render several Blender cameras, write PNGs.
// usage: node tools/qa/shots_multi.mjs outdir URL CAM1,CAM2,... [--w 960 --h 540]
//   (use a static build: `npx vite build --outDir X && npx vite preview --outDir X --port 4173`;
//    the dev server's hot reload interrupts long software-GL renders)
import { chromium } from 'playwright';
import fs from 'fs';
let [outdir, url, camList] = process.argv.slice(2);
// '@golden' = every shot in tools/qa/golden_shots.json; '@golden:A,B' = a subset of it
let extra = null;
if (camList.startsWith('@golden')) {
  extra = JSON.parse(fs.readFileSync(new URL('./golden_shots.json', import.meta.url)));
  camList = camList.includes(':') ? camList.split(':')[1] : Object.keys(extra).join(',');
}
const a = process.argv.slice(2);
const get = (k, d) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
fs.mkdirSync(outdir, { recursive: true });
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome', args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: +get('--w', 960), height: +get('--h', 540) } });
const logs = [];
page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));
page.setDefaultTimeout(1800000);
await page.goto(url);
await page.waitForFunction(() => { const w = window.w3d; return w && w.vegReady && w.camsReady && w.infraReady !== false && (w.gcReady ?? true) && w.backdropReady; }, null, { timeout: 600000 });
await page.waitForTimeout(5000);
if (extra) await page.evaluate((e) => { Object.assign(window.w3d.cams, e); }, extra);
for (const cam of camList.split(',')) {
  const t0 = Date.now();
  const data = await page.evaluate((cam) => {
    const w = window.w3d; w.stop();
    if (cam === 'hero') w.heroView(); else w.setCam(cam);
    w.updateCulling();
    return w.snapshot();
  }, cam);
  fs.writeFileSync(`${outdir}/${cam}.png`, Buffer.from(data.split(',')[1], 'base64'));
  console.log(cam, ((Date.now() - t0) / 1000).toFixed(0) + 's');
}
for (const l of logs.filter((l) => /error|warn/i.test(l)).slice(0, 20)) console.log(l.slice(0, 600));
await browser.close();
