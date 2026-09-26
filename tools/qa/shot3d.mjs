// Single-frame 3D render for QA (software GL is too slow for the live loop).
// usage: node tools/qa/shot3d.mjs out.png "x,y,dist,pitch,yaw" [--mode orbit|top] [--w 1600 --h 900] [--eval js]
import { chromium } from 'playwright';
import fs from 'fs';
const a = process.argv.slice(2);
const get = (k, d) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
const [x, y, dist, pitch, yaw] = (a[1] || '1000,330,1400,0.9,0.6').split(',').map(Number);
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome', args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: +get('--w', 1600), height: +get('--h', 900) } });
const logs = [];
page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));
await page.goto(get('--url', 'http://localhost:5173/#view=3d'));
await page.waitForFunction(() => (window).w3d && document.querySelector('#loading')?.style.display === 'none', null, { timeout: 300000 });
await page.waitForFunction(() => (window).w3d.backdropReady, null, { timeout: 300000 });
await page.waitForFunction(() => (window).w3d.vegReady !== false && (window).w3d.camsReady !== false && (window).w3d.infraReady !== false && ((window).w3d.gcReady ?? true) !== false, null, { timeout: 300000 });
await page.waitForTimeout(4000); // textures
const url = await page.evaluate(([x, y, dist, pitch, yaw, mode, ev, cam]) => {
  const w = (window).w3d; w.stop();
  if (mode) w.setMode(mode);
  if (cam) w.setCam(cam); else if (x < 0) w.heroView(); else w.setView(x, y, dist, pitch, yaw);
  if (ev) eval(ev);
  w.updateCullingPublic?.();
  return w.snapshot();
}, [x, y, dist, pitch, yaw, get('--mode', null), get('--eval', null), get('--cam', null)]);
fs.writeFileSync(a[0], Buffer.from(url.split(',')[1], 'base64'));
for (const l of logs.filter((l) => /error|warn|shader|program|GRADE/i.test(l)).slice(0, 25)) console.log(l.slice(0, 1500));
await browser.close();
