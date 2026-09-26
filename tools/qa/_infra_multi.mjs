// multi-camera QA shots in one page load: node tools/qa/_infra_multi.mjs outdir w h CAM1 CAM2 ... (or free:x,y,dist,pitch,yaw)
import { chromium } from 'playwright';
import fs from 'fs';
const [out, W, H, ...cams] = process.argv.slice(2);
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome', args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: +W, height: +H } });
const logs = [];
page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));
await page.goto(process.env.URL || 'http://localhost:5173/#view=3d');
await page.waitForFunction(() => (window).w3d && document.querySelector('#loading')?.style.display === 'none', null, { timeout: 600000 });
await page.waitForFunction(() => (window).w3d?.backdropReady && (window).w3d?.vegReady === true && (window).w3d?.camsReady === true && (window).w3d?.infraReady === true, null, { timeout: 600000 });
await page.waitForTimeout(6000);
for (const c of cams) {
  const t0 = Date.now();
  const url = await page.evaluate((c) => {
    const w = (window).w3d; w.stop();
    if (c.startsWith('free:')) { const [x, y, d, p, yw] = c.slice(5).split(',').map(Number); w.setMode('free'); w.setView(x, y, d, p, yw); }
    else w.setCam(c);
    w.camera.updateMatrixWorld(); w['updateCulling']();
    w.infra?.update(w.camera);
    const st = w.infra?.stats(); console.log('infra stats', c, JSON.stringify(st), 'renderer', JSON.stringify(w.renderer.info.render));
    return w.snapshot();
  }, c);
  const name = c.replace(/[^A-Za-z0-9_,.-]/g, '_');
  fs.writeFileSync(`${out}/${name}.png`, Buffer.from(url.split(',')[1], 'base64'));
  console.log('shot', name, (Date.now() - t0) / 1000, 's');
}
for (const l of logs.filter((l) => /error|infra stats|warn/i.test(l) && !/groundcover|gcGust|'im'/.test(l)).slice(0, 30)) console.log(l.slice(0, 600));
await browser.close();
