// Headless screenshots of the planner for visual QA.
// usage: node tools/qa/shot.mjs out.png [hash] [--3d] [--eval "js"] [--wait ms] [--w 1600 --h 900]
import { chromium } from 'playwright';
const a = process.argv.slice(2);
const out = a[0];
const hash = a[1] && !a[1].startsWith('--') ? a[1] : '';
const get = (k, d) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome', args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: +get('--w', 1600), height: +get('--h', 900) } });
const logs = [];
page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));
await page.goto(`http://localhost:5173/${hash ? '#' + hash : ''}`);
await page.waitForFunction(() => document.querySelector('#loading')?.style.display === 'none', null, { timeout: 120000 });
const ev = get('--eval', null);
if (ev) { await page.evaluate(ev); }
await page.waitForTimeout(+get('--wait', 1500));
await page.waitForFunction(() => document.querySelector('#loading')?.style.display === 'none', null, { timeout: 300000 });
await page.waitForTimeout(+get('--wait2', 500));
await page.screenshot({ path: out });
for (const l of logs.filter((l) => /error|warn/i.test(l)).slice(0, 15)) console.log(l);
await browser.close();
