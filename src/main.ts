import './styles.css';
import { loadWorld, World, xyz, xy } from './core/data';
import { bus, Selection } from './core/bus';
import { formatPx, pxToWorld } from './core/coords';
import { Map2D, LayerKey } from './components/map/Map2D';
import type { World3D } from './components/world3d/World3D';

const $ = <T extends HTMLElement = HTMLElement>(s: string) => document.querySelector(s) as T;

let world: World;
let map: Map2D;
let w3d: World3D | null = null;
let view: '2d' | '3d' = '2d';

const LAYERS: { key: LayerKey; label: string; on: boolean; in3d?: string }[] = [
  { key: 'base', label: 'Source map (reference)', on: true },
  { key: 'terrain', label: 'Terrain relief', on: false, in3d: 'terrain' },
  { key: 'landuse', label: 'Land use', on: false },
  { key: 'water', label: 'Water', on: true, in3d: 'water' },
  { key: 'roads', label: 'Roads', on: true, in3d: 'roads' },
  { key: 'bridges', label: 'Bridges / structures', on: true, in3d: 'bridges' },
  { key: 'rail', label: 'Railways', on: true, in3d: 'rail' },
  { key: 'junctions', label: 'Junctions', on: false },
  { key: 'regions', label: 'Region boundaries', on: false },
  { key: 'labels', label: 'Labels', on: true },
  { key: 'landmarks', label: 'Landmarks / POIs', on: true },
  { key: 'qa', label: 'QA issues', on: false },
];

async function main() {
  world = await loadWorld((m) => ($('#loadmsg').textContent = `loading ${m}…`));
  map = new Map2D($('#map2d'), world);
  $('#loading').style.display = 'none';
  buildLayerPanel();
  buildLegends();
  buildRegionSelect();
  buildSearch();
  buildStats();
  bus.select.on(inspect);
  bus.cursor.on((c) => ($('#coords').textContent = c ? `${formatPx(c)}  ·  elev ${c.z?.toFixed(1) ?? '–'} m` : ''));
  document.querySelectorAll<HTMLButtonElement>('#viewToggle button').forEach((b) => (b.onclick = () => switchView(b.dataset.v as any)));
  document.querySelectorAll<HTMLButtonElement>('#camMode button').forEach((b) => (b.onclick = () => {
    document.querySelectorAll('#camMode button').forEach((x) => x.classList.toggle('on', x === b));
    w3d?.setMode(b.dataset.m as any);
  }));
  $<HTMLInputElement>('#debug').onchange = (e) => {
    const on = (e.target as HTMLInputElement).checked;
    map.setDebug(on);
    if (on) { map.setLayer('base', true); ($('#lay-base') as HTMLInputElement).checked = true; }
  };
  $<HTMLInputElement>('#baseOpacity').oninput = (e) => map.setBaseOpacity(+(e.target as HTMLInputElement).value);
  $<HTMLInputElement>('#drape').onchange = (e) => w3d?.setDrape((e.target as HTMLInputElement).checked);
  document.body.classList.add('immersive');
  window.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement || e.key.toLowerCase() !== 'h') return;
    document.body.classList.toggle('immersive');
    requestAnimationFrame(() => { w3d?.resize(); map.invalidate(); });
  });
  const hash = new URLSearchParams(location.hash.slice(1));
  if (hash.get('view') !== '2d') switchView('3d', false);
  if (hash.get('at')) { const [x, y] = hash.get('at')!.split(',').map(Number); focusAt(x, y, 80); }
}

async function ensure3D() {
  if (w3d) return w3d;
  $('#loading').style.display = 'flex';
  $('#loadmsg').textContent = 'building 3D world…';
  await new Promise((r) => setTimeout(r, 30));
  const { World3D } = await import('./components/world3d/World3D');
  w3d = new World3D($('#world3d'), world);
  w3d.onSpeed = (v) => ($('#speed').textContent = `fly speed ×${v.toFixed(1)}`);
  for (const l of LAYERS) if (l.in3d) w3d.setLayer(l.in3d, ($(`#lay-${l.key}`) as HTMLInputElement).checked || l.key === 'terrain');
  $('#loading').style.display = 'none';
  (window as any).w3d = w3d;
  return w3d;
}

async function switchView(v: '2d' | '3d', keepFocus = true) {
  if (v === view) return;
  const c = view === '2d' ? map.centerPx() : w3d!.targetPx();
  view = v;
  document.querySelectorAll('#viewToggle button').forEach((b) => b.classList.toggle('on', (b as HTMLElement).dataset.v === v));
  $('#map2d').classList.toggle('on', v === '2d');
  $('#world3d').classList.toggle('on', v === '3d');
  document.body.classList.toggle('mode3d', v === '3d');
  if (v === '3d') {
    const w = await ensure3D();
    w.resize();
    if (keepFocus) w.setView(c.x, c.y, 900, 0.45, 0.5);
    w.start();
  } else {
    w3d?.stop();
    map.invalidate();
    if (keepFocus) map.map.setView([-c.y, c.x], map.map.getZoom());
  }
}

function focusAt(x: number, y: number, r = 60) {
  if (view === '2d') map.focus(x, y, r); else w3d?.focus(x, y, r);
}
(window as any).focusAt = focusAt;

function buildLayerPanel() {
  const el = $('#layers');
  for (const l of LAYERS) {
    const row = document.createElement('label');
    row.className = 'row';
    row.innerHTML = `<input type="checkbox" id="lay-${l.key}" ${l.on ? 'checked' : ''}/> ${l.label}`;
    row.querySelector('input')!.onchange = (e) => {
      const on = (e.target as HTMLInputElement).checked;
      map.setLayer(l.key, on);
      if (l.in3d && w3d) w3d.setLayer(l.in3d, on || l.key === 'terrain');
    };
    el.appendChild(row);
  }
  const t = document.createElement('label');
  t.className = 'row';
  t.innerHTML = `<input type="checkbox" id="lay-trees" checked/> Vegetation (3D)`;
  t.querySelector('input')!.onchange = (e) => w3d?.setLayer('trees', (e.target as HTMLInputElement).checked);
  el.appendChild(t);
}

function buildLegends() {
  const rt = world.manifest.road_types;
  const count: Record<string, number> = {};
  for (const r of world.roads) if (!r.virtual) count[r.type] = (count[r.type] ?? 0) + r.length_m;
  $('#legendRoads').innerHTML = Object.entries(rt).map(([k, t]) =>
    `<div class="lg"><i style="background:${t.color};height:${Math.max(2, Math.min(8, t.width_m / 2.5))}px"></i>${t.label}<em>${((count[k] ?? 0) / 1000).toFixed(1)} km</em></div>`).join('');
  $('#legendLand').innerHTML = world.manifest.landuse.filter((c) => c.name !== 'none').map((c) => `<div class="lg"><i class="sq" style="background:${c.color}"></i>${c.name.replace('_', ' ')}</div>`).join('');
  const j = [['#ffe45a', 'T junction'], ['#6cff6c', 'crossroads'], ['#5affff', 'fork'], ['#ff9f40', 'angled rural'], ['#ffffff', 'city intersection'], ['#ff5aff', 'ramp merge'], ['#ff5a5a', 'dead end']];
  const s = [['#ffffff', 'bridge'], ['#e0c3ff', 'viaduct'], ['#ffd36b', 'overpass']];
  $('#legendJ').innerHTML = j.map(([c, n]) => `<div class="lg"><i class="dot" style="background:${c}"></i>${n}</div>`).join('') + s.map(([c, n]) => `<div class="lg"><i style="background:${c};height:3px"></i>${n}</div>`).join('');
}

function buildRegionSelect() {
  const sel = $<HTMLSelectElement>('#region');
  sel.innerHTML = `<option value="">Whole world</option>` + world.regions.map((r) => `<option value="${r.id}">${r.name}</option>`).join('');
  sel.onchange = () => {
    if (!sel.value) { focusAt(1000, 333, 1000); return; }
    const r = world.regions.find((r) => r.id === sel.value)!;
    const pts = xy(r.c);
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    focusAt(cx, cy, Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) / 2);
    bus.select.emit({ kind: 'region', id: r.id, data: r });
  };
}

function buildSearch() {
  type Item = { label: string; sub: string; sel: Selection; x: number; y: number };
  const items: Item[] = [];
  const seenNames = new Set<string>();
  for (const r of world.roads) {
    if (r.virtual) continue;
    const pts = xyz(r.c); const [x, y] = pts[Math.floor(pts.length / 2)];
    items.push({ label: r.id, sub: `${r.name ?? ''} ${r.type}`, sel: { kind: 'road', id: r.id, data: r, at: { x, y } }, x, y });
    if (r.name && !seenNames.has(r.name)) { seenNames.add(r.name); items.push({ label: r.name, sub: `${r.type} · ${r.route ?? ''}`, sel: { kind: 'road', id: r.id, data: r, at: { x, y } }, x, y }); }
  }
  for (const s of world.settlements) items.push({ label: s.name, sub: s.kind, sel: { kind: 'settlement', id: s.id, data: s, at: { x: s.c[0], y: s.c[1] } }, x: s.c[0], y: s.c[1] });
  for (const l of world.landmarks) items.push({ label: l.name, sub: l.kind, sel: { kind: 'landmark', id: l.id, data: l, at: { x: l.c[0], y: l.c[1] } }, x: l.c[0], y: l.c[1] });
  for (const w of world.water.lines) { const p = xy(w.c)[Math.floor(w.c.length / 4)]; items.push({ label: w.name, sub: `${w.class} · ${w.id}`, sel: { kind: 'water', id: w.id, data: w, at: { x: p[0], y: p[1] } }, x: p[0], y: p[1] }); }
  for (const b of world.bridges) { const p = [b.c[0], b.c[1]]; items.push({ label: b.id, sub: `${b.kind} · ${b.name ?? ''}`, sel: { kind: 'bridge', id: b.id, data: b, at: { x: p[0], y: p[1] } }, x: p[0], y: p[1] }); }
  for (const n of world.nodes) items.push({ label: n.id, sub: `${n.kind} junction`, sel: { kind: 'node', id: n.id, data: n, at: { x: n.c[0], y: n.c[1] } }, x: n.c[0], y: n.c[1] });
  const inp = $<HTMLInputElement>('#search'), res = $('#results');
  inp.oninput = () => {
    const q = inp.value.trim().toLowerCase();
    if (q.length < 2) { res.innerHTML = ''; return; }
    const hits = items.filter((i) => i.label.toLowerCase().includes(q) || i.sub.toLowerCase().includes(q)).slice(0, 12);
    res.innerHTML = hits.map((h, k) => `<div data-k="${k}"><b>${h.label}</b><span>${h.sub}</span></div>`).join('');
    res.querySelectorAll<HTMLElement>('div').forEach((d) => (d.onclick = () => {
      const h = hits[+d.dataset.k!];
      focusAt(h.x, h.y, 50); bus.select.emit(h.sel); res.innerHTML = ''; inp.blur();
    }));
  };
  inp.onkeydown = (e) => { if (e.key === 'Enter') (res.querySelector('div') as HTMLElement | null)?.click(); if (e.key === 'Escape') { res.innerHTML = ''; inp.value = ''; } };
}

function buildStats() {
  const km = world.roads.filter((r) => !r.virtual).reduce((a, r) => a + r.length_m, 0) / 1000;
  const q = world.qa?.roads?.summary ?? {};
  const errs = Object.entries(q).filter(([k]) => k.endsWith(':error')).reduce((a, [, v]) => a + (v as number), 0);
  $('#stats').innerHTML = `<h3>World</h3>
    <div>5.0 × 1.67 km · 2.5 m/px source grid</div>
    <div>${world.roads.length} road segments · ${km.toFixed(1)} km</div>
    <div>${world.nodes.filter((n) => n.degree >= 3).length} junctions · ${world.bridges.length} structures</div>
    <div>${world.water.lines.length} waterways · ${world.manifest.trees.count.toLocaleString()} trees</div>
    <div>Road QA: ${errs} errors · ${world.qa?.roads?.components ?? '?'} network component(s)</div>
    <div class="muted">Buildings: deferred (next phase)</div>`;
}

const FIELDS: Record<string, string[]> = {
  road: ['id', 'name', 'route', 'type', 'lanes', 'width_m', 'surface', 'material', 'speed_mph', 'length_m', 'max_grade_pct', 'z_range_m', 'max_cut_m', 'max_fill_m', 'oneway', 'interchange', 'min_radius_m', 'from', 'to', 'crossings', 'settlement', 'zone', 'source', 'status', 'def_id'],
  bridge: ['id', 'kind', 'structure', 'road', 'name', 'length_m', 'deck_width_m', 'deck_z_m', 'water_level_m', 'clearance_m', 'waterway_class', 'piers', 'over'],
  rail: ['id', 'name', 'tracks', 'gauge_mm', 'length_m', 'max_grade_pct', 'material'],
  water: ['id', 'name', 'class', 'flows_into', 'outlet', 'source', 'length_px', 'hidden_fraction', 'elev_spec_m'],
  node: ['id', 'kind', 'degree', 'road_types'],
};

function inspect(s: Selection) {
  const el = $('#inspector');
  if (!s) { el.innerHTML = '<div class="empty">Nothing selected.</div>'; return; }
  const d = s.data ?? {};
  const keys = FIELDS[s.kind] ?? Object.keys(d).filter((k) => !['c', 'polys', 'pts'].includes(k));
  const rows = keys.filter((k) => d[k] !== undefined && d[k] !== null && d[k] !== '').map((k) => `<tr><th>${k}</th><td>${fmt(d[k])}</td></tr>`).join('');
  const title = d.name ?? d.id ?? s.id;
  let world3 = '';
  if (s.at) { const [X, , Z] = pxToWorld(s.at.x, s.at.y); world3 = `<div class="muted">px ${s.at.x.toFixed(1)}, ${s.at.y.toFixed(1)} · world X ${X.toFixed(1)} Z ${Z.toFixed(1)}</div>`; }
  const jk = s.kind === 'node' ? `<div class="muted">${world.manifest.junction_kinds[d.kind] ?? ''}</div>` : '';
  el.innerHTML = `<div class="kind">${s.kind}</div><h2>${title}</h2>${jk}${world3}
    <div class="btns"><button id="loc2d">Locate in 2D</button><button id="loc3d">Locate in 3D</button></div>
    <table>${rows}</table>`;
  const at = s.at ?? (d.c ? { x: d.c[0], y: d.c[1] } : null);
  $('#loc2d').onclick = async () => { await switchView('2d', false); if (at) map.focus(at.x, at.y, 40); };
  $('#loc3d').onclick = async () => { await switchView('3d', false); if (at) w3d!.focus(at.x, at.y, 40); };
}
function fmt(v: any): string {
  if (Array.isArray(v)) return v.map((x) => (typeof x === 'number' ? +x.toFixed(2) : Array.isArray(x) ? `[${x.join(', ')}]` : x)).join(', ');
  if (typeof v === 'number') return String(+v.toFixed(2));
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

main().catch((e) => { $('#loadmsg').textContent = 'Failed: ' + e.message; console.error(e); });
