import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { IMG_W, IMG_H, pxToLatLng, latLngToPx, MPP } from '../../core/coords';
import { World, assetUrl, xyz, xy } from '../../core/data';
import { bus } from '../../core/bus';

export type LayerKey = 'base' | 'terrain' | 'landuse' | 'water' | 'roads' | 'rail' | 'bridges' | 'junctions' | 'regions' | 'labels' | 'landmarks' | 'qa';

const ll = (x: number, y: number) => L.latLng(pxToLatLng(x, y));

export class Map2D {
  map: L.Map;
  layers: Record<LayerKey, L.LayerGroup | L.ImageOverlay>;
  private baseImg: L.ImageOverlay;
  private roadLines: { line: L.Polyline; casing: L.Polyline; wpx: number; z: number }[] = [];
  private renderer = L.canvas({ padding: 0.5, tolerance: 4 });
  private highlight = L.layerGroup();
  private debugMode = false;

  constructor(el: HTMLElement, private w: World) {
    const bounds = L.latLngBounds(ll(0, IMG_H), ll(IMG_W, 0));
    this.map = L.map(el, {
      crs: L.CRS.Simple, minZoom: -2, maxZoom: 5, zoomSnap: 0.25, zoomDelta: 0.5, wheelPxPerZoomLevel: 90,
      maxBounds: bounds.pad(0.25), preferCanvas: true, attributionControl: false, zoomControl: false,
    });
    L.control.zoom({ position: 'bottomright' }).addTo(this.map);
    L.control.scale({ position: 'bottomleft', imperial: false, metric: true, maxWidth: 160 }).addTo(this.map);
    // scale control assumes metres per map unit = 1; patch to our metres-per-pixel
    (L.Control.Scale.prototype as any)._update = function (this: any) {
      const m = this._map; const y = m.getSize().y / 2;
      const maxMeters = m.containerPointToLatLng([0, y]).distanceTo(m.containerPointToLatLng([this.options.maxWidth, y])) * MPP;
      this._updateScales(maxMeters);
    };
    this.map.fitBounds(bounds);

    this.baseImg = L.imageOverlay(assetUrl('base_map.png'), bounds, { opacity: 1, className: 'img-base' });
    const terrain = L.imageOverlay(assetUrl('hillshade.png'), bounds, { opacity: 1, className: 'img-hillshade' });
    this.layers = {
      base: this.baseImg, terrain, landuse: this.landuseLayer(), water: this.waterLayer(), roads: this.roadLayer(),
      rail: this.railLayer(), bridges: this.bridgeLayer(), junctions: this.junctionLayer(), regions: this.regionLayer(),
      labels: this.labelLayer(), landmarks: this.landmarkLayer(), qa: this.qaLayer(),
    };
    for (const k of ['base', 'water', 'roads', 'rail', 'bridges', 'labels', 'landmarks'] as LayerKey[]) this.layers[k].addTo(this.map);
    this.highlight.addTo(this.map);
    this.map.on('zoomend', () => this.restyle());
    this.restyle();
    this.map.on('mousemove', (e: L.LeafletMouseEvent) => {
      const p = latLngToPx(e.latlng.lat, e.latlng.lng);
      bus.cursor.emit({ ...p, z: this.w.terrain.at(p.x, p.y) });
    });
    this.map.on('mouseout', () => bus.cursor.emit(null));
    this.map.on('click', (e: L.LeafletMouseEvent) => {
      if ((e.originalEvent as any)._handled) return;
      const p = latLngToPx(e.latlng.lat, e.latlng.lng);
      this.pickLanduse(p.x, p.y);
    });
    bus.select.on((s) => this.showHighlight(s));
  }

  setLayer(k: LayerKey, on: boolean) {
    const l = this.layers[k];
    if (on) l.addTo(this.map); else l.remove();
    if (k === 'labels' || k === 'junctions') this.restyle();
  }

  setBaseOpacity(o: number) { this.baseImg.setOpacity(o); }

  /** Debug/alignment mode: vectors drawn as thin outlines over the full-opacity source map. */
  setDebug(on: boolean) { this.debugMode = on; this.restyle(); }

  focus(x: number, y: number, r = 60) {
    this.map.flyToBounds(L.latLngBounds(ll(x - r, y + r), ll(x + r, y - r)), { duration: 0.8 });
  }

  centerPx() { const c = this.map.getCenter(); return latLngToPx(c.lat, c.lng); }

  invalidate() { this.map.invalidateSize(); }

  // ---------------------------------------------------------------- layers
  private roadLayer() {
    const g = L.layerGroup();
    const types = this.w.manifest.road_types;
    const roads = [...this.w.roads].filter((r) => !r.virtual).sort((a, b) => types[a.type].z - types[b.type].z);
    const casings = L.layerGroup().addTo(g);
    const fills = L.layerGroup().addTo(g);
    for (const r of roads) {
      const t = types[r.type];
      const pts = xyz(r.c).map(([x, y]) => ll(x, y));
      const opts = { renderer: this.renderer, interactive: true, lineCap: 'round' as const, lineJoin: 'round' as const };
      const casing = L.polyline(pts, { ...opts, color: '#1b1d1f', opacity: 0.85, interactive: false });
      const line = L.polyline(pts, { ...opts, color: t.color, opacity: 1, dashArray: ['gravel', 'dirt', 'driveway'].includes(r.type) ? undefined : undefined });
      line.on('click', (e) => { (e.originalEvent as any)._handled = true; bus.select.emit({ kind: 'road', id: r.id, data: r, at: latLngToPx(e.latlng.lat, e.latlng.lng) }); });
      line.bindTooltip(() => `${r.name ?? t.label}<br><span class="tt-sub">${r.id} · ${t.label}</span>`, { sticky: true, className: 'tt' });
      casing.addTo(casings); line.addTo(fills);
      this.roadLines.push({ line, casing, wpx: r.width_m / MPP, z: t.z });
    }
    return g;
  }

  private restyle() {
    const z = this.map.getZoom();
    const s = Math.pow(2, z); // screen px per source px
    for (const r of this.roadLines) {
      const w = Math.max(r.z >= 5 ? 1.6 : r.z >= 3 ? 1.1 : 0.8, r.wpx * s);
      if (this.debugMode) {
        r.line.setStyle({ weight: Math.max(1.2, Math.min(2.5, w * 0.25)), opacity: 1 });
        r.casing.setStyle({ weight: 0, opacity: 0 });
      } else {
        r.line.setStyle({ weight: w, opacity: 1 });
        r.casing.setStyle({ weight: w + Math.max(1.2, w * 0.35), opacity: z < -0.5 && r.z < 3 ? 0 : 0.85 });
      }
    }
    const el = this.map.getContainer();
    el.classList.toggle('zoom-far', z < 0);
    el.classList.toggle('zoom-mid', z >= 0 && z < 2);
    el.classList.toggle('zoom-near', z >= 2);
    el.classList.toggle('debug', this.debugMode);
  }

  private waterLayer() {
    const g = L.layerGroup();
    for (const b of this.w.water.bodies) {
      for (const poly of b.polys) {
        const rings = (poly as number[][]).map((r) => xy(r).map(([x, y]) => ll(x, y)));
        L.polygon(rings, { renderer: this.renderer, color: '#2d5f7c', weight: 1, fillColor: '#4d86a8', fillOpacity: 0.85, interactive: false }).addTo(g);
      }
    }
    for (const l of this.w.water.lines) {
      const pl = L.polyline(xy(l.c).map(([x, y]) => ll(x, y)), { renderer: this.renderer, color: '#9fd3f0', weight: 1.2, opacity: 0.9, dashArray: '2 5' });
      pl.on('click', (e) => { (e.originalEvent as any)._handled = true; bus.select.emit({ kind: 'water', id: l.id, data: l, at: latLngToPx(e.latlng.lat, e.latlng.lng) }); });
      pl.bindTooltip(`${l.name}<br><span class="tt-sub">${l.id} · ${l.class}, flows ${l.flows_into ? 'into ' + l.flows_into : 'off-map'}</span>`, { sticky: true, className: 'tt' });
      pl.addTo(g);
    }
    return g;
  }

  private landuseLayer() {
    const g = L.layerGroup();
    for (const f of this.w.landuse) {
      for (const poly of f.polys) {
        const rings = (poly as number[][]).map((r) => xy(r).map(([x, y]) => ll(x, y)));
        L.polygon(rings, { renderer: this.renderer, stroke: false, fillColor: f.color, fillOpacity: 0.55, interactive: false }).addTo(g);
      }
    }
    return g;
  }

  private pickLanduse(x: number, y: number) {
    const i = Math.floor(y) * IMG_W + Math.floor(x);
    const cls = this.w.manifest.landuse[this.w.landuseRaster[i]];
    const region = this.w.regions.find((r) => pointInPoly(x, y, xy(r.c)));
    bus.select.emit({ kind: 'landuse', id: cls?.name ?? 'none', at: { x, y }, data: { class: cls?.name, region: region?.name, elevation_m: +this.w.terrain.at(x, y).toFixed(1) } });
  }

  private railLayer() {
    const g = L.layerGroup();
    for (const r of this.w.rail) {
      const pts = xyz(r.c).map(([x, y]) => ll(x, y));
      L.polyline(pts, { renderer: this.renderer, color: '#2a2320', weight: 4, opacity: 0.9, interactive: false }).addTo(g);
      const l = L.polyline(pts, { renderer: this.renderer, color: '#d9d2c5', weight: 2, dashArray: '6 6', opacity: 1 });
      l.on('click', (e) => { (e.originalEvent as any)._handled = true; bus.select.emit({ kind: 'rail', id: r.id, data: r, at: latLngToPx(e.latlng.lat, e.latlng.lng) }); });
      l.bindTooltip(`${r.name}<br><span class="tt-sub">${r.id} · ${r.tracks} track(s)</span>`, { sticky: true, className: 'tt' });
      l.addTo(g);
    }
    return g;
  }

  private bridgeLayer() {
    const g = L.layerGroup();
    const col: Record<string, string> = { bridge: '#ffffff', viaduct: '#e0c3ff', overpass: '#ffd36b', culvert: '#8fb8c9', ford: '#c9a27a' };
    for (const b of this.w.bridges) {
      const l = L.polyline(xy(b.c).map(([x, y]) => ll(x, y)), { renderer: this.renderer, color: col[b.kind] ?? '#fff', weight: 3, opacity: 0.95, lineCap: 'butt' });
      l.on('click', (e) => { (e.originalEvent as any)._handled = true; bus.select.emit({ kind: 'bridge', id: b.id, data: b, at: latLngToPx(e.latlng.lat, e.latlng.lng) }); });
      l.bindTooltip(`${b.kind}: ${b.structure}<br><span class="tt-sub">${b.id} · ${b.length_m} m</span>`, { sticky: true, className: 'tt' });
      l.addTo(g);
    }
    return g;
  }

  private junctionLayer() {
    const g = L.layerGroup();
    const col: Record<string, string> = { end: '#ff5a5a', T: '#ffe45a', cross: '#6cff6c', fork: '#5affff', angled: '#ff9f40', merge: '#ff5aff', city: '#ffffff', multi: '#6f7bff', type_change: '#777' };
    for (const n of this.w.nodes) {
      if (n.kind === 'type_change') continue;
      const [x, y] = n.c;
      const m = L.circleMarker(ll(x, y), { renderer: this.renderer, radius: 3, color: '#000', weight: 1, fillColor: col[n.kind] ?? '#fff', fillOpacity: 1 });
      m.on('click', (e) => { (e.originalEvent as any)._handled = true; bus.select.emit({ kind: 'node', id: n.id, data: n, at: { x, y } }); });
      m.bindTooltip(`${n.kind} junction · ${n.degree} legs<br><span class="tt-sub">${n.id}</span>`, { className: 'tt' });
      m.addTo(g);
    }
    return g;
  }

  private regionLayer() {
    const g = L.layerGroup();
    for (const r of this.w.regions) {
      const p = L.polygon(xy(r.c).map(([x, y]) => ll(x, y)), { renderer: this.renderer, color: '#f0e6c8', weight: 1.5, dashArray: '8 6', fill: true, fillOpacity: 0.02, opacity: 0.7 });
      p.on('click', (e) => { (e.originalEvent as any)._handled = true; bus.select.emit({ kind: 'region', id: r.id, data: r, at: latLngToPx(e.latlng.lat, e.latlng.lng) }); });
      p.addTo(g);
    }
    return g;
  }

  private labelLayer() {
    const g = L.layerGroup();
    for (const s of this.w.settlements) {
      const cls = `lbl lbl-${s.kind}`;
      L.marker(ll(s.c[0], s.c[1]), { icon: L.divIcon({ className: cls, html: `<span>${s.name}</span>`, iconSize: [0, 0] }), interactive: false }).addTo(g);
    }
    // road name labels at representative points of named major roads
    const seen = new Set<string>();
    const pri = ['freeway', 'highway', 'arterial', 'main_street', 'collector', 'rural'];
    const named = this.w.roads.filter((r) => r.name && pri.includes(r.type) && !r.virtual && r.length_m > 250).sort((a, b) => b.length_m - a.length_m);
    for (const r of named) {
      const key = `${r.name}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const pts = xyz(r.c);
      const [x, y] = pts[Math.floor(pts.length / 2)];
      const route = r.route && r.route !== r.name ? r.route : r.name;
      const shield = ['freeway', 'highway'].includes(r.type);
      L.marker(ll(x, y), { icon: L.divIcon({ className: `lbl lbl-road ${shield ? 'lbl-shield' : ''} lbl-t-${r.type}`, html: `<span>${shield ? route : r.name}</span>`, iconSize: [0, 0] }), interactive: false }).addTo(g);
    }
    for (const rg of this.w.regions) {
      const pts = xy(rg.c);
      if (!['wilderness', 'transition'].includes(rg.kind)) continue;
      const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length, cy = pts.reduce((a, p) => a + p[1], 0) / pts.length;
      L.marker(ll(cx, cy), { icon: L.divIcon({ className: 'lbl lbl-region', html: `<span>${rg.name}</span>`, iconSize: [0, 0] }), interactive: false }).addTo(g);
    }
    return g;
  }

  private landmarkLayer() {
    const g = L.layerGroup();
    const glyph: Record<string, string> = { arena: '◉', church: '✝', civic: '▣', poi: '◆', waterfall: '≋', natural: '▲', farm: '⌂', pass: '⋀', summit: '▲', rail_yard: '═', interchange: '✳' };
    for (const l of this.w.landmarks) {
      const m = L.marker(ll(l.c[0], l.c[1]), { icon: L.divIcon({ className: `poi poi-${l.kind}`, html: `<b>${glyph[l.kind] ?? '●'}</b><span>${l.name}</span>`, iconSize: [0, 0] }) });
      m.on('click', (e) => { (e.originalEvent as any)._handled = true; bus.select.emit({ kind: 'landmark', id: l.id, data: l, at: { x: l.c[0], y: l.c[1] } }); });
      m.addTo(g);
    }
    return g;
  }

  private qaLayer() {
    const g = L.layerGroup();
    const issues = this.w.qa?.roads?.issues ?? [];
    for (const i of issues) {
      const [x, y] = i.at;
      const m = L.circleMarker(ll(x, y), { renderer: this.renderer, radius: i.severity === 'error' ? 7 : 5, color: i.severity === 'error' ? '#ff3b3b' : '#ffb13b', weight: 2, fill: false });
      m.bindTooltip(`${i.kind}: ${i.msg}`, { className: 'tt' });
      m.on('click', (e) => { (e.originalEvent as any)._handled = true; bus.select.emit({ kind: 'qa', id: i.kind, data: i, at: { x, y } }); });
      m.addTo(g);
    }
    for (const i of this.w.qa?.grades?.issues ?? []) {
      const r = this.w.roads.find((r) => r.id === i.id);
      if (!r) continue;
      const pts = xyz(r.c);
      const [x, y] = pts[Math.floor(pts.length / 2)];
      const m = L.circleMarker(ll(x, y), { renderer: this.renderer, radius: 5, color: '#c77dff', weight: 2, fill: false });
      m.bindTooltip(`${i.kind}: ${i.id} ${JSON.stringify(i)}`, { className: 'tt' });
      m.addTo(g);
    }
    return g;
  }

  private showHighlight(s: any) {
    this.highlight.clearLayers();
    if (!s) return;
    const d = s.data;
    const style = { renderer: this.renderer, color: '#00e5ff', weight: 5, opacity: 0.9, interactive: false };
    if (s.kind === 'bridge' && d.c) {
      L.polyline(xy(d.c).map(([x, y]) => ll(x, y)), style).addTo(this.highlight);
    } else if ((s.kind === 'road' || s.kind === 'rail') && d.c) {
      L.polyline(xyz(d.c).map(([x, y]) => ll(x, y)), style).addTo(this.highlight);
    } else if (s.kind === 'water' && d.c) {
      L.polyline(xy(d.c).map(([x, y]) => ll(x, y)), style).addTo(this.highlight);
    } else if (s.kind === 'region') {
      L.polygon(xy(d.c).map(([x, y]) => ll(x, y)), { ...style, fill: false }).addTo(this.highlight);
    } else if (s.at) {
      L.circleMarker(ll(s.at.x, s.at.y), { ...style, radius: 9, fill: false }).addTo(this.highlight);
    }
  }
}

export function pointInPoly(x: number, y: number, pts: [number, number][]) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
