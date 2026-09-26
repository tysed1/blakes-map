import * as THREE from 'three';
import { pxToWorld } from '../../core/coords';
import { World, Road, xyz, Heightfield } from '../../core/data';

/** Surface material category per road type (game-engine material hint in road_types.json). */
export const SURFACE: Record<string, 'hwy' | 'city' | 'local' | 'chip' | 'gravel' | 'dirt'> = {
  freeway: 'hwy', highway: 'hwy', ramp: 'hwy', arterial: 'city', main_street: 'city', collector: 'city', urban_street: 'city',
  residential: 'local', rural: 'chip', gravel: 'gravel', driveway: 'gravel', dirt: 'dirt',
};
const LIFT = 0.12;

type V3 = [number, number, number];

/** Road centreline in world metres, densified so the ribbon follows the graded profile closely. */
let GROUND: Heightfield | null = null;
export function setGround(h: Heightfield) { GROUND = h; }

export function roadWorldPoints(r: Road, spacing = 2): V3[] {
  const p = xyz(r.c).map(([x, y, z]) => pxToWorld(x, y, z) as V3);
  const out: V3[] = [];
  for (let i = 0; i < p.length - 1; i++) {
    const a = p[i], b = p[i + 1];
    const L = Math.hypot(b[0] - a[0], b[2] - a[2]);
    const n = Math.max(1, Math.ceil(L / spacing));
    for (let k = 0; k < n; k++) {
      const t = k / n;
      out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
    }
  }
  out.push(p[p.length - 1]);
  if (GROUND) {
    // never let the ribbon dip under the (graded) ground between profile vertices;
    // bridge decks are always above ground so they are unaffected
    for (const q of out) {
      const g = GROUND.at(q[0] / 2.5 + 1000, q[2] / 2.5 + 333.5) + 0.05;
      if (q[1] < g) q[1] = g;
    }
  }
  return out;
}

/** Ribbon along points; offsets [o0,o1] across (metres, + = right). Returns positions/indices. */
function ribbon(pts: V3[], o0: number, o1: number, lift: number, pos: number[], uv: number[], idx: number[], trim0 = 0, trim1 = 0) {
  const n = pts.length;
  if (n < 2) return 0;
  // cumulative length for trimming + UV
  const s: number[] = [0];
  for (let i = 1; i < n; i++) s.push(s[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][2] - pts[i - 1][2]));
  const L = s[n - 1];
  if (L - trim0 - trim1 < 0.5) return 0;
  const base = pos.length / 3;
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (s[i] < trim0 - 1e-6 && !(i < n - 1 && s[i + 1] > trim0)) continue;
    if (s[i] > L - trim1 + 1e-6 && !(i > 0 && s[i - 1] < L - trim1)) continue;
    let P = pts[i];
    // clamp trimmed ends onto the trim positions
    if (s[i] < trim0 && i < n - 1) { const t = (trim0 - s[i]) / (s[i + 1] - s[i]); P = lerp3(pts[i], pts[i + 1], t); }
    if (s[i] > L - trim1 && i > 0) { const t = (L - trim1 - s[i - 1]) / (s[i] - s[i - 1]); P = lerp3(pts[i - 1], pts[i], t); }
    const a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)];
    let dx = b[0] - a[0], dz = b[2] - a[2];
    const dl = Math.hypot(dx, dz) || 1;
    dx /= dl; dz /= dl;
    const nx = -dz, nz = dx; // right-hand normal in XZ (x east, z south)
    pos.push(P[0] + nx * o0, P[1] + lift, P[2] + nz * o0, P[0] + nx * o1, P[1] + lift, P[2] + nz * o1);
    const v = Math.min(Math.max(s[i], trim0), L - trim1);
    uv.push(0, v / 8, 1, v / 8);
    if (count > 0) {
      const k = base + (count - 1) * 2;
      idx.push(k, k + 1, k + 2, k + 1, k + 3, k + 2); // upward-facing
    }
    count++;
  }
  return count;
}
const lerp3 = (a: V3, b: V3, t: number): V3 => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

export interface RoadMeshes { group: THREE.Group; pick: Map<THREE.Mesh, string[]> }

class Buf {
  pos: number[] = []; uv: number[] = []; idx: number[] = []; owner: string[] = [];
  add(id: string, f: (pos: number[], uv: number[], idx: number[]) => void) {
    const t0 = this.idx.length;
    f(this.pos, this.uv, this.idx);
    for (let k = t0; k < this.idx.length; k += 3) this.owner.push(id);
  }
  mesh(mat: THREE.Material, name: string) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uv, 2));
    g.setIndex(this.idx);
    g.computeVertexNormals();
    g.computeBoundingSphere();
    const m = new THREE.Mesh(g, mat);
    m.name = name;
    m.receiveShadow = true;
    return m;
  }
}

export function buildRoads(w: World, mats: Record<string, THREE.Material>): RoadMeshes {
  const types = w.manifest.road_types;
  const group = new THREE.Group();
  group.name = 'roads';
  const pick = new Map<THREE.Mesh, string[]>();
  const bufs: Record<string, Buf> = {};
  const B = (k: string) => (bufs[k] ??= new Buf());
  // junction radius per node = largest half width of its legs
  const nodeR = new Map<string, number>();
  const nodeMat = new Map<string, { rank: number; surf: string }>();
  const rank = (t: string) => types[t].z;
  for (const r of w.roads) {
    if (r.virtual) continue;
    for (const n of [r.from, r.to]) {
      nodeR.set(n, Math.max(nodeR.get(n) ?? 0, r.width_m / 2));
      const cur = nodeMat.get(n);
      if (!cur || rank(r.type) > cur.rank) nodeMat.set(n, { rank: rank(r.type), surf: SURFACE[r.type] });
    }
  }
  const nodeById = new Map(w.nodes.map((n) => [n.id, n]));
  for (const r of w.roads) {
    if (r.virtual) continue;
    const t = types[r.type];
    const pts = roadWorldPoints(r);
    const hw = r.width_m / 2;
    const surf = SURFACE[r.type];
    const lift = LIFT + t.z * 0.004;
    B(surf).add(r.id, (p, u, i) => ribbon(pts, -hw, hw, lift, p, u, i));
    // markings (trimmed back from junction pads)
    const jn = (id: string) => (nodeById.get(id)?.degree ?? 1) >= 3;
    const tr0 = jn(r.from) ? (nodeR.get(r.from) ?? 0) + 1 : 0, tr1 = jn(r.to) ? (nodeR.get(r.to) ?? 0) + 1 : 0;
    const ml = lift + 0.02;
    if (['freeway'].includes(r.type)) {
      for (const o of [-0.6, 0.6]) B('yellow').add(r.id, (p, u, i) => ribbon(pts, o - 0.08, o + 0.08, ml, p, u, i, tr0, tr1));
      for (const o of [-hw + 1.2, hw - 1.2]) B('white').add(r.id, (p, u, i) => ribbon(pts, o - 0.1, o + 0.1, ml, p, u, i, tr0, tr1));
      B('median').add(r.id, (p, u, i) => ribbon(pts, -0.5, 0.5, ml - 0.01, p, u, i, tr0, tr1));
    } else if (['highway', 'arterial', 'main_street'].includes(r.type)) {
      for (const o of [-0.12, 0.12]) B('yellow').add(r.id, (p, u, i) => ribbon(pts, o - 0.06, o + 0.06, ml, p, u, i, tr0, tr1));
      if (r.type === 'highway') for (const o of [-hw + 0.9, hw - 0.9]) B('white').add(r.id, (p, u, i) => ribbon(pts, o - 0.08, o + 0.08, ml, p, u, i, tr0, tr1));
    } else if (['collector', 'urban_street', 'rural', 'ramp'].includes(r.type)) {
      if (r.type === 'ramp') for (const o of [-hw + 0.5, hw - 0.5]) B('white').add(r.id, (p, u, i) => ribbon(pts, o - 0.08, o + 0.08, ml, p, u, i, tr0, tr1));
      else B('yellow').add(r.id, (p, u, i) => ribbon(pts, -0.06, 0.06, ml, p, u, i, tr0, tr1));
    }
  }
  // junction pads: filled polygon joining leg ends (clean intersections, no z-fighting overlaps)
  for (const n of w.nodes) {
    if (n.degree < 3 || n.kind === 'merge') continue;
    const R = (nodeR.get(n.id) ?? 3) * 1.18 + 0.6;
    const [x, y, z0] = [n.c[0], n.c[1], 0];
    const legs = w.roads.filter((r) => !r.virtual && (r.from === n.id || r.to === n.id));
    if (!legs.length) continue;
    const zc = legs.map((r) => (r.from === n.id ? r.c[2] : r.c[r.c.length - 1])).reduce((a, b) => a + b, 0) / legs.length || z0;
    const [X, , Z] = pxToWorld(x, y);
    const surf = nodeMat.get(n.id)!.surf;
    B(surf).add(legs[0].id, (p, u, i) => {
      const base = p.length / 3;
      const lift = LIFT + 0.05;
      p.push(X, zc + lift, Z); u.push(0.5, 0.5);
      const seg = 20;
      for (let k = 0; k <= seg; k++) {
        const a = (k / seg) * Math.PI * 2;
        p.push(X + Math.cos(a) * R, zc + lift - 0.02, Z + Math.sin(a) * R);
        u.push(0.5 + Math.cos(a) * 0.5, 0.5 + Math.sin(a) * 0.5);
      }
      for (let k = 0; k < seg; k++) i.push(base, base + 1 + k + 1, base + 1 + k);
    });
  }
  for (const [k, b] of Object.entries(bufs)) {
    if (!b.idx.length) continue;
    const m = b.mesh(mats[k], `roads_${k}`);
    m.userData.kind = 'road';
    pick.set(m, b.owner);
    group.add(m);
  }
  return { group, pick };
}

/** Bridge / viaduct / overpass structures: deck slab, parapets, piers down to terrain. */
export function buildBridges(w: World, hf: Heightfield, mats: { concrete: THREE.Material; steel: THREE.Material }): { group: THREE.Group; pick: Map<THREE.Mesh, string[]> } {
  const group = new THREE.Group();
  group.name = 'bridges';
  const roads = new Map(w.roads.map((r) => [r.id, r]));
  const conc = new Buf(), steel = new Buf();
  const box = (buf: Buf, id: string, c: V3, dx: V3, dz: V3, h: number) => buf.add(id, (p, u, i) => {
    // oriented box centred at c (top face at c.y), half-axes dx, dz in XZ, height h downward
    const base = p.length / 3;
    const corners: V3[] = [];
    for (const [sx, sz] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) corners.push([c[0] + dx[0] * sx + dz[0] * sz, c[1], c[2] + dx[2] * sx + dz[2] * sz]);
    for (const q of corners) { p.push(q[0], q[1], q[2]); u.push(0, 0); }
    for (const q of corners) { p.push(q[0], q[1] - h, q[2]); u.push(0, 1); }
    const f = [[0, 1, 2, 3], [7, 6, 5, 4], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]];
    for (const [a, b, cc, d] of f) i.push(base + a, base + b, base + cc, base + a, base + cc, base + d);
  });
  for (const b of w.bridges) {
    if (!['bridge', 'viaduct', 'overpass'].includes(b.kind)) continue;
    const r = roads.get(b.road);
    if (!r) continue;
    const rp = roadWorldPoints(r, 2);
    const bp = xyz(toXYZ(b.c)).map(([x, y]) => pxToWorld(x, y));  // bridge polylines are stored as 2D px
    // deck points = road points nearest to the bridge polyline span
    const first = nearestIdx(rp, bp[0]), last = nearestIdx(rp, bp[bp.length - 1]);
    const [i0, i1] = first < last ? [first, last] : [last, first];
    const deck = rp.slice(Math.max(0, i0 - 1), Math.min(rp.length, i1 + 2));
    if (deck.length < 2) continue;
    const hw = r.width_m / 2 + 0.6;
    const isSteel = /steel|truss/.test(b.structure);
    const buf = isSteel ? steel : conc;
    // slab
    buf.add(b.id, (p, u, i) => {
      ribbon(deck, -hw, hw, LIFT - 0.02, p, u, i);
      // underside
      const base = p.length / 3;
      for (const q of deck) {
        const k = deck.indexOf(q);
        const a = deck[Math.max(0, k - 1)], c = deck[Math.min(deck.length - 1, k + 1)];
        let dx = c[0] - a[0], dz = c[2] - a[2]; const dl = Math.hypot(dx, dz) || 1; dx /= dl; dz /= dl;
        p.push(q[0] - dz * hw, q[1] - 1.3, q[2] + dx * hw, q[0] + dz * hw, q[1] - 1.3, q[2] - dx * hw); u.push(0, 0, 1, 0);
      }
      for (let k = 0; k < deck.length - 1; k++) { const a = base + k * 2; i.push(a, a + 2, a + 1, a + 1, a + 2, a + 3); }
    });
    // parapets
    for (const s of [-1, 1]) buf.add(b.id, (p, u, i) => ribbonWall(deck, s * hw, 1.0, p, u, i));
    // piers
    const L = deck.length;
    const spacing = b.kind === 'overpass' ? 1e9 : isSteel ? 30 : 24;
    let acc = 0;
    for (let k = 1; k < L - 1; k++) {
      acc += Math.hypot(deck[k][0] - deck[k - 1][0], deck[k][2] - deck[k - 1][2]);
      if (acc < spacing) continue;
      acc = 0;
      const q = deck[k];
      const px = q[0] / 2.5 + 1000, py = q[2] / 2.5 + 333.5;
      const ground = Math.min(hf.at(px, py), w.waterLevel.at(px, py) - 2);
      const h = q[1] - 1.3 - ground;
      if (h < 1.5) continue;
      const a = deck[k - 1], c = deck[k + 1];
      let dx = c[0] - a[0], dz = c[2] - a[2]; const dl = Math.hypot(dx, dz) || 1; dx /= dl; dz /= dl;
      box(conc, b.id, [q[0], q[1] - 1.3, q[2]], [dx * 0.9, 0, dz * 0.9], [-dz * (hw - 1), 0, dx * (hw - 1)], h + 1);
    }
    // abutment walls for overpasses
  }
  const pick = new Map<THREE.Mesh, string[]>();
  for (const [buf, mat, name] of [[conc, mats.concrete, 'concrete'], [steel, mats.steel, 'steel']] as const) {
    if (!buf.idx.length) continue;
    const m = buf.mesh(mat, `bridges_${name}`);
    m.castShadow = true;
    m.userData.kind = 'bridge';
    pick.set(m, buf.owner);
    group.add(m);
  }
  return { group, pick };
}

function toXYZ(c: number[]): number[] {
  const o: number[] = [];
  for (let i = 0; i < c.length; i += 2) o.push(c[i], c[i + 1], 0);
  return o;
}
function nearestIdx(pts: V3[], q: V3) {
  let best = 0, bd = Infinity;
  for (let i = 0; i < pts.length; i++) {
    const d = (pts[i][0] - q[0]) ** 2 + (pts[i][2] - q[2]) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

function ribbonWall(pts: V3[], off: number, h: number, pos: number[], uv: number[], idx: number[]) {
  const base = pos.length / 3;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[Math.max(0, i - 1)], b = pts[Math.min(pts.length - 1, i + 1)];
    let dx = b[0] - a[0], dz = b[2] - a[2]; const dl = Math.hypot(dx, dz) || 1; dx /= dl; dz /= dl;
    const x = pts[i][0] - dz * off, z = pts[i][2] + dx * off;
    pos.push(x, pts[i][1] + LIFT, z, x, pts[i][1] + LIFT + h, z); uv.push(0, 0, 0, 1);
  }
  for (let k = 0; k < pts.length - 1; k++) { const a = base + k * 2; idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3, a, a + 1, a + 2, a + 1, a + 3, a + 2); }
}

/** Railways: ballast bed + steel rails per track. */
export function buildRail(w: World, mats: { ballast: THREE.Material; rail: THREE.Material; tie: THREE.Material }) {
  const group = new THREE.Group();
  group.name = 'rail';
  const bal = new Buf(), rail = new Buf(), ties = new Buf();
  for (const r of w.rail) {
    const pts = xyz(r.c).map(([x, y, z]) => pxToWorld(x, y, z) as V3);
    const tracks = r.tracks as number;
    const hw = (r.width_m as number) / 2;
    bal.add(r.id, (p, u, i) => ribbon(pts, -hw, hw, 0.15, p, u, i));
    for (let t = 0; t < tracks; t++) {
      const c = (t - (tracks - 1) / 2) * 4.0;
      ties.add(r.id, (p, u, i) => ribbon(pts, c - 1.3, c + 1.3, 0.22, p, u, i));
      for (const s of [-0.72, 0.72]) rail.add(r.id, (p, u, i) => ribbon(pts, c + s - 0.05, c + s + 0.05, 0.36, p, u, i));
    }
  }
  const pick = new Map<THREE.Mesh, string[]>();
  for (const [b, m, n] of [[bal, mats.ballast, 'ballast'], [ties, mats.tie, 'ties'], [rail, mats.rail, 'rails']] as const) {
    const mesh = b.mesh(m, `rail_${n}`);
    mesh.userData.kind = 'rail';
    pick.set(mesh, b.owner);
    group.add(mesh);
  }
  return { group, pick };
}

/** Highlight ribbon for a selected road/rail/bridge. */
export function highlightMesh(pts: V3[], width: number) {
  const pos: number[] = [], uv: number[] = [], idx: number[] = [];
  ribbon(pts, -width / 2 - 1, width / 2 + 1, LIFT + 0.3, pos, uv, idx);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  return new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.55, depthWrite: false }));
}
