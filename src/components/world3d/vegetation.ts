import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { pxToWorld } from '../../core/coords';
import { CHUNK } from './terrain';

/**
 * Web vegetation matching the Blender scene (tools/blender/lib_trees.py): the same 27
 * species, positions and scales (vegetation_f32.bin, written by tools/pipeline/vegetation.py),
 * and lib_trees' early-autumn palettes. Each family has a clumpy near template and a cheap
 * far template; chunks swap between them by camera distance.
 */

// lib_trees.SPECIES order: [family, height m, crown base m, crown radius m]
type Fam = 'broad' | 'pine' | 'hemlock' | 'shrub' | 'snag';
const SPECIES: [string, Fam, string, number, number, number][] = [
  ['WhiteOak_A', 'broad', 'oak', 20, 5.5, 6.8], ['WhiteOak_B', 'broad', 'oak', 17, 4.5, 6.0], ['Oak_OpenGrown', 'broad', 'oak', 15, 3.2, 8.0],
  ['RedMaple_A', 'broad', 'maple', 17, 4.5, 5.2], ['RedMaple_B', 'broad', 'maple', 14, 3.5, 4.6],
  ['TulipPoplar_A', 'broad', 'poplar', 28, 13, 5.0], ['TulipPoplar_B', 'broad', 'poplar', 23, 9, 4.6], ['Hickory_A', 'broad', 'hickory', 21, 8, 5.2],
  ['Dogwood', 'broad', 'dogwood', 7, 1.8, 3.2], ['Sapling_HW_A', 'broad', 'sapling', 7, 2.5, 2.2], ['Sapling_HW_B', 'broad', 'sapling', 9, 3.5, 2.6],
  ['WhitePine_A', 'pine', 'pine', 26, 8, 5.2], ['WhitePine_B', 'pine', 'pine', 21, 4, 4.6], ['Hemlock_A', 'hemlock', 'hemlock', 21, 1.5, 4.4],
  ['Hemlock_B', 'hemlock', 'hemlock', 16, 1.0, 3.8], ['VirginiaPine', 'pine', 'pine', 13, 3, 3.2], ['Sapling_Pine', 'hemlock', 'pine', 5, 0.3, 1.6],
  ['Snag_A', 'snag', 'snag', 15, 0, 1.2], ['Snag_B', 'snag', 'snag', 18, 0, 1.4],
  ['Rhododendron_A', 'shrub', 'rhodo', 3.0, 0, 2.0], ['Laurel_B', 'shrub', 'rhodo', 2.0, 0, 1.4], ['Brush_A', 'shrub', 'brush', 1.6, 0, 1.3], ['Brush_B', 'shrub', 'brush', 1.1, 0, 0.9],
  ['WhiteOak_C', 'broad', 'oak', 18, 5.0, 6.4], ['RedMaple_C', 'broad', 'maple', 16, 4.0, 4.8], ['Hemlock_C', 'hemlock', 'hemlock', 24, 3.0, 4.8], ['WhitePine_C', 'pine', 'pine', 28, 10, 5.6],
];

// lib_trees.PAL (linear albedo, weight). Brightened for the web's single-bounce lighting.
const PAL: Record<string, [number[], number][]> = {
  oak: [[[0.046, 0.074, 0.028], 0.5], [[0.060, 0.080, 0.030], 0.3], [[0.15, 0.10, 0.030], 0.12], [[0.14, 0.062, 0.024], 0.08]],
  maple: [[[0.050, 0.078, 0.030], 0.6], [[0.25, 0.085, 0.020], 0.14], [[0.22, 0.045, 0.018], 0.1], [[0.24, 0.14, 0.028], 0.16]],
  poplar: [[[0.058, 0.088, 0.032], 0.7], [[0.20, 0.15, 0.035], 0.2], [[0.12, 0.105, 0.030], 0.1]],
  hickory: [[[0.062, 0.088, 0.03], 0.65], [[0.23, 0.165, 0.035], 0.35]],
  dogwood: [[[0.06, 0.085, 0.035], 0.45], [[0.19, 0.045, 0.030], 0.55]],
  sapling: [[[0.055, 0.090, 0.030], 0.72], [[0.17, 0.115, 0.03], 0.16], [[0.19, 0.06, 0.02], 0.12]],
  pine: [[[0.034, 0.064, 0.038], 0.6], [[0.042, 0.072, 0.042], 0.4]],
  hemlock: [[[0.026, 0.050, 0.034], 0.7], [[0.032, 0.056, 0.037], 0.3]],
  rhodo: [[[0.030, 0.058, 0.030], 0.8], [[0.040, 0.066, 0.030], 0.2]],
  brush: [[[0.07, 0.09, 0.03], 0.5], [[0.19, 0.07, 0.025], 0.18], [[0.16, 0.115, 0.035], 0.32]],
  snag: [[[0.1, 0.1, 0.1], 1.0]],
};
export const FOLIAGE_GAIN = 1.2;

function hash(n: number) { const s = Math.sin(n * 12.9898 + 78.233) * 43758.5453; return s - Math.floor(s); }

/** Template helpers: every template has color (ambient-occlusion tint) + leaf (0 bark / 1 foliage). */
function finish(g: THREE.BufferGeometry, col: (p: THREE.Vector3, n: THREE.Vector3) => number, leaf: number, normalFn?: (p: THREE.Vector3, n: THREE.Vector3) => void) {
  g = g.index ? g.toNonIndexed() : g;
  const pos = g.getAttribute('position'), nrm = g.getAttribute('normal');
  const n = pos.count, c = new Float32Array(n * 3), l = new Float32Array(n);
  const p = new THREE.Vector3(), v = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    p.fromBufferAttribute(pos, i); v.fromBufferAttribute(nrm, i);
    if (normalFn) { normalFn(p, v); nrm.setXYZ(i, v.x, v.y, v.z); }
    const a = col(p, v);
    c[i * 3] = c[i * 3 + 1] = c[i * 3 + 2] = a; l[i] = leaf;
  }
  if (!leaf) for (let i = 0; i < n; i++) { c[i * 3] *= 0.075 * 2.2; c[i * 3 + 1] *= 0.058 * 2.2; c[i * 3 + 2] *= 0.042 * 2.2; }
  g.setAttribute('color', new THREE.BufferAttribute(c, 3));
  g.setAttribute('leaf', new THREE.BufferAttribute(l, 1));
  return g;
}

function jitter(g: THREE.BufferGeometry, amt: number, seed: number) {
  const pos = g.getAttribute('position');
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const k = hash(Math.round(x * 97) + Math.round(y * 57) * 31 + Math.round(z * 73) * 17 + seed);
    const s = 1 + (k - 0.5) * amt;
    pos.setXYZ(i, x * s, y * s, z * s);
  }
  return g;
}

/** Broadleaf, unit space: height 1, crown radius 1 (instances scale xz by crown r, y by height). */
function broadTemplate(hi: boolean) {
  const cy = 0.64, ry = 0.34; // crown centre / half-height (fraction of height)
  const vr = 0.36; // vertical/horizontal aspect of a lobe in unit space (crown_r / height ~ 0.3)
  const parts: THREE.BufferGeometry[] = [];
  const crownNormal = (p: THREE.Vector3, n: THREE.Vector3, w: number) => {
    const c = new THREE.Vector3(p.x, (p.y - cy) / (ry * ry), p.z).normalize();
    n.multiplyScalar(1 - w).addScaledVector(c, w).normalize();
  };
  const shade = (p: THREE.Vector3) => THREE.MathUtils.clamp(0.42 + 0.75 * ((p.y - (cy - ry)) / (2 * ry)) + 0.25 * Math.hypot(p.x, p.z), 0.35, 1.15);
  if (!hi) {
    const g = new THREE.IcosahedronGeometry(1, 0).scale(0.95, ry, 0.95).translate(0, cy, 0);
    return finish(g, shade, 1, (p, n) => crownNormal(p, n, 1));
  }
  const trunk = new THREE.CylinderGeometry(0.018, 0.03, cy, 5, 1, true).translate(0, cy / 2, 0);
  parts.push(finish(trunk.scale(1, 1, 1), () => 1, 0));
  const lobes = [[0, cy + ry * 0.35, 0, 0.62], [0.45, cy - 0.02, 0.1, 0.5], [-0.4, cy + 0.02, 0.3, 0.52], [0.05, cy - 0.05, -0.48, 0.5], [-0.2, cy - 0.12, -0.1, 0.55], [0.25, cy + 0.2, -0.25, 0.45], [-0.3, cy + 0.25, -0.2, 0.42]];
  lobes.forEach(([x, y, z, r], k) => {
    const g = jitter(new THREE.IcosahedronGeometry(r, 1), 0.28, k * 11).scale(1, vr * 1.25, 1).translate(x, y, z);
    parts.push(finish(g, shade, 1, (p, n) => crownNormal(p, n, 0.55)));
  });
  return mergeGeometries(parts)!;
}

/** Conifer (white pine / Virginia pine), unit space: height 1, base radius 1. */
function pineTemplate(hi: boolean) {
  const parts: THREE.BufferGeometry[] = [];
  const shade = (p: THREE.Vector3) => THREE.MathUtils.clamp(0.45 + 0.65 * p.y + 0.3 * Math.hypot(p.x, p.z), 0.35, 1.1);
  if (!hi) return finish(new THREE.ConeGeometry(0.8, 0.75, 6).translate(0, 0.62, 0), shade, 1);
  parts.push(finish(new THREE.CylinderGeometry(0.015, 0.028, 0.6, 5, 1, true).translate(0, 0.3, 0), () => 1, 0));
  const tiers = 5;
  for (let t = 0; t < tiers; t++) {
    const f = t / (tiers - 1);
    const y = 0.3 + f * 0.62, r = (1 - f * 0.78) * 0.95, h = 0.2 - f * 0.05;
    const g = jitter(new THREE.ConeGeometry(r, h, 8, 1), 0.3, t * 7).rotateY(t * 0.9).translate(Math.sin(t * 2.3) * 0.08, y, Math.cos(t * 1.7) * 0.08);
    parts.push(finish(g, shade, 1, (p, n) => n.lerp(new THREE.Vector3(p.x, 0.6, p.z).normalize(), 0.5).normalize()));
  }
  return mergeGeometries(parts)!;
}

/** Hemlock: dense cone of drooping tiers almost to the ground. */
function hemlockTemplate(hi: boolean) {
  const parts: THREE.BufferGeometry[] = [];
  const shade = (p: THREE.Vector3) => THREE.MathUtils.clamp(0.42 + 0.6 * p.y + 0.35 * Math.hypot(p.x, p.z), 0.32, 1.05);
  if (!hi) return finish(new THREE.ConeGeometry(0.9, 0.95, 6).translate(0, 0.52, 0), shade, 1);
  parts.push(finish(new THREE.CylinderGeometry(0.015, 0.03, 0.2, 5, 1, true).translate(0, 0.1, 0), () => 1, 0));
  const tiers = 6;
  for (let t = 0; t < tiers; t++) {
    const f = t / (tiers - 1);
    const y = 0.14 + f * 0.8, r = (1 - f * 0.85), h = 0.24 - f * 0.08;
    const g = jitter(new THREE.ConeGeometry(r, h, 9, 1), 0.25, t * 5).rotateY(t * 1.3).translate(0, y, 0);
    parts.push(finish(g, shade, 1, (p, n) => n.lerp(new THREE.Vector3(p.x, 0.5, p.z).normalize(), 0.5).normalize()));
  }
  return mergeGeometries(parts)!;
}

/** Shrub / rhododendron / brush: low dome of lobes. Unit: height 1, radius 1. */
function shrubTemplate(hi: boolean) {
  const shade = (p: THREE.Vector3) => THREE.MathUtils.clamp(0.4 + 0.7 * p.y, 0.35, 1.1);
  const nf = (p: THREE.Vector3, n: THREE.Vector3) => n.lerp(new THREE.Vector3(p.x, p.y - 0.2, p.z).normalize(), 0.6).normalize();
  if (!hi) return finish(new THREE.IcosahedronGeometry(1, 0).scale(1, 0.6, 1).translate(0, 0.4, 0), shade, 1, nf);
  const parts = [[0, 0.5, 0, 0.62], [0.45, 0.35, 0.2, 0.5], [-0.4, 0.35, -0.25, 0.5], [0.1, 0.35, -0.5, 0.45]].map(([x, y, z, r], k) =>
    finish(jitter(new THREE.IcosahedronGeometry(r, 1), 0.3, k * 3).scale(1, 0.85, 1).translate(x, y, z), shade, 1, nf));
  return mergeGeometries(parts)!;
}

/** Snag: bare grey trunk with two stubs. Unit: height 1, radius 1. */
function snagTemplate(_hi: boolean) {
  const parts = [
    new THREE.CylinderGeometry(0.06, 0.12, 1, 5, 1, true).translate(0, 0.5, 0),
    new THREE.CylinderGeometry(0.02, 0.04, 0.3, 4, 1, true).rotateZ(0.9).translate(0.12, 0.7, 0),
    new THREE.CylinderGeometry(0.02, 0.035, 0.22, 4, 1, true).rotateZ(-1.0).translate(-0.1, 0.55, 0.02),
  ].map((g) => finish(g, (p) => 0.7 + 0.4 * p.y, 1));
  return mergeGeometries(parts)!;
}

const TEMPLATES: Record<Fam, (hi: boolean) => THREE.BufferGeometry> = { broad: broadTemplate, pine: pineTemplate, hemlock: hemlockTemplate, shrub: shrubTemplate, snag: snagTemplate };

/** Foliage material: instance colour tints foliage only (bark keeps its colour), gentle wind sway. */
export function foliageMaterial(time: { value: number }) {
  const m = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92, metalness: 0, envMapIntensity: 0.7 });
  m.onBeforeCompile = (s) => {
    s.uniforms.uTime = time;
    s.vertexShader = s.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float leaf;\nuniform float uTime;')
      .replace('#include <color_vertex>', `
        vColor = vec3(1.0);
        vColor *= color.xyz;
        #ifdef USE_INSTANCING_COLOR
          vColor.xyz *= mix(vec3(1.0), instanceColor.xyz, leaf);
        #endif`)
      .replace('#include <begin_vertex>', `
        vec3 transformed = vec3(position);
        #ifdef USE_INSTANCING
          float ph = instanceMatrix[3].x * 0.05 + instanceMatrix[3].z * 0.037;
          float sw = leaf * position.y * position.y * 0.035;
          transformed.x += sin(uTime * 1.3 + ph) * sw;
          transformed.z += cos(uTime * 1.1 + ph * 1.3) * sw * 0.7;
        #endif`);
  };
  return m;
}

export interface Vegetation { group: THREE.Group; update(cam: THREE.Vector3, far: number): void; count: number }

export function buildVegetation(v: Float32Array, time: { value: number }, nearDist = 480): Vegetation {
  const group = new THREE.Group();
  group.name = 'vegetation';
  const S = 6, n = v.length / S;
  const mat = foliageMaterial(time);
  const geos: Record<string, THREE.BufferGeometry> = {};
  for (const f of Object.keys(TEMPLATES) as Fam[]) { geos[f + 'H'] = TEMPLATES[f](true); geos[f + 'L'] = TEMPLATES[f](false); }
  const buckets = new Map<string, number[]>();
  for (let i = 0; i < n; i++) {
    const sp = SPECIES[v[i * S + 4] | 0];
    if (!sp) continue;
    const key = `${Math.floor(v[i * S] / CHUNK)}_${Math.floor(v[i * S + 1] / CHUNK)}_${sp[1]}`;
    let b = buckets.get(key);
    if (!b) buckets.set(key, (b = []));
    b.push(i);
  }
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3(), p = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0), col = new THREE.Color();
  const chunks: { hi: THREE.InstancedMesh; lo: THREE.InstancedMesh; c: THREE.Vector3; r: number; far: number }[] = [];
  const FAR: Record<Fam, number> = { broad: 1e9, pine: 1e9, hemlock: 1e9, shrub: 1300, snag: 1600 };
  for (const [key, ids] of buckets) {
    const fam = key.split('_')[2] as Fam;
    const hi = new THREE.InstancedMesh(geos[fam + 'H'], mat, ids.length);
    const lo = new THREE.InstancedMesh(geos[fam + 'L'], mat, ids.length);
    const center = new THREE.Vector3();
    ids.forEach((i, k) => {
      const x = v[i * S], y = v[i * S + 1], z = v[i * S + 2], s = v[i * S + 3], seed = v[i * S + 5];
      const sp = SPECIES[v[i * S + 4] | 0];
      const [X, , Z] = pxToWorld(x, y);
      const r1 = hash(seed * 1.7 + 3.1), r2 = hash(seed * 2.3 + 9.7), r3 = hash(seed * 5.1 + 1.3);
      p.set(X, z - 0.35, Z);
      center.add(p);
      q.setFromAxisAngle(up, r1 * Math.PI * 2);
      const h = sp[3] * s, cr = sp[5] * s * (0.88 + 0.24 * r2) * (sp[1] === 'broad' ? 1.2 : 1.05);
      sc.set(cr, h, cr * (0.9 + 0.2 * r3));
      m4.compose(p, q, sc);
      hi.setMatrixAt(k, m4); lo.setMatrixAt(k, m4);
      const pal = PAL[sp[2]];
      let acc = 0, pick = pal[pal.length - 1][0];
      for (const [c, w] of pal) { acc += w; if (r2 < acc) { pick = c; break; } }
      const g = FOLIAGE_GAIN * (0.82 + 0.36 * r3);
      col.setRGB(pick[0] * g, pick[1] * g, pick[2] * g);
      hi.setColorAt(k, col); lo.setColorAt(k, col);
    });
    center.divideScalar(ids.length);
    for (const im of [hi, lo]) {
      im.instanceMatrix.needsUpdate = true;
      im.computeBoundingSphere();
      im.castShadow = true; im.receiveShadow = true;
      im.userData.kind = 'tree';
      im.name = `veg_${key}`;
      group.add(im);
    }
    lo.castShadow = false;
    chunks.push({ hi, lo, c: center, r: hi.boundingSphere?.radius ?? 200, far: FAR[fam] });
  }
  return {
    group, count: n,
    update(cam: THREE.Vector3, far: number) {
      for (const ch of chunks) {
        const d = Math.max(0, ch.c.distanceTo(cam) - ch.r * 0.5);
        ch.hi.visible = d < nearDist;
        ch.lo.visible = !ch.hi.visible && d < Math.min(far, ch.far);
      }
    },
  };
}
