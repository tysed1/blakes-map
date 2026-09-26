import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { pxToWorld } from '../../core/coords';
import { CHUNK } from './terrain';

/** Low-poly stylized tree prototypes (planning proxies; replace with real assets later). */
function hardwoodGeo() {
  const trunk = new THREE.CylinderGeometry(0.25, 0.4, 5, 5).translate(0, 2.5, 0);
  const c1 = new THREE.IcosahedronGeometry(4.2, 1).scale(1, 1.05, 1).translate(0, 8.5, 0);
  const c2 = new THREE.IcosahedronGeometry(3.0, 0).translate(1.6, 10.5, 0.8);
  const color = (g: THREE.BufferGeometry, c: THREE.Color) => {
    const n = g.getAttribute('position').count;
    const a = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { a[i * 3] = c.r; a[i * 3 + 1] = c.g; a[i * 3 + 2] = c.b; }
    g.setAttribute('color', new THREE.BufferAttribute(a, 3));
    return g;
  };
  const brown = new THREE.Color(0x4a3a2a), white = new THREE.Color(1, 1, 1);
  return mergeGeometries([color(trunk.toNonIndexed(), brown), color(c1.toNonIndexed(), white), color(c2.toNonIndexed(), white)]);
}
function coniferGeo() {
  const trunk = new THREE.CylinderGeometry(0.2, 0.35, 4, 5).translate(0, 2, 0);
  const c1 = new THREE.ConeGeometry(3.4, 8, 7).translate(0, 7, 0);
  const c2 = new THREE.ConeGeometry(2.5, 7, 7).translate(0, 11.5, 0);
  const c3 = new THREE.ConeGeometry(1.5, 5, 6).translate(0, 15, 0);
  const color = (g: THREE.BufferGeometry, c: THREE.Color) => {
    const n = g.getAttribute('position').count;
    const a = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { a[i * 3] = c.r; a[i * 3 + 1] = c.g; a[i * 3 + 2] = c.b; }
    g.setAttribute('color', new THREE.BufferAttribute(a, 3));
    return g;
  };
  const brown = new THREE.Color(0x3d3024), white = new THREE.Color(1, 1, 1);
  return mergeGeometries([color(trunk.toNonIndexed(), brown), color(c1.toNonIndexed(), white), color(c2.toNonIndexed(), white), color(c3.toNonIndexed(), white)]);
}

// palette following graphics ref.png: mixed greens with warm autumn accents
const HARDWOOD = [0x5f7f34, 0x6b8a3a, 0x7a8a36, 0x56722e, 0x8c8a34, 0xb0852e, 0xc0782a, 0x9a5a26].map((c) => new THREE.Color(c));
const HARD_W = [0.22, 0.2, 0.14, 0.14, 0.1, 0.09, 0.06, 0.05];
const CONIFER = [0x2f4f2c, 0x36582f, 0x2a4428, 0x3e5f34].map((c) => new THREE.Color(c));

function pick(pal: THREE.Color[], w: number[] | null, r: number) {
  if (!w) return pal[Math.floor(r * pal.length) % pal.length];
  let acc = 0;
  for (let i = 0; i < pal.length; i++) { acc += w[i]; if (r < acc) return pal[i]; }
  return pal[pal.length - 1];
}

export function buildTrees(trees: Float32Array, stride: number) {
  const group = new THREE.Group();
  group.name = 'vegetation';
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, metalness: 0, flatShading: true });
  const geos = [hardwoodGeo(), coniferGeo()];
  const buckets = new Map<string, number[]>();
  const n = trees.length / stride;
  for (let i = 0; i < n; i++) {
    const x = trees[i * stride], y = trees[i * stride + 1], kind = trees[i * stride + 4];
    const key = `${Math.floor(x / CHUNK)}_${Math.floor(y / CHUNK)}_${kind}`;
    let b = buckets.get(key);
    if (!b) { b = []; buckets.set(key, b); }
    b.push(i);
  }
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0);
  let seed = 1;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  for (const [key, ids] of buckets) {
    const kind = Number(key.split('_')[2]);
    const im = new THREE.InstancedMesh(geos[kind], mat, ids.length);
    im.name = `trees_${key}`;
    im.castShadow = true;
    im.receiveShadow = true;
    ids.forEach((i, k) => {
      const x = trees[i * stride], y = trees[i * stride + 1], z = trees[i * stride + 2], sc = trees[i * stride + 3];
      const [X, , Z] = pxToWorld(x, y);
      p.set(X, z - 0.4, Z);
      q.setFromAxisAngle(up, rnd() * Math.PI * 2);
      const hs = sc * (kind ? 1.25 : 1.2);
      s.set(sc * (0.9 + 0.2 * rnd()), hs, sc * (0.9 + 0.2 * rnd()));
      m4.compose(p, q, s);
      im.setMatrixAt(k, m4);
      im.setColorAt(k, kind ? pick(CONIFER, null, rnd()) : pick(HARDWOOD, HARD_W, rnd()));
    });
    im.instanceMatrix.needsUpdate = true;
    im.computeBoundingSphere();
    im.userData.kind = 'tree';
    group.add(im);
  }
  return group;
}
