import * as THREE from 'three';
import { IMG_W, IMG_H, MPP, pxToWorld } from '../../core/coords';
import { Heightfield } from '../../core/data';

export const CHUNK = 100; // source px per chunk side (250 m)
const LOD_STEPS = [1, 2, 4, 8];
const LOD_DIST = [0, 450, 1100, 2200];

/** One chunk geometry at a given sample step, with skirts to hide LOD seams. */
function chunkGeometry(hf: Heightfield, cx: number, cy: number, step: number): THREE.BufferGeometry {
  const x0 = cx * CHUNK, y0 = cy * CHUNK;
  const x1 = Math.min(IMG_W, x0 + CHUNK), y1 = Math.min(IMG_H, y0 + CHUNK);
  const nx = Math.ceil((x1 - x0) / step) + 1, ny = Math.ceil((y1 - y0) / step) + 1;
  const pos: number[] = [], uv: number[] = [], idx: number[] = [];
  const gx = (i: number) => Math.min(x0 + i * step, x1);
  const gy = (j: number) => Math.min(y0 + j * step, y1);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const px = gx(i), py = gy(j);
    const [X, Y, Z] = pxToWorld(px, py, hf.at(px, py));
    pos.push(X, Y, Z); uv.push(px / IMG_W, 1 - py / IMG_H);
  }
  for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    const a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
    idx.push(a, c, b, b, c, d);
  }
  // skirts
  const edge = (list: number[]) => {
    const base = pos.length / 3;
    for (const k of list) {
      pos.push(pos[k * 3], pos[k * 3 + 1] - 6 * step, pos[k * 3 + 2]);
      uv.push(uv[k * 2], uv[k * 2 + 1]);
    }
    for (let n = 0; n < list.length - 1; n++) idx.push(list[n], base + n, list[n + 1], list[n + 1], base + n, base + n + 1);
  };
  const top = [...Array(nx).keys()], bottom = top.map((i) => (ny - 1) * nx + i).reverse();
  const left = [...Array(ny).keys()].map((j) => j * nx).reverse(), right = [...Array(ny).keys()].map((j) => j * nx + nx - 1);
  edge(top.slice().reverse()); edge(bottom.slice().reverse()); edge(left.slice().reverse()); edge(right.slice().reverse());
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  // normals from the heightfield itself (seamless across chunks and LODs)
  const nrm = new Float32Array(pos.length);
  const e = Math.max(1, step * 0.5);
  for (let k = 0; k < pos.length / 3; k++) {
    const px = pos[k * 3] / MPP + 1000, py = pos[k * 3 + 2] / MPP + 333.5;
    const dx = (hf.at(px + e, py) - hf.at(px - e, py)) / (2 * e * MPP);
    const dz = (hf.at(px, py + e) - hf.at(px, py - e)) / (2 * e * MPP);
    const l = Math.hypot(dx, 1, dz);
    nrm[k * 3] = -dx / l; nrm[k * 3 + 1] = 1 / l; nrm[k * 3 + 2] = -dz / l;
  }
  g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  g.computeBoundingSphere();
  return g;
}

export function buildTerrain(hf: Heightfield, material: THREE.Material): THREE.Group {
  const group = new THREE.Group();
  group.name = 'terrain';
  for (let cy = 0; cy * CHUNK < IMG_H; cy++) for (let cx = 0; cx * CHUNK < IMG_W; cx++) {
    const lod = new THREE.LOD();
    lod.name = `C${String(cx).padStart(2, '0')}_${String(cy).padStart(2, '0')}`;
    LOD_STEPS.forEach((s, k) => {
      const m = new THREE.Mesh(chunkGeometry(hf, cx, cy, s), material);
      m.receiveShadow = true;
      m.userData.kind = 'terrain';
      lod.addLevel(m, LOD_DIST[k]);
    });
    group.add(lod);
  }
  return group;
}

export function terrainMaterial(albedo: THREE.Texture, base: THREE.Texture) {
  const mat = new THREE.MeshStandardMaterial({ map: albedo, roughness: 0.95, metalness: 0 });
  (mat as any).userData = { base };
  return mat;
}

/** Water surface following the per-pixel water level (river surfaces slope downstream). */
export function buildWater(level: Heightfield, landuse: Uint8Array, material: THREE.Material): THREE.Group {
  const group = new THREE.Group();
  group.name = 'water';
  const wet = new Uint8Array(IMG_W * IMG_H);
  for (let i = 0; i < wet.length; i++) wet[i] = landuse[i] === 1 ? 1 : 0;
  // dilate by 1 px so the surface tucks under the banks
  const dil = wet.slice();
  for (let y = 1; y < IMG_H - 1; y++) for (let x = 1; x < IMG_W - 1; x++) {
    const i = y * IMG_W + x;
    if (!wet[i] && (wet[i - 1] || wet[i + 1] || wet[i - IMG_W] || wet[i + IMG_W])) dil[i] = 1;
  }
  const step = 1;
  for (let cy = 0; cy * CHUNK < IMG_H; cy++) for (let cx = 0; cx * CHUNK < IMG_W; cx++) {
    const pos: number[] = [], idx: number[] = [];
    const vid = new Map<number, number>();
    const v = (px: number, py: number) => {
      const k = py * (IMG_W + 1) + px;
      let id = vid.get(k);
      if (id === undefined) {
        id = pos.length / 3;
        const [X, , Z] = pxToWorld(px, py);
        pos.push(X, level.at(px, py) + 0.05, Z);
        vid.set(k, id);
      }
      return id;
    };
    for (let y = cy * CHUNK; y < Math.min(IMG_H, (cy + 1) * CHUNK); y += step) for (let x = cx * CHUNK; x < Math.min(IMG_W, (cx + 1) * CHUNK); x += step) {
      if (!dil[y * IMG_W + x]) continue;
      const a = v(x, y), b = v(x + step, y), c = v(x, y + step), d = v(x + step, y + step);
      idx.push(a, c, b, b, c, d);
    }
    if (!idx.length) continue;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    const m = new THREE.Mesh(g, material);
    m.userData.kind = 'water';
    group.add(m);
  }
  return group;
}

export const WORLD_SIZE = { x: IMG_W * MPP, z: IMG_H * MPP };
