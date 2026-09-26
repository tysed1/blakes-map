import * as THREE from 'three';
import { pxToWorld } from '../../core/coords';

export interface BackdropMeta { cell_px: number; w: number; h: number; origin_px: [number, number]; min_m: number; max_m: number; map_rect_cells: [number, number, number, number] }

/** Distant terrain ring around the playable map (hole where the real terrain is). */
export function buildBackdrop(meta: BackdropMeta, hBuf: ArrayBuffer, wBuf: ArrayBuffer, step = 2): THREE.Group {
  const g = new THREE.Group();
  g.name = 'backdrop';
  const u = new Uint16Array(hBuf), wet = new Uint8Array(wBuf);
  const k = (meta.max_m - meta.min_m) / 65535;
  const H = (i: number, j: number) => meta.min_m + u[Math.min(meta.h - 1, j) * meta.w + Math.min(meta.w - 1, i)] * k;
  const [rx0, ry0, rx1, ry1] = meta.map_rect_cells;
  const inside = (i: number, j: number) => i >= rx0 + 1 && i < rx1 - 1 && j >= ry0 + 1 && j < ry1 - 1;
  const nx = Math.floor((meta.w - 1) / step) + 1, ny = Math.floor((meta.h - 1) / step) + 1;
  const pos = new Float32Array(nx * ny * 3), col = new Float32Array(nx * ny * 3);
  const cForest = new THREE.Color(0x334f2e), cHigh = new THREE.Color(0x4a6340), cRock = new THREE.Color(0x7c7a6c), c = new THREE.Color();
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const ci = i * step, cj = j * step;
    const px = meta.origin_px[0] + (ci + 0.5) * meta.cell_px, py = meta.origin_px[1] + (cj + 0.5) * meta.cell_px;
    const h = H(ci, cj);
    const [X, , Z] = pxToWorld(px, py);
    const v = j * nx + i;
    pos[v * 3] = X; pos[v * 3 + 1] = h - (inside(ci, cj) ? 30 : 0); pos[v * 3 + 2] = Z;
    const sl = Math.abs(H(ci + step, cj) - H(ci - step, cj)) + Math.abs(H(ci, cj + step) - H(ci, cj - step));
    const t = THREE.MathUtils.clamp((h - 350) / 500, 0, 1);
    // canopy mottling (hardwood / conifer / autumn patches), rock only on steep crests
    const n1 = Math.sin(px * 0.013 + Math.sin(py * 0.021) * 2) * Math.cos(py * 0.017 - px * 0.004);
    const n2 = Math.sin(px * 0.047 + py * 0.031) * Math.sin(py * 0.053 - px * 0.012);
    c.copy(cForest).lerp(cHigh, t * 0.6 + 0.2 * (n1 * 0.5 + 0.5));
    if (n2 > 0.55) c.lerp(new THREE.Color(0x8a6a34), (n2 - 0.55) * 1.2);
    if (n1 < -0.6) c.lerp(new THREE.Color(0x2c4630), 0.5);
    c.lerp(cRock, THREE.MathUtils.clamp((sl / (step * meta.cell_px * 2.5) - 0.9), 0, 0.35));
    col[v * 3] = c.r; col[v * 3 + 1] = c.g; col[v * 3 + 2] = c.b;
  }
  const idx: number[] = [];
  for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    if (inside(i * step, j * step) && inside((i + 1) * step, (j + 1) * step)) continue;
    const a = j * nx + i, b = a + 1, cc = a + nx, d = cc + 1;
    idx.push(a, cc, b, b, cc, d);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, metalness: 0 }));
  m.receiveShadow = false;
  m.userData.kind = 'backdrop';
  g.add(m);
  // mirrored river water in the backdrop
  const wp: number[] = [], wi: number[] = [];
  for (let j = 0; j < meta.h - 1; j += 1) for (let i = 0; i < meta.w - 1; i += 1) {
    if (!wet[j * meta.w + i] || inside(i, j)) continue;
    const base = wp.length / 3;
    for (const [di, dj] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
      const px = meta.origin_px[0] + (i + di + 0.5) * meta.cell_px, py = meta.origin_px[1] + (j + dj + 0.5) * meta.cell_px;
      const [X, , Z] = pxToWorld(px, py);
      wp.push(X, H(i, j) + 1.6, Z);
    }
    wi.push(base, base + 2, base + 1, base + 1, base + 2, base + 3);
  }
  if (wi.length) {
    const wg = new THREE.BufferGeometry();
    wg.setAttribute('position', new THREE.Float32BufferAttribute(wp, 3));
    wg.setIndex(wi);
    wg.computeVertexNormals();
    g.add(new THREE.Mesh(wg, new THREE.MeshStandardMaterial({ color: 0x3d6776, roughness: 0.4 })));
  }
  return g;
}

/** Gradient sky dome with a warm late-afternoon glow toward the sun (graphics ref). */
export function buildSky(sunDir: THREE.Vector3) {
  const geo = new THREE.SphereGeometry(40000, 32, 16);
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: { sunDir: { value: sunDir.clone().normalize() } },
    vertexShader: `varying vec3 vDir; void main(){ vDir = normalize(position); vec4 p = modelViewMatrix * vec4(position,1.0); gl_Position = projectionMatrix * p; gl_Position.z = gl_Position.w; }`,
    fragmentShader: `varying vec3 vDir; uniform vec3 sunDir;
      void main(){
        float h = vDir.y;
        vec3 zenith = vec3(0.36,0.52,0.78), mid = vec3(0.66,0.76,0.88), horizon = vec3(0.95,0.84,0.70), ground = vec3(0.62,0.66,0.66);
        vec3 c = h > 0.0 ? mix(mix(horizon, mid, smoothstep(0.0,0.12,h)), zenith, smoothstep(0.12,0.6,h)) : mix(horizon, ground, smoothstep(0.0,-0.2,h));
        float s = max(dot(normalize(vDir), normalize(sunDir)), 0.0);
        c += vec3(1.0,0.72,0.45) * (pow(s, 8.0)*0.35 + pow(s, 200.0)*1.2);
        gl_FragColor = vec4(c, 1.0);
        #include <colorspace_fragment>
      }`,
  });
  const m = new THREE.Mesh(geo, mat);
  m.renderOrder = -10;
  m.frustumCulled = false;
  m.name = 'sky';
  return m;
}
