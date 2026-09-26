import * as THREE from 'three';
import { IMG_W, IMG_H, MPP, pxToWorld } from '../../core/coords';
import { assetUrl, bin, World } from '../../core/data';

/**
 * Ground cover + props: the A3 Blender ground layer (tools/blender/lib_groundcover.py, lib_props.py)
 * for the web viewer.
 *
 * GRASS: a GPU blade field in a camera-following ring of 8 m tiles. Each tile draws a fixed patch of
 * blade slots (R2 low-discrepancy positions, so any prefix of the slots is evenly spread); in the
 * vertex shader every slot hashes its own clump position, samples the ground-cover weight raster
 * (tools/pipeline/web_groundcover_raster.py: the Blender GN scatter weights per field type /
 * canopy / moisture / hedge / verge, plus road / rail / water distance fields evaluated per blade)
 * and picks a grass kind (pasture, broomsedge, hay stubble, lawn, rush, forest grass: the
 * lib_groundcover.build_protos clump specs) or collapses to nothing. Three LOD patches (7 / 3 / 1
 * tris per blade) share the same slot order: density thins continuously with distance and each LOD
 * starts exactly where the thinning reaches its slot count, so LOD swaps are invisible; beyond
 * that blades shrink into the ground while their colour converges on the terrain albedo bake.
 * One draw call per LOD, no per-frame CPU work besides a uniform (tiles are re-listed only when the
 * camera moves ~2 m or turns ~8 degrees).
 *
 * DECOR: wildflowers (goldenrod, asters, Queen Anne's lace, chicory), roadside weeds and ferns
 * (Blender prototypes, tools/blender/export_web_groundcover.py), scattered per 16 m tile on the CPU
 * from the same raster, 2 LODs, shrink-out by distance.
 *
 * PROPS: boulders, rocks, logs, stumps, branches, roots, bark debris from public/world/props_f32.bin
 * (the Blender instances) with decimated Poly Haven meshes in one texture atlas, 2 LODs, per-kind
 * distance culling.
 *
 * Wind shares the trees' uTime / uWind uniforms.
 */

type Q = 'low' | 'medium' | 'high' | 'ultra';
interface GrassQ { dens: [number, number, number]; rad: [number, number, number, number]; fade: [number, number]; decor: number; props: number }
export const GROUNDCOVER_QUALITY: Record<Q, GrassQ> = {
  low: { dens: [36, 10, 3.5], rad: [6, 12, 22, 32], fade: [45, 75], decor: 35, props: 0.6 },
  medium: { dens: [60, 16, 5], rad: [8, 15, 28, 40], fade: [60, 100], decor: 50, props: 0.8 },
  high: { dens: [96, 24, 6], rad: [9, 18, 32, 45], fade: [80, 130], decor: 65, props: 1.0 },
  ultra: { dens: [150, 32, 8], rad: [12, 24, 40, 56], fade: [100, 160], decor: 85, props: 1.25 },
};
const TILE = 8; // m, grass tile
const DTILE = 16; // m, decor tile
const PCELL = 40; // source px (100 m), prop cell

interface LodInfo { vcount: number; icount: number; pos: number; nrm: number; uv: number; col: number; idx: number }
interface GrassKind { name: string; h: [number, number]; w: number; lean: [number, number]; col0: number[]; col1: number[][]; seed: number; dens: number; scale: [number, number] }
interface DecorInfo { name: string; layer: 'flowers' | 'weeds' | 'fern'; material: 'vcol' | 'fern'; radius: number; height: number; lods: LodInfo[] }
interface PropInfo { name: string; kind: string; asset: string; radius: number; height: number; bottom: number; lods: LodInfo[] }
interface PropKind { name: string; far: number; protos: number[]; rock: boolean }
interface GeoJson { grass: GrassKind[]; decor: DecorInfo[]; prop_kinds: PropKind[]; props: PropInfo[] }

export interface GroundcoverUniforms { uTime: { value: number }; uWind: { value: number } }
export interface GroundcoverOptions { albedo?: THREE.Texture; sunDir?: THREE.Vector3; uniforms?: Partial<GroundcoverUniforms>; quality?: Q }
export interface Groundcover {
  group: THREE.Group;
  uniforms: GroundcoverUniforms;
  update(camera: THREE.Camera, dt?: number): void;
  setQuality(q: Q): void;
  stats(): { tiles: number[]; blades: number; decor: number; props: number; tris: number; draws: number };
}

// ---------------------------------------------------------------- shared GLSL
const GLSL_COMMON = /* glsl */ `
  uint gcS;
  uint pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
  float rnd() { gcS = pcg(gcS); return float(gcS) * 2.3283064365386963e-10; }
  float hash2(vec2 c) { return float(pcg(uint(int(c.x) + 65536) * 1597334677u ^ uint(int(c.y) + 65536) * 3812015801u)) * 2.3283064365386963e-10; }
  float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2(i), hash2(i + vec2(1, 0)), f.x), mix(hash2(i + vec2(0, 1)), hash2(i + vec2(1, 1)), f.x), f.y);
  }
  // wind: slow gust fronts rolling across the fields + per-plant flutter (same uTime / uWind as the trees)
  const vec2 WIND_DIR = vec2(0.88, 0.47);
  float gcGust(vec2 p) {
    float g = sin(dot(p, WIND_DIR) * 0.075 - uTime * 1.45 + vnoise(p * 0.02) * 5.0) * 0.5 + 0.5;
    return g * g * g;
  }
`;
const TRANSLUCENT = (k: number) => THREE.ShaderChunk.lights_physical_pars_fragment.replace(
  'reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );',
  `float back = saturate( dot( - geometryNormal, directLight.direction ) ) * 0.6 + saturate( - dot( geometryViewDir, directLight.direction ) ) * 0.6;
   reflectedLight.directDiffuse += ( irradiance * ${(1 - k).toFixed(2)} + back * directLight.color * vec3( 1.4, 1.25, 0.7 ) * ${k.toFixed(2)} ) * BRDF_Lambert( material.diffuseColor );`);
const NO_FLIP = THREE.ShaderChunk.normal_fragment_begin.replace('float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;', 'float faceDirection = 1.0;');

// ---------------------------------------------------------------- grass blade shader
const GRASS_VS = /* glsl */ `
  uniform sampler2D uHgt, uGcA, uGcB, uGcC, uAlb;
  uniform float uTime, uWind, uTileM, uPix, uLod, uHasAlb;
  uniform vec3 uCam, uDens;
  uniform vec4 uRad;
  uniform vec2 uFade;
  uniform vec4 uKH[6], uKL[6];
  uniform vec3 uKC0[6], uKC1a[6], uKC1b[6];
  uniform float uKD[6];
  attribute vec2 aTile;
  varying vec3 vGcCol;
  varying float vGcT;
  ${GLSL_COMMON}
  float gcGround(vec2 g, out vec3 n) {
    // identical to the terrain mesh at full resolution: vertex (i, j) at source px (i, j), split b-c
    g = clamp(g, vec2(0.0), vec2(${IMG_W - 0.001}, ${IMG_H - 0.001}));
    ivec2 i = ivec2(floor(g)); vec2 f = g - vec2(i);
    float a = texelFetch(uHgt, i, 0).r, b = texelFetch(uHgt, i + ivec2(1, 0), 0).r;
    float c = texelFetch(uHgt, i + ivec2(0, 1), 0).r, d = texelFetch(uHgt, i + ivec2(1, 1), 0).r;
    float h, dx, dz;
    if (f.x + f.y < 1.0) { h = a + (b - a) * f.x + (c - a) * f.y; dx = b - a; dz = c - a; }
    else { h = d + (c - d) * (1.0 - f.x) + (b - d) * (1.0 - f.y); dx = d - c; dz = d - b; }
    n = normalize(vec3(-dx / ${MPP.toFixed(1)}, 1.0, -dz / ${MPP.toFixed(1)}));
    return h;
  }
  void gcBlade(out vec3 P, out vec3 N) {
    float side = position.x, t = position.y, bi = position.z;
    ivec2 ti = ivec2(floor(aTile / uTileM + 0.5));
    gcS = pcg(uint(ti.x + 32768) * 73856093u ^ uint(ti.y + 32768) * 19349663u);
    vec2 toff = vec2(rnd(), rnd());
    gcS = pcg(gcS ^ (uint(bi) * 83492791u + 1u));
    // R2 sequence slot (every prefix is evenly spread) + small jitter, then pulled into 0.45 m clumps
    vec2 lp = fract(toff + bi * vec2(0.7548776662, 0.5698402910) + (vec2(rnd(), rnd()) - 0.5) * 0.06);
    vec2 wp = aTile + lp * uTileM;
    vec2 cc = floor(wp / 0.45);
    vec2 ctr = (cc + vec2(hash2(cc), hash2(cc + 71.0))) * 0.45;
    vec2 root = mix(wp, ctr, 0.35);
    vec2 g = vec2(root.x / ${MPP.toFixed(1)} + 1000.0, root.y / ${MPP.toFixed(1)} + 333.5);
    vec3 tn; float gy = gcGround(g, tn);
    vec3 R = vec3(root.x, gy - 0.03, root.y);
    P = R; N = tn; vGcCol = vec3(0.0); vGcT = 0.0;
    float dist = distance(uCam, R);
    float dens = mix(mix(uDens.x, uDens.y, smoothstep(uRad.x, uRad.y, dist)), uDens.z, smoothstep(uRad.z, uRad.w, dist));
    float fade = 1.0 - smoothstep(uFade.x, uFade.y, dist);
    if (bi / (uTileM * uTileM) >= dens || fade < 0.01) return;
    // layer weights (Blender GN scatter), exclusion masks from the distance fields
    vec2 uv = g / vec2(${IMG_W}.0, ${IMG_H}.0);
    vec4 A = texture(uGcA, uv) * 3.1875, B = texture(uGcB, uv), C = texture(uGcC, uv);
    float offLow = smoothstep(0.5, 1.6, A.r) * smoothstep(0.8, 2.0, A.g) * smoothstep(0.4, 1.2, A.b);
    float offTall = smoothstep(1.2, 2.5, A.r) * smoothstep(1.5, 3.0, A.g) * smoothstep(0.8, 1.8, A.b);
    float atWater = smoothstep(1.2, 2.5, A.r) * smoothstep(1.5, 3.0, A.g) * smoothstep(0.1, 0.5, A.b);
    float flatf = smoothstep(0.72, 0.9, tn.y);
    float w[6];
    w[0] = B.r * offTall * uKD[0]; w[1] = B.g * offTall * uKD[1]; w[2] = B.b * offTall * uKD[2];
    w[3] = B.a * offLow * uKD[3]; w[4] = C.r * atWater * uKD[4]; w[5] = C.g * offLow * uKD[5];
    float tot = 0.0;
    for (int k = 0; k < 6; k++) { w[k] *= flatf; tot += w[k]; }
    // kind picked per clump (mostly): thin cover = scattered tufts, not a uniform haze of lone blades
    float u = fract(hash2(cc + 11.0) + rnd() * 0.18) * max(1.0, tot);
    int k = -1; float acc = 0.0;
    for (int j = 0; j < 6; j++) { acc += w[j]; if (k < 0 && u < acc) k = j; }
    if (k < 0) return;
    if (tot < 0.6) {
      root = mix(wp, ctr, mix(0.82, 0.35, tot / 0.6));
      g = vec2(root.x / ${MPP.toFixed(1)} + 1000.0, root.y / ${MPP.toFixed(1)} + 333.5);
      gy = gcGround(g, tn); R = vec3(root.x, gy - 0.03, root.y); P = R;
    }
    vec4 KH = uKH[k], KL = uKL[k];
    float pn = vnoise(root * 0.11 + float(k) * 13.0);
    float h = mix(KH.x, KH.y, rnd()) * mix(KH.z, KH.w, rnd()) * (0.72 + 0.56 * pn) * mix(0.85, 1.12, C.a);
    float lean = mix(KL.x, KL.y, rnd());
    float wid = KL.z * 1.7 * mix(0.7, 1.3, rnd());
    bool seed = rnd() < KL.w && uLod < 1.5;
    vec2 away = root - ctr;
    float ang = rnd() * 6.2831853;
    vec2 dir = normalize(away * 3.0 + vec2(cos(ang), sin(ang)) * 0.6);
    // wind bend (in the blade's curve) + flutter across it
    float gust = gcGust(root);
    float fl = sin(uTime * 2.6 + rnd() * 6.2831853 + dot(root, vec2(0.9, 0.4)));
    vec2 bend = WIND_DIR * uWind * (0.03 + 0.17 * gust) + vec2(-WIND_DIR.y, WIND_DIR.x) * fl * 0.05 * uWind;
    if (seed) { h *= 1.22; lean *= 0.4; wid *= 0.38; }
    h *= fade;
    // colour: root -> tip gradient, per-blade value jitter, field-scale warm/cool patches, dry blades
    vec3 c0 = uKC0[k];
    vec3 c1 = mix(uKC1a[k], uKC1b[k], rnd()) * mix(0.8, 1.2, rnd());
    float hn = vnoise(root * 0.045 + 31.0);
    c1 *= mix(vec3(1.18, 1.0, 0.72), vec3(0.78, 1.06, 1.02), smoothstep(0.2, 0.8, hn));
    c1 *= mix(0.72, 1.12, hash2(cc + 5.0));                          // clump-to-clump value
    if (k == 0) c1 *= vec3(0.92, 1.06, 0.85);                         // pasture: a touch greener than the clump spec
    float rv = rnd();
    if (rv < 0.06 && k != 4) c1 = mix(c1, vec3(0.40, 0.33, 0.17), 0.65);   // dry straw blades
    else if (rv > 0.72 && k < 2) c1 = mix(c1, vec3(0.1, 0.19, 0.045), 0.55); // fresh green regrowth
    // geometry
    float tt = t;
    vec2 hz = (dir * lean + bend) * h * tt * tt;
    float vy = h * tt * (1.0 - 0.25 * lean * tt) * (1.0 - 0.3 * dot(bend, bend));
    vec3 sd = normalize(vec3(-dir.y, (rnd() - 0.5) * 0.5, dir.x));
    float taper = seed ? (tt > 0.7 ? 3.2 * (1.0 - tt) / 0.3 + 0.2 : 1.0) : (1.0 - 0.85 * tt);
    float comp = sqrt(uDens.x / dens);
    float wv = max(wid * min(comp, 4.0), dist * uPix * 1.1) * taper;
    P = R + vec3(hz.x, vy, hz.y) + sd * side * wv * 0.5;
    vec3 tg = normalize(vec3((dir * lean + bend).x * 2.0 * tt * h, h * (1.0 - 0.5 * lean * tt), (dir * lean + bend).y * 2.0 * tt * h));
    vec3 bn = normalize(cross(sd, tg)); if (bn.y < 0.0) bn = -bn;
    float far = smoothstep(uFade.x * 0.4, uFade.y, dist);
    N = normalize(mix(bn, tn, mix(0.55, 0.95, far)));
    vec3 col = mix(c0, c1, pow(tt, 0.8));
    if (seed && tt > 0.6) col = mix(c1, vec3(0.34, 0.25, 0.11), 0.5);
    col *= 1.0 + 0.18 * gust * tt * uWind; // bent blades show their paler sides
    col *= mix(mix(0.4, 1.0, smoothstep(0.0, 0.65, tt)), 1.0, far); // self-shadowing inside the sward
    if (uHasAlb > 0.5) {
      vec3 alb = textureLod(uAlb, vec2(uv.x, 1.0 - uv.y), 1.5).rgb;
      col = mix(col, alb * mix(0.8, 1.3, tt), 0.08 + 0.92 * far);
    }
    vGcCol = col; vGcT = tt;
  }
`;

function grassMaterial(U: Record<string, THREE.IUniform>) {
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.72, metalness: 0, side: THREE.DoubleSide, envMapIntensity: 0.45 });
  m.onBeforeCompile = (s) => {
    Object.assign(s.uniforms, U);
    s.vertexShader = s.vertexShader
      .replace('#include <common>', `#include <common>\n${GRASS_VS}`)
      .replace('#include <beginnormal_vertex>', 'vec3 objectNormal; vec3 gcP; gcBlade(gcP, objectNormal);')
      .replace('#include <begin_vertex>', 'vec3 transformed = gcP;');
    s.fragmentShader = s.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vGcCol; varying float vGcT;')
      .replace('#include <color_fragment>', 'diffuseColor.rgb *= vGcCol;')
      .replace('#include <normal_fragment_begin>', NO_FLIP)
      .replace('#include <lights_physical_pars_fragment>', TRANSLUCENT(0.35));
  };
  m.customProgramCacheKey = () => 'gc-grass';
  return m;
}

/** Blade slot patch: `n` blades, pairs of vertices at `ts` + a tip vertex. position = (side, t, slot). */
function bladePatch(n: number, ts: number[]) {
  const vpb = ts.length * 2 + 1, ipb = (ts.length - 1) * 6 + 3;
  const pos = new Float32Array(n * vpb * 3), idx = new Uint32Array(n * ipb);
  for (let b = 0; b < n; b++) {
    let v = b * vpb * 3;
    for (const t of ts) { pos.set([-1, t, b, 1, t, b], v); v += 6; }
    pos.set([0, 1, b], v);
    let o = b * ipb; const base = b * vpb;
    for (let k = 0; k < ts.length - 1; k++) {
      const a = base + k * 2;
      idx.set([a, a + 1, a + 3, a, a + 3, a + 2], o); o += 6;
    }
    const a = base + (ts.length - 1) * 2;
    idx.set([a, a + 1, base + vpb - 1], o);
  }
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);
  return { geo: g, ipb };
}
const LOD_TS = [[0, 0.3, 0.55, 0.78], [0, 0.5], [0]];

// ---------------------------------------------------------------- decor / props shaders
const DECOR_VS = /* glsl */ `
  uniform float uTime, uWind;
  uniform vec3 uCam;
  uniform vec2 uDFade;
  attribute vec4 col;
  varying vec3 vGcCol;
  ${GLSL_COMMON}
`;
function decorMaterial(U: Record<string, THREE.IUniform>, fern: THREE.Texture | null) {
  const m = new THREE.MeshStandardMaterial({
    color: 0xffffff, roughness: fern ? 0.7 : 0.65, metalness: 0, side: THREE.DoubleSide, envMapIntensity: 0.5,
    map: fern ?? null, alphaTest: fern ? 0.45 : 0,
  });
  if (fern) m.alphaToCoverage = true;
  m.onBeforeCompile = (s) => {
    Object.assign(s.uniforms, U);
    s.vertexShader = s.vertexShader
      .replace('#include <common>', `#include <common>\n${DECOR_VS}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        vec3 ip = instanceMatrix[3].xyz;
        float dd = distance(ip, uCam);
        transformed *= 1.0 - smoothstep(uDFade.x, uDFade.y, dd);
        float hh = max(position.y, 0.0);
        float gust = gcGust(ip.xz);
        float fl = sin(uTime * 2.2 + ip.x * 1.7 + ip.z * 1.3);
        vec2 bend = (WIND_DIR * (0.08 + 0.45 * gust) + vec2(-WIND_DIR.y, WIND_DIR.x) * fl * 0.08) * uWind * hh * hh;
        mat3 gcIm = mat3(instanceMatrix); // bend in world space: undo the instance rotation / scale
        transformed += inverse(gcIm) * vec3(bend.x, -0.25 * dot(bend, bend), bend.y);
        ${fern ? 'vGcCol = vec3(1.0);' : 'vGcCol = col.rgb * col.rgb;'}
        #ifdef USE_INSTANCING_COLOR
          vGcCol *= instanceColor;
        #endif`);
    s.fragmentShader = s.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vGcCol;')
      .replace('#include <color_fragment>', 'diffuseColor.rgb *= vGcCol;')
      .replace('#include <normal_fragment_begin>', NO_FLIP)
      .replace('#include <lights_physical_pars_fragment>', TRANSLUCENT(fern ? 0.3 : 0.35));
  };
  m.customProgramCacheKey = () => (fern ? 'gc-fern' : 'gc-vcol');
  return m;
}

function propMaterial(map: THREE.Texture, nmap: THREE.Texture) {
  const m = new THREE.MeshStandardMaterial({ map, normalMap: nmap, roughness: 0.88, metalness: 0, envMapIntensity: 0.6 });
  m.normalScale.set(1, 1);
  return m;
}

// ---------------------------------------------------------------- helpers
function mulberry(a: number) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const smooth = (a: number, b: number, x: number) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

function meshGeometry(buf: ArrayBuffer, l: LodInfo, withCol: boolean) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buf, l.pos, l.vcount * 3), 3));
  g.setAttribute('normal', new THREE.InterleavedBufferAttribute(new THREE.InterleavedBuffer(new Int8Array(buf, l.nrm, l.vcount * 4), 4), 3, 0, true));
  g.setAttribute('uv', new THREE.BufferAttribute(new Uint16Array(buf, l.uv, l.vcount * 2), 2, true));
  if (withCol) g.setAttribute('col', new THREE.BufferAttribute(new Uint8Array(buf, l.col, l.vcount * 4), 4, true));
  g.setIndex(new THREE.BufferAttribute(new Uint16Array(buf, l.idx, l.icount), 1));
  g.computeBoundingSphere();
  return g;
}

// ---------------------------------------------------------------- build
export async function buildGroundcover(world: World, opts: GroundcoverOptions = {}): Promise<Groundcover> {
  const [meta, geoBuf, gcBuf, propBuf, rmeta] = await Promise.all([
    fetch(assetUrl('groundcover/geo.json')).then((r) => r.json() as Promise<GeoJson>), bin('groundcover/geo.bin'),
    bin('groundcover/gc_u8.bin'), bin('props_f32.bin'), fetch(assetUrl('groundcover/raster.json')).then((r) => r.json()),
  ]);
  const W = rmeta.w as number, H = rmeta.h as number;
  const hf = world.terrain;
  const uniforms: GroundcoverUniforms = { uTime: opts.uniforms?.uTime ?? { value: 0 }, uWind: opts.uniforms?.uWind ?? { value: 1 } };

  // terrain vertex grid (exactly what terrain.ts triangulates at full resolution)
  const GW = IMG_W + 1, GH = IMG_H + 1;
  const grid = new Float32Array(GW * GH);
  for (let j = 0; j < GH; j++) for (let i = 0; i < GW; i++) grid[j * GW + i] = hf.at(i, j);
  const groundAt = (px: number, py: number) => {
    const x = Math.min(Math.max(px, 0), IMG_W - 0.001), y = Math.min(Math.max(py, 0), IMG_H - 0.001);
    const i = Math.floor(x), j = Math.floor(y), fx = x - i, fy = y - j, o = j * GW + i;
    const a = grid[o], b = grid[o + 1], c = grid[o + GW], d = grid[o + GW + 1];
    return fx + fy < 1 ? a + (b - a) * fx + (c - a) * fy : d + (c - d) * (1 - fx) + (b - d) * (1 - fy);
  };
  const hgtTex = new THREE.DataTexture(grid, GW, GH, THREE.RedFormat, THREE.FloatType);
  hgtTex.minFilter = hgtTex.magFilter = THREE.NearestFilter; hgtTex.needsUpdate = true;

  // weight raster: channel-major planes -> RGBA textures (planes 0..2) + CPU access (all)
  const raw = new Uint8Array(gcBuf), NPX = W * H;
  const ch = (p: number, c: number) => raw.subarray((p * 4 + c) * NPX, (p * 4 + c + 1) * NPX);
  const planeTex = (p: number) => {
    const d = new Uint8Array(NPX * 4);
    for (let c = 0; c < 4; c++) { const s = ch(p, c); for (let i = 0; i < NPX; i++) d[i * 4 + c] = s[i]; }
    const t = new THREE.DataTexture(d, W, H, THREE.RGBAFormat, THREE.UnsignedByteType);
    t.minFilter = t.magFilter = THREE.LinearFilter; t.generateMipmaps = false; t.needsUpdate = true;
    return t;
  };
  const [texA, texB, texC] = [planeTex(0), planeTex(1), planeTex(2)];
  const sample = (p: number, c: number, px: number, py: number) => {
    const x = Math.min(Math.max(px - 0.5, 0), W - 1.001), y = Math.min(Math.max(py - 0.5, 0), H - 1.001);
    const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0, a = ch(p, c), i = y0 * W + x0;
    return (a[i] * (1 - fx) * (1 - fy) + a[i + 1] * fx * (1 - fy) + a[i + W] * (1 - fx) * fy + a[i + W + 1] * fx * fy) / 255;
  };

  const group = new THREE.Group(); group.name = 'groundcover';
  let q: GrassQ = GROUNDCOVER_QUALITY[opts.quality ?? 'high'];

  // ------------------------------------------------ grass
  const K = meta.grass;
  const v4 = (a: number, b: number, c: number, d: number) => new THREE.Vector4(a, b, c, d);
  const v3 = (c: number[]) => new THREE.Vector3(c[0], c[1], c[2]);
  const GU: Record<string, THREE.IUniform> = {
    uTime: uniforms.uTime, uWind: uniforms.uWind,
    uHgt: { value: hgtTex }, uGcA: { value: texA }, uGcB: { value: texB }, uGcC: { value: texC },
    uAlb: { value: opts.albedo ?? null }, uHasAlb: { value: opts.albedo ? 1 : 0 },
    uTileM: { value: TILE }, uPix: { value: 0.001 }, uLod: { value: 0 },
    uCam: { value: new THREE.Vector3() }, uDens: { value: new THREE.Vector3() }, uRad: { value: new THREE.Vector4() }, uFade: { value: new THREE.Vector2() },
    uKH: { value: K.map((k) => v4(k.h[0], k.h[1], k.scale[0], k.scale[1])) },
    uKL: { value: K.map((k) => v4(k.lean[0], k.lean[1], k.w, k.seed)) },
    uKC0: { value: K.map((k) => v3(k.col0)) },
    uKC1a: { value: K.map((k) => v3(k.col1[0])) }, uKC1b: { value: K.map((k) => v3(k.col1[k.col1.length - 1])) },
    uKD: { value: K.map((k) => k.dens) },
  };
  // wind is shared through the uniform objects: keep them live (World3D may pass getters)
  Object.defineProperty(GU, 'uWind', { get: () => uniforms.uWind, enumerable: true });
  Object.defineProperty(GU, 'uTime', { get: () => uniforms.uTime, enumerable: true });
  const maxD = GROUNDCOVER_QUALITY.ultra.dens;
  const grassLods = LOD_TS.map((ts, li) => {
    const n = Math.ceil(maxD[li] * TILE * TILE);
    const { geo, ipb } = bladePatch(n, ts);
    const tiles = new THREE.InstancedBufferAttribute(new Float32Array(2400 * 2), 2);
    tiles.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('aTile', tiles);
    geo.instanceCount = 0;
    const U = { ...GU, uLod: { value: li } };
    Object.defineProperty(U, 'uWind', { get: () => uniforms.uWind, enumerable: true });
    Object.defineProperty(U, 'uTime', { get: () => uniforms.uTime, enumerable: true });
    const mat = grassMaterial(U);
    mat.customProgramCacheKey = () => 'gc-grass';
    const mesh = new THREE.Mesh(geo, mat);
    mesh.frustumCulled = false; mesh.receiveShadow = true; mesh.castShadow = false;
    mesh.name = `grass_lod${li}`;
    mesh.onBeforeRender = (r, _s, cam) => {
      const c = cam as THREE.PerspectiveCamera;
      const hpx = r.getDrawingBufferSize(_v2).y || 1;
      GU.uPix.value = (2 * Math.tan(THREE.MathUtils.degToRad((c.fov ?? 50) / 2))) / hpx;
    };
    group.add(mesh);
    return { mesh, geo, tiles, ipb, n };
  });
  const _v2 = new THREE.Vector2();
  const applyGrassQuality = () => {
    GU.uDens.value.set(q.dens[0], q.dens[1], q.dens[2]);
    GU.uRad.value.set(...q.rad);
    GU.uFade.value.set(...q.fade);
    grassLods.forEach((l, li) => l.geo.setDrawRange(0, Math.min(l.n, Math.ceil(q.dens[li] * TILE * TILE)) * l.ipb));
  };
  applyGrassQuality();

  // ------------------------------------------------ decor (Blender forbs + ferns)
  const loader = new THREE.TextureLoader();
  const fernTex = loader.load(assetUrl('groundcover/fern.png'));
  fernTex.colorSpace = THREE.SRGBColorSpace; fernTex.anisotropy = 4;
  const DU: Record<string, THREE.IUniform> = { uCam: GU.uCam, uDFade: { value: new THREE.Vector2(40, 60) } };
  Object.defineProperty(DU, 'uWind', { get: () => uniforms.uWind, enumerable: true });
  Object.defineProperty(DU, 'uTime', { get: () => uniforms.uTime, enumerable: true });
  const vcolMat = decorMaterial(DU, null), fernMat = decorMaterial(DU, fernTex);
  const fernDepth = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: fernTex, alphaTest: 0.45 });
  const DMAX = 6000;
  const decor = meta.decor.map((d) => {
    const lods = d.lods.map((l, li) => {
      const im = new THREE.InstancedMesh(meshGeometry(geoBuf, l, d.material === 'vcol'), d.material === 'fern' ? fernMat : vcolMat, DMAX);
      im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      im.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(DMAX * 3), 3);
      im.instanceColor.setUsage(THREE.DynamicDrawUsage);
      im.count = 0; im.visible = false; im.frustumCulled = false; im.receiveShadow = true;
      im.castShadow = false; // too small to matter in the 4k / 1.2 km sun shadow map
      if (d.material === 'fern') im.customDepthMaterial = fernDepth;
      im.name = `decor_${d.name}_lod${li}`;
      group.add(im);
      return im;
    });
    return { info: d, lods, tris: d.lods.map((l) => l.icount / 3) };
  });
  const byLayer: Record<string, number[]> = { flowers: [], weeds: [], fern: [] };
  decor.forEach((d, i) => byLayer[d.info.layer].push(i));
  const dname = (n: string) => decor.findIndex((d) => d.info.name === n);
  const FLOWER_W: [number, number, number][] = [ // proto, base weight, fallow / verge bonus
    [dname('goldenrod'), 0.2, 0.35], [dname('goldenrod_b'), 0.15, 0.3], [dname('aster'), 0.2, 0], [dname('aster_w'), 0.15, 0],
    [dname('qalace'), 0.2, 0.15], [dname('chicory'), 0.1, 0.1],
  ].filter((f) => f[0] >= 0) as [number, number, number][];

  interface DTile { n: number; proto: Uint8Array; mats: Float32Array; cols: Float32Array; cx: number; cz: number; y0: number; y1: number }
  const dcache = new Map<number, DTile>();
  const m4 = new THREE.Matrix4(), qt = new THREE.Quaternion(), eu = new THREE.Euler(), sc = new THREE.Vector3(), pv = new THREE.Vector3();
  function decorTile(ix: number, iz: number): DTile {
    const key = (ix + 4096) * 8192 + (iz + 4096);
    const hit = dcache.get(key);
    if (hit) { dcache.delete(key); dcache.set(key, hit); return hit; }
    const rng = mulberry(key * 2654435761);
    const protos: number[] = [], mats: number[] = [], cols: number[] = [];
    let y0 = 1e9, y1 = -1e9;
    const x0 = ix * DTILE, z0 = iz * DTILE;
    const layers: [string, number, number][] = [['flowers', 0.6, 0], ['weeds', 0.3, 1], ['fern', 0.4, 2]];
    for (const [layer, dens, c] of layers) {
      if (!byLayer[layer].length) continue;
      const n = Math.round(dens * DTILE * DTILE);
      for (let k = 0; k < n; k++) {
        const X = x0 + rng() * DTILE, Z = z0 + rng() * DTILE, r = rng(), r2 = rng(), r3 = rng(), r4 = rng(), r5 = rng();
        const px = X / MPP + 1000, py = Z / MPP + 333.5;
        if (px < 1 || py < 1 || px > IMG_W - 1 || py > IMG_H - 1) continue;
        let w = sample(3, c, px, py);
        if (w < 0.02) continue;
        const road = sample(0, 0, px, py) * 3.1875, rail = sample(0, 1, px, py) * 3.1875, water = sample(0, 2, px, py) * 3.1875;
        w *= smooth(1.2, 2.5, road) * smooth(1.5, 3, rail) * smooth(0.8, 1.8, water);
        const y = groundAt(px, py);
        const sl = Math.hypot(groundAt(px + 0.4, py) - groundAt(px - 0.4, py), groundAt(px, py + 0.4) - groundAt(px, py - 0.4)) / (0.8 * MPP);
        w *= layer === 'fern' ? smooth(0.45, 0.7, 1 / Math.sqrt(1 + sl * sl)) : smooth(0.72, 0.9, 1 / Math.sqrt(1 + sl * sl)); // ferns hold steep forest slopes
        if (r >= w) continue;
        let proto: number, s: number, col: [number, number, number];
        if (layer === 'flowers') {
          const bonus = Math.max(sample(3, 3, px, py), sample(0, 3, px, py));
          let tot = 0; for (const f of FLOWER_W) tot += f[1] + f[2] * bonus;
          let a = r2 * tot; proto = FLOWER_W[FLOWER_W.length - 1][0];
          for (const f of FLOWER_W) { a -= f[1] + f[2] * bonus; if (a < 0) { proto = f[0]; break; } }
          s = 0.8 + 0.4 * r3; const v = 0.8 + 0.4 * r4; col = [v, v, v];
        } else if (layer === 'weeds') {
          proto = byLayer.weeds[Math.floor(r2 * byLayer.weeds.length)]; s = 0.8 + 0.5 * r3; const v = 0.8 + 0.4 * r4; col = [v, v, v];
        } else {
          // lib_groundcover._tint_fern: autumn ferns bronze-tinged (object random -> ramp)
          const f = byLayer.fern; proto = f[r2 < 0.4 ? 0 : r2 < 0.8 ? Math.min(1, f.length - 1) : f.length - 1];
          s = 1.0 + 0.8 * r3; col = [0.8 + 0.8 * r4, 0.95 + 0.05 * r4, 0.7 - 0.25 * r4];
        }
        eu.set((r5 - 0.5) * 0.12, r5 * 97.0 % (Math.PI * 2), (r4 - 0.5) * 0.12);
        qt.setFromEuler(eu); sc.setScalar(s); pv.set(X, y - 0.02, Z);
        m4.compose(pv, qt, sc);
        protos.push(proto); mats.push(...m4.elements); cols.push(...col);
        y0 = Math.min(y0, y); y1 = Math.max(y1, y + 1.2 * s);
      }
    }
    const t: DTile = { n: protos.length, proto: Uint8Array.from(protos), mats: Float32Array.from(mats), cols: Float32Array.from(cols), cx: x0 + DTILE / 2, cz: z0 + DTILE / 2, y0, y1 };
    dcache.set(key, t);
    if (dcache.size > 900) dcache.delete(dcache.keys().next().value!);
    return t;
  }

  // ------------------------------------------------ props (lib_props instances)
  const atlas = loader.load(assetUrl('groundcover/props_albedo.jpg'));
  atlas.colorSpace = THREE.SRGBColorSpace; atlas.anisotropy = 8;
  const atlasN = loader.load(assetUrl('groundcover/props_normal.jpg'));
  atlasN.anisotropy = 4;
  const pmat = propMaterial(atlas, atlasN);
  const PV = new Float32Array(propBuf), NP = PV.length / 6;
  const protoKind: number[] = [];
  meta.prop_kinds.forEach((k, ki) => k.protos.forEach((p) => (protoKind[p] = ki)));
  const ncx = Math.ceil(IMG_W / PCELL), ncy = Math.ceil(IMG_H / PCELL), NC = ncx * ncy;
  const pProto = new Uint8Array(NP), pCell = new Uint32Array(NP), order = new Uint32Array(NP);
  let nKept = 0;
  for (let i = 0; i < NP; i++) {
    const k = PV[i * 6 + 4] | 0, pk = meta.prop_kinds[k];
    if (!pk || !pk.protos.length) continue;
    const sd = PV[i * 6 + 5];
    pProto[i] = pk.protos[Math.floor(((Math.sin(sd * 91.7 + i * 0.013) * 43758.5453) % 1 + 1) % 1 * pk.protos.length)];
    pCell[i] = Math.min(ncy - 1, Math.floor(PV[i * 6 + 1] / PCELL)) * ncx + Math.min(ncx - 1, Math.floor(PV[i * 6] / PCELL));
    order[nKept++] = i;
  }
  const ord = order.subarray(0, nKept).sort((a, b) => pCell[a] - pCell[b]);
  const pMats = new Float32Array(nKept * 16), pCols = new Float32Array(nKept * 3), pPos = new Float32Array(nKept * 3), pP = new Uint8Array(nKept);
  const cellStart = new Int32Array(NC + 1).fill(-1), cellMinY = new Float32Array(NC).fill(1e9), cellMaxY = new Float32Array(NC).fill(-1e9);
  for (let k = 0; k < nKept; k++) {
    const i = ord[k], x = PV[i * 6], y = PV[i * 6 + 1], s = Math.min(2.5, Math.max(0.4, PV[i * 6 + 3])), rot = PV[i * 6 + 5];
    const [X, , Z] = pxToWorld(x, y);
    const gy = groundAt(x, y);
    const pr = pProto[i], kind = meta.prop_kinds[protoKind[pr]];
    const r1 = ((Math.sin(rot * 12.9898 + 4.1) * 43758.5453) % 1 + 1) % 1, r2 = ((Math.sin(rot * 78.233 + 1.7) * 12543.853) % 1 + 1) % 1;
    eu.set((r1 - 0.5) * 0.12, rot, (r2 - 0.5) * 0.12);
    qt.setFromEuler(eu); sc.setScalar(s); pv.set(X, gy - 0.05, Z);
    m4.compose(pv, qt, sc).toArray(pMats, k * 16);
    const v = kind.rock ? 0.82 + 0.3 * r1 : 0.85 + 0.25 * r2;
    pCols.set(kind.rock ? [v * 1.02, v, v * 0.96] : [v, v * 0.97, v * 0.93], k * 3);
    pPos.set([X, gy, Z], k * 3); pP[k] = pr;
    const c = pCell[i];
    if (cellStart[c] < 0) cellStart[c] = k;
    cellMinY[c] = Math.min(cellMinY[c], gy - 1); cellMaxY[c] = Math.max(cellMaxY[c], gy + meta.props[pr].height * s);
  }
  cellStart[NC] = nKept;
  for (let c = NC - 1; c >= 0; c--) if (cellStart[c] < 0) cellStart[c] = cellStart[c + 1];
  const perProto = new Int32Array(meta.props.length);
  for (let k = 0; k < nKept; k++) perProto[pP[k]]++;
  const props = meta.props.map((p, pi) => ({
    info: p,
    kind: meta.prop_kinds[protoKind[pi]],
    lods: p.lods.map((l, li) => {
      const cap = Math.max(1, perProto[pi]);
      const im = new THREE.InstancedMesh(meshGeometry(geoBuf, l, false), pmat, cap);
      im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      im.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(cap * 3), 3);
      im.instanceColor.setUsage(THREE.DynamicDrawUsage);
      im.count = 0; im.visible = false; im.frustumCulled = false; im.receiveShadow = true; im.castShadow = li === 0 && ['boulder_large', 'boulder', 'log', 'log_mossy', 'stump', 'roots'].includes(p.kind);
      im.name = `prop_${p.name}_lod${li}`;
      group.add(im);
      return im;
    }),
    tris: p.lods.map((l) => l.icount / 3),
  }));

  // ------------------------------------------------ per-view update
  const frustum = new THREE.Frustum(), pm = new THREE.Matrix4(), box = new THREE.Box3(), sph = new THREE.Sphere();
  const lastPos = new THREE.Vector3(1e9, 0, 0), lastDir = new THREE.Vector3();
  const dir = new THREE.Vector3();
  let st = { tiles: [0, 0, 0], blades: 0, decor: 0, props: 0, tris: 0, draws: 0 };
  const inView = (c: THREE.Vector3, r: number, cam: THREE.Vector3) => {
    // generous frustum test: ~12 degree angular margin so small turns don't need a re-list
    const m = 0.2 * c.distanceTo(cam) + 2 + r;
    for (let i = 0; i < 5; i++) if (frustum.planes[i].distanceToPoint(c) < -m) return false; // skip the far plane
    return true;
  };

  function relist(camera: THREE.Camera) {
    const cp = camera.position;
    pm.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse); frustum.setFromProjectionMatrix(pm);
    st = { tiles: [0, 0, 0], blades: 0, decor: 0, props: 0, tris: 0, draws: 0 };
    // grass tiles
    const Rg = q.fade[1] + TILE;
    const cnt = [0, 0, 0];
    const arrs = grassLods.map((l) => l.tiles.array as Float32Array);
    for (let iz = Math.floor((cp.z - Rg) / TILE); iz <= Math.floor((cp.z + Rg) / TILE); iz++) {
      for (let ix = Math.floor((cp.x - Rg) / TILE); ix <= Math.floor((cp.x + Rg) / TILE); ix++) {
        const x0 = ix * TILE, z0 = iz * TILE;
        const px0 = x0 / MPP + 1000, py0 = z0 / MPP + 333.5;
        if (px0 > IMG_W || py0 > IMG_H || px0 + TILE / MPP < 0 || py0 + TILE / MPP < 0) continue;
        const k = TILE / MPP;
        const ha = groundAt(px0, py0), hb = groundAt(px0 + k, py0), hc = groundAt(px0, py0 + k), hd = groundAt(px0 + k, py0 + k), he = groundAt(px0 + k / 2, py0 + k / 2);
        box.min.set(x0, Math.min(ha, hb, hc, hd, he) - 0.2, z0); box.max.set(x0 + TILE, Math.max(ha, hb, hc, hd, he) + 1.3, z0 + TILE);
        const d = box.distanceToPoint(cp);
        if (d > q.fade[1]) continue;
        box.getBoundingSphere(sph);
        if (d > 6 && !inView(sph.center, sph.radius, cp)) continue;
        const li = d < q.rad[1] ? 0 : d < q.rad[3] ? 1 : 2;
        if (cnt[li] >= 2400) continue;
        arrs[li][cnt[li] * 2] = x0; arrs[li][cnt[li] * 2 + 1] = z0; cnt[li]++;
      }
    }
    grassLods.forEach((l, li) => {
      l.geo.instanceCount = cnt[li]; l.mesh.visible = cnt[li] > 0;
      l.tiles.clearUpdateRanges(); l.tiles.addUpdateRange(0, cnt[li] * 2); l.tiles.needsUpdate = true;
      const slots = Math.min(l.n, Math.ceil(q.dens[li] * TILE * TILE));
      st.tiles[li] = cnt[li]; st.blades += cnt[li] * slots; st.tris += cnt[li] * slots * (l.ipb / 3);
      if (cnt[li]) st.draws++;
    });
    // decor
    const Rd = q.decor;
    DU.uDFade.value.set(Rd * 0.7, Rd);
    const dc = decor.map((d) => d.lods.map(() => 0));
    const near = Math.min(25, Rd * 0.4);
    for (let iz = Math.floor((cp.z - Rd) / DTILE); iz <= Math.floor((cp.z + Rd) / DTILE); iz++) {
      for (let ix = Math.floor((cp.x - Rd) / DTILE); ix <= Math.floor((cp.x + Rd) / DTILE); ix++) {
        const t = decorTile(ix, iz);
        if (!t.n) continue;
        box.min.set(t.cx - DTILE / 2, t.y0, t.cz - DTILE / 2); box.max.set(t.cx + DTILE / 2, t.y1, t.cz + DTILE / 2);
        const d = box.distanceToPoint(cp);
        if (d > Rd) continue;
        box.getBoundingSphere(sph);
        if (d > 4 && !inView(sph.center, sph.radius, cp)) continue;
        for (let k = 0; k < t.n; k++) {
          const p = t.proto[k], dd = decor[p];
          const li = d < near || dd.lods.length < 2 || dd.tris[0] < 100 ? 0 : 1;
          const im = dd.lods[li], o = dc[p][li];
          if (o >= DMAX) continue;
          (im.instanceMatrix.array as Float32Array).set(t.mats.subarray(k * 16, k * 16 + 16), o * 16);
          (im.instanceColor!.array as Float32Array).set(t.cols.subarray(k * 3, k * 3 + 3), o * 3);
          dc[p][li]++;
        }
      }
    }
    decor.forEach((d, p) => d.lods.forEach((im, li) => {
      const n = dc[p][li];
      im.count = n; im.visible = n > 0;
      if (n) {
        im.instanceMatrix.clearUpdateRanges(); im.instanceMatrix.addUpdateRange(0, n * 16); im.instanceMatrix.needsUpdate = true;
        im.instanceColor!.clearUpdateRanges(); im.instanceColor!.addUpdateRange(0, n * 3); im.instanceColor!.needsUpdate = true;
        st.decor += n; st.tris += n * d.tris[li]; st.draws++;
      }
    }));
    // props
    const pc = props.map((p) => p.lods.map(() => 0));
    const maxFar = Math.max(...meta.prop_kinds.map((k) => k.far)) * q.props;
    for (let cy = 0; cy < ncy; cy++) for (let cx = 0; cx < ncx; cx++) {
      const c = cy * ncx + cx, a = cellStart[c], b = cellStart[c + 1];
      if (a >= b) continue;
      const [x0, , z0] = pxToWorld(cx * PCELL, cy * PCELL), [x1, , z1] = pxToWorld((cx + 1) * PCELL, (cy + 1) * PCELL);
      box.min.set(x0 - 5, cellMinY[c], z0 - 5); box.max.set(x1 + 5, cellMaxY[c], z1 + 5);
      if (box.distanceToPoint(cp) > maxFar || !frustum.intersectsBox(box)) continue;
      for (let k = a; k < b; k++) {
        const pr = pP[k], P = props[pr], far = P.kind.far * q.props;
        const dx = pPos[k * 3] - cp.x, dy = pPos[k * 3 + 1] - cp.y, dz = pPos[k * 3 + 2] - cp.z;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > far * far) continue;
        const li = P.kind.far <= 160 || d2 < (far * 0.22) ** 2 ? 0 : 1; // small debris: one LOD (saves draw calls)
        const im = P.lods[li], o = pc[pr][li];
        (im.instanceMatrix.array as Float32Array).set(pMats.subarray(k * 16, k * 16 + 16), o * 16);
        (im.instanceColor!.array as Float32Array).set(pCols.subarray(k * 3, k * 3 + 3), o * 3);
        pc[pr][li]++;
      }
    }
    props.forEach((p, pr) => p.lods.forEach((im, li) => {
      const n = pc[pr][li];
      im.count = n; im.visible = n > 0;
      if (n) {
        im.instanceMatrix.clearUpdateRanges(); im.instanceMatrix.addUpdateRange(0, n * 16); im.instanceMatrix.needsUpdate = true;
        im.instanceColor!.clearUpdateRanges(); im.instanceColor!.addUpdateRange(0, n * 3); im.instanceColor!.needsUpdate = true;
        st.props += n; st.tris += n * p.tris[li]; st.draws++;
      }
    }));
  }

  function update(camera: THREE.Camera) {
    const cp = camera.position;
    GU.uCam.value.copy(cp);
    camera.getWorldDirection(dir);
    if (cp.distanceToSquared(lastPos) < 4 && dir.dot(lastDir) > 0.99) return; // ~2 m / 8 degrees
    lastPos.copy(cp); lastDir.copy(dir);
    relist(camera);
  }

  return {
    group, uniforms, update,
    setQuality(qn) { q = GROUNDCOVER_QUALITY[qn]; applyGrassQuality(); lastPos.set(1e9, 0, 0); },
    stats: () => st,
  };
}
