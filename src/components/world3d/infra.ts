/**
 * Engineered roads, junctions, bridges, rail, roadside infrastructure and rivers for the web viewer.
 *
 * Geometry is exported from the real Blender build (tools/blender/export_web_infra.py: lib_roads,
 * lib_infrastructure, lib_water) into public/world/infra/. This module rebuilds the meshes and
 * reproduces the Blender materials as MeshStandardMaterial + onBeforeCompile patches (fog, shadows
 * and tone mapping stay three's own chunks):
 *   ground   one shader for asphalt / gravel / red clay / shoulder / curb concrete / sidewalk / ballast.
 *            Asphalt markings, oil strips, wheel paths, patches, cracks, tar snakes, raveling and paint
 *            wear come from the lib_roads per-vertex attributes (rl, rs, len-rs, hw, mk, np, age, surf,
 *            esh, lw, uin, sa, sb, xa, xb) exactly like MAT_Road_Asphalt.
 *   verge    ditches / cut and fill slopes: the baked terrain albedo (seamless with the terrain).
 *   struct   bridges, piers, abutments, girders, trusses, rails, walls, portals, culverts, boulders and
 *            all props: one palette shader (cast concrete, weathered steel, wood, masonry, riprap, rock).
 *   bed      riverbed + wet shore film (alpha edges);  water: depth tint, fresnel env reflection,
 *            flow-advected ripples, whitewater lace where lib_water marks foam / boulder wakes.
 *   wires    power / telephone / fence wires as screen-space ribbons (sub-pixel coverage fade).
 * Per 500 m chunk one draw per group; small groups are distance culled, props are instanced and
 * CPU culled around the camera.
 */
import * as THREE from 'three';
import { assetUrl, bin } from '../../core/data';

type Block = { f: string; o: number; n: number; t: string };
interface ChunkMeta {
  cx: number; cz: number; group: string; lo: number[]; ext: number[]; nv: number; nt: number;
  pos: Block; nrm?: Block; idx: Block; attrs: Record<string, Block>; pick: [number, string][];
}
interface PalMeta { name: string; color: number[]; rough: number; metal: number; mode: number; rust: number }
interface InfraMeta {
  version: number; chunk_m: number; origin: [number, number]; palette: PalMeta[]; textures: Record<string, string>;
  chunks: ChunkMeta[];
  protos: Record<string, { nv: number; nt: number; pos: Block; nrm: Block; idx: Block; sA: Block }>;
  instances: { name: string; proto: string; count: number; rows: Block }[];
  wires: { cx: number; cz: number; n: number; pts: Block; cnt: Block; rad: Block }[];
}

export interface InfraOptions {
  /** baked terrain albedo (public/world/albedo.jpg) for the road verges; loaded if omitted */
  albedo?: THREE.Texture;
  /** shared animation clock ({ value: seconds }) */
  time?: { value: number };
  /** pull pavement / verge / riverbed depth towards the camera (fraction of view distance) */
  depthPull?: number;
}

export interface Infra {
  group: THREE.Group;
  /** layer groups (World3D layer toggles): roads = pavement + verges + props, bridges, rail, water */
  roads: THREE.Group; bridges: THREE.Group; rail: THREE.Group; water: THREE.Group;
  /** World3D-compatible pick maps: mesh -> per-face owner id (lazy) */
  pick: Map<THREE.Mesh, string[]>;
  update(camera: THREE.Camera): void;
  stats(): { draws: number; tris: number };
}

// cull distances (m) per draw group / prop kind
const CULL: Record<string, number> = { ground: 1e9, verge: 950, struct: 1e9, detail: 750, bed: 2200, water: 1e9, wires: 520 };
function protoCull(p: string) {
  if (/Pole/.test(p)) return 900;
  if (/RailTie/.test(p)) return 260;
  if (/Sign/.test(p)) return 450;
  if (/Delineator|GuardrailPost|FencePost/.test(p)) return 330;
  return 400;
}

const TYPED: Record<string, any> = { u1: Uint8Array, i1: Int8Array, u2: Uint16Array, i2: Int16Array, u4: Uint32Array, i4: Int32Array, f4: Float32Array };
function view(bufs: Record<string, ArrayBuffer>, b: Block): any {
  return new TYPED[b.t](bufs[b.f], b.o, b.n);
}

// ------------------------------------------------------------------ GLSL
const NOISE = /* glsl */ `
float ih13(vec3 p){ p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
vec2 ih22(vec2 p){ vec3 q = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973)); q += dot(q, q.yzx + 33.33); return fract((q.xx + q.yz) * q.zy); }
vec3 ih33(vec3 p){ p = fract(p * vec3(0.1031, 0.1030, 0.0973)); p += dot(p, p.yxz + 33.33); return fract((p.xxy + p.yxx) * p.zyx); }
float vnoise(vec3 p){
  vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(ih13(i), ih13(i + vec3(1,0,0)), f.x), mix(ih13(i + vec3(0,1,0)), ih13(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(ih13(i + vec3(0,0,1)), ih13(i + vec3(1,0,1)), f.x), mix(ih13(i + vec3(0,1,1)), ih13(i + vec3(1,1,1)), f.x), f.y), f.z);
}
float fbm2(vec3 p){ return (vnoise(p) * 0.66 + vnoise(p * 2.03 + 7.1) * 0.34); }
float fbm4(vec3 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 4; i++){ s += a * vnoise(p); p = p * 2.03 + 3.7; a *= 0.5; } return s / 0.9375; }
// Blender-like noise contrast: value-noise fbm is flatter than Perlin fbm around 0.5
float bn(float n){ return n; }
// 2D cells: x = F1, y = F2 - F1 (edge proximity), z = cell id
vec3 vor2(vec2 p){
  vec2 n = floor(p), f = fract(p); float d1 = 8.0, d2 = 8.0; vec2 id = vec2(0.0);
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
    vec2 g = vec2(float(i), float(j)); vec2 r = g + ih22(n + g) - f; float d = dot(r, r);
    if (d < d1){ d2 = d1; d1 = d; id = n + g; } else if (d < d2) d2 = d;
  }
  d1 = sqrt(d1); d2 = sqrt(d2);
  return vec3(d1, d2 - d1, ih13(vec3(id, 1.7)));
}
vec3 vor3(vec3 p){
  vec3 n = floor(p), f = fract(p); float d1 = 8.0, d2 = 8.0; vec3 id = vec3(0.0);
  for (int k = -1; k <= 1; k++) for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
    vec3 g = vec3(float(i), float(j), float(k)); vec3 r = g + ih33(n + g) - f; float d = dot(r, r);
    if (d < d1){ d2 = d1; d1 = d; id = n + g; } else if (d < d2) d2 = d;
  }
  d1 = sqrt(d1); d2 = sqrt(d2);
  return vec3(d1, d2 - d1, ih13(id + 0.37));
}
// bump from a scalar height field (view-space derivatives; same maths as three's perturbNormalArb)
vec3 bumpN(vec3 p, vec3 n, float h, float s){
  vec3 dpx = dFdx(p), dpy = dFdy(p); float dhx = dFdx(h), dhy = dFdy(h);
  vec3 r1 = cross(dpy, n), r2 = cross(n, dpx); float det = dot(dpx, r1);
  vec3 g = sign(det) * (dhx * r1 + dhy * r2);
  return normalize(abs(det) * n - s * g);
}
float sstep(float a, float b, float x){ return smoothstep(a, b, x); }
float mrange(float x, float a, float b, float c, float d){ return mix(c, d, smoothstep(a, b, x)); }
`;

// depth pull (decal-like bias against the terrain): log depth writes gl_FragDepth, so polygonOffset is inert
const DEPTH_PULL = /* glsl */ `
#include <logdepthbuf_fragment>
#if defined( USE_LOGDEPTHBUF )
  gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth * uDepthPull ) * logDepthBufFC * 0.5;
#endif
`;

function patchVertexWorld(sh: THREE.WebGLProgramParametersWithUniforms, decl: string, body: string) {
  sh.vertexShader = sh.vertexShader
    .replace('#include <common>', `#include <common>\nvarying vec3 vWp;\n${decl}`)
    .replace('#include <project_vertex>', `#include <project_vertex>
  { vec4 wp4 = vec4(transformed, 1.0);
  #ifdef USE_INSTANCING
    wp4 = instanceMatrix * wp4;
  #endif
    vWp = (modelMatrix * wp4).xyz; }
  ${body}`);
}

// ------------------------------------------------------------------ ground (pavement) material
function groundMaterial(tex: Record<string, THREE.Texture>, pull: { value: number }) {
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9, metalness: 0, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4 });
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, { tAsph: { value: tex.asphalt_l }, tAsphH: { value: tex.asphalt_h }, tGrav: { value: tex.gravel_l }, tClay: { value: tex.clay_l }, uDepthPull: pull });
    patchVertexWorld(sh, `attribute vec2 gS; attribute vec2 gR; attribute vec4 gB; attribute vec4 gC; attribute vec4 gD;
varying vec2 vS; varying vec2 vR; varying vec4 vB; varying vec4 vC; varying vec4 vD;`,
      `vS = gS; vR = gR * 0.01; vB = gB; vC = gC; vD = gD;`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
varying vec3 vWp; varying vec2 vS; varying vec2 vR; varying vec4 vB; varying vec4 vC; varying vec4 vD;
uniform sampler2D tAsph; uniform sampler2D tAsphH; uniform sampler2D tGrav; uniform sampler2D tClay; uniform float uDepthPull;
${NOISE}
float gHeight = 0.5; float gBumpS = 0.0;
// anti-aliased band |u - c| < w/2 (coverage-preserving at distance)
float band(float u, float c, float w){ float aa = max(0.012, fwidth(u) * 0.8); return 1.0 - smoothstep(w * 0.5 - aa, w * 0.5 + aa, abs(u - c)); }
float dashf(float s, float on, float period){ float x = fract(s / period); float aa = max(fwidth(s / period), 1e-4); return smoothstep(0.0, aa, x) * (1.0 - smoothstep(on / period, on / period + aa, x)); }
vec3 gravelCol(vec3 p, vec2 bxy, sampler2D t, float size, vec3 c1, vec3 c2, float rl, float ruts){
  float d1 = texture(t, bxy / size).r * 2.0;
  float d2 = texture(t, mat2(0.362, 0.932, -0.932, 0.362) * bxy / (size * 3.3)).r * 2.0;
  float det = mix(d1, d2, sstep(0.4, 0.6, bn(fbm2(p * 0.08))));
  det = pow(max(det, 0.0), 0.85);
  vec3 base = mix(c1, c2, sstep(0.3, 0.7, bn(fbm2(p * 0.05))));
  vec3 c = base * det;
  if (ruts > 0.5) {
    float rut = (1.0 - sstep(0.15, 0.45, abs(abs(rl) - 0.85))) * step(abs(rl), 3.0);
    c = mix(c, vec3(0.07, 0.06, 0.05), rut * 0.3);
  }
  gHeight = d1 * 0.5; gBumpS = 0.6;
  return c;
}
vec3 concreteCol(vec3 p, vec3 col, float rs, float joints){
  float n1 = bn(fbm4(p * 0.9));
  float n2 = bn(fbm2(p * 0.05));
  vec3 c = mix(col, col * 0.78, sstep(0.3, 0.75, n1));
  c = mix(c, col * vec3(0.7, 0.72, 0.66), 0.5 * sstep(0.45, 0.75, n2));
  float st = bn(fbm2(vec3(p.x / 0.35, p.y / 0.35, p.z / 4.0)));
  c = mix(c, vec3(0.14, 0.13, 0.11), 0.35 * sstep(0.55, 0.8, st));
  if (joints > 0.0) { float j = 1.0 - smoothstep(0.0, max(fwidth(rs / joints), 0.004) + 0.012, fract(rs / joints)); c = mix(c, vec3(0.08, 0.08, 0.075), 0.6 * j); }
  gHeight = n1; gBumpS = 0.15;
  return c;
}
`)
      .replace('#include <logdepthbuf_fragment>', DEPTH_PULL)
      .replace('#include <metalnessmap_fragment>', `#include <metalnessmap_fragment>
{
  vec3 P = vec3(vWp.x, -vWp.z, vWp.y);   // Blender object space (Z-up), as in the Blender shaders
  vec2 bxy = P.xy;
  int kind = int(vC.w + 0.5);
  float rl = vR.x, hw = vR.y, rs = vS.x, eb = vS.y;
  vec3 col; float rough = 0.9;
  if (kind == 0) {
    // ---------------- MAT_Road_Asphalt (lib_roads._asphalt_material)
    float mk = vB.x, npz = vB.y / 255.0, age = vB.z / 255.0, surf = vB.w / 255.0;
    float esh = vC.x / 20.0, lw = vC.y / 20.0, uin = vC.z / 20.0;
    float au = abs(rl);
    vec2 ruv = vec2(rl, rs);
    float t1 = texture(tAsph, bxy / 3.2).r * 2.0 * 0.1098;
    float t2 = texture(tAsph, mat2(0.765, 0.644, -0.644, 0.765) * bxy / 11.0).r * 2.0 * 0.1098;
    float det = pow(mix(t1, t2, 0.35) * 2.6, 0.55);
    vec3 base = mix(vec3(0.030, 0.030, 0.032), vec3(0.085, 0.083, 0.078), age);
    base = mix(base, vec3(0.115, 0.100, 0.082), surf);
    float big = bn(fbm2(P * 0.035));
    base = mix(base, mix(base, vec3(0.13, 0.125, 0.115), 0.5), 0.35 * sstep(0.45, 0.75, big));
    col = base * det;
    float ul = max(au - uin, 0.0) / max(lw, 0.5);
    float pp = fract(ul);
    float inlane = step(au, hw - esh);
    float oil = (1.0 - sstep(0.03, 0.16, abs(pp - 0.5))) * sstep(0.35, 0.7, bn(fbm2(vec3(ruv * 0.25, 0.0)))) * inlane;
    oil *= 1.0 - surf * 0.6;
    float traffic = 0.3 + step(0.5, mk) * 0.7;
    col = mix(col, vec3(0.016, 0.016, 0.017), oil * traffic * 0.35);
    float wheel = (1.0 - sstep(0.05, 0.14, abs(abs(pp - 0.5) - 0.27))) * inlane;
    col = mix(col, mix(col, vec3(0.2, 0.19, 0.17), 0.5), wheel * age * 0.12);
    // patches (road-aligned cells)
    vec3 pc = vor2(vec2(rl / 3.4, rs / 8.5));
    float pr = ih13(vec3(floor(vec2(rl / 3.4, rs / 8.5)) + 0.5, pc.z * 17.0));
    float patchm = step(pc.z, 0.03 + age * 0.13) * step(au, hw - 0.3);
    col = mix(col, mix(vec3(0.03, 0.03, 0.032), vec3(0.05, 0.049, 0.046), pr), patchm * 0.6);
    // cracks: alligator clusters, transverse cracks, sealed centre joint
    vec3 ce = vor2(ruv / 0.32);
    float cluster = sstep(0.78 - age * 0.16, 0.86 - age * 0.14, bn(fbm2(vec3(ruv / 7.0, 3.0))));
    float crack = step(ce.y, 0.04) * cluster * step(0.4, age);
    float tr = fract(rs / 13.7 + fbm2(vec3(ruv * 0.6, 5.0)) * 0.15);
    float trans = step(tr, 0.006) * step(0.35, age);
    float joint = step(abs(rl + (fbm2(vec3(ruv * 0.8, 9.0)) - 0.5) * 0.1), 0.05) * step(0.25, age);
    float tar = max(max(crack, trans), joint * step(mk, 5.5)) * step(au, hw - 0.15);
    col = mix(col, vec3(0.012, 0.012, 0.013), tar * 0.7);
    // edge raveling
    float ravel = sstep(hw - 0.45, hw, au) * sstep(0.4, 0.62, bn(fbm4(vec3(ruv * 1.3, 1.0))));
    col = mix(col, vec3(0.12, 0.105, 0.085), ravel * 0.75);
    // markings (1971 MUTCD): yellow centre lines, white edge / lane lines
    #define IS(c) (1.0 - step(0.5, abs(mk - c)))
    float white = band(au, hw - esh, 0.12) * max(IS(1.0), max(IS(6.0), IS(7.0)));
    float dbl = max(band(rl, -0.14, 0.1), band(rl, 0.14, 0.1));
    float sdash = band(rl, 0.0, 0.1) * dashf(rs, 3.05, 12.2);
    float yellow = mix(sdash, dbl, npz) * max(IS(1.0), IS(2.0));
    yellow = max(yellow, band(rl, 0.0, 0.1) * dashf(rs, 3.05, 9.1) * IS(3.0));
    yellow = max(yellow, dbl * max(IS(4.0), IS(5.0)));
    white = max(white, band(au, lw, 0.1) * dashf(rs, 3.05, 12.2) * IS(4.0));
    white = max(white, band(au, uin + lw, 0.1) * dashf(rs, 3.05, 12.2) * IS(6.0));
    yellow = max(yellow, band(au, uin + 0.08, 0.1) * IS(6.0));
    float park = (1.0 - step(0.02, fract(rs / 6.7))) * step(lw + 0.2, au) * IS(5.0);
    white = max(white, park * step(au, hw - 0.2));
    float sa = vD.x * 0.25, sb = vD.y * 0.25, xa = vD.z * 0.25, xb = vD.w * 0.25;
    float stopB = band(eb, sb + 0.3, 0.45) * step(0.1, rl) * step(0.01, sb);
    float stopA = band(rs, sa + 0.3, 0.45) * step(rl, -0.1) * step(0.01, sa);
    float xwB = max(band(eb, xb + 0.2, 0.3), band(eb, xb + 2.8, 0.3)) * step(0.01, xb);
    float xwA = max(band(rs, xa + 0.2, 0.3), band(rs, xa + 2.8, 0.3)) * step(0.01, xa);
    white = max(white, max(max(stopA, stopB), max(xwA, xwB)) * step(au, hw - 0.3));
    float wear = age * 0.55 + max(IS(2.0), IS(3.0)) * 0.25;
    float wn = bn(fbm4(vec3(ruv / 0.6, 2.0)));
    float keep = sstep(wear * 0.75, wear * 0.75 + 0.18, wn + bn(fbm2(P * 0.02)) * 0.2 + 0.1);
    keep *= 1.0 - tar * 0.8;
    float pw = white * keep, py = yellow * keep;
    col = mix(col, vec3(0.62, 0.62, 0.58), pw);
    col = mix(col, vec3(0.62, 0.38, 0.04), py);
    rough = mix(0.9, 0.72, wheel * 0.5);
    rough = mix(rough, 0.55, max(pw, py));
    rough = mix(rough, 0.6, oil * 0.6);
    float hh = texture(tAsphH, bxy / 3.2).r;
    gHeight = hh - tar * 0.6 + max(pw, py) * 0.25; gBumpS = 0.35;
  } else if (kind == 1) {
    col = gravelCol(P, bxy, tGrav, 2.6, vec3(0.15, 0.13, 0.105), vec3(0.17, 0.11, 0.07), rl, 1.0); rough = 0.97;
  } else if (kind == 2) {
    col = gravelCol(P, bxy, tClay, 3.0, vec3(0.24, 0.105, 0.05), vec3(0.2, 0.1, 0.055), rl, 1.0); rough = 0.97;
  } else if (kind == 3) {
    col = gravelCol(P, bxy, tGrav, 2.0, vec3(0.13, 0.12, 0.1), vec3(0.11, 0.1, 0.07), rl, 0.0); rough = 0.97;
  } else if (kind == 4) {
    col = concreteCol(P, vec3(0.3, 0.295, 0.275), rs, 1.5); rough = 0.88;
  } else if (kind == 5) {
    col = concreteCol(P, vec3(0.27, 0.265, 0.245), rs, 1.5); rough = 0.88;
  } else {
    col = gravelCol(P, bxy, tGrav, 1.1, vec3(0.12, 0.115, 0.11), vec3(0.1, 0.085, 0.07), rl, 0.0); rough = 0.97;
  }
  // ACES (web) crushes the toe harder than Blender's AgX: lift the dark pavement albedos to read alike
  diffuseColor.rgb = col * (kind == 0 ? 1.7 : kind <= 3 || kind == 7 ? 1.35 : 1.0);
  roughnessFactor = rough;
  metalnessFactor = 0.0;
}`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
  normal = bumpN(-vViewPosition, normal, gHeight, gBumpS * 0.06);`);
  };
  m.customProgramCacheKey = () => 'infra-ground-1';
  return m;
}

// ------------------------------------------------------------------ verge (terrain albedo) material
function vergeMaterial(albedo: THREE.Texture, pull: { value: number }) {
  const m = new THREE.MeshStandardMaterial({ map: albedo, roughness: 0.96, metalness: 0, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -2 });
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uDepthPull = pull;
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nuniform float uDepthPull;')
      .replace('#include <logdepthbuf_fragment>', DEPTH_PULL);
  };
  m.customProgramCacheKey = () => 'infra-verge-1';
  return m;
}

// ------------------------------------------------------------------ structure palette material (+ instanced props)
function structMaterial(pal: PalMeta[], tex: Record<string, THREE.Texture>) {
  const A = pal.map((p) => new THREE.Vector4(p.color[0], p.color[1], p.color[2], p.rough));
  const B = pal.map((p) => new THREE.Vector4(p.metal, p.mode, p.rust, 0));
  while (A.length < 32) { A.push(new THREE.Vector4(0.3, 0.3, 0.3, 0.8)); B.push(new THREE.Vector4(0, 0, 0, 0)); }
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8, metalness: 0, side: THREE.DoubleSide });
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, { uPalA: { value: A }, uPalB: { value: B }, tRock: { value: tex.rock } });
    patchVertexWorld(sh, 'attribute vec4 sA; varying vec2 vSA; varying vec3 vWn;',
      `vSA = sA.xy; { vec3 on = objectNormal;
  #ifdef USE_INSTANCING
    on = mat3(instanceMatrix) * on;
  #endif
    vWn = normalize(inverse(transpose(mat3(modelMatrix))) * on); }`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
varying vec3 vWp; varying vec2 vSA; varying vec3 vWn;
uniform vec4 uPalA[32]; uniform vec4 uPalB[32]; uniform sampler2D tRock;
${NOISE}`)
      .replace('#include <metalnessmap_fragment>', `#include <metalnessmap_fragment>
float sH = 0.5, sBump = 0.0;
{
  int id = int(vSA.x + 0.5);
  vec4 pa = uPalA[id], pb = uPalB[id];
  vec3 col = pa.rgb; float rough = pa.a, metal = pb.x, rust = pb.z; int mode = int(pb.y + 0.5);
  vec3 P = vec3(vWp.x, -vWp.z, vWp.y);   // Blender object space (Z-up)
  float wet = vSA.y / 255.0;
  if (mode == 1) {            // cast concrete: blotches, large stains, vertical grime streaks
    float n1 = bn(fbm4(P * 0.9)), n2 = bn(fbm2(P * 0.05));
    col = mix(col, col * 0.78, sstep(0.3, 0.75, n1));
    col = mix(col, col * vec3(0.7, 0.72, 0.66), 0.5 * sstep(0.45, 0.75, n2));
    float st = bn(fbm2(vec3(P.x / 0.35, P.y / 0.35, P.z / 4.0)));
    col = mix(col, vec3(0.14, 0.13, 0.11), 0.35 * sstep(0.55, 0.8, st));
    // water / soot runs below deck edges and on pier tops
    col *= mix(1.0, 0.8, sstep(0.5, 0.9, bn(fbm2(vec3(P.x * 1.5, P.y * 1.5, P.z * 0.15)))) * 0.6);
    sH = n1; sBump = 0.15;
  } else if (mode == 2) {     // painted steel, weathered with rust
    float n = bn(fbm4(P * 0.7)), r = bn(fbm4(P * 2.5));
    col = mix(col, col * 0.8, sstep(0.35, 0.7, n));
    float ru = rust * sstep(0.62, 0.78, r);
    col = mix(col, vec3(0.16, 0.06, 0.025), ru);
    rough = mix(rough, 0.85, ru); metal = mix(metal, 0.1, ru);
    sH = r; sBump = 0.05;
  } else if (mode == 3) {     // weathered timber / creosote poles
    float g = bn(fbm4(vec3(P.x / 0.08, P.y / 0.08, P.z / 1.5)));
    col = mix(col, col * 0.62, sstep(0.3, 0.75, g));
    col = mix(col, vec3(0.16, 0.155, 0.15), 0.5 * sstep(0.55, 0.8, bn(fbm2(P * 0.9))));
    sH = g; sBump = 0.4;
  } else if (mode == 4) {     // rubble masonry, recessed mortar
    vec3 c = vor3(vec3(P.x / 0.55, P.y / 0.55, P.z / 0.32));
    float r2 = ih13(vec3(c.z * 91.0, 3.0, 7.0));
    vec3 stone = mix(vec3(0.2, 0.19, 0.17), vec3(0.34, 0.31, 0.27), c.z);
    stone = mix(stone, vec3(0.28, 0.2, 0.14), r2 * 0.5);
    float mortar = 1.0 - smoothstep(0.04, 0.09, c.y);
    col = mix(stone, vec3(0.42, 0.41, 0.38), mortar);
    col = mix(col, vec3(0.08, 0.1, 0.05), 0.5 * sstep(0.55, 0.8, bn(fbm2(P * 0.6))));
    sH = sstep(0.0, 0.12, c.y); sBump = 0.8;
  } else if (mode == 5) {     // granite riprap
    vec3 c = vor3(P / 0.55);
    vec3 st = mix(vec3(0.11, 0.105, 0.1), vec3(0.24, 0.23, 0.21), c.z);
    st = mix(st, vec3(0.06, 0.08, 0.035), 0.6 * sstep(0.55, 0.8, bn(fbm2(P * 0.3))));
    col = mix(st, vec3(0.03, 0.028, 0.024), 0.8 * (1.0 - smoothstep(0.015, 0.05, c.y)));
    col = mix(col, vec3(0.13, 0.11, 0.08), 0.3 * sstep(0.4, 0.7, fbm2(P * 1.3)));   // soil / silt washed between the stones
    sH = sstep(0.0, 0.2, c.y); sBump = 1.2;
  } else if (mode == 6) {     // river boulders: granite / gneiss, lichen + moss on dry tops, dark and glossy when wet
    float n = bn(fbm4(P * 0.9));
    vec3 base = mix(vec3(0.13, 0.125, 0.115), vec3(0.30, 0.285, 0.26), n);
    vec3 an = abs(vWn) + 1e-3; an /= an.x + an.y + an.z;
    vec3 tr = texture(tRock, vWp.zy / 1.2).rgb * an.x + texture(tRock, vWp.xz / 1.2).rgb * an.y + texture(tRock, vWp.xy / 1.2).rgb * an.z;
    vec3 ov = mix(2.0 * base * tr, 1.0 - 2.0 * (1.0 - base) * (1.0 - tr), step(0.5, base));
    col = mix(base, ov, 0.6);
    float up = sstep(0.55, 0.95, vWn.y);
    float moss = up * sstep(0.45, 0.65, bn(fbm2(P * 2.5))) * (1.0 - wet);
    col = mix(col, vec3(0.09, 0.12, 0.04), moss * 0.75);
    col = mix(col, vec3(0.035, 0.035, 0.03), wet * 0.8);
    rough = mix(0.8, 0.18, wet);
    vec3 v = vor3(P * 1.6);
    sH = fbm2(P * 3.0) + sstep(0.0, 0.08, v.y) * 0.6; sBump = 1.0;
  } else if (mode == 7) {     // galvanized steel: spangle + dull patches
    float n = bn(fbm2(P * 6.0));
    col = mix(col, col * 0.7, 0.3 * sstep(0.4, 0.8, n));
    rough = mix(rough, 0.6, sstep(0.5, 0.8, bn(fbm2(P * 0.8))));
  } else {
    float n = bn(fbm2(P * 6.0));
    col = mix(col, col * 0.7, 0.25 * sstep(0.4, 0.8, n));
  }
  diffuseColor.rgb = col; roughnessFactor = rough; metalnessFactor = metal;
}`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
  if (sBump > 0.0) normal = bumpN(-vViewPosition, normal, sH, sBump * 0.05);`);
  };
  m.customProgramCacheKey = () => 'infra-struct-1';
  return m;
}

// ------------------------------------------------------------------ riverbed + wet shore
function bedMaterial(tex: Record<string, THREE.Texture>, pull: { value: number }) {
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8, metalness: 0, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -2 });
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, { tRocks: { value: tex.bed_rocks }, tGravB: { value: tex.bed_gravel }, uDepthPull: pull });
    patchVertexWorld(sh, 'attribute vec4 bA; varying vec4 vBA;', 'vBA = bA;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
varying vec3 vWp; varying vec4 vBA; uniform sampler2D tRocks; uniform sampler2D tGravB; uniform float uDepthPull;
${NOISE}`)
      .replace('#include <logdepthbuf_fragment>', DEPTH_PULL)
      .replace('#include <metalnessmap_fragment>', `#include <metalnessmap_fragment>
float bH = 0.5, bBump = 0.0;
{
  vec3 P = vec3(vWp.x, -vWp.z, vWp.y);
  vec2 bxy = P.xy;
  float kind = vBA.x, depth = vBA.y / 255.0 * 8.0 - 2.0, bank = vBA.z / 255.0 * 2.0 - 1.0, e = vBA.w / 255.0;
  vec3 col; float rough, alpha;
  if (kind < 0.5) {
    vec3 cr = texture(tRocks, bxy / 1.6).rgb;
    vec3 cg = texture(tGravB, mat2(0.765, 0.644, -0.644, 0.765) * bxy / 3.1).rgb;
    cg = mix(cg, vec3(0.26, 0.24, 0.21), 0.55) * vec3(0.78, 0.76, 0.72);
    float gf = sstep(-0.1, -0.6, bank);
    col = mix(cr, cg, gf);
    col = mix(col, col * vec3(0.23, 0.26, 0.12), bn(fbm2(P * 0.05)) * 0.45);
    col = mix(col, vec3(0.05, 0.055, 0.045), mrange(depth, 0.0, 2.5, 0.0, 0.75));
    col = mix(col, col * vec3(0.35, 0.33, 0.30), mrange(depth, -0.5, 0.02, 0.0, 0.55));
    rough = mrange(depth, -0.7, 0.05, 0.85, 0.3);
    alpha = sstep(0.15, 0.45, e + (fbm4(P * 0.6) - 0.5) * 0.6);
    bH = dot(cr, vec3(0.33)); bBump = 0.9;
  } else {
    float n2 = bn(fbm4(P * 6.0));
    col = mix(vec3(0.030, 0.025, 0.018), vec3(0.070, 0.058, 0.040), n2);
    rough = mrange(e, 0.0, 1.0, 0.7, 0.28);
    alpha = mrange(e - bn(fbm4(P * 0.8)) * 0.3, 0.0, 0.45, 0.0, 0.85);
    bH = n2; bBump = 0.3;
  }
  diffuseColor = vec4(col, alpha); roughnessFactor = rough; metalnessFactor = 0.0;
}`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
  normal = bumpN(-vViewPosition, normal, bH, bBump * 0.05);`);
  };
  m.customProgramCacheKey = () => 'infra-bed-1';
  return m;
}

// ------------------------------------------------------------------ water surface
function waterMaterial(time: { value: number }) {
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.05, metalness: 0, transparent: true, depthWrite: true, envMapIntensity: 0.75 });
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, { uTime: time });
    patchVertexWorld(sh, 'attribute vec4 wA; attribute vec4 wF; varying vec4 vWA; varying vec3 vFl;', 'vWA = wA; vFl = wF.xyz;');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
varying vec3 vWp; varying vec4 vWA; varying vec3 vFl; uniform float uTime;
${NOISE}
float wFoam = 0.0, wBnd = 0.0; vec2 wGrad = vec2(0.0);`)
      .replace('#include <metalnessmap_fragment>', `#include <metalnessmap_fragment>
{
  float depth = vWA.x / 255.0 * 6.0, foam = vWA.y / 255.0, wake = vWA.z / 255.0, shore = vWA.w / 255.0 * 6.0;
  vec2 fl = vFl.xy / 127.0; wBnd = clamp(vFl.z / 127.0, 0.0, 1.0); float fll = length(fl); fl = fll > 1e-3 ? fl / fll : vec2(1.0, 0.0);
  vec2 p = vWp.xz;
  float along = dot(p, fl), across = fl.x * p.y - fl.y * p.x;
  float famt = max(foam, wake * 0.9);
  float speed = 0.35 + 1.4 * famt;
  // flow-advected, flow-stretched ripple height (two octaves) -> gradient by finite differences
  vec2 q = vec2(along * 0.18 - uTime * speed * 0.18, across * 0.9);
  vec2 q2 = vec2(along * 0.8 - uTime * speed * 0.9, across * 3.2);
  float e = 0.05;
  #define RH(v, w) (vnoise(vec3(v, uTime * 0.12)) + vnoise(vec3(w, 3.0 + uTime * 0.3)) * (0.25 + famt * 1.2))
  float h0 = RH(q, q2);
  float hx = RH(q + vec2(e * 0.18, 0.0), q2 + vec2(e * 0.8, 0.0));
  float hy = RH(q + vec2(0.0, e * 0.9), q2 + vec2(0.0, e * 3.2));
  vec2 g = vec2(hx - h0, hy - h0) / e;               // d/d(along), d/d(across)
  float amp = (0.035 + 0.16 * max(foam, wake)) / (1.0 + length(vViewPosition) / 90.0);   // fade ripples (aliasing) with distance
  wGrad = (fl * g.x + vec2(-fl.y, fl.x) * g.y) * amp;
  // whitewater lace (flow-aligned streaks + cells), only where lib_water marks it
  vec2 fc2 = vec2(along * 0.6 - uTime * speed * 0.6, across * 2.4);
  float fn = fbm4(vec3(fc2 * 1.1, uTime * 0.2));
  vec3 fv = vor2(fc2 * 1.4 + vec2(uTime * 0.1, 0.0));
  float lace = fn * 0.75 + (0.25 - fv.y * 0.5) * 0.9;
  float thr = 1.12 - famt * 0.42;
  wFoam = sstep(0.0, 0.12, lace - thr) * sstep(0.08, 0.35, famt);
  // body: clear tinted shallows (the bed shows through) -> dark tea-green pools
  float dfac = (1.0 - exp(-depth / 1.1)) * mrange(shore, 0.0, 2.5, 0.35, 0.92);
  vec3 body = mix(vec3(0.16, 0.2, 0.17), vec3(0.010, 0.030, 0.028), sstep(0.0, 1.0, dfac));
  float alpha = mix(0.18, 0.93, dfac);
  diffuseColor = vec4(mix(body, vec3(0.82, 0.85, 0.84), wFoam), mix(alpha, 0.96, wFoam) * (1.0 - wBnd));
  roughnessFactor = mix(0.04, 0.55, wFoam);
  metalnessFactor = 0.0;
}`)
      .replace('#include <normal_fragment_begin>', `
  float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
  vec3 nW = normalize(vec3(-wGrad.x, 1.0, -wGrad.y));
  vec3 normal = normalize((viewMatrix * vec4(nW, 0.0)).xyz);
  vec3 nonPerturbedNormal = normalize((viewMatrix * vec4(0.0, 1.0, 0.0, 0.0)).xyz);`)
      .replace('#include <opaque_fragment>', `
  { vec3 V = normalize(vViewPosition); float F = 0.02 + 0.98 * pow(1.0 - clamp(dot(normal, V), 0.0, 1.0), 5.0);
    diffuseColor.a = clamp(diffuseColor.a + F * (1.0 - diffuseColor.a), 0.0, 1.0) * (1.0 - wBnd); }
#include <opaque_fragment>`);
  };
  m.customProgramCacheKey = () => 'infra-water-1';
  return m;
}

// ------------------------------------------------------------------ wires (screen-space ribbons)
function wireMaterial(pixK: { value: number }) {
  const m = new THREE.MeshBasicMaterial({ color: new THREE.Color().setRGB(0.035, 0.034, 0.032), transparent: true, depthWrite: false, side: THREE.DoubleSide });
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uPixK = pixK;
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nattribute vec3 wT; attribute vec2 wS; uniform float uPixK; varying float vCov;')
      .replace('#include <begin_vertex>', `
  vec3 transformed = vec3(position);
  { vec3 toC = cameraPosition - position; float d = length(toC);
    vec3 s = normalize(cross(wT, toC / max(d, 1e-3)));
    float px = uPixK * d;                    // metres per pixel at this distance
    float w = max(wS.y, px * 0.9);
    vCov = clamp(max(wS.y / w, 0.35), 0.0, 1.0) * (1.0 - smoothstep(380.0, 520.0, d));
    transformed += s * wS.x * w; }`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vCov;')
      .replace('#include <opaque_fragment>', 'diffuseColor.a *= vCov;\n#include <opaque_fragment>');
  };
  m.customProgramCacheKey = () => 'infra-wire-1';
  return m;
}

// ------------------------------------------------------------------ build
export async function buildInfra(opts: InfraOptions = {}): Promise<Infra> {
  const meta: InfraMeta = await fetch(assetUrl('infra/infra.json')).then((r) => r.json());
  const files = new Set<string>();
  for (const c of meta.chunks) files.add(c.pos.f);
  for (const p of Object.values(meta.protos)) files.add(p.pos.f);
  const bufs: Record<string, ArrayBuffer> = {};
  const loader = new THREE.TextureLoader();
  const loadTex = (name: string, srgb: boolean, repeat = true) => {
    const t = loader.load(assetUrl('infra/' + name));
    t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.anisotropy = 8;
    return t;
  };
  const tex: Record<string, THREE.Texture> = {
    asphalt_l: loadTex('asphalt_l.jpg', false), asphalt_h: loadTex('asphalt_h.jpg', false), gravel_l: loadTex('gravel_l.jpg', false),
    clay_l: loadTex('clay_l.jpg', false), bed_rocks: loadTex('bed_rocks.jpg', true), bed_gravel: loadTex('bed_gravel.jpg', true), rock: loadTex('rock.jpg', true),
  };
  let albedo = opts.albedo;
  if (!albedo) { albedo = loader.load(assetUrl('albedo.jpg')); albedo.colorSpace = THREE.SRGBColorSpace; albedo.anisotropy = 8; }
  await Promise.all([...files].map(async (f) => { bufs[f] = await bin('infra/' + f); }));

  const time = opts.time ?? { value: 0 };
  const pull = { value: 1 - (opts.depthPull ?? 0.003) };
  const pixK = { value: 0.001 };
  const mats: Record<string, THREE.Material> = {
    ground: groundMaterial(tex, pull), verge: vergeMaterial(albedo, pull), struct: structMaterial(meta.palette, tex),
    bed: bedMaterial(tex, pull), water: waterMaterial(time), wires: wireMaterial(pixK),
  };
  mats.detail = mats.struct;

  const group = new THREE.Group(); group.name = 'infra';
  const roads = new THREE.Group(); roads.name = 'roads';
  const bridges = new THREE.Group(); bridges.name = 'bridges';
  const rail = new THREE.Group(); rail.name = 'rail';
  const water = new THREE.Group(); water.name = 'water';
  group.add(roads, bridges, rail, water);
  const pick = new Map<THREE.Mesh, string[]>();
  const culled: { obj: THREE.Object3D; box: THREE.Box3; dist: number }[] = [];
  let tris = 0;

  // ---- chunk meshes
  const unit = new THREE.Box3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(1, 1, 1));
  const unitS = new THREE.Sphere(new THREE.Vector3(0.5, 0.5, 0.5), Math.sqrt(3) / 2);
  for (const c of meta.chunks) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(view(bufs, c.pos), 3, true));
    if (c.nrm) {
      // stored world normals; mesh.scale = ext -> pre-scale so normalMatrix (S^-1) restores them
      const src: Int8Array = view(bufs, c.nrm);
      const out = new Int8Array(src.length);
      const [ex, ey, ez] = c.ext;
      for (let i = 0; i < src.length; i += 4) {
        const x = src[i] * ex, y = src[i + 1] * ey, z = src[i + 2] * ez;
        const l = Math.hypot(x, y, z) || 1;
        out[i] = Math.round((x / l) * 127); out[i + 1] = Math.round((y / l) * 127); out[i + 2] = Math.round((z / l) * 127);
      }
      g.setAttribute('normal', new THREE.InterleavedBufferAttribute(new THREE.InterleavedBuffer(out, 4), 3, 0, true));
    } else {
      const up = new Int8Array(c.nv * 4);
      for (let i = 1; i < up.length; i += 4) up[i] = 127;
      g.setAttribute('normal', new THREE.InterleavedBufferAttribute(new THREE.InterleavedBuffer(up, 4), 3, 0, true));
    }
    const itemSize: Record<string, number> = { gS: 2, gR: 2, gB: 4, gC: 4, gD: 4, wA: 4, wF: 4, bA: 4, sA: 4 };
    for (const [k, b] of Object.entries(c.attrs)) {
      g.setAttribute(k, new THREE.BufferAttribute(view(bufs, b), itemSize[k] ?? 4, false));
    }
    g.setIndex(new THREE.BufferAttribute(view(bufs, c.idx), 1));
    g.boundingBox = unit.clone(); g.boundingSphere = unitS.clone();
    const mesh = new THREE.Mesh(g, mats[c.group]);
    mesh.position.fromArray(c.lo); mesh.scale.fromArray(c.ext);
    mesh.matrixAutoUpdate = false; mesh.updateMatrix();
    mesh.name = `infra_${c.group}_${c.cx}_${c.cz}`;
    mesh.receiveShadow = true;
    mesh.castShadow = c.group === 'struct' || c.group === 'detail';
    mesh.renderOrder = c.group === 'bed' ? 1 : c.group === 'water' ? 2 : 0;
    mesh.userData.kind = c.group === 'struct' ? 'bridge' : c.group === 'water' || c.group === 'bed' ? 'water' : 'road';
    tris += c.nt;
    const parent = c.group === 'water' || c.group === 'bed' ? water : c.group === 'struct' ? bridges : roads;
    parent.add(mesh);
    const box = new THREE.Box3(new THREE.Vector3().fromArray(c.lo), new THREE.Vector3().fromArray(c.lo).add(new THREE.Vector3().fromArray(c.ext)));
    culled.push({ obj: mesh, box, dist: CULL[c.group] ?? 1e9 });
    if (c.group === 'ground' || c.group === 'struct' || c.group === 'verge') {
      const starts = c.pick.map((p) => p[0]);
      const ids = c.pick.map((p) => p[1].replace(/^(ROAD|JUNCTION|BRIDGE|RAIL) /, '').replace(/ TIES$/, ''));
      const lookup = (f: number) => {
        let lo = 0, hi = starts.length - 1;
        while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (starts[mid] <= f) lo = mid; else hi = mid - 1; }
        return ids[lo];
      };
      pick.set(mesh, new Proxy([] as string[], { get: (t, k) => (typeof k === 'string' && /^\d+$/.test(k) ? lookup(+k) : (t as any)[k]) }));
    }
  }

  // ---- instanced props
  const protoGeo: Record<string, THREE.BufferGeometry> = {};
  for (const [name, p] of Object.entries(meta.protos)) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(view(bufs, p.pos), 3));
    g.setAttribute('normal', new THREE.InterleavedBufferAttribute(new THREE.InterleavedBuffer(view(bufs, p.nrm), 4), 3, 0, true));
    g.setAttribute('sA', new THREE.BufferAttribute(view(bufs, p.sA), 4, false));
    g.setIndex(new THREE.BufferAttribute(view(bufs, p.idx), 1));
    protoGeo[name] = g;
  }
  interface InstSet { mesh: THREE.InstancedMesh; rows: Float32Array; n: number; r2: number }
  const insts: InstSet[] = [];
  for (const it of meta.instances) {
    const g = protoGeo[it.proto];
    if (!g) continue;
    const rows: Float32Array = view(bufs, it.rows);
    const cap = Math.min(it.count, /RailTie/.test(it.proto) ? 2400 : 1200);
    const mesh = new THREE.InstancedMesh(g, mats.struct, cap);
    mesh.count = 0; mesh.frustumCulled = false;
    mesh.castShadow = !/RailTie|Delineator/.test(it.proto); mesh.receiveShadow = true;
    mesh.name = 'infra_inst_' + it.name;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    (/RailTie/.test(it.proto) ? rail : roads).add(mesh);
    const r = protoCull(it.proto);
    insts.push({ mesh, rows, n: it.count, r2: r * r });
  }

  // ---- wires
  for (const w of meta.wires) {
    const pts: Float32Array = view(bufs, w.pts), cnt: Uint16Array = view(bufs, w.cnt), rad: Uint8Array = view(bufs, w.rad);
    const np = pts.length / 3;
    const pos = new Float32Array(np * 6), tan = new Float32Array(np * 6), ws = new Float32Array(np * 4);
    const idx: number[] = [];
    let k = 0;
    for (let li = 0; li < cnt.length; li++) {
      const n = cnt[li], r = rad[li] / 1000;
      for (let j = 0; j < n; j++) {
        const i = k + j, a = k + Math.max(j - 1, 0), b = k + Math.min(j + 1, n - 1);
        let tx = pts[b * 3] - pts[a * 3], ty = pts[b * 3 + 1] - pts[a * 3 + 1], tz = pts[b * 3 + 2] - pts[a * 3 + 2];
        const l = Math.hypot(tx, ty, tz) || 1; tx /= l; ty /= l; tz /= l;
        for (let s = 0; s < 2; s++) {
          const v = i * 2 + s;
          pos.set([pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]], v * 3);
          tan.set([tx, ty, tz], v * 3);
          ws.set([s ? 1 : -1, r], v * 2);
        }
        if (j < n - 1) { const v = i * 2; idx.push(v, v + 1, v + 2, v + 1, v + 3, v + 2); }
      }
      k += n;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('wT', new THREE.BufferAttribute(tan, 3));
    g.setAttribute('wS', new THREE.BufferAttribute(ws, 2));
    g.setIndex(idx);
    g.computeBoundingBox(); g.computeBoundingSphere();
    g.boundingSphere!.radius += 2;
    const mesh = new THREE.Mesh(g, mats.wires);
    mesh.name = `infra_wires_${w.cx}_${w.cz}`; mesh.renderOrder = 3;
    mesh.onBeforeRender = (renderer, _s, cam) => {
      const pc = cam as THREE.PerspectiveCamera;
      const h = renderer.getDrawingBufferSize(new THREE.Vector2()).y || 1;
      pixK.value = (2 * Math.tan(THREE.MathUtils.degToRad(pc.fov ?? 50) / 2)) / h;   // metres per device pixel at 1 m
    };
    roads.add(mesh);
    culled.push({ obj: mesh, box: g.boundingBox!.clone(), dist: CULL.wires });
    tris += idx.length / 3;
  }

  // ---- culling / LOD
  const last = new THREE.Vector3(1e9, 0, 0);
  const tmpM = new THREE.Matrix4(), tmpQ = new THREE.Quaternion(), tmpE = new THREE.Euler(0, 0, 0, 'YZX'), tmpP = new THREE.Vector3(), tmpS = new THREE.Vector3();
  function update(camera: THREE.Camera) {
    const cp = camera.getWorldPosition(tmpP.clone());
    for (const c of culled) c.obj.visible = c.dist >= 1e8 || c.box.distanceToPoint(cp) < c.dist;
    if (cp.distanceToSquared(last) < 36) return;
    last.copy(cp);
    for (const s of insts) {
      const R = s.rows, cap = s.mesh.instanceMatrix.count;
      let n = 0;
      for (let i = 0; i < s.n && n < cap; i++) {
        const o = i * 7;
        const dx = R[o] - cp.x, dy = R[o + 1] - cp.y, dz = R[o + 2] - cp.z;
        if (dx * dx + dy * dy + dz * dz > s.r2) continue;
        tmpE.set(R[o + 3], R[o + 4], R[o + 5], 'YZX'); tmpQ.setFromEuler(tmpE);
        tmpS.setScalar(R[o + 6]);
        tmpM.compose(new THREE.Vector3(R[o], R[o + 1], R[o + 2]), tmpQ, tmpS);
        s.mesh.setMatrixAt(n++, tmpM);
      }
      s.mesh.count = n;
      s.mesh.instanceMatrix.needsUpdate = true;
    }
  }

  function stats() {
    let draws = 0, t = 0;
    for (const l of [roads, bridges, rail, water]) l.traverseVisible((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      draws++;
      const ic = (m as any).isInstancedMesh ? (m as THREE.InstancedMesh).count : 1;
      t += ((m.geometry.index ? m.geometry.index.count : m.geometry.attributes.position.count) / 3) * ic;
    });
    return { draws, tris: Math.round(t) };
  }
  void tris;
  return { group, roads, bridges, rail, water, pick, update, stats };
}
