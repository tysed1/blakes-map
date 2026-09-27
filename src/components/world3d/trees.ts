import * as THREE from 'three';
import { pxToWorld, IMG_W, IMG_H } from '../../core/coords';
import { assetUrl, bin, Heightfield } from '../../core/data';

/**
 * The Blender leaf-card forest (tools/blender/lib_trees.py) in three.js.
 *
 * Geometry: tools/blender/export_web_trees.py exports every species at 5 LODs from the same seed
 * (cards + noise cores + limbs near, core-only ellipsoids at the horizon). Instances are the
 * ecosystem scatter (vegetation_f32.bin) that Blender instances, with lib_trees' per-instance
 * tints, sink and tilt. Instances are pre-sorted by (species, cell); every few frames each
 * 50 m cell picks the LODs its distance range overlaps (frustum-culled), and the visible ranges
 * are block-copied into one InstancedMesh per species x LOD (~250 draw calls for 215k plants).
 * LOD switches are invisible: every LOD owns a distance band and fades in / out per instance with
 * a screen-space dither over a short cross-fade zone (complementary noise, so the outgoing and
 * incoming LODs tile the pixels exactly); cells inside a fade zone are emitted to both LODs and the
 * vertex shader collapses instances outside their LOD's band. Shadows switch at the band middle.
 *
 * Shading follows the Blender materials: leaf = card texture x instance tint x LEAF_K x crown AO,
 * per-card value jitter, crown-volume normals, 32 % warm translucency toward the sun; core = the
 * tinted lifted mass behind the cards; bark = triplanar Poly Haven bark with the species tint.
 */

interface LodInfo { vcount: number; icount: number; groups: [number, number, number][]; pos: number; nrm: number; uv: number; al: number; idx: number }
interface SpeciesInfo { name: string; family: string; card: string | null; bark: string; radius: number; height: number; lods: LodInfo[] }
interface GeoJson { species: SpeciesInfo[]; lods: number; palettes: Record<string, [number[], number][]>; leaf_k: number }

const CELL = 20; // source px (50 m)
const BARK_TINT: Record<string, [number, number, number, string]> = {
  brown: [0.55, 0.5, 0.45, 'brown'], grey: [0.55, 0.55, 0.52, 'brown'], pine: [0.6, 0.5, 0.44, 'pine'], dead: [0.75, 0.73, 0.7, 'brown'],
  pale: [0.95, 0.92, 0.84, 'brown'], cinnamon: [0.78, 0.55, 0.42, 'brown'],
};
/** lib_trees.autumn_weight: warm palette entries x0.25 (green coves) .. x3 (high south crests). */
const autumnWeight = (a: number) => 0.25 + 2.75 * Math.pow(Math.min(Math.max(a, 0), 1), 1.2);
const SMALL = new Set(['Sapling_HW_A', 'Sapling_HW_B', 'Sapling_Pine', 'Rhododendron_A', 'Laurel_B', 'Brush_A', 'Brush_B', 'Dogwood']);

export interface TreeQuality { lodDist: [number, number, number, number]; smallFar: number; shadowLod: number }
// LOD switch distances (m): 0 full | 1 30 % cards | 2 cores + sparse cards | 3 four cores | 4 shared crown billboard.
// Cross-fades hide the switches, so they sit where the pixel footprint allows (triangle budget, R4 impostors next).
export const TREE_QUALITY: Record<string, TreeQuality> = {
  low: { lodDist: [30, 90, 220, 450], smallFar: 110, shadowLod: -1 },
  medium: { lodDist: [40, 120, 280, 580], smallFar: 150, shadowLod: 0 },
  high: { lodDist: [38, 130, 320, 700], smallFar: 180, shadowLod: 1 },
  ultra: { lodDist: [80, 230, 500, 1000], smallFar: 280, shadowLod: 1 },
};

// ---------------------------------------------------------------- shading
export interface FoliageUniforms { uTime: { value: number }; uSunDir: { value: THREE.Vector3 }; uWind: { value: number }; uGain: { value: number } }
/** Per-LOD cross-fade band (x..y fade in, z..w fade out, metres from uCamPos) + small-plant far fade. */
interface FadeUniforms { uFade: { value: THREE.Vector4 }; uFar: { value: THREE.Vector2 }; uCamPos: { value: THREE.Vector3 } }

const FADE_VERT_PARS = `
  uniform vec4 uFade; uniform vec2 uFar; uniform vec3 uCamPos;
  // > 0: visible fraction on a fade-out edge; < 0: -fraction on a fade-in edge (complementary dither)
  float treeFade(vec3 ip) {
    float d = distance(ip, uCamPos);
    float fin = clamp((d - uFade.x) / (uFade.y - uFade.x), 0.0, 1.0);
    if (fin < 1.0) return -fin;
    return (1.0 - clamp((d - uFade.z) / (uFade.w - uFade.z), 0.0, 1.0)) * (1.0 - clamp((d - uFar.x) / (uFar.y - uFar.x), 0.0, 1.0));
  }`;
const FADE_FRAG = `
  {
    float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    if ((vFade < 0.0 ? 1.0 - ign : ign) >= abs(vFade)) discard;
  }`;

/**
 * One merged material per species mesh (card x bark kind x LOD): bark, crown cores and leaf cards in a
 * single draw call. The per-vertex material id rides in the normal's 4th byte (export_web_trees.py:
 * 0 bark, 64 core, 127 card) and selects the shading branch:
 *   card: card texture x instance tint x LEAF_K x crown AO, per-card value jitter, normal map, crown-volume
 *         normals (no back-face flip), 32 % warm translucency toward the sun (MAT_Foliage_<card>)
 *   core: tinted mass with the leaf-card pattern projected on it, silhouette dissolved into leaf clusters
 *   bark: triplanar bark texture (object coords, Blender box projection) x the species bark tint
 */
function treeMaterial(map: THREE.Texture, nmap: THREE.Texture, leafK: number, bark: THREE.Texture, barkTint: [number, number, number],
  barkScale: number, coreLeaf: THREE.Texture, u: FoliageUniforms, f: FadeUniforms) {
  const m = new THREE.MeshStandardMaterial({
    map, normalMap: nmap, normalScale: new THREE.Vector2(0.55, 0.55), alphaTest: 0.5, side: THREE.DoubleSide,
    roughness: 0.62, metalness: 0, envMapIntensity: 0.55,
  });
  m.alphaToCoverage = true;
  m.defines = { FOLIAGE: '' };
  m.onBeforeCompile = (s) => {
    Object.assign(s.uniforms, u, f, {
      uBark: { value: bark }, uBarkTint: { value: new THREE.Vector3(...barkTint) }, uBarkScale: { value: barkScale }, tLeaf: { value: coreLeaf },
    });
    s.vertexShader = s.vertexShader
      .replace('#include <common>', `#include <common>
        attribute vec2 al; attribute float mid;
        uniform float uTime; uniform float uWind;
        varying vec2 vAL; varying vec3 vTint; varying vec3 vObj; varying float vFade; varying float vMid;
        ${FADE_VERT_PARS}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        vAL = al; vObj = position; vMid = mid;
        #ifdef USE_INSTANCING_COLOR
          vTint = instanceColor;
        #else
          vTint = vec3(1.0);
        #endif
        #ifdef USE_INSTANCING
        {
          vec3 ip = instanceMatrix[3].xyz;
          float ph = ip.x * 0.043 + ip.z * 0.031;
          float hh = max(position.y, 0.0);
          float sw = uWind * hh * hh * 0.0009;
          transformed.x += sin(uTime * 0.9 + ph) * sw;
          transformed.z += cos(uTime * 0.7 + ph * 1.3) * sw * 0.8;
          transformed += normal * sin(uTime * 3.1 + ph * 7.0 + position.y * 1.7 + position.x) * 0.045 * uWind * step(0.75, mid);
        }
        #endif`)
      .replace('#include <project_vertex>', `#include <project_vertex>
        #ifdef USE_INSTANCING
          vFade = treeFade((modelMatrix * vec4(instanceMatrix[3].xyz, 1.0)).xyz);
          if (abs(vFade) < 0.004) gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // outside this LOD's band: collapse
        #else
          vFade = 1.0;
        #endif`);
    s.fragmentShader = s.fragmentShader
      .replace('#include <common>', `#include <common>
        varying vec2 vAL; varying vec3 vTint; varying vec3 vObj; varying float vFade; varying float vMid;
        uniform vec3 uSunDir; uniform float uGain;
        uniform sampler2D uBark, tLeaf; uniform vec3 uBarkTint; uniform float uBarkScale;
        float gLeaf;`)
      .replace('#include <clipping_planes_fragment>', FADE_FRAG + '\n#include <clipping_planes_fragment>')
      .replace('#include <color_fragment>', '')
      .replace('#include <normal_fragment_begin>', THREE.ShaderChunk.normal_fragment_begin.replace('float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;', 'float faceDirection = 1.0;'))
      .replace('#include <map_fragment>', `#include <map_fragment>
        gLeaf = step(0.75, vMid);
        if (vMid > 0.75) {
          diffuseColor.rgb *= vTint * ${leafK.toFixed(3)} * vAL.x * mix(0.85, 1.15, vAL.y) * uGain;
        } else if (vMid > 0.25) {
          // core: dense crown interior, leaf-card pattern projected on it (dark gaps between clusters)
          vec3 q = vObj / 1.35;
          vec3 w = abs(normalize(cross(dFdx(vObj), dFdy(vObj)))); w = pow(w, vec3(3.0)); w /= dot(w, vec3(1.0));
          vec4 lf = texture2D(tLeaf, q.zy) * w.x + texture2D(tLeaf, q.xz) * w.y + texture2D(tLeaf, q.xy) * w.z;
          float lum = dot(lf.rgb, vec3(0.3, 0.55, 0.15)) * 1.9;
          float br = mix(0.28, 1.0, lf.a) * mix(1.0, clamp(lum, 0.6, 1.4), lf.a);
          diffuseColor = vec4(vTint * vAL.x * br * uGain, 1.0);
          // dissolve the core's silhouette into leaf clusters (no smooth 'balloon' outline)
          vec3 vn = normalize(cross(dFdx(vViewPosition), dFdy(vViewPosition)));
          float rim = 1.0 - abs(dot(vn, normalize(vViewPosition)));
          if (rim > 0.35 && lf.a < smoothstep(0.35, 0.9, rim)) discard;
        } else {
          vec3 p = vObj / uBarkScale;
          vec3 w = abs(normalize(cross(dFdx(vObj), dFdy(vObj)))); w = pow(w, vec3(4.0)); w /= dot(w, vec3(1.0));
          vec3 t = texture2D(uBark, p.yz * vec2(0.5, 1.0)).rgb * w.x + texture2D(uBark, p.xy * vec2(1.0, 0.5)).rgb * w.z + texture2D(uBark, p.xz).rgb * w.y;
          diffuseColor = vec4(uBarkTint * t * vAL.x * 1.1, 1.0);
        }`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
        if (vMid < 0.75) normal = nonPerturbedNormal;`)
      .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
        roughnessFactor = vMid > 0.75 ? 0.62 : vMid > 0.25 ? 0.85 : 0.9;`)
      // cards: 68 % principled + 32 % translucent BSDF with a warm transmission tint (backlit leaves glow);
      // directLight.color already carries the shadow term
      .replace('#include <lights_physical_pars_fragment>', THREE.ShaderChunk.lights_physical_pars_fragment.replace(
        'reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );',
        `float back = ( saturate( dot( - geometryNormal, directLight.direction ) ) * 0.6 + pow( saturate( - dot( geometryViewDir, directLight.direction ) ), 3.0 ) * 0.5 ) * gLeaf;
        reflectedLight.directDiffuse += ( irradiance * mix( 1.0, 0.68, gLeaf ) + back * directLight.color * vec3( 1.35, 1.2, 0.62 ) * 0.26 ) * BRDF_Lambert( material.diffuseColor );`));
  };
  m.customProgramCacheKey = () => 'tree';
  return m;
}

/**
 * Horizon LOD: one camera-facing crown card per tree for every species in a single draw call (2 tris).
 * Per-instance ibb = (crown centre height, half width, half height, shape 0 round .. 1 conical); the card is
 * lit as a sphere-ish crown (view-space normal from the card position), its outline frayed by the leaf-card
 * alpha and its body broken up by the leaf-card colour, like the core material it replaces.
 */
function billboardMaterial(u: FoliageUniforms, leafTex: THREE.Texture, f: FadeUniforms) {
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9, metalness: 0, envMapIntensity: 0.5 });
  m.alphaToCoverage = true;
  m.onBeforeCompile = (s) => {
    Object.assign(s.uniforms, u, f, { tLeaf: { value: leafTex } });
    s.vertexShader = s.vertexShader
      .replace('#include <common>', `#include <common>
        attribute vec4 ibb;
        varying vec3 vTint; varying vec4 vBB; varying float vFade;
        ${FADE_VERT_PARS}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        #ifdef USE_INSTANCING_COLOR
          vTint = instanceColor;
        #else
          vTint = vec3(1.0);
        #endif
        // object-space stand-in on the crown (worldpos / shadow / baked-sun chunks read 'transformed':
        // the card must sit at crown height, not at the trunk base)
        transformed = vec3(0.0, ibb.x + position.y * ibb.z, 0.0);`)
      .replace('#include <project_vertex>', `
        vec4 mvPosition = vec4(0.0, 0.0, 0.0, 1.0);
        #ifdef USE_INSTANCING
          vec3 ip = (modelMatrix * vec4(instanceMatrix[3].xyz, 1.0)).xyz;
          float isc = length(instanceMatrix[0].xyz);
          vec3 cc = ip + vec3(0.0, ibb.x * isc, 0.0);
          vec3 vd = normalize(cc - uCamPos);
          // seen from above the crown is round: the card's height blends from crown height to crown width
          float rup = mix(ibb.z, ibb.y, abs(vd.y));
          mvPosition = viewMatrix * vec4(cc, 1.0) + vec4(position.x * ibb.y * isc, position.y * rup * isc, 0.0, 0.0);
          vFade = ibb.y > 0.0 ? treeFade(ip) : 0.0;
          vBB = vec4(position.xy, ibb.w, fract(ip.x * 0.371 + ip.z * 0.613));
        #endif
        gl_Position = projectionMatrix * mvPosition;
        #ifdef USE_INSTANCING
          if (abs(vFade) < 0.004) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        #endif`);
    s.fragmentShader = s.fragmentShader
      .replace('#include <common>', `#include <common>
        uniform sampler2D tLeaf; uniform float uGain;
        varying vec3 vTint; varying vec4 vBB; varying float vFade;
        float bbR;`)
      .replace('#include <clipping_planes_fragment>', FADE_FRAG + `
        {
          vec2 bp = vBB.xy;
          // conical crowns narrow toward the top
          float wdt = mix(1.0, clamp(0.55 - 0.5 * bp.y, 0.08, 1.0), vBB.z);
          bbR = length(vec2(bp.x / wdt, bp.y));
          vec4 lf = texture2D(tLeaf, bp * 0.9 + vBB.w * 7.0);
          if (bbR > 0.62 + 0.4 * lf.a) discard;
        }
        #include <clipping_planes_fragment>`)
      .replace('#include <map_fragment>', `#include <map_fragment>
        {
          vec4 lf = texture2D(tLeaf, vBB.xy * 1.3 + vBB.w * 5.0);
          float lum = dot(lf.rgb, vec3(0.3, 0.55, 0.15)) * 1.9;
          float br = mix(0.28, 1.0, lf.a) * mix(1.0, clamp(lum, 0.6, 1.4), lf.a);
          // crown AO: darker underside and core, like the LOD-0 cards' AO ramp
          float ao = (0.42 + 0.38 * (vBB.y * 0.5 + 0.5)) * mix(1.0, 0.8, 1.0 - clamp(bbR, 0.0, 1.0));
          diffuseColor.rgb *= vTint * ao * br * uGain;
        }`)
      .replace('#include <color_fragment>', '')
      .replace('#include <normal_fragment_begin>', `
        float faceDirection = 1.0;
        vec2 bn = vBB.xy; float rr = min(dot(bn, bn), 1.0);
        // crown dome: sphere normal on the card, bent toward the sky (tops catch the low sun / sky light)
        vec3 upV = normalize((viewMatrix * vec4(0.0, 1.0, 0.0, 0.0)).xyz);
        vec3 normal = normalize(vec3(bn.x, bn.y, sqrt(1.0 - rr) + 0.1) + upV * 0.7);
        vec3 nonPerturbedNormal = normal;`)
      // same warm translucency as the leaf cards (backlit crowns glow)
      .replace('#include <lights_physical_pars_fragment>', THREE.ShaderChunk.lights_physical_pars_fragment.replace(
        'reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );',
        `float back = pow( saturate( - dot( geometryViewDir, directLight.direction ) ), 3.0 ) * 0.5 + 0.25;
        reflectedLight.directDiffuse += ( irradiance * 0.75 + back * directLight.color * vec3( 1.35, 1.2, 0.62 ) * 0.3 ) * BRDF_Lambert( material.diffuseColor );`));
  };
  m.customProgramCacheKey = () => 'treebb';
  return m;
}

/** Shadow depth: optional card alpha, and the LOD band with a hard switch at the fade middle (one caster). */
function depthMaterial(map: THREE.Texture | null, f: FadeUniforms) {
  const m = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map, alphaTest: map ? 0.5 : 0 });
  m.onBeforeCompile = (s) => {
    Object.assign(s.uniforms, f);
    // card alpha applies to card vertices only (bark / cores are solid casters)
    s.fragmentShader = s.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vMid;')
      .replace('#include <map_fragment>', '#include <map_fragment>\n  if (vMid < 0.75) diffuseColor.a = 1.0;');
    s.vertexShader = s.vertexShader
      .replace('#include <common>', `#include <common>\nattribute float mid; varying float vMid;\n${FADE_VERT_PARS}`)
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n  vMid = mid;')
      .replace('#include <project_vertex>', `#include <project_vertex>
        #ifdef USE_INSTANCING
          if (abs(treeFade((modelMatrix * vec4(instanceMatrix[3].xyz, 1.0)).xyz)) < 0.5) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        #endif`);
  };
  m.customProgramCacheKey = () => 'treedepth' + (map ? 'm' : '');
  return m;
}

// ---------------------------------------------------------------- instances
function hash(n: number) { const s = Math.sin(n * 12.9898 + 78.233) * 43758.5453; return s - Math.floor(s); }

export interface Trees {
  group: THREE.Group;
  count: number;
  uniforms: FoliageUniforms;
  update(camera: THREE.Camera, force?: boolean): void;
  setQuality(q: keyof typeof TREE_QUALITY): void;
  stats(): { instances: number; tris: number };
}

export async function buildTrees(hf: Heightfield, sunDir: THREE.Vector3): Promise<Trees> {
  const [meta, geoBuf, vegBuf] = await Promise.all([
    fetch(assetUrl('trees/geo.json')).then((r) => r.json() as Promise<GeoJson>), bin('trees/geo.bin'), bin('vegetation_f32.bin'),
  ]);
  // uGain: Cycles gets multiple scattering inside the crowns; a single-bounce rasteriser needs a lift
  const uniforms: FoliageUniforms = { uTime: { value: 0 }, uSunDir: { value: sunDir.clone().normalize() }, uWind: { value: 1 }, uGain: { value: 2.2 } };
  const loader = new THREE.TextureLoader();
  const tex = (f: string, srgb: boolean) => {
    const t = loader.load(assetUrl('trees/' + f));
    t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace; t.anisotropy = 4;
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    return t;
  };
  // one material set per LOD (+ small-plant far fade): same programs, per-LOD fade uniforms
  const camU = { value: new THREE.Vector3(1e9, 0, 0) };
  const fades: FadeUniforms[][] = [0, 1].map(() => [0, 1, 2, 3, 4].map(() => ({ uFade: { value: new THREE.Vector4() }, uFar: { value: new THREE.Vector2() }, uCamPos: camU })));
  const matCache = new Map<string, THREE.Material>();
  const cached = <T extends THREE.Material>(k: string, make: () => T) => { if (!matCache.has(k)) matCache.set(k, make()); return matCache.get(k)! as T; };
  const texCache = new Map<string, THREE.Texture>();
  const ctex = (f: string, srgb: boolean) => { if (!texCache.has(f)) texCache.set(f, tex(f, srgb)); return texCache.get(f)!; };
  const barkTex: Record<string, THREE.Texture> = { brown: tex('bark_brown.jpg', true), pine: tex('bark_pine.jpg', true) };
  const coreLeaf = tex('card_oak.png', true);
  const fk = (li: number, sm: number) => `${li}:${sm}`;
  // merged material per (card, bark kind, LOD, small-plant fade); snags (no card) borrow the oak card
  // texture but have no card vertices
  const treeMat = (card: string | null, b: string, li: number, sm: number) => cached(`tree:${card}:${b}:${fk(li, sm)}`, () => {
    const c = card ?? 'oak', [r, g, bb, t] = BARK_TINT[b] ?? BARK_TINT.brown;
    return treeMaterial(ctex(`card_${c}.png`, true), ctex(`card_${c}_n.jpg`, false), meta.leaf_k, barkTex[t], [r, g, bb], t === 'pine' ? 0.7 : 0.6,
      coreLeaf, uniforms, fades[sm][li]);
  });
  const depth = (card: string | null, li: number, sm: number) => cached(`depth:${card}:${fk(li, sm)}`, () =>
    depthMaterial(card ? ctex(`card_${card}.png`, true) : null, fades[sm][li]));

  // geometries
  const geos: THREE.BufferGeometry[][] = meta.species.map((sp) => sp.lods.map((l) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(geoBuf, l.pos, l.vcount * 3), 3));
    const n4 = new Int8Array(geoBuf, l.nrm, l.vcount * 4);
    const ib = new THREE.InterleavedBuffer(n4, 4);
    g.setAttribute('normal', new THREE.InterleavedBufferAttribute(ib, 3, 0, true));
    g.setAttribute('mid', new THREE.InterleavedBufferAttribute(ib, 1, 3, true)); // material id (0 bark, 0.5 core, 1 card)
    g.setAttribute('uv', new THREE.BufferAttribute(new Uint8Array(geoBuf, l.uv, l.vcount * 2), 2, true));
    g.setAttribute('al', new THREE.BufferAttribute(new Uint8Array(geoBuf, l.al, l.vcount * 2), 2, true));
    g.setIndex(new THREE.BufferAttribute(new Uint16Array(geoBuf, l.idx, l.icount), 1));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, sp.height / 2, 0), Math.hypot(sp.radius, sp.height / 2));
    return g;
  }));
  const matsFor = (sp: SpeciesInfo, li: number, sm: number) => treeMat(sp.card, sp.bark, li, sm);

  // instances sorted by (species, cell)
  const S = 6, V = new Float32Array(vegBuf), n = V.length / S;
  const ncx = Math.ceil(IMG_W / CELL), ncy = Math.ceil(IMG_H / CELL), NC = ncx * ncy;
  const nsp = meta.species.length;
  const order = new Uint32Array(n);
  const key = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const sp = V[i * S + 4] | 0;
    const c = Math.min(ncy - 1, Math.floor(V[i * S + 1] / CELL)) * ncx + Math.min(ncx - 1, Math.floor(V[i * S] / CELL));
    key[i] = sp * NC + c; order[i] = i;
  }
  order.sort((a, b) => key[a] - key[b]);
  const mats = new Float32Array(n * 16), cols = new Float32Array(n * 3);
  const range = new Int32Array(nsp * NC * 2).fill(-1); // [start, end) per species x cell
  const cellMinY = new Float32Array(NC).fill(1e9), cellMaxY = new Float32Array(NC).fill(-1e9);
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), sc = new THREE.Vector3(), p = new THREE.Vector3();
  const pal = meta.palettes;
  let seed = 5;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  // palette pick weighted by autumn progress (vegetation seed: int part = autumn * 100)
  const pick = (pl: [number[], number][], r: number, k: number) => {
    let tot = 0; for (const [c, w] of pl) tot += c[0] > c[1] ? w * k : w;
    let a = 0; for (const [c, w] of pl) { a += (c[0] > c[1] ? w * k : w) / tot; if (r < a) return c; } return pl[pl.length - 1][0];
  };
  for (let k = 0; k < n; k++) {
    const i = order[k], kk = key[i], spi = Math.floor(kk / NC), c = kk % NC;
    const x = V[i * S], y = V[i * S + 1], s = V[i * S + 3], sd0 = V[i * S + 5];
    const aw = autumnWeight(Math.floor(sd0) / 100), sd = sd0 - Math.floor(sd0);
    const z = hf.at(x, y);
    // lib_trees.build_vegetation: sink on slopes so the root flare never floats
    const sl = Math.hypot(hf.at(x + 1, y) - hf.at(x - 1, y), hf.at(x, y + 1) - hf.at(x, y - 1)) / (2 * 2.5);
    const [X, , Z] = pxToWorld(x, y);
    p.set(X, z - 0.25 - Math.min(sl, 1.5) * 0.6, Z);
    const r1 = hash(sd * 1.7 + 3.1), r2 = hash(sd * 2.3 + 9.7), r3 = hash(sd * 5.1 + 1.3), r4 = hash(sd * 7.7 + 4.4);
    e.set((r3 - 0.5) * 0.05, r1 * Math.PI * 2, (r4 - 0.5) * 0.05);
    q.setFromEuler(e);
    sc.setScalar(s);
    m4.compose(p, q, sc).toArray(mats, k * 16);
    const sp = meta.species[spi];
    const pl = pal[sp.family] ?? pal.oak;
    let col = pick(pl, r2, aw);
    if (rnd() < 0.25) { const c2 = pick(pl, rnd(), aw); const t = rnd() * 0.5; col = col.map((v, j) => v * (1 - t) + c2[j] * t); }
    const j = 0.85 + 0.3 * r3;
    cols[k * 3] = col[0] * j; cols[k * 3 + 1] = col[1] * j; cols[k * 3 + 2] = col[2] * j;
    const ri = (spi * NC + c) * 2;
    if (range[ri] < 0) range[ri] = k;
    range[ri + 1] = k + 1;
    cellMinY[c] = Math.min(cellMinY[c], p.y); cellMaxY[c] = Math.max(cellMaxY[c], p.y + sp.height * s);
  }
  const perSpecies = new Int32Array(nsp);
  for (let k = 0; k < n; k++) perSpecies[Math.floor(key[order[k]] / NC)]++;

  // one InstancedMesh per species x LOD, sized for the whole species
  const group = new THREE.Group(); group.name = 'trees';
  const NL = 4; // geometry LODs per species; LOD 4 is the shared billboard
  const meshes: THREE.InstancedMesh[][] = meta.species.map((sp, si) => sp.lods.slice(0, NL).map((_, li) => {
    const sm = SMALL.has(sp.name) ? 1 : 0;
    const im = new THREE.InstancedMesh(geos[si][li], matsFor(sp, li, sm), Math.max(1, perSpecies[si]));
    im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    im.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, perSpecies[si]) * 3), 3);
    im.instanceColor.setUsage(THREE.DynamicDrawUsage);
    im.count = 0; im.frustumCulled = false; im.receiveShadow = true;
    im.customDepthMaterial = depth(sp.card, li, sm);
    im.userData.kind = 'tree';
    im.name = `tree_${sp.name}_lod${li}`;
    group.add(im);
    return im;
  }));
  const small = meta.species.map((sp) => SMALL.has(sp.name));
  // billboard: crown centre / radii from the exported horizon ellipsoid (lod 4), slightly grown to the card shell
  const bbSp = new Float32Array(nsp * 4);
  meta.species.forEach((sp, si) => {
    const l = sp.lods[sp.lods.length - 1], P = new Float32Array(geoBuf, l.pos, l.vcount * 3);
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9, z0 = 1e9, z1 = -1e9;
    for (let v = 0; v < l.vcount; v++) {
      x0 = Math.min(x0, P[v * 3]); x1 = Math.max(x1, P[v * 3]); y0 = Math.min(y0, P[v * 3 + 1]); y1 = Math.max(y1, P[v * 3 + 1]); z0 = Math.min(z0, P[v * 3 + 2]); z1 = Math.max(z1, P[v * 3 + 2]);
    }
    const conical = sp.family === 'hemlock' || sp.name === 'Sapling_Pine' ? 1 : sp.family === 'pine' ? 0.35 : 0;
    bbSp.set([(y0 + y1) / 2, Math.max(x1 - x0, z1 - z0) / 2 * 1.15, (y1 - y0) / 2 * 1.12, conical], si * 4);
  });
  // the billboard layer is static: every tree once, in the shared sorted buffers; the vertex shader keeps
  // only the instances inside LOD 4's distance band (no CPU copies; GPU clips the rest). Shrubs and
  // saplings get a zero-size card (they are gone long before the billboard band).
  const bbAll = new Float32Array(n * 4); // per sorted instance
  const cellCount = new Int32Array(NC);
  for (let k = 0; k < n; k++) {
    const kk = key[order[k]], si = Math.floor(kk / NC);
    cellCount[kk % NC]++;
    if (!SMALL.has(meta.species[si].name)) for (let j = 0; j < 4; j++) bbAll[k * 4 + j] = bbSp[si * 4 + j];
  }
  const quad = new THREE.BufferGeometry();
  quad.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]), 3));
  quad.setAttribute('normal', new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]), 3));
  quad.setIndex([0, 1, 2, 0, 2, 3]);
  quad.setAttribute('ibb', new THREE.InstancedBufferAttribute(bbAll, 4));
  const bbMesh = new THREE.InstancedMesh(quad, billboardMaterial(uniforms, coreLeaf, fades[0][4]), Math.max(1, n));
  bbMesh.instanceMatrix = new THREE.InstancedBufferAttribute(mats, 16);
  bbMesh.instanceColor = new THREE.InstancedBufferAttribute(cols, 3);
  bbMesh.count = n; bbMesh.frustumCulled = false; bbMesh.castShadow = false; bbMesh.receiveShadow = false;
  bbMesh.userData.kind = 'tree'; bbMesh.name = 'tree_billboards_lod4';
  group.add(bbMesh);

  let quality: TreeQuality = TREE_QUALITY.high;
  // LOD l is drawn over [lo[l], hi[l]] (metres); fade zones are +-FADE_W around each switch distance
  const lo = new Float32Array(5), hi = new Float32Array(5);
  const fadeW = (d: number) => Math.min(Math.max(0.06 * d, 3), 30); // narrow: a static frame shows the dither
  const applyQuality = () => {
    const d = quality.lodDist;
    for (let l = 0; l < 5; l++) {
      const wi = l > 0 ? fadeW(d[l - 1]) : 0, wo = l < 4 ? fadeW(d[l]) : 0;
      lo[l] = l > 0 ? d[l - 1] - wi : -1; hi[l] = l < 4 ? d[l] + wo : 1e9;
      for (const sm of [0, 1]) {
        fades[sm][l].uFade.value.set(l > 0 ? lo[l] : -2, l > 0 ? d[l - 1] + wi : -1, l < 4 ? d[l] - wo : 1e9, l < 4 ? hi[l] : 2e9);
        fades[sm][l].uFar.value.set(sm ? quality.smallFar * 0.8 : 1e9, sm ? quality.smallFar : 2e9);
      }
    }
    meshes.forEach((l) => l.forEach((im, li) => (im.castShadow = li <= quality.shadowLod)));
  };
  applyQuality();
  // cell bounds in world metres (no per-frame allocation)
  const cellX = new Float32Array(ncx + 1), cellZ = new Float32Array(ncy + 1);
  for (let i = 0; i <= ncx; i++) cellX[i] = pxToWorld(i * CELL, 0)[0];
  for (let i = 0; i <= ncy; i++) cellZ[i] = pxToWorld(0, i * CELL)[2];
  const cellMask = new Uint8Array(NC), prevMask = new Uint8Array(NC).fill(255), cellDmin = new Float32Array(NC);
  const active = new Int32Array(NC); let nActive = 0; // cells with any LOD this pass
  const frustum = new THREE.Frustum(), pm = new THREE.Matrix4(), box = new THREE.Box3();
  const lastPos = new THREE.Vector3(1e9, 0, 0), lastDir = new THREE.Vector3(), dir = new THREE.Vector3(), lastEmit = new THREE.Vector3(1e9, 0, 0);
  const cnt = new Int32Array(5);
  const PAD = 15, SLACK = 3; // crown overhang (m); camera travel allowed between culling passes (m)
  let visible = 0, tris = 0;

  function update(camera: THREE.Camera, force = false) {
    const cp = camera.position;
    camU.value.copy(cp);
    camera.getWorldDirection(dir);
    if (!force && cp.distanceToSquared(lastPos) < 4 && dir.dot(lastDir) > 0.9995) return;
    lastPos.copy(cp); lastDir.copy(dir);
    pm.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse); frustum.setFromProjectionMatrix(pm);
    let changed = force;
    for (let cy = 0; cy < ncy; cy++) for (let cx = 0; cx < ncx; cx++) {
      const c = cy * ncx + cx;
      let m = 0;
      if (cellMaxY[c] >= cellMinY[c]) {
        box.min.set(cellX[cx] - PAD, cellMinY[c], cellZ[cy] - PAD); box.max.set(cellX[cx + 1] + PAD, cellMaxY[c], cellZ[cy + 1] + PAD);
        if (frustum.intersectsBox(box)) {
          const dmin = box.distanceToPoint(cp) - SLACK;
          const ex = Math.max(Math.abs(cp.x - box.min.x), Math.abs(cp.x - box.max.x));
          const ey = Math.max(Math.abs(cp.y - box.min.y), Math.abs(cp.y - box.max.y));
          const ez = Math.max(Math.abs(cp.z - box.min.z), Math.abs(cp.z - box.max.z));
          const dmax = Math.sqrt(ex * ex + ey * ey + ez * ez) + SLACK;
          for (let l = 0; l < 5; l++) if (dmin <= hi[l] && dmax >= lo[l]) m |= 1 << l;
          cellDmin[c] = dmin;
        }
      }
      cellMask[c] = m;
      if (m !== prevMask[c]) changed = true;
    }
    nActive = 0; let nbb = 0;
    for (let c = 0; c < NC; c++) { if (cellMask[c] & 15) active[nActive++] = c; if (cellMask[c] & 16) nbb += cellCount[c]; }
    // near LODs are filtered per instance: re-emit once the camera has moved past the slack
    if (cp.distanceToSquared(lastEmit) > (SLACK - 1) * (SLACK - 1)) changed = true;
    if (!changed) return;
    lastEmit.copy(cp);
    prevMask.set(cellMask);
    visible = 0; tris = 0;
    const smallFar = quality.smallFar + SLACK;
    for (let si = 0; si < nsp; si++) {
      const sp = meta.species[si], sm = small[si];
      const lm = meshes[si];
      cnt.fill(0);
      for (let ai = 0; ai < nActive; ai++) {
        const c = active[ai], m = cellMask[c];
        const ri = (si * NC + c) * 2, a = range[ri];
        if (a < 0) continue;
        if (sm && cellDmin[c] > smallFar) continue; // shrubs / saplings dissolve into the canopy with distance
        const b = range[ri + 1];
        for (let l = 0; l < NL; l++) {
          if (!(m & (1 << l))) continue;
          const im = lm[l];
          const am = im.instanceMatrix.array as Float32Array, ac = im.instanceColor!.array as Float32Array;
          if (l <= 2 && m !== 1 << l) {
            // near LODs in a cell that straddles a band edge: per instance, only those inside this LOD's band
            // (+ slack), so the submitted near-LOD count follows the band instead of the 50 m cell
            const lo2 = lo[l] - SLACK, hi2 = hi[l] + SLACK;
            for (let k = a; k < b; k++) {
              const dx = mats[k * 16 + 12] - cp.x, dy = mats[k * 16 + 13] - cp.y, dz = mats[k * 16 + 14] - cp.z;
              const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
              if (d < lo2 || d > hi2) continue;
              const t = cnt[l]++;
              for (let j = 0; j < 16; j++) am[t * 16 + j] = mats[k * 16 + j];
              ac[t * 3] = cols[k * 3]; ac[t * 3 + 1] = cols[k * 3 + 1]; ac[t * 3 + 2] = cols[k * 3 + 2];
            }
            continue;
          }
          const o = cnt[l];
          for (let j = a * 16, e = b * 16, t = o * 16; j < e; j++, t++) am[t] = mats[j];
          for (let j = a * 3, e = b * 3, t = o * 3; j < e; j++, t++) ac[t] = cols[j];
          cnt[l] += b - a;
        }
      }
      for (let l = 0; l < NL; l++) {
        const im = lm[l];
        im.count = cnt[l];
        im.visible = cnt[l] > 0;
        if (cnt[l]) {
          im.instanceMatrix.clearUpdateRanges(); im.instanceMatrix.addUpdateRange(0, cnt[l] * 16); im.instanceMatrix.needsUpdate = true;
          im.instanceColor!.clearUpdateRanges(); im.instanceColor!.addUpdateRange(0, cnt[l] * 3); im.instanceColor!.needsUpdate = true;
          visible += cnt[l]; tris += cnt[l] * (sp.lods[l].icount / 3);
        }
      }
    }
    visible += nbb; tris += n * 2; // billboard layer: all instances submitted, off-band ones collapse in the vertex shader
  }

  return {
    group, count: n, uniforms, update,
    setQuality(qn) { quality = TREE_QUALITY[qn]; applyQuality(); prevMask.fill(255); lastPos.set(1e9, 0, 0); },
    stats: () => ({ instances: visible, tris }),
  };
}
