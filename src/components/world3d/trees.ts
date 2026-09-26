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
 * 100 m cell picks a LOD from its distance and frustum visibility, and the visible ranges are
 * block-copied into one InstancedMesh per species x LOD (~250 draw calls for 215k plants).
 *
 * Shading follows the Blender materials: leaf = card texture x instance tint x LEAF_K x crown AO,
 * per-card value jitter, crown-volume normals, 32 % warm translucency toward the sun; core = the
 * tinted lifted mass behind the cards; bark = triplanar Poly Haven bark with the species tint.
 */

interface LodInfo { vcount: number; icount: number; groups: [number, number, number][]; pos: number; nrm: number; uv: number; al: number; idx: number }
interface SpeciesInfo { name: string; family: string; card: string | null; bark: string; radius: number; height: number; lods: LodInfo[] }
interface GeoJson { species: SpeciesInfo[]; lods: number; palettes: Record<string, [number[], number][]>; leaf_k: number }

const CELL = 40; // source px (100 m)
const BARK_TINT: Record<string, [number, number, number, string]> = {
  brown: [0.55, 0.5, 0.45, 'brown'], grey: [0.55, 0.55, 0.52, 'brown'], pine: [0.6, 0.5, 0.44, 'pine'], dead: [0.75, 0.73, 0.7, 'brown'],
};
const SMALL = new Set(['Sapling_HW_A', 'Sapling_HW_B', 'Sapling_Pine', 'Rhododendron_A', 'Laurel_B', 'Brush_A', 'Brush_B', 'Dogwood']);

export interface TreeQuality { lodDist: [number, number, number, number]; smallFar: number; shadowLod: number }
export const TREE_QUALITY: Record<string, TreeQuality> = {
  low: { lodDist: [45, 180, 520, 1100], smallFar: 350, shadowLod: 0 },
  medium: { lodDist: [70, 260, 750, 1500], smallFar: 550, shadowLod: 1 },
  high: { lodDist: [100, 340, 950, 1900], smallFar: 800, shadowLod: 2 },
  ultra: { lodDist: [140, 450, 1200, 2400], smallFar: 1100, shadowLod: 2 },
};

// ---------------------------------------------------------------- shading
export interface FoliageUniforms { uTime: { value: number }; uSunDir: { value: THREE.Vector3 }; uWind: { value: number }; uGain: { value: number } }

function windChunk(leafy: boolean) {
  return `
    #ifdef USE_INSTANCING
      vec3 ip = instanceMatrix[3].xyz;
      float ph = ip.x * 0.043 + ip.z * 0.031;
      float hh = max(position.y, 0.0);
      float sw = uWind * hh * hh * 0.0009;
      transformed.x += sin(uTime * 0.9 + ph) * sw;
      transformed.z += cos(uTime * 0.7 + ph * 1.3) * sw * 0.8;
      ${leafy ? 'transformed += normal * sin(uTime * 3.1 + ph * 7.0 + position.y * 1.7 + position.x) * 0.045 * uWind;' : ''}
    #endif`;
}

function patchCommon(s: THREE.WebGLProgramParametersWithUniforms, u: FoliageUniforms, leafy: boolean) {
  Object.assign(s.uniforms, u);
  s.vertexShader = s.vertexShader
    .replace('#include <common>', `#include <common>
      attribute vec2 al;
      uniform float uTime; uniform float uWind;
      varying vec2 vAL; varying vec3 vTint; varying vec3 vObj;`)
    .replace('#include <begin_vertex>', `#include <begin_vertex>
      vAL = al; vObj = position;
      #ifdef USE_INSTANCING_COLOR
        vTint = instanceColor;
      #else
        vTint = vec3(1.0);
      #endif
      ${windChunk(leafy)}`);
  s.fragmentShader = s.fragmentShader.replace('#include <common>', `#include <common>
      varying vec2 vAL; varying vec3 vTint; varying vec3 vObj;
      uniform vec3 uSunDir; uniform float uGain;`)
    .replace('#include <color_fragment>', ''); // instance tint is applied explicitly (bark is untinted)
}

/** Leaf cards: MAT_Foliage_<card>. */
function leafMaterial(map: THREE.Texture, nmap: THREE.Texture, leafK: number, u: FoliageUniforms) {
  const m = new THREE.MeshStandardMaterial({
    map, normalMap: nmap, normalScale: new THREE.Vector2(0.55, 0.55), alphaTest: 0.5, side: THREE.DoubleSide,
    roughness: 0.62, metalness: 0, envMapIntensity: 0.6,
  });
  m.alphaToCoverage = true;
  m.defines = { FOLIAGE: '' };
  m.onBeforeCompile = (s) => {
    patchCommon(s, u, true);
    s.fragmentShader = s.fragmentShader
      // crown-volume normals must not flip on back faces (Blender cards: custom normals, no flip)
      .replace('#include <normal_fragment_begin>', THREE.ShaderChunk.normal_fragment_begin.replace('float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;', 'float faceDirection = 1.0;'))
      .replace('#include <map_fragment>', `#include <map_fragment>
        // card colour x instance tint x LEAF_K x crown AO, per-card value jitter (lib_trees._tinted)
        diffuseColor.rgb *= vTint * ${leafK.toFixed(3)} * vAL.x * mix(0.85, 1.15, vAL.y) * uGain;`)
      // 68 % principled + 32 % translucent BSDF with a warm transmission tint (backlit leaves glow);
      // directLight.color already carries the shadow term
      .replace('#include <lights_physical_pars_fragment>', THREE.ShaderChunk.lights_physical_pars_fragment.replace(
        'reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );',
        `float back = saturate( dot( - geometryNormal, directLight.direction ) ) * 0.6 + pow( saturate( - dot( geometryViewDir, directLight.direction ) ), 3.0 ) * 0.5;
        reflectedLight.directDiffuse += ( irradiance * 0.68 + back * directLight.color * vec3( 1.35, 1.2, 0.62 ) * 0.26 ) * BRDF_Lambert( material.diffuseColor );`));
  };
  m.customProgramCacheKey = () => 'leaf';
  return m;
}

/** Core masses behind the cards: MAT_Foliage_Core (tint x AO, soft voronoi-ish breakup). */
function coreMaterial(u: FoliageUniforms) {
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.85, metalness: 0, envMapIntensity: 0.5 });
  m.onBeforeCompile = (s) => {
    patchCommon(s, u, false);
    s.fragmentShader = s.fragmentShader.replace('#include <map_fragment>', `#include <map_fragment>
        vec3 q = vObj * 1.5;
        float br = 0.7 + 0.45 * (0.5 + 0.5 * sin(q.x * 2.1 + sin(q.y * 1.7)) * sin(q.z * 2.3 + sin(q.x * 1.3)));
        diffuseColor.rgb *= vTint * vAL.x * br * uGain;`);
  };
  m.customProgramCacheKey = () => 'core';
  return m;
}

/** Bark: triplanar bark texture (object coords, Blender box projection) x species tint. */
function barkMaterial(tex: THREE.Texture, tint: [number, number, number], scale: number, u: FoliageUniforms) {
  const m = new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(tint[0], tint[1], tint[2]), roughness: 0.9, metalness: 0, envMapIntensity: 0.4 });
  m.onBeforeCompile = (s) => {
    patchCommon(s, u, false);
    s.uniforms.uBark = { value: tex };
    s.fragmentShader = s.fragmentShader
      .replace('uniform vec3 uSunDir;', 'uniform vec3 uSunDir; uniform sampler2D uBark;')
      .replace('#include <map_fragment>', `#include <map_fragment>
        {
          vec3 p = vObj / ${scale.toFixed(2)};
          vec3 w = abs(normalize(cross(dFdx(vObj), dFdy(vObj)))); w = pow(w, vec3(4.0)); w /= dot(w, vec3(1.0));
          vec3 t = texture2D(uBark, p.yz * vec2(0.5, 1.0)).rgb * w.x + texture2D(uBark, p.xy * vec2(1.0, 0.5)).rgb * w.z + texture2D(uBark, p.xz).rgb * w.y;
          diffuseColor.rgb *= t * vAL.x * 1.1;
        }`);
  };
  m.customProgramCacheKey = () => 'bark' + scale;
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
  const leafMats = new Map<string, THREE.Material>(), barkMats = new Map<string, THREE.Material>();
  const barkTex: Record<string, THREE.Texture> = { brown: tex('bark_brown.jpg', true), pine: tex('bark_pine.jpg', true) };
  const core = coreMaterial(uniforms);
  const depthMats = new Map<string, THREE.Material>();
  const leaf = (card: string) => {
    if (!leafMats.has(card)) {
      const map = tex(`card_${card}.png`, true);
      leafMats.set(card, leafMaterial(map, tex(`card_${card}_n.jpg`, false), meta.leaf_k, uniforms));
      depthMats.set(card, new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map, alphaTest: 0.5 }));
    }
    return leafMats.get(card)!;
  };
  const bark = (b: string) => {
    if (!barkMats.has(b)) { const [r, g, bb, t] = BARK_TINT[b]; barkMats.set(b, barkMaterial(barkTex[t], [r, g, bb], t === 'pine' ? 0.7 : 0.6, uniforms)); }
    return barkMats.get(b)!;
  };

  // geometries
  const geos: THREE.BufferGeometry[][] = meta.species.map((sp) => sp.lods.map((l) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(geoBuf, l.pos, l.vcount * 3), 3));
    const n4 = new Int8Array(geoBuf, l.nrm, l.vcount * 4);
    g.setAttribute('normal', new THREE.InterleavedBufferAttribute(new THREE.InterleavedBuffer(n4, 4), 3, 0, true));
    g.setAttribute('uv', new THREE.BufferAttribute(new Uint8Array(geoBuf, l.uv, l.vcount * 2), 2, true));
    g.setAttribute('al', new THREE.BufferAttribute(new Uint8Array(geoBuf, l.al, l.vcount * 2), 2, true));
    g.setIndex(new THREE.BufferAttribute(new Uint16Array(geoBuf, l.idx, l.icount), 1));
    for (const [start, count, mi] of l.groups) g.addGroup(start, count, mi);
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, sp.height / 2, 0), Math.hypot(sp.radius, sp.height / 2));
    return g;
  }));
  const matsFor = (sp: SpeciesInfo) => [bark(sp.bark), sp.card ? leaf(sp.card) : bark(sp.bark), core];

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
  const pick = (pl: [number[], number][], r: number) => { let a = 0; const tot = pl.reduce((s, x) => s + x[1], 0); for (const [c, w] of pl) { a += w / tot; if (r < a) return c; } return pl[pl.length - 1][0]; };
  for (let k = 0; k < n; k++) {
    const i = order[k], kk = key[i], spi = Math.floor(kk / NC), c = kk % NC;
    const x = V[i * S], y = V[i * S + 1], s = V[i * S + 3], sd = V[i * S + 5];
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
    let col = pick(pl, r2);
    if (rnd() < 0.25) { const c2 = pick(pl, rnd()); const t = rnd() * 0.5; col = col.map((v, j) => v * (1 - t) + c2[j] * t); }
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
  const meshes: THREE.InstancedMesh[][] = meta.species.map((sp, si) => sp.lods.map((_, li) => {
    const im = new THREE.InstancedMesh(geos[si][li], matsFor(sp), Math.max(1, perSpecies[si]));
    im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    im.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, perSpecies[si]) * 3), 3);
    im.instanceColor.setUsage(THREE.DynamicDrawUsage);
    im.count = 0; im.frustumCulled = false; im.receiveShadow = true;
    if (sp.card) im.customDepthMaterial = depthMats.get(sp.card);
    im.userData.kind = 'tree';
    im.name = `tree_${sp.name}_lod${li}`;
    group.add(im);
    return im;
  }));

  let quality: TreeQuality = TREE_QUALITY.high;
  const setShadows = () => meshes.forEach((l) => l.forEach((im, li) => (im.castShadow = li <= quality.shadowLod)));
  setShadows();
  const cellLod = new Int8Array(NC), prevLod = new Int8Array(NC).fill(-2);
  const frustum = new THREE.Frustum(), pm = new THREE.Matrix4(), box = new THREE.Box3();
  const lastPos = new THREE.Vector3(1e9, 0, 0), lastDir = new THREE.Vector3();
  let visible = 0, tris = 0;

  function update(camera: THREE.Camera, force = false) {
    const cp = camera.position, dir = new THREE.Vector3(); camera.getWorldDirection(dir);
    if (!force && cp.distanceToSquared(lastPos) < 4 && dir.dot(lastDir) > 0.9995) return;
    lastPos.copy(cp); lastDir.copy(dir);
    pm.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse); frustum.setFromProjectionMatrix(pm);
    const d = quality.lodDist;
    let changed = force;
    for (let cy = 0; cy < ncy; cy++) for (let cx = 0; cx < ncx; cx++) {
      const c = cy * ncx + cx;
      if (cellMaxY[c] < cellMinY[c]) { cellLod[c] = -1; continue; }
      const [x0, , z0] = pxToWorld(cx * CELL, cy * CELL), [x1, , z1] = pxToWorld((cx + 1) * CELL, (cy + 1) * CELL);
      box.min.set(x0 - 15, cellMinY[c], z0 - 15); box.max.set(x1 + 15, cellMaxY[c], z1 + 15);
      let l = -1;
      if (frustum.intersectsBox(box)) {
        const dist = box.distanceToPoint(cp);
        l = dist < d[0] ? 0 : dist < d[1] ? 1 : dist < d[2] ? 2 : dist < d[3] ? 3 : 4;
      }
      cellLod[c] = l;
      if (l !== prevLod[c]) changed = true;
    }
    if (!changed) return;
    prevLod.set(cellLod);
    visible = 0; tris = 0;
    const smallFar = quality.smallFar;
    for (let si = 0; si < nsp; si++) {
      const sp = meta.species[si], small = SMALL.has(sp.name);
      const lm = meshes[si];
      const cnt = [0, 0, 0, 0, 0];
      for (let c = 0; c < NC; c++) {
        let l = cellLod[c];
        if (l < 0) continue;
        const ri = (si * NC + c) * 2, a = range[ri];
        if (a < 0) continue;
        if (small && l >= 2) {
          // shrubs / saplings vanish into the canopy with distance
          const cx = c % ncx, cy = Math.floor(c / ncx), [X, , Z] = pxToWorld((cx + 0.5) * CELL, (cy + 0.5) * CELL);
          if (Math.hypot(X - cp.x, Z - cp.z) > smallFar) continue;
        }
        const b = range[ri + 1], im = lm[l], o = cnt[l];
        (im.instanceMatrix.array as Float32Array).set(mats.subarray(a * 16, b * 16), o * 16);
        (im.instanceColor!.array as Float32Array).set(cols.subarray(a * 3, b * 3), o * 3);
        cnt[l] += b - a;
      }
      for (let l = 0; l < 5; l++) {
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
  }

  return {
    group, count: n, uniforms, update,
    setQuality(qn) { quality = TREE_QUALITY[qn]; setShadows(); prevLod.fill(-2); },
    stats: () => ({ instances: visible, tris }),
  };
}
