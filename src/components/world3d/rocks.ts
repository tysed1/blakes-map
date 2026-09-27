import * as THREE from 'three';
import { pxToWorld } from '../../core/coords';
import { assetUrl, bin } from '../../core/data';
import { loadTexture } from '../../engine/textures';

/**
 * Rock kit (E2): crags, cliff-band ledge blocks, talus boulders, scree and road rock cuts.
 *
 * Meshes: tools/blender/lib_rocks.py -> tools/blender/export_web_rocks.py (public/world/rocks/geo.*),
 * 4 kinds x 4 variants x 3 LODs. Placement: tools/pipeline/rocks.py (public/world/rocks_f32.bin).
 * Shading: triplanar world-space rock detail (the terrain's own detail_rock map, so crags and rocky
 * terrain read as one material) x a grey-brown gneiss base x per-instance tint x baked cavity AO,
 * lichen / moss on up-facing ledges, height-derived bump. LODs cross-fade with the same complementary
 * screen dither as the trees; the far LOD dissolves out at the kind's far distance.
 * Draw calls: one per kind x LOD x variant slot (LOD0 4 variants, LOD1 2, LOD2 1) that has instances.
 */

interface LodInfo { vcount: number; icount: number; pos: number; nrm: number; ao: number; idx: number }
interface KindInfo { name: string; dims: [number, number, number]; variants: { lods: LodInfo[]; radius: number; height: number }[] }
interface GeoJson { kinds: KindInfo[]; variants: number; lods: number }

type Q = 'low' | 'medium' | 'high' | 'ultra';
// per kind: LOD switch distances [0->1, 1->2] and far cut (m) at High
const DIST: Record<string, [number, number, number]> = {
  cliff_block: [70, 260, 2600], crag: [60, 240, 2200], boulder: [40, 140, 700], scree: [22, 60, 140],
};
const QSCALE: Record<Q, number> = { low: 0.55, medium: 0.8, high: 1.0, ultra: 1.25 };
const SLOTS = [4, 2, 1]; // variant meshes per LOD

export interface Rocks {
  group: THREE.Group;
  update(camera: THREE.Camera, force?: boolean): void;
  setQuality(q: Q): void;
  stats(): { instances: number; tris: number; draws: number };
}

const FADE = `
  uniform vec4 uFade; uniform vec3 uCamPos;
  float rockFade(vec3 ip) {
    float d = distance(ip, uCamPos);
    float fin = clamp((d - uFade.x) / max(uFade.y - uFade.x, 1e-3), 0.0, 1.0);
    if (fin < 1.0) return -fin;
    return 1.0 - clamp((d - uFade.z) / max(uFade.w - uFade.z, 1e-3), 0.0, 1.0);
  }`;

function rockMaterial(tex: { value: THREE.Texture }, texRefs: { value: THREE.Texture }[], fade: { value: THREE.Vector4 }, cam: { value: THREE.Vector3 }) {
  const m = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.86, metalness: 0, envMapIntensity: 0.5 });
  m.onBeforeCompile = (s) => {
    const tu = { value: tex.value }; texRefs.push(tu);
    Object.assign(s.uniforms, { uFade: fade, uCamPos: cam, tRock: tu });
    s.vertexShader = s.vertexShader
      .replace('#include <common>', `#include <common>
        attribute float ao; varying float vAO; varying vec3 vRW; varying vec3 vRN; varying vec3 vTint; varying float vFade;
        ${FADE}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        vAO = ao;
        #ifdef USE_INSTANCING_COLOR
          vTint = instanceColor;
        #else
          vTint = vec3(1.0);
        #endif`)
      .replace('#include <project_vertex>', `#include <project_vertex>
        #ifdef USE_INSTANCING
          vec4 rw = modelMatrix * instanceMatrix * vec4(transformed, 1.0);
          vRW = rw.xyz;
          vRN = normalize(mat3(modelMatrix) * mat3(instanceMatrix) * objectNormal);
          vFade = rockFade((modelMatrix * vec4(instanceMatrix[3].xyz, 1.0)).xyz);
          if (abs(vFade) < 0.004) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        #else
          vRW = (modelMatrix * vec4(transformed, 1.0)).xyz; vRN = normalize(mat3(modelMatrix) * objectNormal); vFade = 1.0;
        #endif`);
    s.fragmentShader = s.fragmentShader
      .replace('#include <common>', `#include <common>
        uniform sampler2D tRock;
        varying float vAO; varying vec3 vRW; varying vec3 vRN; varying vec3 vTint; varying float vFade;
        float rH;
        vec4 triR(vec3 p, vec3 n) {
          vec3 w = pow(abs(n), vec3(4.0)); w /= dot(w, vec3(1.0));
          return texture2D(tRock, p.zy) * w.x + texture2D(tRock, p.xz) * w.y + texture2D(tRock, p.xy) * w.z;
        }
        vec3 perturbR(vec3 surf_pos, vec3 surf_norm, vec2 dHdxy) {
          vec3 vSigmaX = dFdx(surf_pos); vec3 vSigmaY = dFdy(surf_pos);
          vec3 R1 = cross(vSigmaY, surf_norm); vec3 R2 = cross(surf_norm, vSigmaX);
          float fDet = dot(vSigmaX, R1);
          vec3 vGrad = sign(fDet) * (dHdxy.x * R1 + dHdxy.y * R2);
          return normalize(abs(fDet) * surf_norm - vGrad);
        }`)
      .replace('#include <clipping_planes_fragment>', `{
          float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
          if ((vFade < 0.0 ? 1.0 - ign : ign) >= abs(vFade)) discard;
        }
        #include <clipping_planes_fragment>`)
      .replace('#include <map_fragment>', `#include <map_fragment>
        {
          vec3 n = normalize(vRN);
          // two scales: 4 m joints / grain and a 17 m macro tone so big faces never repeat
          vec4 t = triR(vRW / 4.2, n);
          vec4 t2 = triR(vRW / 17.0 + 0.37, n);
          // detail_rock mean ~ (0.069, 0.053, 0.037): normalise, then a grey-brown gneiss base
          vec3 c = mix(vec3(1.0), t.rgb / vec3(0.069, 0.053, 0.037), 0.55) * vec3(0.14, 0.128, 0.11); // tamed contrast
          c *= mix(0.8, 1.2, dot(t2.rgb / vec3(0.069, 0.053, 0.037), vec3(0.333)) * 0.5);
          // lichen (pale grey-green) + moss on ledges facing the sky, darker streaks down vertical faces
          float up = smoothstep(0.45, 0.85, n.y);
          float lich = up * smoothstep(0.35, 0.65, t2.a + 0.2 * t.a);
          c = mix(c, vec3(0.135, 0.14, 0.105), lich * 0.5);
          c = mix(c, vec3(0.05, 0.07, 0.03), up * (1.0 - lich) * 0.35);
          c *= mix(1.0, 0.78, (1.0 - abs(n.y)) * smoothstep(0.4, 0.6, fract(vRW.x * 0.37 + vRW.z * 0.29 + t.a)));
          diffuseColor.rgb = c * vTint * (0.4 + 0.65 * vAO * vAO); // cavity darkening seats the blocks
          rH = t.a + 0.5 * t2.a;
        }`)
      .replace('#include <color_fragment>', '')
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
        {
          float fd = 1.0 - smoothstep(40.0, 160.0, length(vRW - cameraPosition));
          vec2 dH = vec2(dFdx(rH), dFdy(rH)) * 1.6 * fd;
          normal = perturbR(-vViewPosition, normal, dH);
        }`);
  };
  m.customProgramCacheKey = () => 'rock';
  return m;
}

export async function buildRocks(opts: { quality?: Q } = {}): Promise<Rocks> {
  const [meta, geoBuf, rBuf, rj] = await Promise.all([
    fetch(assetUrl('rocks/geo.json')).then((r) => r.json() as Promise<GeoJson>), bin('rocks/geo.bin'), bin('rocks_f32.bin'),
    fetch(assetUrl('rocks.json')).then((r) => r.json() as Promise<{ stride: number }>),
  ]);
  const S = rj.stride, R = new Float32Array(rBuf), n = R.length / S;
  // the terrain's rock detail (KTX2 when baked, src/engine/textures.ts); swapped in when it arrives
  const tex = new THREE.Texture();
  const texU: { value: THREE.Texture }[] = [];
  const camU = { value: new THREE.Vector3(1e9, 0, 0) };
  const K = meta.kinds.length;
  const fades = meta.kinds.map(() => [0, 1, 2].map(() => ({ value: new THREE.Vector4() })));
  const rockTex = { value: tex };
  const mats = fades.map((fl) => fl.map((f) => rockMaterial(rockTex, texU, f, camU)));
  loadTexture('terrain/detail_rock', 'terrain/detail_rock.webp', { srgb: true, anisotropy: 4, wrap: THREE.RepeatWrapping })
    .then((t) => { rockTex.value = t; for (const u of texU) u.value = t; });

  // geometries [kind][variant][lod]
  const geos = meta.kinds.map((k) => k.variants.map((v) => v.lods.map((l) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(geoBuf, l.pos, l.vcount * 3), 3));
    g.setAttribute('normal', new THREE.InterleavedBufferAttribute(new THREE.InterleavedBuffer(new Int8Array(geoBuf, l.nrm, l.vcount * 4), 4), 3, 0, true));
    g.setAttribute('ao', new THREE.BufferAttribute(new Uint8Array(geoBuf, l.ao, l.vcount), 1, true));
    g.setIndex(new THREE.BufferAttribute(new Uint16Array(geoBuf, l.idx, l.icount), 1));
    return g;
  })));

  // instances sorted by kind; world matrices + tints precomputed
  // sorted by (kind, 100 m cell): update() culls whole cells before touching instances
  const CELLR = 40, ncx = Math.ceil(2000 / CELLR), ncy = Math.ceil(667 / CELLR), NCR = ncx * ncy;
  const cellOf = (i: number) => Math.min(ncy - 1, Math.max(0, Math.floor(R[i * S + 1] / CELLR))) * ncx + Math.min(ncx - 1, Math.max(0, Math.floor(R[i * S] / CELLR)));
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (R[a * S + 4] - R[b * S + 4]) * NCR + cellOf(a) - cellOf(b));
  const M = new Float32Array(n * 16), C = new Float32Array(n * 3), P = new Float32Array(n * 3), RAD = new Float32Array(n);
  const kindOf = new Uint8Array(n), varOf = new Uint8Array(n);
  const kStart = new Int32Array(K + 1);
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(0, 0, 0, 'YXZ'), sc = new THREE.Vector3(), p = new THREE.Vector3();
  const perKind = new Int32Array(K);
  order.forEach((i, k) => {
    const o = i * S, kind = R[o + 4] | 0, s = R[o + 3];
    const [X, , Z] = pxToWorld(R[o], R[o + 1]);
    p.set(X, R[o + 2], Z); e.set(-R[o + 7], R[o + 6], 0); q.setFromEuler(e); sc.setScalar(s);
    m4.compose(p, q, sc).toArray(M, k * 16);
    const h = Math.sin(i * 12.9898) * 43758.5453, r = h - Math.floor(h);
    const warm = 0.9 + 0.2 * r;
    C[k * 3] = warm * 1.02; C[k * 3 + 1] = 0.98 + 0.06 * (1 - r); C[k * 3 + 2] = 0.95 + 0.08 * (1 - r);
    P.set([X, R[o + 2], Z], k * 3);
    kindOf[k] = kind; varOf[k] = (R[o + 5] | 0) % meta.variants;
    if (R[o + 5] >= 10) { C[k * 3] *= 0.55; C[k * 3 + 1] *= 0.58; C[k * 3 + 2] *= 0.6; } // spray-wet ledge rock (rocks.py: variant + 10)
    const d = meta.kinds[kind].dims; RAD[k] = Math.max(d[0], d[1], d[2]) * 0.75 * s;
    perKind[kind]++;
  });
  for (let k = 0; k < K; k++) kStart[k + 1] = kStart[k] + perKind[k];
  // per kind x cell instance ranges + cell vertical extents
  const cr = new Int32Array(K * NCR * 2).fill(-1), cMinY = new Float32Array(NCR).fill(1e9), cMaxY = new Float32Array(NCR).fill(-1e9);
  order.forEach((i, k) => {
    const c = cellOf(i), ri = (kindOf[k] * NCR + c) * 2;
    if (cr[ri] < 0) cr[ri] = k;
    cr[ri + 1] = k + 1;
    cMinY[c] = Math.min(cMinY[c], P[k * 3 + 1] - RAD[k]); cMaxY[c] = Math.max(cMaxY[c], P[k * 3 + 1] + RAD[k]);
  });
  const cX0 = (cx: number) => (cx * CELLR - 1000) * 2.5, cZ0 = (cy: number) => (cy * CELLR - 333.5) * 2.5;
  const cbox = new THREE.Box3();

  const group = new THREE.Group(); group.name = 'rocks';
  // meshes[kind][lod][slot]
  const meshes = meta.kinds.map((kd, ki) => [0, 1, 2].map((li) => Array.from({ length: SLOTS[li] }, (_, s) => {
    const im = new THREE.InstancedMesh(geos[ki][s][li], mats[ki][li], Math.max(1, perKind[ki]));
    im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    im.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, perKind[ki]) * 3), 3);
    im.instanceColor.setUsage(THREE.DynamicDrawUsage);
    im.count = 0; im.frustumCulled = false; im.visible = false;
    im.castShadow = li === 0; im.receiveShadow = true;
    im.name = `rock_${kd.name}_lod${li}_${s}`; im.userData.kind = 'rock';
    group.add(im);
    return im;
  })));

  let qs = QSCALE[opts.quality ?? 'high'];
  const lo = new Float32Array(K * 3), hi = new Float32Array(K * 3);
  const applyQuality = () => {
    meta.kinds.forEach((kd, ki) => {
      const [d0, d1, far] = (DIST[kd.name] ?? [60, 240, 1500]).map((v) => v * qs);
      const w0 = Math.max(3, d0 * 0.12), w1 = Math.max(6, d1 * 0.1), wf = far * 0.15;
      const bands: [number, number, number, number][] = [[-2, -1, d0 - w0, d0 + w0], [d0 - w0, d0 + w0, d1 - w1, d1 + w1], [d1 - w1, d1 + w1, far - wf, far]];
      bands.forEach((b, li) => { fades[ki][li].value.set(...b); lo[ki * 3 + li] = li ? b[0] : -1; hi[ki * 3 + li] = b[3]; });
    });
  };
  applyQuality();

  const frustum = new THREE.Frustum(), pm = new THREE.Matrix4(), sph = new THREE.Sphere();
  const lastPos = new THREE.Vector3(1e9, 0, 0), lastDir = new THREE.Vector3(), dir = new THREE.Vector3();
  const cnt = new Int32Array(K * 3 * 4);
  const st = { instances: 0, tris: 0, draws: 0 };
  const SLACK = 3;

  function update(camera: THREE.Camera, force = false) {
    const cp = camera.position;
    camU.value.copy(cp);
    camera.getWorldDirection(dir);
    if (!force && cp.distanceToSquared(lastPos) < 4 && dir.dot(lastDir) > 0.9995) return;
    lastPos.copy(cp); lastDir.copy(dir);
    pm.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse); frustum.setFromProjectionMatrix(pm);
    cnt.fill(0);
    st.instances = 0; st.tris = 0; st.draws = 0;
    for (let ki = 0; ki < K; ki++) {
      const far = hi[ki * 3 + 2] + SLACK;
      for (let c = 0; c < NCR; c++) {
       const ri = (ki * NCR + c) * 2;
       if (cr[ri] < 0) continue;
       const cx = c % ncx, cy = (c - cx) / ncx;
       cbox.min.set(cX0(cx) - 10, cMinY[c], cZ0(cy) - 10); cbox.max.set(cX0(cx + 1) + 10, cMaxY[c], cZ0(cy + 1) + 10);
       if (cbox.distanceToPoint(cp) > far || !frustum.intersectsBox(cbox)) continue;
       for (let k = cr[ri]; k < cr[ri + 1]; k++) {
        const dx = P[k * 3] - cp.x, dy = P[k * 3 + 1] - cp.y, dz = P[k * 3 + 2] - cp.z;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (d > far) continue;
        sph.center.set(P[k * 3], P[k * 3 + 1], P[k * 3 + 2]); sph.radius = RAD[k];
        if (!frustum.intersectsSphere(sph)) continue;
        for (let li = 0; li < 3; li++) {
          if (d < lo[ki * 3 + li] - SLACK || d > hi[ki * 3 + li] + SLACK) continue;
          const s = varOf[k] % SLOTS[li], ci = (ki * 3 + li) * 4 + s, im = meshes[ki][li][s], o = cnt[ci]++;
          const am = im.instanceMatrix.array as Float32Array, ac = im.instanceColor!.array as Float32Array;
          for (let j = 0; j < 16; j++) am[o * 16 + j] = M[k * 16 + j];
          ac[o * 3] = C[k * 3]; ac[o * 3 + 1] = C[k * 3 + 1]; ac[o * 3 + 2] = C[k * 3 + 2];
        }
       }
      }
    }
    for (let ki = 0; ki < K; ki++) for (let li = 0; li < 3; li++) for (let s = 0; s < SLOTS[li]; s++) {
      const im = meshes[ki][li][s], c = cnt[(ki * 3 + li) * 4 + s];
      im.count = c; im.visible = c > 0;
      if (c) {
        im.instanceMatrix.clearUpdateRanges(); im.instanceMatrix.addUpdateRange(0, c * 16); im.instanceMatrix.needsUpdate = true;
        im.instanceColor!.clearUpdateRanges(); im.instanceColor!.addUpdateRange(0, c * 3); im.instanceColor!.needsUpdate = true;
        st.instances += c; st.tris += c * (im.geometry.index!.count / 3); st.draws++;
      }
    }
  }

  return {
    group, update,
    setQuality(qn) { qs = QSCALE[qn]; applyQuality(); lastPos.set(1e9, 0, 0); },
    stats: () => st,
  };
}
