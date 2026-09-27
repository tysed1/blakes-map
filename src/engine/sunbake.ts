import * as THREE from 'three';
// the typings declare CSMShader as an interface only; the module exports the chunk object
import * as CSMShaderModule from 'three/examples/jsm/csm/CSMShader.js';
const CSMShader = (CSMShaderModule as unknown as { CSMShader: { lights_fragment_begin: string } }).CSMShader;
import { IMG_W, IMG_H, MPP, ORIGIN_PX } from '../core/coords';
import { assetUrl } from '../core/data';

/**
 * Baked static-sun shadow + ground AO (board item R1; bake: tools/pipeline/sun_shadow_bake.py).
 *
 * The golden-hour sun never moves, so terrain self-shadowing and the canopy's long low-sun shadows
 * are baked once into world-space textures:
 *   sunbake.jpg    (R8, 1.25 m)  D_all: metres a point must rise above the ground to see the sun
 *                                (terrain + canopy occluders)
 *   sunbake_lo.webp (RG8, 2.5 m) D_ter: same, terrain occluders only; AO: ground ambient occlusion
 *                                (canopy cover + terrain concavity)
 * with D stored as sqrt(D / 64 m). Real-time cascaded shadows only cover the near field (<= ~300 m);
 * there the bake contributes the terrain term only (the CSM has the exact leaf-card canopy), and it
 * takes over completely past the last cascade.
 *
 * Installation is global: every lit three.js material (Standard / Physical / Lambert / Phong) hooked
 * by cinematic.ts `prepare()` gets `USE_SUNBAKE` + these uniforms, and the shared shader chunks
 * multiply the sun (directional lights) by the baked visibility and the indirect diffuse by the AO.
 * Receivers: non-instanced meshes are ground-attached (height 0); instanced meshes (trees, props,
 * grass) use their height above the instance origin, so crowns rise out of the ground shadow.
 *
 * Custom ShaderMaterials can include SUNBAKE_PARS + call `sunBakeEval(worldPos, heightAboveGround,
 * viewDepth, lit, ao)` with `sunBake.uniforms` merged into their uniforms.
 */

export const SUNBAKE_DMAX = 64.0;

const uniforms = {
  tSunBake: { value: null as THREE.Texture | null },     // R8: D_all, 1.25 m
  tSunBakeLo: { value: null as THREE.Texture | null },   // RG8: D_ter, AO, 2.5 m
  // uv = world.xz * xf.xy + xf.zw (row 0 = source py 0, texture uploaded without flipY)
  uSunBakeXf: { value: new THREE.Vector4(1 / (IMG_W * MPP), 1 / (IMG_H * MPP), ORIGIN_PX[0] / IMG_W, ORIGIN_PX[1] / IMG_H) },
  // x, y: view depth where the canopy bake fades in (end of the real-time shadows); z: terrain-shadow
  // weight; w: 1 = bake loaded
  uSunBakeK: { value: new THREE.Vector4(120, 160, 0.45, 0) },
  // xy: horizontal unit vector toward the sun (world XZ), z: tan(sun elevation), w: AO strength
  uSunBakeDir: { value: new THREE.Vector4(0, 0, 0, 1) },
  // x: canopy shadow strength on the ground (leaf-card crowns let light through), y: crown receiver
  // bias per metre above the ground (sun-side crown tops stay lit inside a closed canopy), z: crown
  // lookup shift (m), w: canopy shadow strength on crowns
  uSunBakeT: { value: new THREE.Vector4(0.3, 0.8, 5, 0.2) },
};

export const SUNBAKE_PARS = /* glsl */`
uniform sampler2D tSunBake, tSunBakeLo;
uniform vec4 uSunBakeXf;
uniform vec4 uSunBakeK;
uniform vec4 uSunBakeDir;
uniform vec4 uSunBakeT;
// lit: sun visibility from the bake (terrain always, canopy past the real-time cascades); ao: ground AO
void sunBakeEval(vec3 wp, float h, float viewDepth, out float lit, out float ao) {
  lit = 1.0; ao = 1.0;
  if (uSunBakeK.w < 0.5) return;
  // crowns / tall props: step the lookup toward the sun past the object's own bulk (the bake treats
  // every crown as a solid dome, so a crown would otherwise shadow itself)
  float crown = smoothstep(1.5, 5.0, h);
  float selfk = crown * uSunBakeT.z;
  vec2 p = wp.xz + uSunBakeDir.xy * selfk;
  h += selfk * uSunBakeDir.z;
  vec2 uv = p * uSunBakeXf.xy + uSunBakeXf.zw;
  vec3 s = vec3(texture2D(tSunBake, uv).r, texture2D(tSunBakeLo, uv).rg);   // uniform control flow (mips)
  // outside the baked map (backdrop ring): open sky
  float inside = step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);
  s = mix(vec3(0.0, 0.0, 1.0), s, inside);
  vec2 D = s.rg * s.rg * ${SUNBAKE_DMAX.toFixed(1)};
  // penumbra widens with occluder height (~ distance to the occluder)
  vec2 soft = 1.0 + 0.06 * D;
  float bias = 0.25 + uSunBakeT.y * h * crown;
  vec2 L = 1.0 - smoothstep(vec2(bias), vec2(bias) + soft, D - h);
  float litTer = mix(1.0, L.y, uSunBakeK.z);
  // canopy shadow strength: ground (x) vs crowns (w; crowns already carry their own crown AO)
  float canopy = 1.0 - mix(uSunBakeT.x, uSunBakeT.w, crown) * (1.0 - (L.y > 0.02 ? clamp(L.x / L.y, 0.0, 1.0) : 1.0));
  float far = smoothstep(uSunBakeK.x, uSunBakeK.y, viewDepth);
  lit = litTer * mix(1.0, canopy, far);
  ao = mix(1.0, mix(s.b, 1.0, smoothstep(0.5, 6.0, h)), uSunBakeDir.w);
}
`;

let installed = false;

/** Patch the shared shader chunks (call before any material compiles, and before the CSM is built). */
export function installSunBake(sunDir: THREE.Vector3) {
  if (installed) return;
  installed = true;
  const hz = Math.hypot(sunDir.x, sunDir.z);
  uniforms.uSunBakeDir.value.set(sunDir.x / hz, sunDir.z / hz, sunDir.y / hz, 0.5);
  const C = THREE.ShaderChunk as any;
  C.shadowmap_pars_vertex = C.shadowmap_pars_vertex + `
#ifdef USE_SUNBAKE
  varying float vSunBakeH;
#endif`;
  C.shadowmap_vertex = C.shadowmap_vertex + `
#ifdef USE_SUNBAKE
  {
    vSunBakeH = 0.0;
    #ifdef USE_INSTANCING
      vec4 sbP = instanceMatrix * vec4(transformed, 1.0), sbO = instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
      #ifdef USE_BATCHING
        sbP = batchingMatrix * sbP; sbO = batchingMatrix * sbO;
      #endif
      vSunBakeH = max(0.0, (modelMatrix * sbP).y - (modelMatrix * sbO).y);
    #endif
  }
#endif`;
  C.shadowmap_pars_fragment = C.shadowmap_pars_fragment + `
#ifdef USE_SUNBAKE
  varying float vSunBakeH;
  ${SUNBAKE_PARS}
#endif`;
  const head = `
#ifdef USE_SUNBAKE
  float sunBakeLit, sunBakeAO;
  sunBakeEval(cameraPosition + transpose(mat3(viewMatrix)) * (-vViewPosition), vSunBakeH, vViewPosition.z, sunBakeLit, sunBakeAO);
#endif
`;
  const apply = `
#ifdef USE_SUNBAKE
  directLight.color *= sunBakeLit;
#endif
`;
  const patch = (src: string) => head + src
    .replace(/getDirectionalLightInfo\( directionalLight, directLight \);/g, (m) => m + apply)
    .replace(/getDirectionalLightInfo\( directionalLights\[0\], directLight \);/g, (m) => m + apply);
  C.lights_fragment_begin = patch(C.lights_fragment_begin);
  CSMShader.lights_fragment_begin = patch(CSMShader.lights_fragment_begin);
  C.aomap_fragment = C.aomap_fragment + `
#ifdef USE_SUNBAKE
  reflectedLight.indirectDiffuse *= sunBakeAO;
#endif`;
}

/** Start loading the bake; materials render unshadowed-by-bake (uSunBakeK.w = 0) until it arrives. */
export function loadSunBake(renderer: THREE.WebGLRenderer): Promise<void> {
  const L = new THREE.TextureLoader();
  const load = (f: string, format: THREE.PixelFormat) => new Promise<THREE.Texture | null>((res) => {
    L.load(assetUrl(f), (t) => {
      // single / dual channel GPU formats straight from the PNG (R8 / RG8: 1/4 and 1/2 of RGBA)
      t.flipY = false; t.colorSpace = THREE.NoColorSpace; t.format = format;
      t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
      t.minFilter = THREE.LinearMipmapLinearFilter; t.magFilter = THREE.LinearFilter;
      t.anisotropy = Math.min(4, renderer.capabilities.getMaxAnisotropy());
      t.needsUpdate = true;
      res(t);
    }, undefined, () => { console.warn(`${f} missing: baked sun shadows off`); res(null); });
  });
  return Promise.all([load('sunbake.jpg', THREE.RedFormat), load('sunbake_lo.webp', THREE.RGFormat)]).then(([a, b]) => {
    if (!a || !b) return;
    uniforms.tSunBake.value = a; uniforms.tSunBakeLo.value = b; uniforms.uSunBakeK.value.w = 1;
  });
}

export const sunBake = {
  uniforms,
  /** Hook a lit material (idempotent per compile: adds the define + shared uniforms). */
  onCompile(shader: THREE.WebGLProgramParametersWithUniforms) { Object.assign(shader.uniforms, uniforms); },
  /** View depth range over which the baked canopy shadows replace the real-time cascades. */
  setNearField(from: number, to: number) { uniforms.uSunBakeK.value.x = from; uniforms.uSunBakeK.value.y = to; },
  setTerrainWeight(w: number) { uniforms.uSunBakeK.value.z = w; },
  setAO(k: number) { uniforms.uSunBakeDir.value.w = k; },
};
