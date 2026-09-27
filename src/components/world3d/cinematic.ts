import * as THREE from 'three';
import { CSM } from 'three/examples/jsm/csm/CSM.js';
import { installSunBake, loadSunBake, sunBake } from '../../engine/sunbake';
import { FusedPost, GradeParams } from '../../engine/post';

/**
 * Cinematic look: physically-flavoured aerial perspective, golden-hour sun, cascaded shadows,
 * bloom and a filmic colour grade. Everything here is global (shader chunks + post chain), so
 * every material in the scene (terrain, trees, roads, water, ground cover) shares one atmosphere.
 */

export interface AtmosphereParams {
  sunDir: THREE.Vector3;       // toward the sun (normalized)
  haze: THREE.Color;           // blue aerial-perspective colour (linear) = scene.fog.color
  sunHaze: THREE.Color;        // warm in-scatter colour toward the sun (linear)
  baseHeight: number;          // metres: fog density reference height (valley floor)
  falloff: number;             // 1/m: exponential height falloff
  mist?: { base: number; scale: number; density: number };   // valley mist layer (R5): metres, metres, 1/m
}

/**
 * Replace three's fog with height-exponential aerial perspective + sun in-scattering.
 * Uses FogExp2.density as the base extinction and fog.color as the ambient haze colour; the sun
 * direction / warm colour / height falloff are compile-time constants (static sun).
 * Must be called before any material compiles.
 */
export function installAtmosphere(p: AtmosphereParams) {
  const v3 = (v: { x: number; y: number; z: number } | THREE.Color) => 'r' in v ? `vec3(${v.r.toFixed(4)}, ${v.g.toFixed(4)}, ${v.b.toFixed(4)})` : `vec3(${v.x.toFixed(4)}, ${v.y.toFixed(4)}, ${v.z.toFixed(4)})`;
  const C = THREE.ShaderChunk as any;
  C.fog_pars_vertex = `#ifdef USE_FOG
  varying float vFogDepth;
  varying vec3 vFogView;
#endif`;
  C.fog_vertex = `#ifdef USE_FOG
  vFogDepth = - mvPosition.z;
  vFogView = mvPosition.xyz;
#endif`;
  C.fog_pars_fragment = `#ifdef USE_FOG
  uniform vec3 fogColor;
  varying float vFogDepth;
  varying vec3 vFogView;
  #ifdef FOG_EXP2
    uniform float fogDensity;
  #else
    uniform float fogNear;
    uniform float fogFar;
  #endif
  #define ATM_SUN ${v3(p.sunDir)}
  #define ATM_SUNHAZE ${v3(p.sunHaze)}
  ${p.mist ? `#define ATM_MIST vec3(${p.mist.base.toFixed(1)}, ${p.mist.scale.toFixed(1)}, ${p.mist.density.toFixed(5)})
  #define ATM_MISTCOL vec3(0.36, 0.4, 0.46)` : ''}
  vec3 atmosphere(vec3 col, vec3 viewPos) {
    vec3 rd = transpose(mat3(viewMatrix)) * viewPos;      // world-space camera -> fragment
    float dist = length(rd);
    vec3 dir = rd / max(dist, 1e-3);
    #ifdef FOG_EXP2
      float dens = fogDensity;
    #else
      float dens = 1.0 / max(fogFar, 1.0);
    #endif
    // height fog integral along the ray (denser in the valleys, thin over the ridges)
    float k = ${p.falloff.toFixed(5)};
    float h0 = cameraPosition.y - ${p.baseHeight.toFixed(1)};
    float dy = rd.y;
    float fh = exp(-k * h0) * (abs(k * dy) > 1e-4 ? (1.0 - exp(-k * dy)) / (k * dy) : 1.0);
    float od = dens * dist * clamp(fh, 0.05, 6.0);
    float ext = 1.0 - exp(-od);
    float inscat = 1.0 - exp(-od * 0.55);
    float mu = max(dot(dir, ATM_SUN), 0.0);
    vec3 fc = mix(fogColor, ATM_SUNHAZE, clamp(pow(mu, 6.0) * 0.85 + pow(mu, 1.5) * 0.18, 0.0, 1.0));
    // aerial perspective: extinction toward the haze colour, plus a warm forward-scatter glow
    col = col * (1.0 - ext) + fc * ext;
    col += ATM_SUNHAZE * pow(mu, 12.0) * inscat * 0.35;
    #ifdef ATM_MIST
    {
      // valley mist (R5): a thin layer hugging the valley floors (exponential above ATM_MIST.x, scale
      // height ATM_MIST.y), broken into drifting-free pockets by low-frequency noise at the fragment
      float km = 1.0 / ATM_MIST.y;
      float hc = max(cameraPosition.y - ATM_MIST.x, -15.0), hf = max(hc + dy, -15.0);
      float ec = exp(-km * hc), ef = exp(-km * hf);
      float fm = abs(hf - hc) > 0.5 ? (ec - ef) / (km * (hf - hc)) : ec;
      vec2 q = (cameraPosition.xz + rd.xz) / 650.0;
      vec2 i = floor(q), f = fract(q); f = f * f * (3.0 - 2.0 * f);
      #define ATM_H(v) fract(sin(dot(v, vec2(127.1, 311.7))) * 43758.5453)
      float nz = mix(mix(ATM_H(i), ATM_H(i + vec2(1.0, 0.0)), f.x), mix(ATM_H(i + vec2(0.0, 1.0)), ATM_H(i + vec2(1.0, 1.0)), f.x), f.y);
      float pocket = 0.25 + 0.75 * smoothstep(0.3, 0.75, nz);
      float odm = ATM_MIST.z * dist * fm * pocket;
      float em = 1.0 - exp(-odm);
      // lit from above-behind: a soft cool white, only mildly warmer toward the sun (no sun-side bloom)
      vec3 mc = mix(ATM_MISTCOL, ATM_SUNHAZE * 0.75, clamp(pow(mu, 8.0) * 0.45, 0.0, 1.0));
      em *= smoothstep(-2.0, 25.0, hc + 40.0 * (1.0 - pocket));   // thinner when the camera is inside the layer
      col = col * (1.0 - em) + mc * em;
    }
    #endif
    return col;
  }
#endif`;
  // clamp: sun glints on water can exceed half-float range (Inf -> NaN in tone mapping / shafts)
  C.fog_fragment = `#ifdef USE_FOG
  gl_FragColor.rgb = atmosphere(min(gl_FragColor.rgb, vec3(256.0)), vFogView);
#endif`;
}

/**
 * Post: one fused full-screen pass (src/engine/post.ts, board item R2): quarter-res sun shafts and
 * bloom, then ACES + sRGB + unsharp mask + filmic grade (warm highlights / cool shadows, saturation,
 * contrast, lift, vignette) at full resolution.
 */

export type ShadowQuality = 'low' | 'medium' | 'high' | 'ultra';
/**
 * Near-field real-time shadows (board item R1). Everything past the last cascade (and the terrain
 * everywhere) comes from the static-sun bake (src/engine/sunbake.ts), so the cascades stay short:
 * Low none, Medium 1 x 160 m, High 2 x 300 m, Ultra 3 x 500 m.
 */
export const SHADOW_PRESET: Record<ShadowQuality, { cascades: number; far: number; size: number }> = {
  low: { cascades: 0, far: 0, size: 1024 },
  medium: { cascades: 1, far: 160, size: 2048 },
  high: { cascades: 2, far: 300, size: 2048 },
  ultra: { cascades: 3, far: 500, size: 2048 },
};

/**
 * Objects on this layer (and not on layer 0) cast only into the nearest cascade: the nearest tree
 * LOD is drawn within ~55 m, where the first cascade covers everything; skipping it in the farther
 * cascades saves ~60 shadow draw calls per cascade. The main camera renders the layer as usual.
 */
export const NEAR_CASTER_LAYER = 5;

export interface Cinematic {
  post: FusedPost;
  /** bloom on/off (quality presets) */
  bloom: { enabled: boolean };
  grade: GradeParams;
  csm: CSM | null;
  /** the sun (first cascade light) */
  readonly sun: THREE.DirectionalLight;
  setSize(w: number, h: number, dpr: number): void;
  render(): void;
  prepare(scene: THREE.Object3D): void;
  /** casterRange: distance up to which trees cast real-time shadows (the bake's canopy term takes over there) */
  setShadowQuality(q: ShadowQuality, casterRange?: number): void;
}

export function createCinematic(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera, sunDir: THREE.Vector3, sunColor: THREE.Color, sunIntensity: number): Cinematic {
  camera.layers.enable(NEAR_CASTER_LAYER);
  const post = new FusedPost(renderer);
  const size = renderer.getDrawingBufferSize(new THREE.Vector2());
  post.setSize(size.x, size.y);
  const bloom = { get enabled() { return post.bloomEnabled; }, set enabled(v: boolean) { post.bloomEnabled = v; } };

  installSunBake(sunDir);
  loadSunBake(renderer);
  // one CSM per cascade count (the count is baked into the shaders); only the active one is in the scene
  const rigs = new Map<number, CSM>();
  const rig = (n: number) => {
    let c = rigs.get(n);
    if (!c) {
      c = new CSM({
        camera, parent: scene, cascades: n, maxFar: 300, mode: 'practical', shadowMapSize: 2048,
        lightDirection: sunDir.clone().negate(), lightIntensity: sunIntensity, lightNear: 1, lightFar: 2500, lightMargin: 250, shadowBias: -0.00045, // ~1.1 m in depth (as before at lightFar 9000)
      } as any);
      c.fade = true;
      for (const l of c.lights) { l.color.copy(sunColor); l.shadow.normalBias = 0.5; }
      c.lights[0].shadow.camera.layers.enable(NEAR_CASTER_LAYER);
      // uniform arrays always padded to the largest cascade count: three reuses a material's cached
      // program when its key comes back (quality switched away and back) WITHOUT re-running
      // onBeforeCompile, so the live CSM_cascades array may belong to another rig
      const ext = (c as any).getExtendedBreaks.bind(c);
      (c as any).getExtendedBreaks = (t: THREE.Vector2[]) => { ext(t); while (t.length < 3) t.push(t[t.length - 1]?.clone() ?? new THREE.Vector2()); };
      c.remove();
      rigs.set(n, c);
    }
    return c;
  };
  let csm = rig(2);
  const attach = (c: CSM) => { for (const l of c.lights) { scene.add(l); scene.add(l.target); } };
  attach(csm);
  // every lit material: CSM hook + baked sun (defines / uniforms); re-hooked when the cascade count changes
  const hooked = new Set<THREE.Material>();
  const orig = new WeakMap<THREE.Material, { obc: THREE.Material['onBeforeCompile']; key: () => string }>();
  const hook = (m: THREE.Material) => {
    let o = orig.get(m);
    if (!o) { o = { obc: m.onBeforeCompile, key: m.customProgramCacheKey.bind(m) }; orig.set(m, o); }
    const { obc, key } = o;
    csm.setupMaterial(m);
    const h = m.onBeforeCompile, n = csm.cascades;
    m.onBeforeCompile = (s, r) => { obc.call(m, s, r); h.call(m, s, r); sunBake.onCompile(s); };
    m.defines!.USE_SUNBAKE = '';
    m.customProgramCacheKey = () => key() + '|csm' + n;
    m.needsUpdate = true;
  };
  const setup = (m: THREE.Material) => {
    if (hooked.has(m) || !(m as any).isMeshStandardMaterial && !(m as any).isMeshLambertMaterial && !(m as any).isMeshPhongMaterial && !(m as any).isMeshPhysicalMaterial) return;
    hooked.add(m);
    hook(m);
  };
  const sp = new THREE.Vector3(), fwd = new THREE.Vector3(), sunUv = new THREE.Vector2();
  return {
    post, bloom, grade: post.grade,
    get csm() { return csm; },
    get sun() { return csm.lights[0]; },
    setSize(w, h, dpr) {
      post.setSize(w * dpr, h * dpr);
      csm.updateFrustums();
    },
    render() {
      if (renderer.shadowMap.enabled) csm.update();
      // shafts only when the sun is in front of the camera
      sp.copy(sunDir).multiplyScalar(10000).add(camera.position).project(camera);
      camera.getWorldDirection(fwd);
      const facing = fwd.dot(sunDir);
      post.setShafts(sunUv.set(sp.x * 0.5 + 0.5, sp.y * 0.5 + 0.5), THREE.MathUtils.smoothstep(facing, 0.1, 0.6) * post.shaftGain, camera.aspect);
      post.render(scene, camera);
    },
    prepare(root) {
      root.traverse((o) => {
        const mm = (o as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
        if (!mm || (o as any).isSprite) return;
        if (Array.isArray(mm)) mm.forEach(setup); else setup(mm);
      });
    },
    setShadowQuality(q, casterRange = Infinity) {
      const P = SHADOW_PRESET[q];
      const n = Math.max(1, P.cascades);
      if (n !== csm.cascades) {
        csm.remove();
        csm = rig(n);
        attach(csm);
        hooked.forEach(hook);
        // re-register each material's live uniforms with the active rig so its breaks stay current
        hooked.forEach((m) => {
          const u = (renderer.properties.get(m) as any).uniforms;
          if (u?.CSM_cascades) (csm as any).shaders.set(m, { uniforms: u });
        });
        (csm as any).updateUniforms?.();
      }
      const on = P.cascades > 0;
      if (renderer.shadowMap.enabled !== on) { renderer.shadowMap.enabled = on; hooked.forEach((m) => (m.needsUpdate = true)); }
      for (const l of csm.lights) {
        l.castShadow = on;
        if (l.shadow.mapSize.x !== P.size) { l.shadow.mapSize.set(P.size, P.size); l.shadow.map?.dispose(); (l.shadow as any).map = null; }
      }
      csm.maxFar = Math.max(1, P.far);
      csm.updateFrustums();
      // the CSM fades its last cascade out from the cascade's centre; the baked canopy shadows fade in
      // over the same depth range (everywhere when there are no cascades)
      const b = csm.breaks, c0 = on ? ((b.length > 1 ? b[b.length - 2] : 0) + 1) * 0.5 * P.far : 0;
      const end = Math.min(P.far * 1.02, casterRange);
      sunBake.setNearField(on ? Math.min(c0, end * 0.75) : 0, on ? end : 1);
    },
  };
}
