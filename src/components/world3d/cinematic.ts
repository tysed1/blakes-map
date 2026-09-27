import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { CSM } from 'three/examples/jsm/csm/CSM.js';
import { installSunBake, loadSunBake, sunBake } from '../../engine/sunbake';

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
    return col;
  }
#endif`;
  // clamp: sun glints on water can exceed half-float range (Inf -> NaN in tone mapping / shafts)
  C.fog_fragment = `#ifdef USE_FOG
  gl_FragColor.rgb = atmosphere(min(gl_FragColor.rgb, vec3(256.0)), vFogView);
#endif`;
}

/** Filmic grade in display space: warm highlights / cool shadows, saturation, contrast, vignette. */
const GradeShader = {
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    uSat: { value: 1.12 }, uContrast: { value: 1.08 }, uVignette: { value: 0.32 },
    uShadowTint: { value: new THREE.Vector3(-0.012, 0.0, 0.03) }, uHighTint: { value: new THREE.Vector3(0.035, 0.012, -0.03) },
    uLift: { value: 0.012 }, uRes: { value: new THREE.Vector2(1, 1) }, uSharpen: { value: 0.18 },
  },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `uniform sampler2D tDiffuse; uniform float uSat, uContrast, uVignette, uLift, uSharpen; uniform vec3 uShadowTint, uHighTint; uniform vec2 uRes;
    varying vec2 vUv;
    void main(){
      vec2 px = 1.0 / vec2(textureSize(tDiffuse, 0));
      vec3 c = texture2D(tDiffuse, vUv).rgb;
      // light unsharp mask (recovers leaf-card crispness lost to MSAA resolve)
      vec3 bl = (texture2D(tDiffuse, vUv + vec2(px.x, 0.0)).rgb + texture2D(tDiffuse, vUv - vec2(px.x, 0.0)).rgb +
                 texture2D(tDiffuse, vUv + vec2(0.0, px.y)).rgb + texture2D(tDiffuse, vUv - vec2(0.0, px.y)).rgb) * 0.25;
      c = max(c + (c - bl) * uSharpen, 0.0);
      float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
      c = mix(vec3(l), c, uSat);
      c = (c - 0.5) * uContrast + 0.5;
      c += uShadowTint * (1.0 - smoothstep(0.0, 0.5, l)) + uHighTint * smoothstep(0.45, 1.0, l);
      c = c * (1.0 - uLift) + uLift;
      vec2 q = vUv - 0.5; q.x *= px.y / px.x;
      c *= 1.0 - uVignette * smoothstep(0.35, 1.05, length(q));
      gl_FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
    }`,
};

/** Sun shafts: radial scatter of the bright (sky) pixels toward the sun's screen position, in HDR. */
const ShaftsShader = {
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    uSun: { value: new THREE.Vector2(0.5, 0.5) }, uStrength: { value: 0.0 }, uColor: { value: new THREE.Vector3(1.0, 0.72, 0.45) },
    uAspect: { value: 1.0 },
  },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `uniform sampler2D tDiffuse; uniform vec2 uSun; uniform float uStrength, uAspect; uniform vec3 uColor; varying vec2 vUv;
    void main(){
      vec4 base = texture2D(tDiffuse, vUv);
      base.rgb = any(isnan(base.rgb)) ? vec3(0.0) : min(base.rgb, vec3(256.0));
      if (uStrength <= 0.001) { gl_FragColor = base; return; }
      vec2 d = (uSun - vUv) / 40.0;
      vec2 p = vUv; float acc = 0.0, w = 1.0;
      float n = fract(sin(dot(vUv, vec2(12.9898, 78.233))) * 43758.5453);
      p += d * n;
      for (int i = 0; i < 40; i++) {
        vec3 c = min(texture2D(tDiffuse, p).rgb, vec3(16.0));
        if (any(isnan(c))) c = vec3(0.0);
        float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
        acc += smoothstep(1.2, 3.0, l) * w;
        w *= 0.96; p += d;
      }
      vec2 q = vUv - uSun; q.x *= uAspect;
      float fall = exp(-length(q) * 2.2);
      gl_FragColor = vec4(base.rgb + uColor * acc / 40.0 * uStrength * fall, base.a);
    }`,
};

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

export interface Cinematic {
  shafts: ShaderPass;
  composer: EffectComposer;
  bloom: UnrealBloomPass;
  grade: ShaderPass;
  csm: CSM | null;
  /** the sun (first cascade light) */
  readonly sun: THREE.DirectionalLight;
  setSize(w: number, h: number, dpr: number): void;
  render(): void;
  prepare(scene: THREE.Object3D): void;
  setShadowQuality(q: ShadowQuality): void;
}

export function createCinematic(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera, sunDir: THREE.Vector3, sunColor: THREE.Color, sunIntensity: number): Cinematic {
  const size = renderer.getSize(new THREE.Vector2());
  const rt = new THREE.WebGLRenderTarget(size.x, size.y, { type: THREE.HalfFloatType, samples: 4 });
  const composer = new EffectComposer(renderer, rt);
  composer.addPass(new RenderPass(scene, camera));
  const shafts = new ShaderPass(ShaftsShader);
  composer.addPass(shafts);
  const bloom = new UnrealBloomPass(size, 0.22, 0.55, 0.92);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());
  const grade = new ShaderPass(GradeShader);
  composer.addPass(grade);

  installSunBake(sunDir);
  loadSunBake(renderer);
  // one CSM per cascade count (the count is baked into the shaders); only the active one is in the scene
  const rigs = new Map<number, CSM>();
  const rig = (n: number) => {
    let c = rigs.get(n);
    if (!c) {
      c = new CSM({
        camera, parent: scene, cascades: n, maxFar: 300, mode: 'practical', shadowMapSize: 2048,
        lightDirection: sunDir.clone().negate(), lightIntensity: sunIntensity, lightNear: 1, lightFar: 2500, lightMargin: 250, shadowBias: -0.00012,
      } as any);
      c.fade = true;
      for (const l of c.lights) { l.color.copy(sunColor); l.shadow.normalBias = 0.5; }
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
  const sp = new THREE.Vector3(), fwd = new THREE.Vector3();
  return {
    composer, bloom, grade, shafts,
    get csm() { return csm; },
    get sun() { return csm.lights[0]; },
    setSize(w, h, dpr) {
      composer.setPixelRatio(dpr); composer.setSize(w, h);
      grade.uniforms.uRes.value.set(w * dpr, h * dpr);
      csm.updateFrustums();
    },
    render() {
      if (renderer.shadowMap.enabled) csm.update();
      // shafts only when the sun is in front of the camera
      sp.copy(sunDir).multiplyScalar(10000).add(camera.position).project(camera);
      camera.getWorldDirection(fwd);
      const facing = fwd.dot(sunDir);
      shafts.uniforms.uSun.value.set(sp.x * 0.5 + 0.5, sp.y * 0.5 + 0.5);
      shafts.uniforms.uStrength.value = THREE.MathUtils.smoothstep(facing, 0.1, 0.6) * 1.6;
      shafts.uniforms.uAspect.value = camera.aspect;
      shafts.enabled = shafts.uniforms.uStrength.value > 0.001;
      composer.render();
    },
    prepare(root) {
      root.traverse((o) => {
        const mm = (o as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
        if (!mm || (o as any).isSprite) return;
        if (Array.isArray(mm)) mm.forEach(setup); else setup(mm);
      });
    },
    setShadowQuality(q) {
      const P = SHADOW_PRESET[q];
      const n = Math.max(1, P.cascades);
      if (n !== csm.cascades) {
        csm.remove();
        csm = rig(n);
        attach(csm);
        hooked.forEach(hook);
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
      sunBake.setNearField(c0, on ? P.far * 1.02 : 1);
    },
  };
}
