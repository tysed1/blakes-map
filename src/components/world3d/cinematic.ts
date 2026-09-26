import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { CSM } from 'three/examples/jsm/csm/CSM.js';

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

export interface Cinematic {
  shafts: ShaderPass;
  composer: EffectComposer;
  bloom: UnrealBloomPass;
  grade: ShaderPass;
  csm: CSM | null;
  setSize(w: number, h: number, dpr: number): void;
  render(): void;
  prepare(scene: THREE.Object3D): void;
  setShadowQuality(q: 'low' | 'medium' | 'high' | 'ultra'): void;
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

  // cascaded shadows: crisp contact shadows up close, tree/terrain shadow patterns to the horizon
  const csm = new CSM({
    camera, parent: scene, cascades: 4, maxFar: 3200, mode: 'practical', shadowMapSize: 2048,
    lightDirection: sunDir.clone().negate(), lightIntensity: sunIntensity, lightNear: 1, lightFar: 9000, lightMargin: 400, shadowBias: -0.00012,
  } as any);
  csm.fade = true;
  for (const l of csm.lights) { l.color.copy(sunColor); l.shadow.normalBias = 0.5; }
  const done = new WeakSet<THREE.Material>();
  const setup = (m: THREE.Material) => {
    if (done.has(m) || !(m as any).isMeshStandardMaterial && !(m as any).isMeshLambertMaterial && !(m as any).isMeshPhongMaterial && !(m as any).isMeshPhysicalMaterial) return;
    done.add(m);
    const prev = m.onBeforeCompile;
    csm.setupMaterial(m);
    const hook = m.onBeforeCompile;
    m.onBeforeCompile = (s, r) => { prev.call(m, s, r); hook.call(m, s, r); };
    const key = m.customProgramCacheKey.bind(m);
    m.customProgramCacheKey = () => key() + '|csm';
    m.needsUpdate = true;
  };
  return {
    composer, bloom, grade, csm, shafts,
    setSize(w, h, dpr) {
      composer.setPixelRatio(dpr); composer.setSize(w, h);
      grade.uniforms.uRes.value.set(w * dpr, h * dpr);
      csm.updateFrustums();
    },
    render() {
      csm.update();
      // shafts only when the sun is in front of the camera
      const sp = sunDir.clone().multiplyScalar(10000).add(camera.position).project(camera);
      const fwd = new THREE.Vector3(); camera.getWorldDirection(fwd);
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
        (Array.isArray(mm) ? mm : [mm]).forEach(setup);
      });
    },
    setShadowQuality(q) {
      const s = { low: 1024, medium: 2048, high: 2048, ultra: 4096 }[q];
      csm.maxFar = { low: 1200, medium: 2000, high: 3200, ultra: 4000 }[q];
      for (const l of csm.lights) { if (l.shadow.mapSize.x !== s) { l.shadow.mapSize.set(s, s); l.shadow.map?.dispose(); (l.shadow as any).map = null; } }
      csm.updateFrustums();
    },
  };
}
