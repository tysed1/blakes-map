import * as THREE from 'three';
import { FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js';

/**
 * Fused post chain (board item R2). Replaces EffectComposer + shafts + UnrealBloom + OutputPass +
 * grade (4 full-screen passes after the scene) with ONE full-resolution uber pass:
 *
 *   scene (MSAA HDR half-float)
 *     -> sun shafts, quarter res (40-tap radial scatter of the bright sky toward the sun)
 *     -> bloom, quarter res: 4x4-box prefilter + threshold, 5 separable-gaussian levels, composite
 *   uber (full res, to the canvas): scene + shafts + bloom -> ACES filmic (exposure) -> sRGB
 *     -> unsharp mask (on tone-mapped neighbours) -> saturation / contrast / split-tone / lift / vignette
 *
 * The maths reproduce the old passes (same shaders and constants); bloom levels 1-4 sit at the
 * same resolutions as UnrealBloomPass's mips, level 0 is computed at quarter instead of half
 * resolution with the same pixel footprint.
 */

const VERT = /* glsl */`varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

export interface GradeParams { sat: number; contrast: number; vignette: number; lift: number; sharpen: number; shadowTint: THREE.Vector3; highTint: THREE.Vector3 }

const shaftsMat = () => new THREE.ShaderMaterial({
  uniforms: { tScene: { value: null }, uSun: { value: new THREE.Vector2(0.5, 0.5) }, uAspect: { value: 1 } },
  vertexShader: VERT, depthTest: false, depthWrite: false,
  fragmentShader: /* glsl */`uniform sampler2D tScene; uniform vec2 uSun; uniform float uAspect; varying vec2 vUv;
    void main(){
      vec2 d = (uSun - vUv) / 40.0;
      vec2 p = vUv; float acc = 0.0, w = 1.0;
      float n = fract(sin(dot(vUv, vec2(12.9898, 78.233))) * 43758.5453);
      p += d * n;
      for (int i = 0; i < 40; i++) {
        vec3 c = min(texture2D(tScene, p).rgb, vec3(16.0));
        if (any(isnan(c))) c = vec3(0.0);
        float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
        acc += smoothstep(1.2, 3.0, l) * w;
        w *= 0.96; p += d;
      }
      vec2 q = vUv - uSun; q.x *= uAspect;
      gl_FragColor = vec4(vec3(acc / 40.0 * exp(-length(q) * 2.2)), 1.0);
    }`,
});

// UnrealBloomPass's luminosity high pass, as a 4x4 box prefilter (4 bilinear taps) for quarter res
const brightMat = () => new THREE.ShaderMaterial({
  uniforms: { tScene: { value: null }, tShafts: { value: null }, uShaft: { value: new THREE.Vector3() }, uTexel: { value: new THREE.Vector2() }, uThreshold: { value: 0.92 } },
  vertexShader: VERT, depthTest: false, depthWrite: false,
  fragmentShader: /* glsl */`uniform sampler2D tScene, tShafts; uniform vec3 uShaft; uniform vec2 uTexel; uniform float uThreshold; varying vec2 vUv;
    vec3 tap(vec2 uv) {
      vec3 c = texture2D(tScene, uv).rgb;
      c = any(isnan(c)) ? vec3(0.0) : min(c, vec3(256.0));
      c += uShaft * texture2D(tShafts, uv).r;
      float v = dot(c, vec3(0.299, 0.587, 0.114));
      return c * smoothstep(uThreshold, uThreshold + 0.01, v);
    }
    void main(){
      vec2 o = uTexel;
      gl_FragColor = vec4((tap(vUv + vec2(-o.x, -o.y)) + tap(vUv + vec2(o.x, -o.y)) + tap(vUv + vec2(-o.x, o.y)) + tap(vUv + o)) * 0.25, 1.0);
    }`,
});

// UnrealBloomPass.getSeperableBlurMaterial
const blurMat = (kernelRadius: number) => {
  const coefficients: number[] = [];
  for (let i = 0; i < kernelRadius; i++) coefficients.push(0.39894 * Math.exp(-0.5 * i * i / (kernelRadius * kernelRadius)) / kernelRadius);
  return new THREE.ShaderMaterial({
    defines: { KERNEL_RADIUS: kernelRadius },
    uniforms: { colorTexture: { value: null }, invSize: { value: new THREE.Vector2() }, direction: { value: new THREE.Vector2() }, gaussianCoefficients: { value: coefficients } },
    vertexShader: VERT, depthTest: false, depthWrite: false,
    fragmentShader: /* glsl */`varying vec2 vUv; uniform sampler2D colorTexture; uniform vec2 invSize; uniform vec2 direction; uniform float gaussianCoefficients[KERNEL_RADIUS];
      void main() {
        float weightSum = gaussianCoefficients[0];
        vec3 diffuseSum = texture2D(colorTexture, vUv).rgb * weightSum;
        for (int i = 1; i < KERNEL_RADIUS; i ++) {
          float x = float(i); float w = gaussianCoefficients[i];
          vec2 uvOffset = direction * invSize * x;
          diffuseSum += (texture2D(colorTexture, vUv + uvOffset).rgb + texture2D(colorTexture, vUv - uvOffset).rgb) * w;
          weightSum += 2.0 * w;
        }
        gl_FragColor = vec4(diffuseSum / weightSum, 1.0);
      }`,
  });
};

// UnrealBloomPass composite (strength / radius / factors / white tints)
const compositeMat = () => new THREE.ShaderMaterial({
  uniforms: { b1: { value: null }, b2: { value: null }, b3: { value: null }, b4: { value: null }, b5: { value: null }, uF: { value: [0, 0, 0, 0, 0] } },
  vertexShader: VERT, depthTest: false, depthWrite: false,
  fragmentShader: /* glsl */`varying vec2 vUv; uniform sampler2D b1, b2, b3, b4, b5; uniform float uF[5];
    void main(){
      gl_FragColor = vec4(uF[0] * texture2D(b1, vUv).rgb + uF[1] * texture2D(b2, vUv).rgb + uF[2] * texture2D(b3, vUv).rgb
        + uF[3] * texture2D(b4, vUv).rgb + uF[4] * texture2D(b5, vUv).rgb, 1.0);
    }`,
});

const uberMat = () => new THREE.ShaderMaterial({
  uniforms: {
    tScene: { value: null }, tShafts: { value: null }, tBloom: { value: null },
    uShaft: { value: new THREE.Vector3() }, uBloomK: { value: 1 }, uExposure: { value: 1 },
    uSat: { value: 1 }, uContrast: { value: 1 }, uVignette: { value: 0 }, uLift: { value: 0 }, uSharpen: { value: 0 },
    uShadowTint: { value: new THREE.Vector3() }, uHighTint: { value: new THREE.Vector3() },
  },
  vertexShader: VERT, depthTest: false, depthWrite: false, toneMapped: false,
  fragmentShader: /* glsl */`uniform sampler2D tScene, tShafts, tBloom; uniform vec3 uShaft, uShadowTint, uHighTint;
    uniform float uBloomK, uExposure, uSat, uContrast, uVignette, uLift, uSharpen; varying vec2 vUv;
    // three.js ACESFilmicToneMapping (r169)
    vec3 RRTAndODTFit(vec3 v) { vec3 a = v * (v + 0.0245786) - 0.000090537; vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081; return a / b; }
    vec3 aces(vec3 color) {
      const mat3 ACESInputMat = mat3(vec3(0.59719, 0.07600, 0.02840), vec3(0.35458, 0.90834, 0.13383), vec3(0.04823, 0.01566, 0.83777));
      const mat3 ACESOutputMat = mat3(vec3(1.60475, -0.10208, -0.00327), vec3(-0.53108, 1.10813, -0.07276), vec3(-0.07367, -0.00605, 1.07602));
      color *= uExposure / 0.6;
      color = ACESOutputMat * RRTAndODTFit(ACESInputMat * color);
      return clamp(color, 0.0, 1.0);
    }
    vec3 srgb(vec3 c) { return mix(c * 12.92, pow(c, vec3(0.41666)) * 1.055 - vec3(0.055), vec3(lessThanEqual(vec3(0.0031308), c))); }
    vec3 hdr(vec2 uv) { vec3 c = texture2D(tScene, uv).rgb; return any(isnan(c)) ? vec3(0.0) : min(c, vec3(256.0)); }
    void main(){
      vec2 px = 1.0 / vec2(textureSize(tScene, 0));
      // shafts + bloom are low frequency: one sample each, shared by the sharpen neighbourhood
      vec3 add = uShaft * texture2D(tShafts, vUv).r + uBloomK * texture2D(tBloom, vUv).rgb;
      vec3 c = srgb(aces(hdr(vUv) + add));
      vec3 bl = (srgb(aces(hdr(vUv + vec2(px.x, 0.0)) + add)) + srgb(aces(hdr(vUv - vec2(px.x, 0.0)) + add)) +
                 srgb(aces(hdr(vUv + vec2(0.0, px.y)) + add)) + srgb(aces(hdr(vUv - vec2(0.0, px.y)) + add))) * 0.25;
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
});

const KERNELS = [3, 5, 7, 9, 11];
const FACTORS = [1.0, 0.8, 0.6, 0.4, 0.2];

export class FusedPost {
  readonly scene: THREE.WebGLRenderTarget;
  private shaftsRT: THREE.WebGLRenderTarget;
  private bright: THREE.WebGLRenderTarget;
  private h: THREE.WebGLRenderTarget[] = [];
  private v: THREE.WebGLRenderTarget[] = [];
  private bloomRT: THREE.WebGLRenderTarget;
  private quad = new FullScreenQuad();
  private mShafts = shaftsMat(); private mBright = brightMat(); private mComp = compositeMat(); private mUber = uberMat();
  private mBlur = KERNELS.map(blurMat);
  private shaftStrength = 0;
  bloomEnabled = true;
  bloomStrength = 0.22; bloomRadius = 0.55; bloomThreshold = 0.92;
  shaftColor = new THREE.Vector3(1.0, 0.72, 0.45);
  readonly grade: GradeParams = {
    sat: 1.12, contrast: 1.08, vignette: 0.32, lift: 0.012, sharpen: 0.18,
    shadowTint: new THREE.Vector3(-0.012, 0.0, 0.03), highTint: new THREE.Vector3(0.035, 0.012, -0.03),
  };

  constructor(private renderer: THREE.WebGLRenderer) {
    const o = { type: THREE.HalfFloatType, depthBuffer: false };
    this.scene = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
    this.shaftsRT = new THREE.WebGLRenderTarget(1, 1, o);
    this.bright = new THREE.WebGLRenderTarget(1, 1, o);
    this.bloomRT = new THREE.WebGLRenderTarget(1, 1, o);
    for (let i = 0; i < 5; i++) { this.h.push(new THREE.WebGLRenderTarget(1, 1, o)); this.v.push(new THREE.WebGLRenderTarget(1, 1, o)); }
    const u = this.mComp.uniforms;
    [u.b1, u.b2, u.b3, u.b4, u.b5].forEach((x, i) => (x.value = this.v[i].texture));
    this.mUber.uniforms.tScene.value = this.scene.texture;
    this.mUber.uniforms.tShafts.value = this.shaftsRT.texture;
    this.mUber.uniforms.tBloom.value = this.bloomRT.texture;
  }

  /** Size in device pixels. */
  setSize(w: number, h: number) {
    w = Math.max(1, Math.round(w)); h = Math.max(1, Math.round(h));
    this.scene.setSize(w, h);
    const qw = Math.max(1, Math.round(w / 4)), qh = Math.max(1, Math.round(h / 4));
    this.shaftsRT.setSize(qw, qh); this.bright.setSize(qw, qh); this.bloomRT.setSize(qw, qh);
    this.mBright.uniforms.uTexel.value.set(1 / w, 1 / h);
    // level 0 stands in for UnrealBloom's half-res mip 0 (half-texel steps), levels 1-4 = its mips 1-4
    let lw = qw, lh = qh;
    for (let i = 0; i < 5; i++) {
      if (i >= 2) { lw = Math.max(1, Math.round(lw / 2)); lh = Math.max(1, Math.round(lh / 2)); }
      this.h[i].setSize(lw, lh); this.v[i].setSize(lw, lh);
    }
  }

  /** Sun screen position (uv), shaft strength (0 = off) and aspect for this frame. */
  setShafts(sunUv: THREE.Vector2, strength: number, aspect: number) {
    this.mShafts.uniforms.uSun.value.copy(sunUv);
    this.mShafts.uniforms.uAspect.value = aspect;
    this.shaftStrength = strength;
  }

  render(scene: THREE.Scene, camera: THREE.Camera) {
    const r = this.renderer;
    const oldTarget = r.getRenderTarget();
    r.setRenderTarget(this.scene);
    r.clear();
    r.render(scene, camera);
    const sceneTex = this.scene.texture;
    const shaftK = this.mUber.uniforms.uShaft.value as THREE.Vector3;
    if (this.shaftStrength > 0.001) {
      this.mShafts.uniforms.tScene.value = sceneTex;
      this.pass(this.mShafts, this.shaftsRT);
      shaftK.copy(this.shaftColor).multiplyScalar(this.shaftStrength);
    } else shaftK.set(0, 0, 0);
    if (this.bloomEnabled) {
      const b = this.mBright.uniforms;
      b.tScene.value = sceneTex; b.tShafts.value = this.shaftsRT.texture; b.uShaft.value.copy(shaftK); b.uThreshold.value = this.bloomThreshold;
      this.pass(this.mBright, this.bright);
      let input = this.bright;
      for (let i = 0; i < 5; i++) {
        const m = this.mBlur[i], u = m.uniforms, s = i === 0 ? 0.5 : 1;
        u.invSize.value.set(1 / this.h[i].width, 1 / this.h[i].height);
        u.colorTexture.value = input.texture; u.direction.value.set(s, 0);
        this.pass(m, this.h[i]);
        u.colorTexture.value = this.h[i].texture; u.direction.value.set(0, s);
        this.pass(m, this.v[i]);
        input = this.v[i];
      }
      const f = this.mComp.uniforms.uF.value as number[];
      for (let i = 0; i < 5; i++) f[i] = this.bloomStrength * THREE.MathUtils.lerp(FACTORS[i], 1.2 - FACTORS[i], this.bloomRadius);
      this.pass(this.mComp, this.bloomRT);
    }
    const u = this.mUber.uniforms, g = this.grade;
    u.uBloomK.value = this.bloomEnabled ? 1 : 0;
    u.uExposure.value = r.toneMappingExposure;
    u.uSat.value = g.sat; u.uContrast.value = g.contrast; u.uVignette.value = g.vignette; u.uLift.value = g.lift; u.uSharpen.value = g.sharpen;
    u.uShadowTint.value.copy(g.shadowTint); u.uHighTint.value.copy(g.highTint);
    this.pass(this.mUber, oldTarget);
  }

  private pass(m: THREE.Material, target: THREE.WebGLRenderTarget | null) {
    this.renderer.setRenderTarget(target);
    this.quad.material = m;
    this.quad.render(this.renderer);
  }
}
