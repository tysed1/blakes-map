import * as THREE from 'three';
import { SUNBAKE_PARS, sunBake } from './sunbake';

/**
 * Far-tree impostors (board item R4): octahedral (upper hemisphere) impostors baked at load time from
 * the trees' own LOD mesh + material, so the far band keeps exactly the look it replaces.
 *
 * Bake: for every species the LOD mesh is rendered (plain Mesh: no instance tint, no wind, no fade)
 * from G x G hemi-octahedral directions into one tile of a shared atlas, twice: sun only (direct) and
 * sky/environment only (indirect), both scaled by 1/K into sRGB8 targets (hardware encode keeps the
 * darks). The sun never moves, so the direct pass already holds the leaf translucency, crown AO,
 * specular and the core material's leaf-cluster breakup for that view.
 * Runtime: one camera-facing quad per tree (2 triangles instead of the LOD's cores), the 4 nearest
 * views blended, lit = tint x K x (direct x baked sun visibility x clouds + indirect x ground AO),
 * then the shared aerial perspective; the LOD cross-fade dither is the tree LOD's own.
 */

const G = 8;         // views per side (hemi-octahedral)
const T = 64;        // px per view
const K = 4;         // radiance scale stored in the atlases
const COLS = 6;

export interface ImpostorSet { atlasD: THREE.Texture; atlasI: THREE.Texture; tiles: Map<THREE.BufferGeometry, THREE.Vector4>; dispose(): void }

interface BakeItem { geometry: THREE.BufferGeometry; material: THREE.Material }

/** Hemi-octahedral direction for grid cell centre (i, j). */
function cellDir(i: number, j: number, out: THREE.Vector3) {
  const u = ((i + 0.5) / G) * 2 - 1, v = ((j + 0.5) / G) * 2 - 1;
  const x = (u + v) * 0.5, z = (u - v) * 0.5, y = 1 - Math.abs(x) - Math.abs(z);
  return out.set(x, Math.max(y, 0.02), z).normalize();
}

/** Bounds used for a species' tile: ortho half-size and crown centre (object space). */
function bounds(g: THREE.BufferGeometry) {
  g.computeBoundingBox();
  const b = g.boundingBox!, c = b.getCenter(new THREE.Vector3()), s = b.getSize(new THREE.Vector3());
  const r = Math.max(s.x, s.z, s.y) * 0.5 * 1.04;
  return { c, r };
}

export function bakeImpostors(renderer: THREE.WebGLRenderer, items: BakeItem[], env: { sunDir: THREE.Vector3; sunColor: THREE.Color; sunIntensity: number; hemi: THREE.HemisphereLight | null; environment: THREE.Texture | null; envIntensity: number; envRotation: THREE.Euler }): ImpostorSet {
  const rows = Math.ceil(items.length / COLS);
  const W = COLS * G * T, H = rows * G * T;
  const mk = () => {
    const rt = new THREE.WebGLRenderTarget(W, H, { depthBuffer: true });
    rt.texture.colorSpace = THREE.SRGBColorSpace;
    // nearest: each view is combined by coverage in the shader (bilinear would bleed the empty
    // background into silhouettes; mips would bleed across views)
    rt.texture.generateMipmaps = false; rt.texture.minFilter = THREE.NearestFilter; rt.texture.magFilter = THREE.NearestFilter;
    return rt;
  };
  const rtD = mk(), rtI = mk();
  const scene = new THREE.Scene();
  const sun = new THREE.DirectionalLight(env.sunColor, env.sunIntensity / K);
  sun.position.copy(env.sunDir).multiplyScalar(100); sun.target.position.set(0, 0, 0);
  scene.add(sun, sun.target);
  const hemi = env.hemi ? new THREE.HemisphereLight(env.hemi.color, env.hemi.groundColor, env.hemi.intensity / K) : null;
  const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 1000);
  const mesh = new THREE.Mesh();
  scene.add(mesh);
  const tiles = new Map<THREE.BufferGeometry, THREE.Vector4>();
  const prev = { target: renderer.getRenderTarget(), shadow: renderer.shadowMap.enabled, autoClear: renderer.autoClear, bake: sunBake.uniforms.uSunBakeK.value.w, clear: renderer.getClearColor(new THREE.Color()), clearA: renderer.getClearAlpha() };
  // lit materials must take the non-shadow light loop (the CSM chunk only lights shadowed cascades
  // when shadow maps are on) and must not sample the world-space sun bake at the bake origin
  renderer.shadowMap.enabled = false;
  sunBake.uniforms.uSunBakeK.value.w = 0;
  renderer.autoClear = false;
  renderer.setClearColor(0x000000, 0);
  const d = new THREE.Vector3();
  for (const pass of [0, 1] as const) {
    const rt = pass === 0 ? rtD : rtI;
    sun.visible = pass === 0;
    if (hemi) { if (pass === 1) scene.add(hemi); else scene.remove(hemi); }
    scene.environment = pass === 1 ? env.environment : null;
    scene.environmentIntensity = env.envIntensity / K;
    (scene as any).environmentRotation = env.envRotation;
    renderer.setRenderTarget(rt);
    renderer.clear(true, true, true);
    items.forEach((it, k) => {
      mesh.geometry = it.geometry; mesh.material = it.material;
      const { c, r } = bounds(it.geometry);
      const tx = (k % COLS) * G * T, ty = Math.floor(k / COLS) * G * T;
      if (pass === 0) tiles.set(it.geometry, new THREE.Vector4(tx / W, ty / H, c.y, r));
      cam.left = -r; cam.right = r; cam.top = r; cam.bottom = -r; cam.near = 0.1; cam.far = r * 4; cam.updateProjectionMatrix();
      for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
        cellDir(i, j, d);
        cam.position.copy(c).addScaledVector(d, r * 2);
        cam.up.set(0, 1, 0);
        if (d.y > 0.999) cam.up.set(0, 0, -1);
        cam.lookAt(c);
        cam.updateMatrixWorld();
        // row 0 of a render target is its bottom: tile (i, j) sits at x = i, y = j
        rt.viewport.set(tx + i * T, ty + j * T, T, T); rt.scissor.set(tx + i * T, ty + j * T, T, T); rt.scissorTest = true;
        renderer.setRenderTarget(rt);
        renderer.clearDepth();
        renderer.render(scene, cam);
      }
    });
    rt.scissorTest = false; rt.viewport.set(0, 0, W, H); rt.scissor.set(0, 0, W, H);
  }
  renderer.setRenderTarget(prev.target);
  renderer.shadowMap.enabled = prev.shadow;
  renderer.autoClear = prev.autoClear;
  renderer.setClearColor(prev.clear, prev.clearA);
  sunBake.uniforms.uSunBakeK.value.w = prev.bake;
  mesh.geometry = new THREE.BufferGeometry();
  return { atlasD: rtD.texture, atlasI: rtI.texture, tiles, dispose() { rtD.dispose(); rtI.dispose(); } };
}

/** Uniforms of the tree LOD being replaced (cross-fade band) + per-species tile. */
export interface ImpostorFade { uFade: { value: THREE.Vector4 }; uFar: { value: THREE.Vector2 }; uCamPos: { value: THREE.Vector3 } }

export function impostorMaterial(set: ImpostorSet, tile: THREE.Vector4, fade: ImpostorFade) {
  const m = new THREE.ShaderMaterial({
    fog: true,
    uniforms: THREE.UniformsUtils.merge([THREE.UniformsLib.fog]),
    defines: { G_VIEWS: G.toFixed(1), K_SCALE: K.toFixed(1) },
    vertexShader: /* glsl */`
      #include <common>
      #include <fog_pars_vertex>
      uniform vec4 uTile;          // atlas origin (uv), crown centre height, half-size (object space)
      uniform vec2 uTileSize;      // one view in atlas uv
      uniform vec4 uFade; uniform vec2 uFar; uniform vec3 uCamPos;
      varying vec2 vQ; varying vec2 vOct; varying vec3 vTint; varying float vFade; varying vec3 vW; varying float vH;
      float treeFade(vec3 ip) {
        float d = distance(ip, uCamPos);
        float fin = clamp((d - uFade.x) / (uFade.y - uFade.x), 0.0, 1.0);
        if (fin < 1.0) return -fin;
        return (1.0 - clamp((d - uFade.z) / (uFade.w - uFade.z), 0.0, 1.0)) * (1.0 - clamp((d - uFar.x) / (uFar.y - uFar.x), 0.0, 1.0));
      }
      void main() {
        vec3 ip = (modelMatrix * vec4(instanceMatrix[3].xyz, 1.0)).xyz;
        mat3 R = mat3(instanceMatrix);
        float sc = length(R[0]);
        R = mat3(R[0] / sc, R[1] / length(R[1]), R[2] / length(R[2]));
        vec3 cc = ip + R * vec3(0.0, uTile.z, 0.0) * sc;
        vec3 toCam = normalize(cameraPosition - cc);
        // view direction in the tree's own frame -> hemi-octahedral coordinates
        vec3 dl = transpose(R) * toCam;
        dl.y = max(dl.y, 0.02); dl = normalize(dl);
        dl /= abs(dl.x) + abs(dl.y) + abs(dl.z);
        vOct = vec2(dl.x + dl.z, dl.x - dl.z);
        // camera-facing quad; its 'up' follows the bake cameras (world up projected)
        vec3 fw = -toCam;
        vec3 rt = normalize(cross(fw, abs(fw.y) > 0.999 ? vec3(0.0, 0.0, -1.0) : vec3(0.0, 1.0, 0.0)));
        vec3 up = cross(rt, fw);
        vec3 wp = cc + (rt * position.x + up * position.y) * uTile.w * sc;
        vQ = position.xy * 0.5 + 0.5;
        #ifdef USE_INSTANCING_COLOR
          vTint = instanceColor;
        #else
          vTint = vec3(1.0);
        #endif
        vW = wp; vH = max(0.0, wp.y - ip.y);
        vFade = treeFade(ip);
        vec4 mvPosition = viewMatrix * vec4(wp, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        if (abs(vFade) < 0.004) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        #include <fog_vertex>
      }`,
    fragmentShader: /* glsl */`
      #include <common>
      #include <fog_pars_fragment>
      uniform sampler2D tD, tI;
      uniform vec4 uTile; uniform vec2 uTileSize;
      varying vec2 vQ; varying vec2 vOct; varying vec3 vTint; varying float vFade; varying vec3 vW; varying float vH;
      ${SUNBAKE_PARS}
      vec4 view(sampler2D t, vec2 cell) {
        return texture2D(t, uTile.xy + (cell + clamp(vQ, 0.01, 0.99)) * uTileSize);
      }
      void main() {
        float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
        if ((vFade < 0.0 ? 1.0 - ign : ign) >= abs(vFade)) discard;
        // 4 nearest baked views, bilinear in the view grid, each weighted by its own coverage
        vec2 g = clamp((vOct * 0.5 + 0.5) * G_VIEWS - 0.5, 0.0, G_VIEWS - 1.0);
        vec2 g0 = floor(g), f = g - g0, g1 = min(g0 + 1.0, G_VIEWS - 1.0);
        vec2 c[4]; c[0] = g0; c[1] = vec2(g1.x, g0.y); c[2] = vec2(g0.x, g1.y); c[3] = g1;
        float w[4]; w[0] = (1.0 - f.x) * (1.0 - f.y); w[1] = f.x * (1.0 - f.y); w[2] = (1.0 - f.x) * f.y; w[3] = f.x * f.y;
        vec3 sd = vec3(0.0), si = vec3(0.0); float cov = 0.0;
        for (int k = 0; k < 4; k++) {
          vec4 dk = view(tD, c[k]);
          float ck = w[k] * step(0.02, dk.a);
          sd += dk.rgb * ck; si += view(tI, c[k]).rgb * ck; cov += ck;
        }
        if (cov < 0.5) discard;
        sd /= cov; si /= cov;
        float lit, ao;
        sunBakeEval(vW, vH, -(viewMatrix * vec4(vW, 1.0)).z, lit, ao);
        vec3 col = vTint * K_SCALE * (sd * lit + si * ao);
        gl_FragColor = vec4(col, 1.0);
        #include <fog_fragment>
      }`,
  });
  Object.assign(m.uniforms, sunBake.uniforms, fade, {
    tD: { value: set.atlasD }, tI: { value: set.atlasI }, uTile: { value: tile },
    uTileSize: { value: new THREE.Vector2(T / set.atlasD.image.width, T / set.atlasD.image.height) },
  });
  return m;
}

/** Camera-facing unit quad (x, y in -1..1). */
export function impostorQuad() {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]), 3));
  g.setIndex([0, 1, 2, 0, 2, 3]);
  g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);
  return g;
}
