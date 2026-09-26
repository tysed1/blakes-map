import * as THREE from 'three';
import { assetUrl } from '../../core/data';

/**
 * Terrain surface: the Blender-baked albedo (A3's ecology-driven terrain material, 0.83 m/texel)
 * carries the large-scale colour; up close, tiling Poly Haven detail maps (grass / leaf litter /
 * lichen rock / plowed soil, tools/pipeline/web_terrain_detail.py) modulate it and add bump
 * lighting, weighted by the same ecology rasters (rock exposure + slope, canopy, field type).
 * Detail fades out by ~260 m so distant terrain is exactly the bake (no tiling patterns).
 */
const MEANS: Record<string, [number, number, number]> = {
  grass: [0.326, 0.2405, 0.1053], forest: [0.2001, 0.1186, 0.0442], rock: [0.0689, 0.0526, 0.0371], soil: [0.0972, 0.0483, 0.0174],
};

export function terrainMaterial(albedo: THREE.Texture, anisotropy: number) {
  const L = new THREE.TextureLoader();
  const t = (f: string, srgb = true) => {
    const x = L.load(assetUrl('terrain/' + f));
    x.wrapS = x.wrapT = THREE.RepeatWrapping; x.anisotropy = anisotropy;
    x.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    return x;
  };
  const mask = t('mask.png', false); mask.wrapS = mask.wrapT = THREE.ClampToEdgeWrapping;
  // detail maps: RGB albedo (sRGB) + A height. Alpha is decoded linearly by three either way.
  const tex = { grass: t('detail_grass.webp'), forest: t('detail_forest.webp'), rock: t('detail_rock.webp'), soil: t('detail_soil.webp') };
  const m = new THREE.MeshStandardMaterial({ map: albedo, roughness: 0.95, metalness: 0 });
  const v3 = (a: number[]) => new THREE.Vector3(a[0], a[1], a[2]);
  m.onBeforeCompile = (s) => {
    Object.assign(s.uniforms, {
      tMask: { value: mask }, tGrass: { value: tex.grass }, tForest: { value: tex.forest }, tRock: { value: tex.rock }, tSoil: { value: tex.soil },
      mGrass: { value: v3(MEANS.grass) }, mForest: { value: v3(MEANS.forest) }, mRock: { value: v3(MEANS.rock) }, mSoil: { value: v3(MEANS.soil) },
      uDetail: { value: 1.0 },
    });
    s.vertexShader = s.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vTW;')
      .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvTW = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    s.fragmentShader = s.fragmentShader
      .replace('#include <common>', `#include <common>
        varying vec3 vTW;
        uniform sampler2D tMask, tGrass, tForest, tRock, tSoil;
        uniform vec3 mGrass, mForest, mRock, mSoil;
        uniform float uDetail;
        float tH; float tFade; float tRockW;
        vec4 triRock(vec3 p, vec3 n) {
          vec3 w = pow(abs(n), vec3(4.0)); w /= dot(w, vec3(1.0));
          return texture2D(tRock, p.zy) * w.x + texture2D(tRock, p.xz) * w.y + texture2D(tRock, p.xy) * w.z;
        }
        vec3 perturbT(vec3 surf_pos, vec3 surf_norm, vec2 dHdxy) {
          vec3 vSigmaX = dFdx(surf_pos); vec3 vSigmaY = dFdy(surf_pos);
          vec3 R1 = cross(vSigmaY, surf_norm); vec3 R2 = cross(surf_norm, vSigmaX);
          float fDet = dot(vSigmaX, R1);
          vec3 vGrad = sign(fDet) * (dHdxy.x * R1 + dHdxy.y * R2);
          return normalize(abs(fDet) * surf_norm - vGrad);
        }`)
      .replace('#include <map_fragment>', `#include <map_fragment>
        {
          vec4 msk = texture2D(tMask, vMapUv);
          vec3 wn = normalize(transpose(mat3(viewMatrix)) * normalize(vNormal));
          float slope = 1.0 - wn.y;
          float rockW = max(smoothstep(0.22, 0.6, msk.r), smoothstep(0.3, 0.55, slope));
          float forestW = (1.0 - rockW) * smoothstep(0.25, 0.75, msk.g);
          float ft = msk.b * 255.0 / 40.0;
          float soilW = (1.0 - rockW) * (1.0 - forestW) * (1.0 - smoothstep(0.2, 0.6, abs(ft - 3.0)));
          float grassW = max(0.0, 1.0 - rockW - forestW - soilW);
          float dist = length(vTW - cameraPosition);
          tFade = (1.0 - smoothstep(70.0, 280.0, dist)) * uDetail;
          vec2 uv = vTW.xz / 3.2;
          vec4 dg = texture2D(tGrass, uv), df = texture2D(tForest, uv * 1.15), ds = texture2D(tSoil, uv * 0.9);
          vec4 dr = triRock(vTW / 5.5, wn);
          vec3 mod3 = dg.rgb / mGrass * grassW + df.rgb / mForest * forestW + ds.rgb / mSoil * soilW + dr.rgb / mRock * rockW;
          // macro breakup (kills tiling on the mid-distance) from a far-scaled grass sample
          float macro = dot(texture2D(tGrass, vTW.xz / 41.0).rgb / mGrass, vec3(0.333));
          mod3 *= mix(1.0, clamp(macro, 0.7, 1.3), 0.5);
          diffuseColor.rgb *= mix(vec3(1.0), clamp(mod3, 0.0, 2.2), 0.82 * tFade);
          // rock outcrops: lichen grey-green texture colour shows through the bake
          diffuseColor.rgb = mix(diffuseColor.rgb, dr.rgb * vec3(1.15, 1.2, 1.1), rockW * 0.35 * tFade);
          tH = dg.a * grassW + df.a * forestW + ds.a * soilW + dr.a * rockW;
          tRockW = rockW;
        }`)
      .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
        roughnessFactor = mix(roughnessFactor, 0.82, tRockW);`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
        {
          vec2 dH = vec2(dFdx(tH), dFdy(tH)) * (1.4 + tRockW * 2.0) * tFade;
          normal = perturbT(-vViewPosition, normal, dH);
        }`);
  };
  m.customProgramCacheKey = () => 'terrain-detail';
  return m;
}
