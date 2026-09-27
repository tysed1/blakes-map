import * as THREE from 'three';
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js';
import { assetUrl, bin } from '../core/data';

/**
 * Texture loading with GPU compression (board item R3).
 *
 * `loadTexture('albedo', 'albedo.jpg')` resolves to the Basis Universal KTX2 build of the texture
 * (tools/pipeline/ktx2_bake.mjs -> public/world/albedo.ktx2), transcoded to the GPU's native
 * block format (ASTC / BC7 / BC1 / ETC2: 4-8x less VRAM than RGBA8, no 6000 px RGBA uploads), and
 * falls back to the source image when there is no .ktx2 or no transcoder. The KTX2 files are
 * Y-flipped at encode, so UVs match the flipY=true image path.
 * KTX2 bytes go through `bin()` (packed as .wasm on the artifact host).
 */

let ktx2: KTX2Loader | null = null;
let available: Promise<Set<string>> | null = null;

export function initTextures(renderer: THREE.WebGLRenderer) {
  if (ktx2) return;
  ktx2 = new KTX2Loader().setTranscoderPath(`${import.meta.env.BASE_URL}basis/`).detectSupport(renderer);
  // world/ktx2.json lists the live KTX2 textures {name: source image} (ktx2_bake.mjs), so missing ones cost no 404s
  available = fetch(assetUrl('ktx2.json')).then((r) => (r.ok ? r.json() : {})).then((l: Record<string, string>) => new Set(Object.keys(l))).catch(() => new Set<string>());
}

export interface TexOpts { srgb?: boolean; anisotropy?: number; wrap?: THREE.Wrapping }

function apply<T extends THREE.Texture>(t: T, o: TexOpts): T {
  t.colorSpace = o.srgb === false ? THREE.NoColorSpace : THREE.SRGBColorSpace;
  if (o.anisotropy) t.anisotropy = o.anisotropy;
  if (o.wrap !== undefined) t.wrapS = t.wrapT = o.wrap;
  t.needsUpdate = true;
  return t;
}

export async function loadTexture(name: string, fallback: string, o: TexOpts = {}): Promise<THREE.Texture> {
  if (ktx2 && available && (await available).has(name)) {
    try {
      const buf = await bin(name + '.ktx2');
      const t = await new Promise<THREE.CompressedTexture>((res, rej) => ktx2!.parse(buf, (x) => res(x as THREE.CompressedTexture), rej));
      t.minFilter = t.mipmaps && t.mipmaps.length > 1 ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter;
      return apply(t, o);
    } catch (e) { console.warn(`ktx2 ${name}: falling back to ${fallback}`, e); }
  }
  const t = await new THREE.TextureLoader().loadAsync(assetUrl(fallback));
  return apply(t, o);
}
