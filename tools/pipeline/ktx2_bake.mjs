// KTX2 (Basis Universal) texture bake for the web viewer (board item R3).
// usage: node tools/pipeline/ktx2_bake.mjs [name ...]      (default: every entry of TEXTURES)
// Writes public/world/<name>.ktx2 next to the source image. The viewer (src/engine/textures.ts)
// loads the .ktx2 when it exists and falls back to the source image otherwise; the artifact
// deploy packs .ktx2 like *.bin (tools/deploy/prepare_artifact.py).
// Images are Y-flipped at encode so compressed textures keep the flipY=true UV convention of the
// PNG/JPEG path, and resized to multiples of 4 (block-compressed formats need whole blocks).
import { encodeToKTX2 } from 'ktx2-encoder';
import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', '..');
const PUB = path.join(ROOT, 'public', 'world');
// name: [source, mode, srgb, live]; etc1s = colour (small, ~0.5 B/px on GPU), uastc = normals / data.
// live = the viewer loads it through src/engine/textures.ts loadTexture(name, source): only live
// entries go into world/ktx2.json (the viewer's and the artifact packer's list; the packer then
// ships the KTX2 instead of the source image). Flip `live` when the owning module switches loaders.
export const TEXTURES = {
  albedo: ['albedo.jpg', 'etc1s', true, true],
  'groundcover/props_albedo': ['groundcover/props_albedo.jpg', 'etc1s', true, false],
  'groundcover/props_normal': ['groundcover/props_normal.jpg', 'uastc', false, false],
  'terrain/detail_grass': ['terrain/detail_grass.webp', 'uastc', true, false],
  'terrain/detail_forest': ['terrain/detail_forest.webp', 'uastc', true, false],
  'terrain/detail_rock': ['terrain/detail_rock.webp', 'uastc', true, false],
  'terrain/detail_soil': ['terrain/detail_soil.webp', 'uastc', true, false],
};

function decode(src) {
  const py = `import sys
from PIL import Image
im = Image.open(sys.argv[1]).convert('RGBA')
w, h = im.size
W, H = (w + 3) // 4 * 4 if w % 4 else w, h - h % 4 if h % 4 and h % 4 < 2 else ((h + 3) // 4 * 4 if h % 4 else h)
if (W, H) != (w, h):
    im = im.resize((W, H), Image.LANCZOS)
sys.stdout.buffer.write(W.to_bytes(4, 'little') + H.to_bytes(4, 'little') + im.tobytes())`;
  const raw = execFileSync('python3', ['-c', py, src], { maxBuffer: 1 << 30 });
  return { width: raw.readUInt32LE(0), height: raw.readUInt32LE(4), data: new Uint8Array(raw.buffer, raw.byteOffset + 8, raw.length - 8) };
}

const names = process.argv.slice(2).length ? process.argv.slice(2) : Object.keys(TEXTURES);
for (const n of names) {
  const [src, mode, srgb] = TEXTURES[n];
  const t0 = Date.now();
  const img = decode(path.join(PUB, src));
  const common = { generateMipmap: true, isYFlip: true, isPerceptual: srgb, isSetKTX2SRGBTransferFunc: srgb, imageDecoder: async () => img };
  const opts = mode === 'uastc'
    ? { ...common, isUASTC: true, needSupercompression: true, enableRDO: true, rdoQualityLevel: 1 }
    : { ...common, isUASTC: false, qualityLevel: 200, compressionLevel: 2 };
  const out = await encodeToKTX2(new Uint8Array(1), opts);
  const dst = path.join(PUB, n + '.ktx2');
  fs.writeFileSync(dst, out);
  console.log(`${n}.ktx2 ${img.width}x${img.height} ${mode} ${(out.length / 1e6).toFixed(2)} MB (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}
// index of live .ktx2 textures (name -> source image)
const list = Object.fromEntries(Object.entries(TEXTURES).filter(([n, t]) => t[3] && fs.existsSync(path.join(PUB, n + '.ktx2'))).map(([n, t]) => [n, t[0]]));
fs.writeFileSync(path.join(PUB, 'ktx2.json'), JSON.stringify(list, null, 1));
