"""Down-sized texture cache for Blender renders (memory: 4 agents share ~14 GB).

Poly Haven 2k maps load as 16 MB (8-bit) or 67 MB (16-bit displacement -> float) images, and Cycles
keeps a second copy. For world-scale terrain and props 1k is plenty. This writes
assets/external/polyhaven/_cache1k/<file> for every diffuse / displacement / roughness map:
diffuse -> 1024 px JPEG, displacement / roughness -> 1024 px 8-bit PNG (greyscale).
tools/blender/lib_materials._img() picks the cached file automatically when present.
    python3 tools/assets/make_tex_cache.py [--size 1024]
"""
import os, sys, glob
import numpy as np
from PIL import Image
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PH = os.path.join(ROOT, 'assets/external/polyhaven')
OUT = os.path.join(PH, '_cache1k')


def main(size=1024):
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(PH, '*', 'textures', '*'))):
        b = os.path.basename(f)
        if not any(k in b for k in ('_diff_', '_diffuse_', '_disp_', '_rough_')) or b.endswith('.exr'):
            continue
        dst = os.path.join(OUT, os.path.splitext(b)[0] + ('.jpg' if ('_diff' in b) else '.png'))
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(f):
            continue
        im = Image.open(f)
        if '_diff' in b:
            im = im.convert('RGB')
            if max(im.size) > size:
                im = im.resize((size, size), Image.LANCZOS)
            im.save(dst, quality=92)
        else:
            a = np.asarray(im).astype(np.float32)
            if a.ndim == 3:
                a = a[..., 0]
            a = a / (65535.0 if a.max() > 255 else 255.0)
            g = Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8), 'L')
            if max(g.size) > size:
                g = g.resize((size, size), Image.LANCZOS)
            g.save(dst)
        n += 1
    print('texture cache:', n, 'new files in', OUT)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(int(a[a.index('--size') + 1]) if '--size' in a else 1024)
