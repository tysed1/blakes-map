"""Web terrain albedo: Blender bake (tools/blender/bake_web_albedo.py) graded for the three.js viewer.

Under canopy the bake shows bare leaf litter; in Cycles the dense crowns hide it, but the web
trees are sparser proxies, so canopy cover is pre-tinted toward the shadowed crown colour.
Exposed rock is lifted toward the pale grey crags of the renders.
usage: python3 tools/pipeline/web_albedo.py exports/web/albedo_bake.png data/terrain/albedo_web.jpg
"""
import sys
import numpy as np
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src).convert('RGB')
W, H = im.size
a = np.asarray(im, np.float32) / 255.0
lin = a ** 2.2
eco = np.fromfile('public/world/eco_u8.bin', np.uint8).reshape(667, 2000, -1)


def up(ch):
    return np.asarray(Image.fromarray(eco[:, :, ch]).resize((W, H), Image.BILINEAR), np.float32) / 255.0


canopy, rock = up(2), up(10)
crown = np.array([0.055, 0.075, 0.03], np.float32)  # lib_trees oak/poplar greens, in shade
rng = np.random.default_rng(3)
n = np.asarray(Image.fromarray((rng.random((H // 8, W // 8)) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), np.float32) / 255
k = np.clip(canopy * 0.75 * (0.8 + 0.4 * n), 0, 0.8)[..., None]
lin = lin * (1 - k) + crown * (0.7 + 0.6 * n[..., None]) * k
r = np.clip(rock * 1.2, 0, 1)[..., None] * (1 - k)
lin = lin * (1 - 0.5 * r) + np.array([0.30, 0.29, 0.26], np.float32) * 0.5 * r * (0.7 + 0.6 * lin.mean(2, keepdims=True) / 0.1).clip(0.5, 1.5)
out = np.clip(lin, 0, 1) ** (1 / 2.2)
Image.fromarray((out * 255).astype(np.uint8)).save(dst, quality=86)
print('wrote', dst, W, H)
