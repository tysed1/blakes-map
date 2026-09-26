"""Textures for the web ground cover / props (public/world/groundcover/), after
tools/blender/export_web_groundcover.py has written geo.json (atlas layout).

    python3 tools/pipeline/web_groundcover_textures.py

  props_albedo.jpg  2048 atlas of the Poly Haven prop diffuse maps; rocks re-tinted toward grey-brown
                    Appalachian sandstone / gneiss exactly like lib_props._retint_rock (sat 0.55, x (0.95, 0.9, 0.82))
  props_normal.jpg  matching OpenGL normal-map atlas (from the *_nor_gl EXRs)
  fern.png          fern_02 diffuse + alpha (512, colour bled into the cut-out)
"""
import json, os
import cv2
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'public/world/groundcover')
PH = os.path.join(ROOT, 'assets/external/polyhaven')
A = 2048


def srgb2lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def retint(rgb, tint=(0.95, 0.9, 0.82), sat=0.55):
    """Blender Hue/Saturation (sat) then multiply, in linear space."""
    lin = srgb2lin(rgb)
    hsv = cv2.cvtColor(lin.astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 1] *= sat
    lin = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * np.array(tint, np.float32)
    return lin2srgb(lin)


def tile_img(path, T, pad, normal=False):
    im = np.asarray(Image.open(path).convert('RGB'), np.float32) / 255
    inner = T - 2 * pad
    im = cv2.resize(im, (inner, inner), interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_REPLICATE)


def main():
    meta = json.load(open(os.path.join(OUT, 'geo.json')))
    at = meta['atlas']
    g = at['grid']; T = A // g; pad = int(round(at['pad'] * T))
    alb = np.zeros((A, A, 3), np.float32); alb[:] = (0.35, 0.33, 0.3)
    nrm = np.zeros((A, A, 3), np.float32); nrm[:] = (0.5, 0.5, 1.0)
    for aid, (col, row) in at['tiles'].items():
        d = os.path.join(PH, aid, 'textures')
        a = tile_img(os.path.join(d, f'{aid}_diff_1k.jpg'), T, pad)
        if aid in at['rock']:
            a = retint(a)
        alb[row * T:(row + 1) * T, col * T:(col + 1) * T] = a
        nf = os.path.join(ROOT, 'tools/.cache/web_groundcover', f'{aid}_nor_gl.png')  # converted by the Blender export
        if os.path.exists(nf):
            nrm[row * T:(row + 1) * T, col * T:(col + 1) * T] = tile_img(nf, T, pad, True)
    Image.fromarray((np.clip(alb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(OUT, 'props_albedo.jpg'), quality=86)
    Image.fromarray((np.clip(nrm, 0, 1) * 255 + 0.5).astype(np.uint8)).resize((A // 2, A // 2), Image.LANCZOS).save(os.path.join(OUT, 'props_normal.jpg'), quality=86)
    # fern: diffuse + alpha, colour bled into transparent texels so mips don't fringe
    d = os.path.join(PH, 'fern_02', 'textures')
    rgb = Image.open(os.path.join(d, 'fern_02_diff_1k.jpg')).convert('RGB').resize((512, 512), Image.LANCZOS)
    a = Image.open(os.path.join(d, 'fern_02_alpha_1k.png')).convert('L').resize((512, 512), Image.LANCZOS)
    blur = rgb.resize((32, 32), Image.BILINEAR).resize((512, 512), Image.BILINEAR)
    rgb = Image.composite(rgb, blur, a.point(lambda v: 255 if v > 8 else 0))
    rgb.putalpha(a)
    rgb.save(os.path.join(OUT, 'fern.png'), optimize=True)
    for f in ('props_albedo.jpg', 'props_normal.jpg', 'fern.png'):
        print(f, round(os.path.getsize(os.path.join(OUT, f)) / 1e6, 2), 'MB')


if __name__ == '__main__':
    main()
