"""Backdrop terrain surrounding the playable map (art: graphics ref.png - layered,
hazy blue Appalachian ridges to the horizon).

Near the map edge the real terrain is mirrored (valleys, rivers and ridges carry on
naturally), then it blends into procedural SW-NE trending ridges (typical of the
Blue Ridge / Valley-and-Ridge) that rise with distance to form a horizon.
Output: public-ready data/terrain/backdrop_u16.bin (+ backdrop.json), 10 m cells,
covering the map with PAD metres on every side. Water mask for mirrored rivers.
"""
import sys, os, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import cv2
from PIL import Image
from tools.lib.common import path, load_json, save_json, W, H

CELL = 4       # source px per backdrop cell (10 m)
PAD_PX = 2800  # 7 km on each side


def ridged(shape, scale, seed, angle_deg=35, stretch=2.6):
    rng = np.random.default_rng(seed)
    h, w = shape
    big = int(max(h, w) * 1.5)
    n = rng.standard_normal((int(big / scale) + 4, int(big / scale / stretch) + 4)).astype(np.float32)
    n = cv2.resize(n, (int(big / stretch) + 1, big + 1), interpolation=cv2.INTER_CUBIC)
    n = cv2.resize(n, (big + 1, big + 1), interpolation=cv2.INTER_LINEAR)
    M = cv2.getRotationMatrix2D((big / 2, big / 2), angle_deg, 1.0)
    n = cv2.warpAffine(n, M, (big, big), borderMode=cv2.BORDER_REFLECT)
    y0, x0 = (big - h) // 2, (big - w) // 2
    n = n[y0:y0 + h, x0:x0 + w]
    n = n / (np.abs(n).max() + 1e-6)
    return 1 - np.abs(n)


def main():
    T = np.fromfile(path('data/terrain/height_graded_f32.bin'), np.float32).reshape(H, W)
    lu = np.array(Image.open(path('data/landuse/landuse_classes.png')))
    small = cv2.resize(T, (W // CELL, H // CELL), interpolation=cv2.INTER_AREA)
    wet = cv2.resize((lu == 1).astype(np.float32), (W // CELL, H // CELL), interpolation=cv2.INTER_AREA)
    pad = PAD_PX // CELL
    mir = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    mwet = cv2.copyMakeBorder(wet, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    hh, ww = mir.shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    # distance (cells) outside the map rectangle
    dx = np.maximum(0, np.maximum(pad - xx, xx - (pad + W // CELL - 1)))
    dy = np.maximum(0, np.maximum(pad - yy, yy - (pad + H // CELL - 1)))
    d = np.hypot(dx, dy)
    base = float(np.median(small))
    proc = (0.55 * ridged((hh, ww), 60, 1) + 0.3 * ridged((hh, ww), 25, 2, 40) + 0.15 * ridged((hh, ww), 10, 3, 30))
    proc = cv2.GaussianBlur(proc, (0, 0), 1.2)
    rise = np.clip(d / (pad * 0.9), 0, 1)
    amp = 120 + 520 * rise ** 1.3                      # ridges grow toward the horizon
    procH = base - 30 + amp * (proc - 0.35) + 260 * rise ** 1.6
    blend = np.clip((d - 20) / 130, 0, 1) ** 1.5       # mirror for ~200 m, blend over ~1.3 km
    blend = cv2.GaussianBlur(blend, (0, 0), 8)
    out = mir * (1 - blend) + procH * blend
    water = (mwet > 0.5) & (blend < 0.35)
    out = np.where(water, out - 2.0, out)
    out[pad:pad + H // CELL, pad:pad + W // CELL] = small
    lo, hi = float(np.floor(out.min())), float(np.ceil(out.max()))
    ((out - lo) / (hi - lo) * 65535).round().astype('<u2').tofile(path('data/terrain/backdrop_u16.bin'))
    water.astype(np.uint8).tofile(path('data/terrain/backdrop_water_u8.bin'))
    meta = {'cell_px': CELL, 'cell_m': CELL * 2.5, 'w': ww, 'h': hh, 'origin_px': [-PAD_PX, -PAD_PX], 'min_m': lo, 'max_m': hi,
            'map_rect_cells': [pad, pad, pad + W // CELL, pad + H // CELL],
            'note': 'cell (i,j) centre at source px (origin + (i+0.5)*cell). Inside map_rect the real terrain is used by the viewer instead.'}
    save_json(path('data/terrain/backdrop.json'), meta, indent=1)
    Image.fromarray(((out - lo) / (hi - lo) * 255).astype(np.uint8)).save(path('assets/maps/debug/backdrop.png'))
    print('backdrop', out.shape, lo, hi)


if __name__ == '__main__':
    main()
