#!/usr/bin/env python3
"""Static-sun shadow + ambient-occlusion bake for the web viewer (board item R1).

The golden-hour sun never moves, so far shadows (terrain self-shadowing, forest canopy, the long
low-sun shadows of every tree) are baked once into a world-space texture that every lit material
samples (src/engine/sunbake.ts). Real-time cascaded shadow maps then only cover the near field.

Encoding (flipY-free: row 0 = py 0):
  public/world/sunbake.jpg     gray (JPEG q90), 2 texels per source px (1.25 m):
      D_all : how far above the ground a point must be to see the sun, occluders = terrain + canopy
  public/world/sunbake_lo.webp RGB lossless, 1 texel per source px (2.5 m):
      R D_ter : same, terrain only (the near field gets canopy shadows from the CSM instead)
      G AO    : ground ambient occlusion (canopy cover + terrain concavity), 255 = open
  D is metres, stored as 255 * sqrt(D / DMAX) (fine steps near the ground where penumbrae live).

For a receiver at world XZ and height h above the ground: lit = 1 - smoothstep(bias, bias + soft, D - h).
Canopy: every tree of the ecosystem scatter (vegetation_f32.bin, species radius/height from
trees/geo.json) is a solid dome; the lit-height field is H(p) = max_s(O(p + s*u) - s*tanE), with O the
occluder height field and u the horizontal direction toward the sun, computed with log2 doubling
steps (each a bilinear shift of the whole grid).

usage: python3 tools/pipeline/sun_shadow_bake.py [--scale 2] [--out public/world/sunbake.jpg]
Rerun whenever the terrain (terrain_u16.bin) or the vegetation scatter changes.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PUB = os.path.join(ROOT, 'public', 'world')
MPP = 2.5
# must match World3D.sunDir (three.js world, Y up): (-1400, 360, -560)
SUN = np.array([-1400.0, 360.0, -560.0])
DMAX = 64.0
# crown opacity footprint: leaf-card crowns are ragged; the dome radius is a bit inside the bound
CROWN_R = 0.82
SNAG_R = 0.35


def load_terrain():
    man = json.load(open(os.path.join(PUB, 'manifest.json')))
    t = man['terrain']
    u = np.fromfile(os.path.join(PUB, t['file']), dtype=np.uint16).reshape(t['h'], t['w'])
    return t['min_m'] + u.astype(np.float32) * ((t['max_m'] - t['min_m']) / 65535.0), t['w'], t['h']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', type=int, default=2, help='bake texels per source px')
    ap.add_argument('--out', default=os.path.join(PUB, 'sunbake.jpg'))
    ap.add_argument('--debug', default=None, help='optional preview png (lit shading)')
    a = ap.parse_args()
    t0 = time.time()
    T0, W0, H0 = load_terrain()
    f = a.scale
    W, H = W0 * f, H0 * f
    texel = MPP / f
    # terrain at bake texel centres (Heightfield.at: sample (i,j) sits at px (i+0.5, j+0.5), bilinear)
    px = (np.arange(W) + 0.5) / f
    py = (np.arange(H) + 0.5) / f
    gy, gx = np.meshgrid(py - 0.5, px - 0.5, indexing='ij')
    T = ndimage.map_coordinates(T0, [gy, gx], order=1, mode='nearest').astype(np.float32)
    del gx, gy
    print(f'terrain {W}x{H} @ {texel} m  ({time.time() - t0:.1f}s)')

    # ---- canopy domes
    geo = json.load(open(os.path.join(PUB, 'trees', 'geo.json')))
    sp_r = np.array([s['radius'] for s in geo['species']], np.float32)
    sp_h = np.array([s['height'] for s in geo['species']], np.float32)
    sp_k = np.array([SNAG_R if s['family'] == 'snag' else CROWN_R for s in geo['species']], np.float32)
    V = np.fromfile(os.path.join(PUB, 'vegetation_f32.bin'), dtype=np.float32).reshape(-1, 6)
    spi = V[:, 4].astype(int)
    R = sp_r[spi] * V[:, 3] * sp_k[spi] / texel            # crown radius, texels
    Ht = sp_h[spi] * V[:, 3]                               # tree height, m
    cx = V[:, 0] * f - 0.5
    cy = V[:, 1] * f - 0.5
    C = np.zeros((H, W), np.float32)                       # canopy height above ground
    cover = np.zeros((H, W), np.float32)                   # crown coverage (for AO)
    rr = np.ceil(R).astype(int)
    for r in np.unique(rr):
        sel = np.nonzero(rr == r)[0]
        oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
        ox = ox.ravel(); oy = oy.ravel()
        ix = np.rint(cx[sel])[:, None].astype(int) + ox[None, :]
        iy = np.rint(cy[sel])[:, None].astype(int) + oy[None, :]
        d = np.hypot(ix - cx[sel][:, None], iy - cy[sel][:, None]) / R[sel][:, None]
        h = Ht[sel][:, None] * np.sqrt(np.clip(1.0 - d * d, 0.0, 1.0))
        ok = (d < 1.0) & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        flat = (iy * W + ix)[ok]
        np.maximum.at(C.ravel(), flat, h[ok])
        np.maximum.at(cover.ravel(), flat, np.clip(1.2 - d[ok], 0, 1) * np.minimum(1.0, Ht[sel][:, None] / 8.0).repeat(ox.size, 1)[ok])
    print(f'canopy {len(V)} trees  ({time.time() - t0:.1f}s)')

    # ---- lit height by log2 doubling along the sun's horizontal direction
    hz = math.hypot(SUN[0], SUN[2])
    ux, uz = SUN[0] / hz, SUN[2] / hz                       # toward the sun (world X, Z == px x, y)
    tanE = SUN[1] / hz
    jj, ii = np.mgrid[0:H, 0:W].astype(np.float32)

    def shift(A, L):
        return ndimage.map_coordinates(A, [jj + L * uz, ii + L * ux], order=1, mode='constant', cval=-1e6).astype(np.float32)

    def lit_height(O):
        # M_1(p) = O(p + u) - tanE ; M_2L(p) = max(M_L(p), M_L(p + L*u) - L*tanE)   (distances in texels)
        M = shift(O, 1) - np.float32(texel * tanE)
        L = 1
        while L * texel < 3000.0:
            M = np.maximum(M, shift(M, L) - np.float32(L * texel * tanE))
            L *= 2
        return M

    Hall = lit_height(np.maximum(T, T + C))
    print(f'lit height (all)  ({time.time() - t0:.1f}s)')
    Hter = lit_height(T)
    print(f'lit height (terrain)  ({time.time() - t0:.1f}s)')
    Dall = np.clip(Hall - T, 0, DMAX)
    Dter = np.clip(Hter - T, 0, DMAX)

    # ---- ambient occlusion on the ground: canopy cover (blurred ~4 m) + terrain concavity (~40 m)
    cov = ndimage.gaussian_filter(np.minimum(cover, 1.0), 4.0 / texel)
    conc = ndimage.gaussian_filter(T, 40.0 / texel) - T
    ao = 1.0 - 0.6 * np.clip(cov, 0, 1) - np.clip(conc * 0.02, 0, 0.18)
    ao = np.clip(ao, 0.25, 1.0)

    enc = lambda D: np.rint(255.0 * np.sqrt(D / DMAX)).astype(np.uint8)
    # full res: D_all (the detailed canopy term); half res (2.5 m): D_ter + AO (both smooth)
    # canopy term as gray JPEG q90 (1.1 MB vs 2.1 MB PNG; D error p99 ~1 m, inside the penumbra width)
    Image.fromarray(enc(Dall), 'L').save(a.out, quality=90, optimize=True)
    lo = os.path.splitext(a.out)[0] + '_lo.webp'
    box = lambda A: A.reshape(H0, f, W0, f).mean(axis=(1, 3))   # texel-centre aligned 2x2 average
    Dt, Ao = box(Dter), box(ao)
    Image.fromarray(np.dstack([enc(np.clip(Dt, 0, DMAX)), np.rint(np.clip(Ao, 0, 1) * 255).astype(np.uint8), np.zeros_like(enc(Dt))]), 'RGB').save(lo, lossless=True, method=6)
    for old in ('sunbake.png', 'sunbake_lo.png'):   # earlier output names
        if os.path.exists(os.path.join(os.path.dirname(a.out), old)):
            os.remove(os.path.join(os.path.dirname(a.out), old))
    print(f'wrote {a.out} {os.path.getsize(a.out) / 1e6:.1f} MB + {lo} {os.path.getsize(lo) / 1e6:.1f} MB  ({time.time() - t0:.1f}s)')
    if a.debug:
        lit = 1.0 - np.clip((Dall - 0.3) / 1.5, 0, 1)
        litT = 1.0 - np.clip((Dter - 0.3) / 1.5, 0, 1)
        Image.fromarray(np.rint(np.dstack([lit, litT, ao]) * 255).astype(np.uint8)).save(a.debug)
    return 0


if __name__ == '__main__':
    sys.exit(main())
