"""Stylized terrain albedo (2x source resolution = 1.25 m/texel) shared by the web
3D view and Blender. Art direction: graphics ref.png - warm late-afternoon Appalachia,
rich blended greens with autumn accents, golden fields, grey-ochre rock on steep
ground, soft baked AO. Roads/water are separate meshes (not painted), but road
shoulders and yards get a subtle worn-ground tint.
Outputs data/terrain/albedo_2x.jpg (+ public copy via export_web)."""
import sys, os, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import cv2
from PIL import Image
from tools.lib.common import path, load_json, W, H

S = 2  # upscale
C = {  # linear-ish sRGB palette (0..255)
    'forest_hardwood': [(62, 84, 38), (84, 94, 40), (132, 96, 40)],  # green, olive, autumn ochre
    'forest_conifer': [(36, 58, 36), (46, 68, 42), (32, 50, 34)],
    'farmland': [(196, 176, 96), (174, 168, 84), (150, 158, 78)],
    'meadow': [(140, 150, 78), (160, 156, 86), (122, 136, 70)],
    'rock': [(132, 126, 112), (150, 140, 118), (112, 108, 98)],
    'residential': [(118, 128, 70), (132, 136, 78), (104, 116, 64)],   # lawns + trees
    'town': [(126, 130, 76), (140, 136, 82), (112, 118, 68)],
    'commercial': [(124, 124, 96), (136, 130, 104), (116, 118, 88)],  # lawns + lots
    'industrial': [(124, 120, 98), (134, 128, 104), (112, 110, 90)],
    'rail_yard': [(96, 86, 76), (108, 96, 84), (86, 78, 70)],
    'water': [(70, 76, 60), (70, 76, 60), (70, 76, 60)],  # river bed / banks (water mesh on top)
    'none': [(100, 110, 60)] * 3,
}


def noise(shape, scale, seed):
    rng = np.random.default_rng(seed)
    h, w = shape
    n = rng.standard_normal((int(h / scale) + 3, int(w / scale) + 3)).astype(np.float32)
    return cv2.resize(n, (w + int(2 * scale), h + int(2 * scale)), interpolation=cv2.INTER_CUBIC)[int(scale):int(scale) + h, int(scale):int(scale) + w]


def main():
    lu = load_json(path('data/landuse/landuse.json'))['classes']
    cls = np.array(Image.open(path('data/landuse/landuse_classes.png')))
    T = np.fromfile(path('data/terrain/height_graded_f32.bin'), np.float32).reshape(H, W)
    HS, WS = H * S, W * S
    Tb = cv2.resize(T, (WS, HS), interpolation=cv2.INTER_CUBIC)
    gy, gx = np.gradient(cv2.GaussianBlur(Tb, (0, 0), 1.2), 2.5 / S)
    slope = np.hypot(gx, gy)
    # soft blend between classes: one-hot -> blurred weights
    col = np.zeros((HS, WS, 3), np.float32)
    wsum = np.zeros((HS, WS), np.float32)
    n1 = noise((HS, WS), 30, 1); n2 = noise((HS, WS), 8, 2); n3 = cv2.GaussianBlur(noise((HS, WS), 4, 3), (0, 0), 1.0)
    for c in lu:
        m = (cls == c['id']).astype(np.float32)
        if m.sum() == 0:
            continue
        m = cv2.GaussianBlur(cv2.resize(m, (WS, HS), interpolation=cv2.INTER_NEAREST), (0, 0), 2.2)
        pal = np.array(C.get(c['name'], C['none']), np.float32)
        t1 = np.clip(0.5 + 0.45 * n1, 0, 1)[..., None]
        t2 = np.clip(0.5 + 0.5 * n2, 0, 1)[..., None]
        base = pal[0] * (1 - t1) + pal[1] * t1
        if c['name'] == 'forest_hardwood':
            aut = np.clip((cv2.GaussianBlur(n2, (0, 0), 2) - 0.7) * 1.2, 0, 0.7)[..., None]  # scattered autumn patches
            base = base * (1 - aut) + pal[2] * aut
        else:
            base = base * (1 - 0.35 * t2) + pal[2] * 0.35 * t2
        col += base * m[..., None]
        wsum += m
    col /= np.maximum(wsum, 1e-6)[..., None]
    # steep ground: rock / bare earth regardless of class (except water)
    wet = cv2.resize((cls == 1).astype(np.float32), (WS, HS), interpolation=cv2.INTER_LINEAR)
    rockw = np.clip((slope - 0.55) / 0.35, 0, 1) * (1 - wet)
    rock = np.array(C['rock'][0], np.float32) * (1 + 0.06 * n3[..., None])
    col = col * (1 - rockw[..., None]) + rock * rockw[..., None]
    dirt = np.clip((slope - 0.35) / 0.3, 0, 1) * (1 - rockw) * 0.4 * (1 - wet)
    col = col * (1 - dirt[..., None]) + np.array([128, 108, 76], np.float32) * dirt[..., None]
    # river banks: damp darker, sandy/rocky edge
    near_w = cv2.GaussianBlur(wet, (0, 0), 3) * (1 - wet)
    col = col * (1 - 0.35 * near_w[..., None]) + np.array([120, 112, 92], np.float32) * 0.35 * near_w[..., None]
    # road shoulders: worn ground tint
    rtypes = load_json(path('data/roads/road_types.json'))['types']
    rm = np.zeros((HS, WS), np.uint8)
    for f in load_json(path('data/roads/roads.geojson'))['features']:
        wpx = rtypes[f['properties']['type']]['width_m'] / 2.5 * S
        cv2.polylines(rm, [(np.asarray(f['geometry']['coordinates'])[:, :2] * S).round().astype(np.int32)], False, 1, int(wpx + 5))
    rmw = cv2.GaussianBlur(rm.astype(np.float32), (0, 0), 2) * 0.35
    col = col * (1 - rmw[..., None]) + np.array([126, 116, 88], np.float32) * rmw[..., None]
    # elevation tint: cooler/bluer high ridges (atmospheric), warmer valleys
    e = (Tb - Tb.min()) / (Tb.max() - Tb.min())
    col *= (1 + 0.06 * (0.5 - e))[..., None]
    # baked AO from local relief + sun-facing warm light (sun from WSW, late afternoon)
    blur = cv2.GaussianBlur(Tb, (0, 0), 12)
    ao = np.clip(1 + (Tb - blur) * 0.025, 0.72, 1.12)
    sun = np.array([-0.8, -0.35])  # direction toward sun in image xy (west, slightly north)
    lit = np.clip(0.82 + 0.35 * (-(gx * sun[0] + gy * sun[1])), 0.55, 1.2)
    col *= (ao * lit)[..., None]
    col *= (1 + 0.025 * n3)[..., None]
    img = np.clip(col, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path('data/terrain/albedo_2x.jpg'), quality=90)
    print('albedo', img.shape)


if __name__ == '__main__':
    main()
