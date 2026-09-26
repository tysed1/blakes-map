"""Street-grid detection for dense urban cores (downtowns, industrial districts,
Hollow Ridge town centre).

For each grid zone (data/manual/zones.json, kind in GRID_KINDS):
  1. estimate the two dominant street orientations from the road-likeness
     field's structure tensor,
  2. project the field onto each orientation's normal -> 1-D profile whose
     peaks are street lines,
  3. walk each line and keep only the runs where a street is actually visible,
  4. refine each run with the centre tracer.
Output: tools/.cache/grid_streets.json (consumed by roads.py as source 'grid').
Per-zone overrides live in data/manual/grid_overrides.json.
"""
import sys, os, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.signal import find_peaks
from shapely.geometry import Polygon, LineString, Point

from tools.lib.common import path, load_json, save_json, W, H
from tools.lib.features import road_prob, hsv, water_mask
from tools.lib.geom import rasterize_polys

GRID_KINDS = {'downtown': 'urban_street', 'industrial': 'urban_street', 'town_center': 'urban_street'}


def field():
    Hh, S, V = hsv()
    pave = ((S < 0.30) & (V > 0.45) & (V < 0.95)).astype(np.float32)
    P = np.maximum(road_prob() * 1.6, pave * 0.6)
    P[water_mask()] = 0
    return np.clip(P, 0, 1)


def orientations(F, mask):
    G = cv2.GaussianBlur(F, (0, 0), 1.2)
    gx = cv2.Sobel(G, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(G, cv2.CV_32F, 0, 1, ksize=3)
    m = mask > 0
    ang = np.arctan2(gy[m], gx[m])  # gradient direction = street normal
    mag = np.hypot(gx[m], gy[m])
    # doubled-angle histogram (orientation, 180-periodic) of street DIRECTION
    d = (ang + math.pi / 2) % math.pi
    hist, edges = np.histogram(d, bins=180, range=(0, math.pi), weights=mag)
    hist = gaussian_filter1d(np.r_[hist, hist, hist], 3)[180:360]
    a1 = int(np.argmax(hist))
    # second orientation: best peak at least 50 deg away
    dist = np.minimum(np.abs(np.arange(180) - a1), 180 - np.abs(np.arange(180) - a1))
    h2 = np.where(dist >= 50, hist, 0)
    a2 = int(np.argmax(h2))
    return [(a1 + 0.5) * math.pi / 180, (a2 + 0.5) * math.pi / 180]


def detect_lines(F, mask, poly, theta, min_sep=11, thr_run=0.30, min_len=14, gap=7):
    d = np.array([math.cos(theta), math.sin(theta)])
    n = np.array([-d[1], d[0]])
    ys, xs = np.nonzero(mask)
    P = np.stack([xs + 0.5, ys + 0.5], 1)
    s = P @ n
    t = P @ d
    s0, s1 = s.min(), s.max()
    bins = np.arange(math.floor(s0), math.ceil(s1) + 2)
    val = F[ys, xs]
    num, _ = np.histogram(s, bins=bins, weights=val)
    den, _ = np.histogram(s, bins=bins)
    prof = gaussian_filter1d(num / np.maximum(den, 1), 1.0)
    base = gaussian_filter1d(prof, 8)
    pk, props = find_peaks(prof - base, distance=min_sep, prominence=0.035)
    lines = []
    tmin, tmax = t.min() - 5, t.max() + 5
    for k in pk:
        sv = bins[k] + 0.5
        # refine peak position (parabolic)
        if 0 < k < len(prof) - 1:
            a, b, c = prof[k - 1], prof[k], prof[k + 1]
            den2 = a - 2 * b + c
            if abs(den2) > 1e-9:
                sv += 0.5 * (a - c) / den2
        ts = np.arange(tmin, tmax, 1.0)
        pts = sv * n[None] + ts[:, None] * d[None]
        # value across a +-1.5 px band (max)
        vals = []
        for off in (-1.5, -0.75, 0, 0.75, 1.5):
            q = pts + off * n
            xi = np.clip(q[:, 0].astype(int), 0, W - 1)
            yi = np.clip(q[:, 1].astype(int), 0, H - 1)
            vals.append(F[yi, xi] * (mask[yi, xi] > 0))
        v = gaussian_filter1d(np.max(vals, 0), 1.5)
        on = v > thr_run
        # close gaps
        on = maximum_filter1d(on.astype(np.uint8), gap).astype(bool)
        on = ~maximum_filter1d((~on).astype(np.uint8), gap).astype(bool) | on & False | on
        # runs
        i = 0
        while i < len(on):
            if not on[i]:
                i += 1
                continue
            j = i
            while j < len(on) and on[j]:
                j += 1
            if j - i >= min_len:
                seg = pts[i:j]
                ls = LineString(seg).intersection(poly.buffer(3))
                if not ls.is_empty and ls.length >= min_len:
                    geoms = [ls] if ls.geom_type == 'LineString' else [g for g in ls.geoms if g.geom_type == 'LineString']
                    for g in geoms:
                        if g.length >= min_len:
                            lines.append(np.asarray(g.coords))
            i = j
    return lines


def main():
    zones = load_json(path('data/manual/zones.json'))['zones']
    ov = load_json(path('data/manual/grid_overrides.json')) if os.path.exists(path('data/manual/grid_overrides.json')) else {}
    F = field()
    out = []
    for z in zones:
        if z['kind'] not in GRID_KINDS:
            continue
        poly = Polygon(z['pts'])
        mask = rasterize_polys([poly], (H, W))
        zo = ov.get(z['id'], {})
        thetas = zo.get('orientations_deg')
        thetas = [math.radians(a) for a in thetas] if thetas else orientations(F, mask)
        for ti, th in enumerate(thetas):
            ls = detect_lines(F, mask, poly, th, min_sep=zo.get('min_sep', 11), thr_run=zo.get('thr', 0.30))
            for k, l in enumerate(ls):
                out.append({'zone': z['id'], 'orient': ti, 'theta_deg': round(math.degrees(th), 1),
                            'type': GRID_KINDS[z['kind']], 'pts': [[round(x, 2), round(y, 2)] for x, y in l]})
        print(z['id'], [round(math.degrees(t), 1) for t in thetas], sum(1 for o in out if o['zone'] == z['id']), 'segments')
    save_json(path('tools/.cache/grid_streets.json'), out)


if __name__ == '__main__':
    main()
