"""Water extraction: river/creek centrelines (from hand waypoints snapped to
the source map), per-vertex channel widths, and water-body polygons.

Outputs
  data/water/waterways.geojson   centrelines, ordered downstream, with widths
  data/water/water_bodies.geojson polygons (islands as holes)
  tools/.cache/water_mask.npy     final raster mask (source grid) for later stages
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import cv2
from scipy import ndimage as ndi
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from skimage.graph import route_through_array

from tools.lib.common import load_json, save_json, path, W, H, rnd
from tools.lib.features import hsv, load_rgb, water_mask
from tools.lib.trace import smooth_polyline, rdp, resample
from tools.lib.geom import mask_to_polygons, geojson_polygon, geojson_line, fc, rasterize_polys, polyline_length


def raw_water():
    Hh, S, V = hsv()
    a = load_rgb()
    r, g, b = [a[..., i].astype(int) for i in range(3)]
    w = water_mask()
    w2 = (Hh >= 175) & (Hh <= 235) & (S > 0.2) & (V > 0.18) & (b >= r + 5)
    white = (S < 0.22) & (V > 0.72)
    near = cv2.dilate(w.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    cand = w | w2 | (white & near)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
    return cand, white & near


def water_cost(mask):
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    s = np.clip(dt / 4.0, 0, 1)
    c = np.where(mask, 1.0 / (0.05 + s), 60.0)
    return c.astype(np.float64)


def snap(wp, cost, margin=20):
    out = [tuple(wp[0])]
    for a, b in zip(wp[:-1], wp[1:]):
        x0, y0 = int(min(W - 1, a[0])), int(min(H - 1, a[1]))
        x1, y1 = int(min(W - 1, b[0])), int(min(H - 1, b[1]))
        bx0, bx1 = max(0, min(x0, x1) - margin), min(W, max(x0, x1) + margin + 1)
        by0, by1 = max(0, min(y0, y1) - margin), min(H, max(y0, y1) + margin + 1)
        idx, _ = route_through_array(cost[by0:by1, bx0:bx1], (y0 - by0, x0 - bx0), (y1 - by0, x1 - bx0),
                                     fully_connected=True, geometric=True)
        out += [(c + bx0 + 0.5, r + by0 + 0.5) for r, c in idx][1:]
    out[-1] = tuple(wp[-1])
    return out


def main():
    man = load_json(path('data/manual/waterways.json'))
    cand, rapids = raw_water()
    excl = [Polygon(e['pts']) for e in man['exclude_polygons']]
    if excl:
        cand &= ~(rasterize_polys(excl, (H, W)) > 0)
    cost = water_cost(cand)

    # --- centrelines
    lines = []
    for r in man['rivers']:
        wp = [tuple(map(float, p)) for p in r['wp']]
        raw = snap(wp, cost)
        sm = smooth_polyline(raw, 4.0 if r['class'] == 'river' else 2.5)
        lines.append((r, sm))

    # --- keep only water components touching a centreline (drops glass towers, pools, blue roofs)
    lab, n = ndi.label(cand)
    line_mask = np.zeros((H, W), np.uint8)
    for r, sm in lines:
        cv2.polylines(line_mask, [(sm - 0.5).round().astype(np.int32)], False, 1, 3)
    touched = np.unique(lab[(line_mask > 0) & (lab > 0)])
    keep_ids = set(touched.tolist())
    for kp in man.get('keep_points', []):
        keep_ids.add(int(lab[int(kp[1]), int(kp[0])]))
    keep = np.isin(lab, list(keep_ids - {0}))
    # fill tiny holes only (rapids/boats), keep real islands
    holes = ndi.binary_fill_holes(keep) & ~keep
    hl, hn = ndi.label(holes)
    hs = ndi.sum(holes, hl, range(1, hn + 1))
    small = np.isin(hl, 1 + np.where(hs < 25)[0])
    keep |= small

    # --- widths from distance transform along the snapped centreline
    dt = cv2.distanceTransform(keep.astype(np.uint8), cv2.DIST_L2, 5)
    feats = []
    channel_polys = []
    for r, sm in lines:
        pts = resample(sm, 1.0)
        xi = np.clip(pts[:, 0].astype(int), 0, W - 1)
        yi = np.clip(pts[:, 1].astype(int), 0, H - 1)
        # search a small window for the local max distance (centre may be off by a pixel)
        wv = np.array([dt[max(0, y - 2):y + 3, max(0, x - 2):x + 3].max() for x, y in zip(xi, yi)]) * 2.0
        visible = wv >= 1.5
        minw = 3.0 if r['class'] == 'creek' else 5.0
        wv = np.where(visible, wv, np.nan)
        # interpolate through hidden stretches (bridges / canopy), then smooth
        idx = np.arange(len(wv))
        if np.isnan(wv).all():
            wv[:] = minw
        else:
            good = ~np.isnan(wv)
            wv = np.interp(idx, idx[good], wv[good])
        wv = np.maximum(ndi.gaussian_filter1d(wv, 6), minw)
        simp = rdp(pts, 0.25)
        # map widths to simplified vertices
        d_all = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        d_s = np.r_[0, np.cumsum(np.hypot(*np.diff(simp, axis=0).T))]
        w_s = np.interp(d_s, d_all, wv)
        hidden_frac = float(1 - visible.mean())
        props = {
            'id': r['id'], 'name': r['name'], 'class': r['class'],
            'flows_into': r.get('flows_into'), 'outlet': r.get('outlet'), 'source': r.get('source'),
            'backwater': bool(r.get('backwater', False)),
            'length_px': rnd(polyline_length(simp)), 'hidden_fraction': rnd(hidden_frac, 3),
            'width_px': [rnd(x, 1) for x in w_s],
            'direction': 'coordinates ordered downstream',
        }
        feats.append(geojson_line(simp, props))
        # variable-width channel polygon to keep water continuous under bridges/canopy
        segs = []
        for i in range(len(pts) - 1):
            ww = max(minw, 0.9 * min(wv[i], wv[i + 1])) / 2
            segs.append(LineString([pts[i], pts[i + 1]]).buffer(ww, cap_style='round', resolution=4))
        channel_polys.append(unary_union(segs))

    mask_polys = mask_to_polygons(keep, blur=0.8, simplify=0.3, min_area=6)
    allw = unary_union(mask_polys + channel_polys).buffer(0.6).buffer(-0.6)
    allw = allw.simplify(0.3, preserve_topology=True)
    bodies = []
    geoms = allw.geoms if allw.geom_type == 'MultiPolygon' else [allw]
    geoms = sorted(geoms, key=lambda g: -g.area)
    for i, g in enumerate(geoms):
        if g.area < 8:
            continue
        # which river does it belong to?
        best, bd = None, 1e9
        for (r, sm), f in zip(lines, feats):
            d = g.distance(LineString(sm))
            if d < bd:
                bd, best = d, r['id']
        bodies.append(geojson_polygon(g, {'id': f'WTR_B{i + 1:03d}', 'kind': 'river_channel' if bd < 1 else 'pond',
                                          'waterway': best, 'area_px': rnd(g.area, 1)}))
    save_json(path('data/water/waterways.geojson'), fc(feats, 'waterways'))
    save_json(path('data/water/water_bodies.geojson'), fc(bodies, 'water_bodies'))
    final = rasterize_polys(geoms, (H, W), value=255, ups=4) > 127
    np.save(path('tools/.cache/water_mask.npy'), final)
    np.save(path('tools/.cache/rapids_mask.npy'), rapids & final)
    print(f'waterways: {len(feats)}  water bodies: {len(bodies)}  water px: {final.sum()}')


if __name__ == '__main__':
    main()
