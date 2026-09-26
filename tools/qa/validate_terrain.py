"""Terrain & water validation (A1). Writes data/qa/terrain_report.json and assets/maps/debug/terrain_qa.png.

Checks
  river_uphill      water surface along each centreline never rises downstream (tol 5 cm)
  confluence        a tributary's mouth level matches its parent's level at the junction
  surface_step      adjacent water pixels differ by more than a cascade step (water 'walls')
  perched_river     river not in its valley: land within the lateral window lower than the water
  ridge_crossing    river crosses a designed crest line (ridge_skeleton) away from a saddle / gorge
  pit               closed depressions (raw terrain: should be none; graded: road embankments -> culverts)
  spike             single-cell spikes / holes relative to the 3x3 neighbourhood
  zone_slope        slopes above the usable limit inside developed zones (graded terrain)
  seam              discontinuities along the 100 px Blender chunk borders; NaN/inf; backdrop rim mismatch
  road_corridor     how much grading had to move the raw terrain along roads (+ grading issue counts)

usage: python3 tools/qa/validate_terrain.py [--strict]
"""
import sys, os, json, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import cv2
from scipy import ndimage as ndi
from shapely.geometry import Polygon, LineString, Point
from tools.lib.common import path, load_json, save_json, W, H, rnd
from tools.lib.geom import rasterize_polys
from tools.lib.trace import resample

MPP = 2.5
ZONE_SLOPE = {'downtown': 0.10, 'industrial': 0.08, 'town_center': 0.10, 'town': 0.18, 'city': 0.25}
STEP_MAX = 0.8          # m between adjacent water pixels (steeper = a water wall, needs a real waterfall asset)
SPIKE_M = 3.0
PIT_M = 0.3


def load_f32(p):
    return np.fromfile(path(p), np.float32).reshape(H, W)


def clusters(mask, max_items=25, min_px=1):
    lab, n = ndi.label(mask)
    if n == 0:
        return [], 0
    idx = np.arange(1, n + 1)
    sz = ndi.sum(mask, lab, idx)
    com = ndi.center_of_mass(mask, lab, idx)
    items = sorted(((int(s), (rnd(c[1], 1), rnd(c[0], 1))) for s, c in zip(sz, com) if s >= min_px), reverse=True)
    return [{'px': s, 'at': at} for s, at in items[:max_items]], len(items)


def main():
    strict = '--strict' in sys.argv
    T = load_f32('data/terrain/height_f32.bin')
    TG = load_f32('data/terrain/height_graded_f32.bin')
    WL = load_f32('data/terrain/water_level_f32.bin')
    wet = np.isfinite(WL)
    rep = {'summary': {}, 'checks': {}}
    marks = []  # (x, y, colour)

    def add(name, severity, items, extra=None):
        rep['checks'][name] = {'severity': severity, 'count': len(items) if isinstance(items, list) else items, 'items': items if isinstance(items, list) else [], **(extra or {})}

    # ---------------- rivers
    ww = load_json(path('data/water/waterways.geojson'))['features']
    lines = {f['properties']['id']: (resample(np.asarray(f['geometry']['coordinates'], float), 1.0), f['properties']) for f in ww}
    WLn = WL.copy()
    _, (iy, ix) = ndi.distance_transform_edt(~wet, return_indices=True)
    WLn = WL[iy, ix]
    uphill, conf, perched = [], [], []
    for k, (L, p) in lines.items():
        xi = np.clip(L[:, 0].astype(int), 0, W - 1); yi = np.clip(L[:, 1].astype(int), 0, H - 1)
        z = WLn[yi, xi]
        rise = np.maximum.accumulate(z[::-1])[::-1]  # max of downstream remainder
        bad = np.nonzero(z[:-1] + 0.05 < rise[1:])[0]
        if len(bad):
            i = int(bad[np.argmax(rise[1:][bad] - z[:-1][bad])])
            uphill.append({'id': k, 'at': [rnd(L[i, 0], 1), rnd(L[i, 1], 1)], 'rise_m': rnd(float((rise[1:] - z[:-1]).max()), 2), 'samples': int(len(bad))})
            marks.append((L[i, 0], L[i, 1], (0, 0, 255)))
        par = p.get('flows_into')
        if par and par in lines:
            Lp = lines[par][0]
            j = int(np.argmin(np.hypot(*(Lp - L[-1]).T)))
            zp = WLn[int(np.clip(Lp[j, 1], 0, H - 1)), int(np.clip(Lp[j, 0], 0, W - 1))]
            dz = float(z[-1] - zp)
            if abs(dz) > 0.3:
                conf.append({'id': k, 'parent': par, 'at': [rnd(L[-1, 0], 1), rnd(L[-1, 1], 1)], 'mouth_minus_parent_m': rnd(dz, 2)})
                marks.append((L[-1, 0], L[-1, 1], (0, 128, 255)))
        # perched: lateral windows (both sides, 6..30 px beyond the bank) must stay above the water
        d1 = np.gradient(L, axis=0); d1 /= np.maximum(np.hypot(*d1.T), 1e-9)[:, None]
        nrm = np.stack([-d1[:, 1], d1[:, 0]], 1)
        worst = 0.0; wat = None; cnt = 0
        for s in range(0, len(L), 4):
            for sgn in (1, -1):
                for off in range(6, 31, 3):
                    q = L[s] + nrm[s] * sgn * off
                    qx, qy = int(q[0]), int(q[1])
                    if not (0 <= qx < W and 0 <= qy < H) or wet[qy, qx]:
                        continue
                    dz = float(T[qy, qx] - z[s])
                    if dz < -0.5:
                        cnt += 1
                        if dz < worst:
                            worst, wat = dz, (q[0], q[1])
                    break  # first dry sample beyond the bank on this side
        if cnt:
            perched.append({'id': k, 'samples': cnt, 'worst_m': rnd(worst, 2), 'at': [rnd(wat[0], 1), rnd(wat[1], 1)]})
            marks.append((wat[0], wat[1], (255, 0, 255)))
    add('river_uphill', 'error', uphill)
    add('confluence', 'error', conf)
    add('perched_river', 'warn', perched)

    # surface steps
    gx = np.abs(np.diff(WL, axis=1)); gy = np.abs(np.diff(WL, axis=0))
    stepm = np.zeros((H, W), bool)
    stepm[:, :-1] |= np.nan_to_num(gx) > STEP_MAX
    stepm[:-1, :] |= np.nan_to_num(gy) > STEP_MAX
    items, n = clusters(stepm)
    add('surface_step', 'warn', items, {'clusters': n, 'max_step_m': rnd(float(max(np.nanmax(gx), np.nanmax(gy))), 2)})
    for it in items:
        marks.append((it['at'][0], it['at'][1], (0, 200, 255)))

    # ridge crossings (designed crest vs river)
    rc = []
    sk = path('data/terrain/ridge_skeleton.geojson')
    if os.path.exists(sk):
        man = load_json(path('data/manual/ridges.json'))
        saddles = [Point(s['at']).buffer(s['radius_px']) for s in man.get('saddles', [])]
        for f in load_json(sk)['features']:
            if f['properties'].get('kind') == 'spur':
                continue
            cl = LineString(f['geometry']['coordinates'])
            for k, (L, p) in lines.items():
                rl = LineString(L)
                if cl.distance(rl) < 3:
                    x = cl.intersection(rl.buffer(3))
                    c = x.centroid if not x.is_empty else cl.interpolate(cl.project(rl.centroid))
                    if any(sd.contains(c) for sd in saddles):
                        continue
                    rc.append({'ridge': f['properties']['id'], 'river': k, 'at': [rnd(c.x, 1), rnd(c.y, 1)]})
                    marks.append((c.x, c.y, (0, 0, 180)))
    add('ridge_crossing', 'warn', rc)

    # ---------------- pits & spikes
    from tools.pipeline.terrain import _flood
    outlet = wet.copy()
    outlet[0, :] = outlet[-1, :] = outlet[:, 0] = outlet[:, -1] = True
    for name, Z in (('pit_raw', T), ('pit_graded', TG)):
        zf, _ = _flood(Z.astype(np.float64), outlet, 0.0)
        pit = (zf - Z) > PIT_M
        items, n = clusters(pit, 30, 2)
        add(name, 'error' if name == 'pit_raw' else 'info', items, {'clusters': n, 'pit_px': int(pit.sum()), 'max_depth_m': rnd(float((zf - Z).max()), 2)})
        if name == 'pit_raw':
            for it in items:
                marks.append((it['at'][0], it['at'][1], (0, 255, 255)))
    for name, Z in (('spike_raw', T), ('spike_graded', TG)):
        med = ndi.median_filter(Z, size=3)
        dev = Z - med
        gyz, gxz = np.gradient(ndi.uniform_filter(Z, 5), MPP)
        tol = SPIKE_M + 2.5 * np.hypot(gxz, gyz) * MPP
        sp = (np.abs(dev) > tol) & ~wet
        items, n = clusters(sp)
        add(name, 'warn', items, {'clusters': n})

    # ---------------- developed-zone slopes (graded terrain)
    gy_, gx_ = np.gradient(cv2.GaussianBlur(TG, (0, 0), 1.0), MPP)
    slope = np.hypot(gx_, gy_)
    zs = []
    for z in load_json(path('data/manual/zones.json'))['zones']:
        lim = ZONE_SLOPE.get(z['kind'])
        if lim is None:
            continue
        m = rasterize_polys([Polygon(z['pts'])], (H, W)) > 0
        m &= ~(cv2.dilate(wet.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
        if not m.any():
            continue
        over = m & (slope > lim)
        frac = float(over.sum() / m.sum())
        if frac > 0.03:
            c = ndi.center_of_mass(over)
            zs.append({'zone': z.get('name', z['kind']), 'kind': z['kind'], 'limit': lim, 'frac_over': rnd(frac, 3), 'p95_slope': rnd(float(np.percentile(slope[m], 95)), 3),
                       'at': [rnd(c[1], 1), rnd(c[0], 1)]})
    add('zone_slope', 'warn', zs)

    # ---------------- seams / finiteness / backdrop rim
    seam = []
    for nm, Z in (('raw', T), ('graded', TG)):
        if not np.isfinite(Z).all():
            seam.append({'what': f'{nm} has NaN/inf', 'count': int((~np.isfinite(Z)).sum())})
        d2x = np.abs(np.diff(Z, 2, axis=1)); d2y = np.abs(np.diff(Z, 2, axis=0))
        base = float(np.percentile(np.concatenate([d2x.ravel(), d2y.ravel()]), 99.5))
        for c in range(100, W, 100):
            v = float(d2x[:, c - 1].max())
            if v > max(4 * base, 3.0):
                seam.append({'what': f'{nm} chunk seam x={c}', 'max_d2_m': rnd(v, 2)})
        for r in range(100, H, 100):
            v = float(d2y[r - 1, :].max())
            if v > max(4 * base, 3.0):
                seam.append({'what': f'{nm} chunk seam y={r}', 'max_d2_m': rnd(v, 2)})
    bj = path('data/terrain/backdrop.json')
    if os.path.exists(bj):
        m = load_json(bj)
        u = np.fromfile(path('data/terrain/backdrop_u16.bin'), '<u2').reshape(m['h'], m['w'])
        z = m['min_m'] + u.astype(np.float32) * (m['max_m'] - m['min_m']) / 65535
        x0, y0, x1, y1 = m['map_rect_cells']
        small = cv2.resize(TG, (W // m['cell_px'], H // m['cell_px']), interpolation=cv2.INTER_AREA)
        ring = np.concatenate([z[y0 - 1, x0:x1] - small[0, :], z[y1, x0:x1] - small[-1, :], z[y0:y1, x0 - 1] - small[:, 0], z[y0:y1, x1] - small[:, -1]])
        rim = float(np.abs(ring).max())
        rep['checks']['backdrop_rim'] = {'severity': 'warn' if rim > 8 else 'info', 'count': int(rim > 8), 'max_abs_m': rnd(rim, 2), 'p95_abs_m': rnd(float(np.percentile(np.abs(ring), 95)), 2), 'items': []}
    add('seam', 'error', seam)

    # ---------------- road corridors: how much grading moved the raw terrain
    rm = np.zeros((H, W), np.uint8)
    rt = {}
    for f in load_json(path('data/roads/roads.geojson'))['features']:
        t = f['properties']['type']
        c = np.asarray(f['geometry']['coordinates'])[:, :2].round().astype(np.int32)
        cv2.polylines(rm, [c], False, 1, 3)
        rt.setdefault(t, np.zeros((H, W), np.uint8))
        cv2.polylines(rt[t], [c], False, 1, 3)
    dz = np.abs(TG - T)
    road = (rm > 0) & ~wet
    corr = {'mean_abs_m': rnd(float(dz[road].mean()), 2), 'p95_abs_m': rnd(float(np.percentile(dz[road], 95)), 2), 'max_abs_m': rnd(float(dz[road].max()), 2), 'by_type': {}}
    for t, m in rt.items():
        mm = (m > 0) & ~wet
        if mm.sum() > 20:
            corr['by_type'][t] = {'mean_abs_m': rnd(float(dz[mm].mean()), 2), 'p95_abs_m': rnd(float(np.percentile(dz[mm], 95)), 2)}
    gr = path('data/qa/grade_report.json')
    if os.path.exists(gr):
        from collections import Counter
        corr['grading_issues'] = dict(Counter(i['kind'] for i in load_json(gr)['issues']))
    rep['checks']['road_corridor'] = {'severity': 'info', 'count': 0, 'items': [], **corr}

    # ---------------- summary + debug image
    sev = {}
    for k, v in rep['checks'].items():
        sev.setdefault(v['severity'], {})[k] = v['count']
    rep['summary'] = sev
    save_json(path('data/qa/terrain_report.json'), rep, indent=1)
    from tools.pipeline.terrain import hillshade
    tmp = path('assets/maps/debug/terrain_qa.png')
    hillshade(TG, tmp)
    img = cv2.imread(tmp)
    img[wet] = (img[wet] * 0.4 + np.array([200, 120, 40]) * 0.6).astype(np.uint8)
    for x, y, col in marks:
        cv2.circle(img, (int(x), int(y)), 7, col, 2)
    cv2.imwrite(tmp, img)
    for s in ('error', 'warn', 'info'):
        if s in sev:
            print(f'{s:5s}', {k: v for k, v in sev[s].items()})
    print('road corridor |graded-raw|:', {k: corr[k] for k in ('mean_abs_m', 'p95_abs_m', 'max_abs_m')}, 'grading issues:', corr.get('grading_issues'))
    if strict and any(v for v in sev.get('error', {}).values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
