"""Building proxies.

Two sources, merged with priority:
  A. detected  : large roofs (industrial, commercial, barns, towers) = connected
                 components of the roof mask, fitted with min-area rectangles.
  B. frontage  : lots generated along every street (both sides), oriented to the
                 street, kept only where the base map shows development there
                 (evidence map). This reproduces the map's building density with
                 clean, believable, non-overlapping footprints.
  +  landmarks from data/manual/landmarks.json (custom footprints).

Output: data/buildings/buildings.geojson (+ id registry for stable IDs)
"""
import sys, os, math, hashlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import cv2
from scipy import ndimage as ndi
from shapely.geometry import Polygon, Point, LineString, shape, box
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.prepared import prep

from tools.lib.common import path, load_json, save_json, W, H, rnd
from tools.lib.features import hsv, load_rgb, cream_score, rock_density
from tools.lib.geom import geojson_polygon, fc, rasterize_polys
from tools.lib.trace import resample

BT = load_json(path('data/buildings/building_types.json'))['types']
RT = load_json(path('data/roads/road_types.json'))['types']
ZONES = [(z, Polygon(z['pts'])) for z in load_json(path('data/manual/zones.json'))['zones']]
PREFIX = {'laurel_city': 'LC', 'hollow_ridge': 'HR', 'tannersville': 'TV', None: 'RU'}
NO_FRONTAGE = {'freeway', 'ramp', 'highway'}


def zone_at(x, y):
    p = Point(x, y)
    for z, poly in ZONES:
        if poly.contains(p):
            return z
    return None


def hrand(*k):
    h = hashlib.md5(repr(k).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


# ------------------------------------------------------------------ rasters
def road_polys():
    polys = []
    for f in load_json(path('data/roads/roads.geojson'))['features']:
        p = f['properties']
        w = RT[p['type']]['width_m'] / 2.5
        polys.append(LineString(f['geometry']['coordinates']).buffer(w / 2, cap_style='flat' if p['type'] in NO_FRONTAGE else 'round'))
    for f in load_json(path('data/railways/railways.geojson'))['features']:
        polys.append(LineString(f['geometry']['coordinates']).buffer(f['properties']['width_m'] / 2.5 / 2 + 0.6))
    return unary_union(polys)


def masks():
    Hh, S, V = hsv()
    a = load_rgb().astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    wm = np.load(path('tools/.cache/water_mask.npy'))
    veg = ((Hh >= 45) & (Hh <= 160) & (S > 0.28) & (g >= r - 5)) | ((V < 0.28) & (g > r) & (g > b))
    red = ((r > g + 25) & (r > b + 25) & (V > 0.3)) & ((Hh < 25) | (Hh > 340))
    grey = (S < 0.22) & (V > 0.35)
    white = (S < 0.16) & (V > 0.72)
    glass = (b > r + 12) & (V < 0.55) & (V > 0.12) & (S > 0.18) & ~wm
    dark = (V < 0.40) & (S < 0.35) & (b >= g - 8)
    roof = (red | grey | white | dark | glass) & ~veg & ~wm
    cream = cream_score() > 0.35
    return dict(roof=roof, red=red, white=white, glass=glass, veg=veg, water=wm, cream=cream, grey=grey)


# ------------------------------------------------------------------ shapes
def rect(cx, cy, w, d, ang):
    p = box(-w / 2, -d / 2, w / 2, d / 2)
    return translate(rotate(p, ang, origin=(0, 0)), cx, cy)


def detected(M, road_area):
    """Large roofs from connected components."""
    roof = M['roof'] & ~(rasterize_polys([road_area], (H, W), ups=2) > 100)
    roof = cv2.morphologyEx(roof.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    lab, n = ndi.label(roof)
    out = []
    rock = rock_density()
    objs = ndi.find_objects(lab)
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        comp = (lab[sl] == i + 1)
        area = int(comp.sum())
        if area < 16:
            continue
        ys, xs = np.nonzero(comp)
        xs = xs + sl[1].start + 0.5
        ys = ys + sl[0].start + 0.5
        cx, cy = xs.mean(), ys.mean()
        z = zone_at(cx, cy)
        if z is None and rock[int(cy), int(cx)] > 0.18:
            continue
        pts = np.stack([xs, ys], 1).astype(np.float32)
        (rx, ry), (rw, rh), ang = cv2.minAreaRect(pts)
        rw, rh = rw + 1, rh + 1
        fill = area / (rw * rh)
        if fill < 0.45 or max(rw, rh) > 70:
            continue
        redf = M['red'][ys.astype(int), xs.astype(int)].mean()
        glassf = M['glass'][ys.astype(int), xs.astype(int)].mean()
        out.append(dict(cx=rx, cy=ry, w=rw * 0.92, d=rh * 0.92, ang=ang, area=area, red=redf, glass=glassf, zone=z, src='detected'))
    return out


def classify_detected(b):
    z = b['zone']
    k = z['kind'] if z else None
    A = b['w'] * b['d'] * 6.25  # m^2
    if k == 'downtown':
        if b['glass'] > 0.25 and A > 250:
            return 'tower'
        return 'urban_midrise' if A > 900 else 'downtown_lowrise'
    if k == 'industrial':
        return 'industrial' if A > 500 else 'commercial_small'
    if k in ('town_center',):
        return 'commercial_small' if A < 900 else 'civic'
    if k in ('town', 'city'):
        if A > 1500:
            return 'industrial' if b['red'] < 0.2 else 'civic'
        if A > 450:
            return 'commercial_small'
        return 'house_large'
    # rural
    if b['red'] > 0.35:
        return 'barn'
    return 'house_large' if A < 450 else 'barn'


# ------------------------------------------------------------------ frontage
LOT = {  # zone kind -> (lot width px, depth px, setback px, evidence threshold, types)
    'downtown':    (7.0, 9.0, 0.3, 0.30, 'downtown'),
    'town_center': (6.5, 7.0, 0.6, 0.30, 'town_center'),
    'industrial':  (12.0, 10.0, 1.5, 0.40, 'industrial'),
    'city':        (5.6, 4.0, 1.4, 0.26, 'residential'),
    'town':        (6.5, 4.0, 1.8, 0.24, 'residential'),
    None:          (9.0, 4.0, 2.5, 0.40, 'rural'),
}


def evidence(M, road_area):
    rr = rasterize_polys([road_area.buffer(0.5)], (H, W), ups=2) > 100
    e = (M['roof'] | M['cream'] | M['red'] | M['white']) & ~rr & ~M['water']
    e = e.astype(np.float32)
    e[M['veg']] *= 0.0
    return cv2.GaussianBlur(e, (0, 0), 1.2)


def frontage(E, road_area, blocked, M):
    out = []
    feats = load_json(path('data/roads/roads.geojson'))['features']
    forest = cv2.GaussianBlur(M['veg'].astype(np.float32), (0, 0), 2.0)
    for f in feats:
        p = f['properties']
        if p['type'] in NO_FRONTAGE or p.get('grade_separated'):
            continue
        pts = np.asarray(f['geometry']['coordinates'], float)
        if len(pts) < 2:
            continue
        hw = RT[p['type']]['width_m'] / 2.5 / 2
        mid = pts[len(pts) // 2]
        z0 = zone_at(*mid)
        lw, dp, sb, thr, fam = LOT[z0['kind'] if z0 else None]
        s = resample(pts, 1.0)
        if len(s) < lw:
            continue
        t = np.gradient(s, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        for side in (-1, 1):
            i = int(lw / 2)
            while i < len(s) - lw / 2:
                zc = zone_at(*s[i])
                kind = zc['kind'] if zc else None
                lw, dp, sb, thr, fam = LOT[kind]
                j = hrand(p['id'], side, i)
                w = lw * (0.72 + 0.2 * j)
                d = dp * (0.85 + 0.3 * hrand(p['id'], side, i, 'd'))
                if fam == 'residential':
                    w = min(w, 4.6 + 1.2 * j)
                if fam == 'rural':
                    w = 4.4 + 1.4 * j
                    d = 3.6 + 1.0 * hrand(i, 'r')
                n = np.array([-t[i, 1], t[i, 0]]) * side
                off = hw + sb + d / 2
                c = s[i] + n * off
                ang = math.degrees(math.atan2(t[i, 1], t[i, 0]))
                fp = rect(c[0], c[1], w, d, ang)
                xi, yi = int(np.clip(c[0], 0, W - 1)), int(np.clip(c[1], 0, H - 1))
                ev = E[yi, xi]
                ok = ev >= thr and not fp.intersects(road_area) and not blocked(fp)
                if fam in ('rural', 'residential') and forest[yi, xi] > 0.82:
                    ok = False
                if ok:
                    out.append(dict(poly=fp, cx=c[0], cy=c[1], w=w, d=d, ang=ang, zone=zc, src='frontage', fam=fam, ev=float(ev), road=p['id']))
                    blocked.add(fp)
                    i += int(round(w + (1.0 if fam in ('residential', 'rural') else 0.4) + (4 * hrand(i, side, 'gap') if fam == 'rural' else 0)))
                else:
                    i += 2
    return out


class Blocker:
    def __init__(self, geoms, pad=0.6):
        self.geoms = list(geoms)
        self.pad = pad
        self.grid = {}
        for g in self.geoms:
            self._ins(g)

    def _cells(self, g):
        x0, y0, x1, y1 = g.bounds
        for gx in range(int(x0 // 20), int(x1 // 20) + 1):
            for gy in range(int(y0 // 20), int(y1 // 20) + 1):
                yield (gx, gy)

    def _ins(self, g):
        for c in self._cells(g):
            self.grid.setdefault(c, []).append(g)

    def add(self, g):
        self.geoms.append(g)
        self._ins(g)

    def __call__(self, g):
        gb = g.buffer(self.pad)
        seen = set()
        for c in self._cells(gb):
            for o in self.grid.get(c, ()):
                if id(o) in seen:
                    continue
                seen.add(id(o))
                if o.intersects(gb):
                    return True
        return False


def frontage_type(b):
    fam = b['fam']
    j = hrand(round(b['cx'], 1), round(b['cy'], 1))
    if fam == 'downtown':
        return 'downtown_lowrise' if j < 0.75 else 'urban_midrise'
    if fam == 'town_center':
        return 'commercial_small'
    if fam == 'industrial':
        return 'industrial' if b['w'] * b['d'] * 6.25 > 400 else 'commercial_small'
    if fam == 'residential':
        return 'house' if j < 0.82 else ('house_large' if j < 0.95 else 'trailer')
    return 'house' if j < 0.55 else ('trailer' if j < 0.8 else 'house_large')


# ------------------------------------------------------------------ main
def main():
    M = masks()
    road_area = road_polys()
    water = unary_union([shape(f['geometry']) for f in load_json(path('data/water/water_bodies.geojson'))['features']])
    blocked = Blocker([water.buffer(0.8)] if not water.is_empty else [], pad=0.6)
    items = []
    # landmarks first
    for l in load_json(path('data/manual/landmarks.json'))['landmarks']:
        if not l.get('building'):
            continue
        cx, cy = l['center']
        if 'radius_px' in l:
            fp = Point(cx, cy).buffer(l['radius_px'], resolution=16)
            w = d = 2 * l['radius_px']; ang = 0.0
        else:
            w, d = l['size_px']; ang = l.get('rotation_deg', 0.0)
            fp = rect(cx, cy, w, d, ang)
        items.append(dict(poly=fp, cx=cx, cy=cy, w=w, d=d, ang=ang, zone=zone_at(cx, cy), src='landmark', type=l['kind'], name=l['name'], lid=l['id']))
        blocked.add(fp)
    # detected large roofs
    det = sorted(detected(M, road_area), key=lambda b: -b['area'])
    for b in det:
        fp = rect(b['cx'], b['cy'], b['w'], b['d'], b['ang'])
        if fp.intersects(road_area):
            fp = fp.difference(road_area.buffer(0.5))
            if fp.is_empty or fp.area < 0.6 * b['w'] * b['d'] or fp.geom_type != 'Polygon':
                continue
            fp = fp.minimum_rotated_rectangle
        if blocked(fp):
            continue
        b['poly'] = fp
        b['type'] = classify_detected(b)
        items.append(b)
        blocked.add(fp)
    E = evidence(M, road_area)
    items += frontage(E, road_area, blocked, M)
    for b in items:
        if 'type' not in b:
            b['type'] = frontage_type(b)
    write(items)


def load_registry():
    p = path('data/buildings/id_registry.json')
    return load_json(p) if os.path.exists(p) else {'ids': {}, 'next': {}}


def write(items):
    reg = load_registry()
    cells = {}
    for k, v in reg['ids'].items():
        v.pop('_used', None)
        cells.setdefault((int(v['c'][0] // 4), int(v['c'][1] // 4)), []).append(k)
    feats = []
    items.sort(key=lambda b: (round(b['cy'] / 25), b['cx']))
    for b in items:
        z = b['zone']
        pre = PREFIX[z['settlement'] if z else None]
        cx, cy = b['cx'], b['cy']
        bid = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for k in cells.get((int(cx // 4) + dx, int(cy // 4) + dy), []):
                    v = reg['ids'][k]
                    if not v.get('_used') and k.startswith(pre) and math.hypot(v['c'][0] - cx, v['c'][1] - cy) < 1.5 and v['t'] == b['type']:
                        bid = k
        if bid is None:
            n = reg['next'].get(pre, 1)
            reg['next'][pre] = n + 1
            bid = f'{pre}_BLDG_{n:04d}'
            reg['ids'][bid] = {}
        reg['ids'][bid].update({'c': [rnd(cx, 2), rnd(cy, 2)], 't': b['type'], '_used': True})
        t = BT[b['type']]
        lo, hi = t['height_m']
        j = hrand(bid)
        hgt = lo + (hi - lo) * j
        if b['type'] == 'tower':
            hgt = lo + (hi - lo) * (0.3 + 0.7 * j) * min(1.0, 0.6 + b.get('area', 40) / 200)
        poly = b['poly']
        props = {
            'id': bid, 'type': b['type'], 'name': b.get('name'),
            'region': z['settlement'] if z else 'rural', 'zone': z['id'] if z else None,
            'center_px': [rnd(cx, 2), rnd(cy, 2)],
            'rotation_deg': rnd(((b['ang'] + 90) % 180) - 90, 1),
            'width_m': rnd(b['w'] * 2.5, 1), 'depth_m': rnd(b['d'] * 2.5, 1),
            'footprint_m2': rnd(poly.area * 6.25, 1),
            'height_m': rnd(hgt, 1), 'roof': t['roof'],
            'source': b['src'], 'landmark': b.get('lid'), 'fronts_road': b.get('road'),
            'replace_with': t['replace_with'], 'replacement_status': 'proxy',
        }
        feats.append(geojson_polygon(poly, props))
    for k in list(reg['ids']):
        if not reg['ids'][k].pop('_used', False):
            reg['ids'][k]['retired'] = True
    save_json(path('data/buildings/buildings.geojson'), fc(feats, 'buildings'))
    save_json(path('data/buildings/id_registry.json'), reg)
    from collections import Counter
    c = Counter(f['properties']['type'] for f in feats)
    r = Counter(f['properties']['region'] for f in feats)
    s = Counter(f['properties']['source'] for f in feats)
    print(f'buildings: {len(feats)}', dict(c), dict(r), dict(s))


if __name__ == '__main__':
    main()
