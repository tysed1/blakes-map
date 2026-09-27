"""Land use / land cover + world hierarchy.

Per-pixel classes derived from the map colours, then made geographically sensible:
farmland only on usable (gentle) ground, rock only on steep ground, forest type from
canopy colour, developed classes from zones. Outputs:
  data/landuse/landuse.geojson       polygons per class
  data/landuse/landuse_classes.png   class raster (source grid), ids in landuse.json
  data/landuse/tree_density.png      0..255 canopy density (for vegetation scattering)
  data/landuse/landuse.json          class table
  data/world/regions.geojson, data/world/settlements.geojson, data/poi/landmarks.geojson
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import cv2
from scipy import ndimage as ndi
from shapely.geometry import Polygon, Point
from PIL import Image
from tools.lib.common import path, load_json, save_json, W, H, rnd
from tools.lib.features import hsv, load_rgb, rock_density
from tools.lib.geom import mask_to_polygons, geojson_polygon, geojson_point, fc, rasterize_polys

CLASSES = [
    (0, 'none', '#000000'), (1, 'water', '#3b6e8f'), (2, 'forest_hardwood', '#5d7a32'), (3, 'forest_conifer', '#2f5234'),
    (4, 'farmland', '#b9b56a'), (5, 'meadow', '#8fa55a'), (6, 'rock', '#9a968a'), (7, 'residential', '#a8a37e'),
    (8, 'commercial', '#9d8f80'), (9, 'industrial', '#8c8a86'), (10, 'rail_yard', '#6f6258'), (11, 'town', '#a3a07a'),
]
ID = {n: i for i, n, _ in CLASSES}


def main():
    Hh, S, V = hsv()
    a = load_rgb().astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    T = np.fromfile(path('data/terrain/height_graded_f32.bin'), np.float32).reshape(H, W)
    gy, gx = np.gradient(cv2.GaussianBlur(T, (0, 0), 2), 2.5)
    slope = np.hypot(gx, gy)
    wm = ~np.isnan(np.fromfile(path('data/terrain/water_level_f32.bin'), np.float32).reshape(H, W))
    rock = rock_density()
    canopy = ((Hh >= 50) & (Hh <= 170) & (V < 0.50) & (S > 0.25)) | (V < 0.22)
    conifer = canopy & ((Hh >= 95) | (V < 0.22))
    hard = canopy & ~conifer
    tree_d = cv2.GaussianBlur(canopy.astype(np.float32), (0, 0), 2.0)
    con_d = cv2.GaussianBlur(conifer.astype(np.float32), (0, 0), 3.0)
    hard_d = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 3.0)
    openf = (Hh >= 38) & (Hh <= 95) & (S > 0.25) & (V > 0.42) & ~canopy
    open_d = cv2.GaussianBlur(openf.astype(np.float32), (0, 0), 3.0)

    cls = np.full((H, W), ID['forest_hardwood'], np.uint8)
    cls[con_d > hard_d] = ID['forest_conifer']
    # open land: farmland where usable (gentle), meadow otherwise
    op = (open_d > 0.35) & (tree_d < 0.5)
    cls[op & (slope < 0.14)] = ID['farmland']
    cls[op & (slope >= 0.14) & (slope < 0.35)] = ID['meadow']
    # steep 'open' paint on mountainsides is the map's stylized rock / light foliage, not pasture:
    # Appalachian slopes of this steepness are wooded (bare crags come from terrain rock exposure)
    cls[op & (slope >= 0.35)] = ID['forest_hardwood']
    # rock: only on steep ground
    rk = (cv2.GaussianBlur(rock, (0, 0), 1.5) > 0.30) & (slope > 0.22)
    cls[rk] = ID['rock']
    # developed land = inside a zone AND actually served by streets (road density)
    rr = np.zeros((H, W), np.uint8)
    for f in load_json(path('data/roads/roads.geojson'))['features']:
        if f['properties']['type'] in ('freeway', 'ramp', 'highway', 'driveway', 'gravel', 'dirt', 'rural'):
            continue
        cv2.polylines(rr, [np.asarray(f['geometry']['coordinates'])[:, :2].round().astype(np.int32)], False, 1, 2)
    dens = cv2.GaussianBlur(rr.astype(np.float32), (0, 0), 7)
    served = cv2.GaussianBlur((dens > 0.09).astype(np.float32), (0, 0), 3) > 0.5
    for z in load_json(path('data/manual/zones.json'))['zones']:
        m = rasterize_polys([Polygon(z['pts'])], (H, W)) > 0
        if z['kind'] in ('city', 'town'):
            m &= served
        k = {'downtown': 'commercial', 'town_center': 'commercial', 'industrial': 'industrial', 'town': 'town', 'city': 'residential'}[z['kind']]
        if z['kind'] in ('city', 'town'):
            # woodland inside city limits stays woodland (ravines, steep lots)
            m &= ~((tree_d > 0.75) & (slope > 0.12)) & ~rk
            m &= ~(op & (slope < 0.14) & (open_d > 0.7))  # large open fields in city limits stay farm/park
        cls[m & (cls != ID['water'])] = ID[k]
    # rail yards
    for f in load_json(path('data/railways/railways.geojson'))['features']:
        if f['properties']['tracks'] > 1:
            m = np.zeros((H, W), np.uint8)
            cv2.polylines(m, [np.asarray(f['geometry']['coordinates'])[:, :2].round().astype(np.int32)], False, 1, int(f['properties']['width_m'] / 2.5) + 4)
            cls[m > 0] = ID['rail_yard']
    # grassy balds (data/manual/terrain.json 'balds'): treeless crest meadows; bare crags stay rock
    bald = bald_mask(T) > 0.5
    cls[bald & (cls != ID['rock']) & ~wm] = ID['meadow']
    cls[wm] = ID['water']
    # clean speckle: majority filter
    cls = ndi.generic_filter(cls, lambda v: np.bincount(v.astype(int), minlength=12).argmax(), size=5, mode='nearest').astype(np.uint8) if False else majority(cls)
    cls[wm] = ID['water']
    os.makedirs(path('data/landuse'), exist_ok=True)
    Image.fromarray(cls).save(path('data/landuse/landuse_classes.png'))
    td = np.where(np.isin(cls, [ID['forest_hardwood'], ID['forest_conifer']]), np.clip(tree_d * 1.15, 0.35, 1), tree_d * 0.6)
    td[np.isin(cls, [ID['water'], ID['rock'], ID['rail_yard'], ID['commercial'], ID['industrial']])] *= 0.15
    td[cls == ID['farmland']] *= 0.1
    Image.fromarray((np.clip(td, 0, 1) * 255).astype(np.uint8)).save(path('data/landuse/tree_density.png'))
    Image.fromarray((np.clip(con_d / np.maximum(con_d + hard_d, 1e-3), 0, 1) * 255).astype(np.uint8)).save(path('data/landuse/conifer_ratio.png'))
    feats = []
    for i, n, col in CLASSES:
        if n in ('none', 'water'):
            continue
        polys = mask_to_polygons(cls == i, blur=1.2, simplify=0.8, min_area=40)
        for k, p in enumerate(polys):
            feats.append(geojson_polygon(p, {'id': f'LU_{n.upper()}_{k + 1:04d}', 'class': n, 'color': col, 'area_m2': rnd(p.area * 6.25, 0)}, n=1))
    save_json(path('data/landuse/landuse.geojson'), fc(feats, 'landuse'))
    save_json(path('data/landuse/landuse.json'), {'classes': [{'id': i, 'name': n, 'color': c} for i, n, c in CLASSES],
                                                  'raster': 'data/landuse/landuse_classes.png', 'tree_density': 'data/landuse/tree_density.png',
                                                  'conifer_ratio': 'data/landuse/conifer_ratio.png'}, indent=1)
    # hierarchy
    R = load_json(path('data/manual/regions.json'))
    rf = []
    for rg in R['regions']:
        p = Polygon(rg['pts']).buffer(0)
        rf.append(geojson_polygon(p, {k: v for k, v in rg.items() if k != 'pts'} | {'area_km2': rnd(p.area * 6.25 / 1e6, 2)}))
    save_json(path('data/world/regions.geojson'), fc(rf, 'regions'))
    save_json(path('data/world/settlements.geojson'), fc([geojson_point(s['center'], {k: v for k, v in s.items() if k != 'center'}) for s in R['settlements']], 'settlements'))
    L = load_json(path('data/manual/landmarks.json'))['landmarks']
    os.makedirs(path('data/poi'), exist_ok=True)
    save_json(path('data/poi/landmarks.geojson'), fc([geojson_point(l['center'], {k: v for k, v in l.items() if k not in ('center',)}) for l in L], 'landmarks'))
    from collections import Counter
    c = Counter(cls.ravel().tolist())
    print('landuse px:', {CLASSES[k][1]: v for k, v in sorted(c.items())}, 'polygons', len(feats))


def bald_mask(T, seed=31):
    """0..1 grassy-bald cover: crest ground within radius_px of each bald centre and less than drop_m below
    its local summit, with noise-wobbled edges (tongues of grass down the spurs, forest up the hollows)."""
    out = np.zeros((H, W), np.float32)
    cfg = load_json(path('data/manual/terrain.json')).get('balds', {}).get('items', [])
    if not cfg:
        return out
    rng = np.random.default_rng(seed)
    n = cv2.resize(rng.standard_normal((H // 8 + 2, W // 8 + 2)).astype(np.float32), None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)[:H, :W]
    n += 0.3 * cv2.resize(rng.standard_normal((H // 3 + 2, W // 3 + 2)).astype(np.float32), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)[:H, :W]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for b in cfg:
        cx, cy = b['center']; r = float(b['radius_px']); drop = float(b['drop_m'])
        d = np.hypot(xx - cx, yy - cy)
        near = d < r * 1.3
        top = float(T[near].max())
        below = top - T
        m = (1 - np.clip((below - drop * (1 + 0.25 * n)) / 4.0, 0, 1)) * (1 - np.clip((d - r * (1 + 0.15 * n)) / 6.0, 0, 1))
        out = np.maximum(out, np.where(near, m, 0))
    return np.clip(cv2.GaussianBlur(out, (0, 0), 1.0), 0, 1)


def majority(cls, r=2, n=12):
    k = 2 * r + 1
    best = np.zeros(cls.shape, np.float32)
    out = cls.copy()
    for i in range(n):
        m = cv2.boxFilter((cls == i).astype(np.float32), -1, (k, k))
        sel = m > best
        out[sel] = i
        best[sel] = m[sel]
    return out


if __name__ == '__main__':
    main()
