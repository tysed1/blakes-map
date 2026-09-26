"""A3 validation cameras (environment art): driver eye 1.2 m, pedestrian 1.7 m, low cinematic 5 m,
medium aerial 100 m, high 1000 m - in forest, along roads, at the river and in fields.
Positions are derived from the world data so they stay valid when terrain / roads change.

    blender -b exports/a3/world.blend --python tools/blender/a3_testcams.py --python tools/blender/render.py -- \
        --cams TC_road_driver,TC_field_ped --res 960x540 --samples 16 --outdir exports/a3/renders/tc ...
Cameras: TC_road_driver (1.2 m), TC_hwy_low (5 m), TC_field_ped (1.7 m), TC_field_low (5 m), TC_forest_ped (1.7 m),
TC_river_low (5 m), TC_aerial_100, TC_high_1000.
"""
import bpy, os, sys, json, math
import numpy as np
from mathutils import Vector
ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
def ARG(k, d=None):
    return ARGS[ARGS.index(k) + 1] if k in ARGS else d
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
H, W = 667, 2000
T = np.fromfile(os.path.join(ROOT, 'data/terrain/height_graded_f32.bin'), np.float32).reshape(H, W)


def hgt(x, y):
    x = min(max(x - 0.5, 0), W - 1.001); y = min(max(y - 0.5, 0), H - 1.001)
    x0, y0 = int(x), int(y); fx, fy = x - x0, y - y0
    return float(T[y0, x0] * (1 - fx) * (1 - fy) + T[y0, x0 + 1] * fx * (1 - fy) + T[y0 + 1, x0] * (1 - fx) * fy + T[y0 + 1, x0 + 1] * fx * fy)


def b(x, y, z):
    return Vector(((x - 1000) * 2.5, -(y - 333.5) * 2.5, z))


def feature(path_, key, val):
    for f in json.load(open(os.path.join(ROOT, path_)))['features']:
        if f['properties'].get(key) == val:
            return np.asarray(f['geometry']['coordinates'], float)
    return None


def along(c, t):
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(c[:, :2], axis=0).T))]
    L = s[-1] * t
    i = min(max(np.searchsorted(s, L) - 1, 0), len(c) - 2)
    f = (L - s[i]) / max(s[i + 1] - s[i], 1e-6)
    p = c[i, :2] + (c[i + 1, :2] - c[i, :2]) * f
    d = c[i + 1, :2] - c[i, :2]
    return p, d / max(np.hypot(*d), 1e-9)


def road_z(c, p):
    if c.shape[1] > 2:
        i = np.hypot(c[:, 0] - p[0], c[:, 1] - p[1]).argmin()
        return float(c[i, 2])
    return hgt(*p)


def make(name, loc, look, lens=35):
    cd = bpy.data.cameras.get(name) or bpy.data.cameras.new(name)
    cd.lens = lens; cd.clip_start = 0.1; cd.clip_end = 40000
    ob = bpy.data.objects.get(name) or bpy.data.objects.new(name, cd)
    if ob.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (look - loc).to_track_quat('-Z', 'Y').to_euler()
    return ob


def cams():
    out = {}
    # rural road through forest, driver eye 1.2 m, looking along the road (longest rural road near Hollow Ridge)
    best = None
    for f in json.load(open(os.path.join(ROOT, 'data/roads/roads.geojson')))['features']:
        if f['properties'].get('type') != 'rural' or f['properties'].get('virtual'):
            continue
        c = np.asarray(f['geometry']['coordinates'], float)
        L = np.hypot(*np.diff(c[:, :2], axis=0).T).sum()
        if L < 120:
            continue
        m = c[len(c) // 2]
        sc_ = np.hypot(m[0] - 960, m[1] - 430) - L * 0.3
        if best is None or sc_ < best[0]:
            best = (sc_, c)
    rd = best[1] if best else None
    if rd is not None:
        p, d = along(rd, 0.45)
        z = road_z(rd, p)
        lp = b(p[0] - d[1] * 0.6, p[1] + d[0] * 0.6, z + 1.2)  # right lane
        out['road_driver'] = make('TC_road_driver', lp, b(p[0] + d[0] * 40, p[1] + d[1] * 40, z + 1.0), 28)
    # US 19 highway, low cinematic 5 m over the shoulder
    hw = feature('data/roads/roads.geojson', 'id', 'HR_RD_0308')
    if hw is not None:
        p, d = along(hw, 0.5)
        z = road_z(hw, p)
        out['hwy_low'] = make('TC_hwy_low', b(p[0] + d[1] * 6, p[1] - d[0] * 6, z + 5), b(p[0] + d[0] * 60, p[1] + d[1] * 60, z + 2), 28)
    # pasture / hay field, pedestrian 1.7 m looking toward the forest edge
    eco = np.fromfile(os.path.join(ROOT, 'public/world/eco_u8.bin'), np.uint8).reshape(H, W, -1)
    ft = eco[..., 3]
    for key, types, z0 in (('field_ped', (1, 2), 1.7), ('field_low', (4, 1), 5.0)):
        ys, xs = np.nonzero(np.isin(ft, types))
        if len(xs):
            dd = np.hypot(xs - 1000, ys - 330) + (0 if key == 'field_ped' else 300) * (xs < 1000)
            i = dd.argmin()
            x, y = float(xs[i]) + 0.5, float(ys[i]) + 0.5
            # look toward the nearest forest
            cls = np.fromfile(os.path.join(ROOT, 'public/world/landuse_u8.bin'), np.uint8).reshape(H, W)
            fy, fx = np.nonzero(np.isin(cls, [2, 3]))
            j = np.hypot(fx - x, fy - y).argmin()
            dx, dy = fx[j] - x, fy[j] - y; n = max(math.hypot(dx, dy), 1)
            cx, cy = x - dx / n * 12, y - dy / n * 12
            out[key] = make('TC_' + key, b(cx, cy, hgt(cx, cy) + z0), b(x + dx / n * 60, y + dy / n * 60, hgt(x + dx / n * 60, y + dy / n * 60) + z0 + 2), 30)
    # forest interior, pedestrian (a moist hollow if possible)
    cov = eco[..., 2].astype(float); tpi = eco[..., 6].astype(float)
    gy, gx = np.gradient(T, 2.5); slope = np.hypot(gx, gy)
    sc = cov - np.abs(tpi - 135) * 0.3 - slope * 400   # mid-slope hardwoods (not a hemlock hollow)
    if eco.shape[2] > 9:
        sc -= (eco[..., 8] < 250) * 500 + (eco[..., 9] < 250) * 500   # away from roads / rail
    sc[:, :800] = -1e9; sc[:, 1300:] = -1e9; sc[:200] = -1e9; sc[500:] = -1e9
    # candidate spots, best first; reject any within 5 m of a tree / shrub instance (camera inside a crown)
    vf = os.path.join(ROOT, 'public/world/vegetation_f32.bin')
    V = np.fromfile(vf, '<f4').reshape(-1, 6) if os.path.exists(vf) else np.zeros((0, 6), np.float32)
    order = np.argsort(sc.ravel())[::-1][:4000]
    y, x = np.unravel_index(order[0], sc.shape)
    CONI = {11, 12, 13, 14, 15, 16, 19, 20, 25, 26}   # conifers + rhododendron / laurel (view blockers)
    for k in order:
        yy, xx = np.unravel_index(k, sc.shape)
        if not len(V):
            y, x = yy, xx; break
        d = np.hypot(V[:, 0] - xx - 0.5, V[:, 1] - yy - 0.5) * 2.5
        near = d < 25
        if d.min() > 6.0 and np.isin(V[near, 4].astype(int), list(CONI)).mean() < 0.15:
            y, x = yy, xx
            break
    out['forest_ped'] = make('TC_forest_ped', b(x, y, hgt(x, y) + 1.7), b(x + 20, y - 8, hgt(x + 20, y - 8) + 4), 26)
    # river: over the channel ~45 m downstream of the Depot Street bridge, 4 m above the water, looking
    # upstream at the bridge (bridge, rapids, clear water, dense banks - the key reference motif)
    brs = json.load(open(os.path.join(ROOT, 'data/roads/bridges.geojson')))['features']
    br = next((f for f in brs if f['properties']['id'].startswith('HR_RD_0318')), None) or \
        next((f for f in brs if f['properties'].get('waterway_class') == 'river'), None)
    if br is not None:
        bc = np.asarray(br['geometry']['coordinates'], float)[:, :2].mean(0)
        riv, bd = None, 1e9
        for f in json.load(open(os.path.join(ROOT, 'data/water/waterways.geojson')))['features']:
            c = np.asarray(f['geometry']['coordinates'], float)
            dd = np.hypot(c[:, 0] - bc[0], c[:, 1] - bc[1])
            if dd.min() < bd:
                bd, riv, i0 = dd.min(), c, int(dd.argmin())
        s_ = np.r_[0, np.cumsum(np.hypot(*np.diff(riv[:, :2], axis=0).T))]
        j = int(np.searchsorted(s_, s_[i0] + 18))   # 18 px = 45 m downstream
        j = min(j, len(riv) - 1)
        p = riv[j, :2]
        wl = np.fromfile(os.path.join(ROOT, 'data/terrain/water_level_f32.bin'), np.float32).reshape(H, W)
        zw = float(np.nan_to_num(wl[int(p[1]), int(p[0])], nan=hgt(*p)))
        out['river_low'] = make('TC_river_low', b(p[0], p[1], zw + 4.0), b(bc[0], bc[1], zw + 3.0), 26)
    # medium aerial 100 m over Hollow Ridge valley, high map view 1000 m
    out['aerial_100'] = make('TC_aerial_100', b(1000, 420, hgt(1000, 420) + 100), b(1060, 350, hgt(1060, 350)), 30)
    out['high_1000'] = make('TC_high_1000', b(1000, 700, 1400), b(1000, 250, 300), 30)
    return out


C = cams()
print('A3 test cameras:', ', '.join(o.name for o in C.values()))
