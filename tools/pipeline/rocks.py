"""Rock placement (E2): crags, cliff bands, talus, scree fans and road rock cuts from the terrain.

    python3 tools/pipeline/rocks.py        (after terrain / grading / roads; reads N's walls.geojson)

Instances of the rock kit (tools/blender/lib_rocks.py, web meshes: tools/blender/export_web_rocks.py):
  * cliff bands: ledge blocks where rock exposure and slope are high (crest crags, spur noses, gorge
    walls), faces turned downslope, rows following the contour, pitched a little with the slope
  * crags / tors: blocky outcrops on crests and knobs with exposed rock
  * talus: boulders on the apron below the crags (decaying with distance from the rock)
  * scree fans: small angular stones streaming downslope from the cliff bands
  * road rock cuts: stacked ledge blocks along N's rock_cut walls (data/roads/walls.geojson), faces to the road
Clear of pavement / shoulders, rail ballast, bridge decks and open water.

Writes public/world/rocks_f32.bin  float32 [x_px, y_px, z_m, scale, kind, variant, yaw, pitch]
(kind = lib_rocks.KINDS index; yaw about +Y so the block's exposed face (-y in Blender = +z web) points
along the downslope / toward the road; pitch = forward tilt with the slope) and rocks.json (counts).
"""
import os, sys, json, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import cv2
from scipy import ndimage as ndi
from PIL import Image
from tools.lib.common import path, load_json, W, H
from tools.pipeline.vegetation import corridor_edge_distance, clear_zone_distance, fbm, smoothstep, crag_score, MPP, WATER

KINDS = ['cliff_block', 'crag', 'boulder', 'scree']   # tools/blender/lib_rocks.KINDS
DIMS = {'cliff_block': (6.0, 3.0, 4.2), 'crag': (4.0, 3.4, 3.2), 'boulder': (1.8, 1.5, 1.2), 'scree': (0.55, 0.45, 0.32)}   # lib_rocks.DIMS
VARIANTS = 4
STRIDE = 8


def jitter(cell, rng, prob):
    gx, gy = np.meshgrid(np.arange(0, W, cell), np.arange(0, H, cell))
    x = (gx + rng.random(gx.shape) * cell).ravel(); y = (gy + rng.random(gy.shape) * cell).ravel()
    xi, yi = np.clip(x.astype(int), 0, W - 1), np.clip(y.astype(int), 0, H - 1)
    keep = rng.random(len(x)) < prob[yi, xi]
    return x[keep], y[keep], xi[keep], yi[keep]


def bilinear(a, x, y):
    x = np.clip(np.asarray(x, float) - 0.5, 0, a.shape[1] - 1.001); y = np.clip(np.asarray(y, float) - 0.5, 0, a.shape[0] - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); fx = x - x0; fy = y - y0
    return a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy) + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy


def main(seed=23):
    rng = np.random.default_rng(seed)
    T = np.fromfile(path('data/terrain/height_graded_f32.bin'), np.float32).reshape(H, W)
    cls = np.array(Image.open(path('data/landuse/landuse_classes.png')))
    rexp = np.asarray(Image.open(path('data/terrain/rock_exposure_u8.png')), np.float32) / 255
    wet = cls == WATER
    dwater = ndi.distance_transform_edt(~wet) * MPP
    Ts = cv2.GaussianBlur(T, (0, 0), 1.0)
    gy, gx = np.gradient(Ts, MPP)
    slope = np.hypot(gx, gy)
    tpi = np.clip((T - cv2.GaussianBlur(T, (0, 0), 8)) / 6, -1.5, 1.5)
    # corridors: nothing on pavement / shoulders / ballast / decks (2x res distance to the corridor edge)
    rt = load_json(path('data/roads/road_types.json'))['types']
    def road_hw(f):
        p = f['properties']
        if p.get('virtual'):
            return None
        return (p.get('width_m') or rt[p['type']]['width_m']) / 2 + (p.get('shoulder_m') or 0), 0, 0
    roads = load_json(path('data/roads/roads.geojson'))['features']
    road_d, _, _ = corridor_edge_distance(roads, road_hw)
    rail_d, _, _ = corridor_edge_distance(load_json(path('data/railways/railways.geojson'))['features'], lambda f: (f['properties']['width_m'] / 2, 0, 0))
    br = [f for f in load_json(path('data/roads/bridges.geojson'))['features'] if f['geometry']['coordinates']]
    bridge_d, _, _ = corridor_edge_distance(br, lambda f: ((f['properties'].get('deck_width_m') or 10) / 2, 0, 0))

    cz_d = clear_zone_distance()   # N's interchange infields etc.

    def s2(a, x, y):
        return a[np.clip((y * 2).astype(int), 0, H * 2 - 1), np.clip((x * 2).astype(int), 0, W * 2 - 1)]

    def clear(x, y, margin):
        xi, yi = np.clip(x.astype(int), 0, W - 1), np.clip(y.astype(int), 0, H - 1)
        return (s2(road_d, x, y) > 2.5 + margin) & (s2(rail_d, x, y) > 3 + margin) & (s2(bridge_d, x, y) > 4 + margin) & (dwater[yi, xi] > 1.5) & ~wet[yi, xi] & (s2(cz_d, x, y) > margin)

    out = []

    def emit(x, y, kind, scale, yaw, pitch, sink):
        n = len(x)
        if not n:
            return
        z = bilinear(T, x, y) - sink
        var = rng.integers(0, VARIANTS, n)
        out.append(np.stack([x, y, z, scale, np.full(n, KINDS.index(kind)), var, yaw, pitch], 1))

    def downslope_yaw(xi, yi):
        # web: X = east (+px x), Z = south (+px y); downslope = -gradient. yaw so local +Z (the exposed face) faces it
        dx, dz = -gx[yi, xi], -gy[yi, xi]
        return np.arctan2(dx, dz)

    n12 = fbm((H, W), 5, seed + 1, 2)
    # ---- cliff bands: high exposure on steep ground; bands (not polka dots): stronger along the contour
    crag = crag_score(rexp, slope, dwater, T) * np.clip(0.6 + 0.5 * n12, 0, 1)
    x, y, xi, yi = jitter(1.7, rng, np.clip(crag * 1.1, 0, 1).astype(np.float32))
    ok = clear(x, y, 7); x, y, xi, yi = x[ok], y[ok], xi[ok], yi[ok]   # road cuts get their own courses below
    sc = np.exp(rng.normal(0, 0.2, len(x))) * (0.9 + 0.8 * rexp[yi, xi])   # big ledges read from the valley
    pitch = np.clip(np.arctan(slope[yi, xi]) * 0.35, 0, 0.45) + rng.normal(0, 0.05, len(x))
    # the steeper the ground, the deeper the block is set (its downhill base must never show)
    sink = sc * (0.9 + 1.4 * np.clip(slope[yi, xi] - 0.7, 0, 1.5))
    emit(x, y, 'cliff_block', sc, downslope_yaw(xi, yi) + rng.normal(0, 0.18, len(x)), pitch, sink)
    ncliff = len(x)
    # ---- crags / tors on crests and knobs with exposed rock (moderate slope)
    tor = smoothstep(0.3, 0.7, rexp) * smoothstep(0.1, 0.5, tpi) * smoothstep(0.15, 0.4, slope) * (1 - smoothstep(0.9, 1.3, slope))
    x, y, xi, yi = jitter(3.5, rng, np.clip(tor * 0.35, 0, 1).astype(np.float32))
    ok = clear(x, y, 3); x, y, xi, yi = x[ok], y[ok], xi[ok], yi[ok]
    sc = np.exp(rng.normal(0, 0.25, len(x)))
    emit(x, y, 'crag', sc, rng.uniform(0, 2 * math.pi, len(x)), rng.normal(0, 0.08, len(x)), 0.7 * sc)
    # ---- talus boulders below the crags + scree fans downslope of cliff bands
    rock = rexp > 0.6
    drock = ndi.distance_transform_edt(~rock) * MPP
    # 'below' the rock: terrain lower than the nearest rock cell
    _, (iy, ix) = ndi.distance_transform_edt(~rock, return_indices=True)
    below = np.clip((T[iy, ix] - T) / 8, 0, 1)
    tal = np.exp(-drock / 18) * below * smoothstep(0.25, 0.6, slope) * ~rock
    x, y, xi, yi = jitter(2.6, rng, np.clip(tal * 0.35, 0, 1).astype(np.float32))
    ok = clear(x, y, 1); x, y, xi, yi = x[ok], y[ok], xi[ok], yi[ok]
    sc = np.exp(rng.normal(0, 0.35, len(x))) * (0.6 + 0.6 * np.exp(-drock[yi, xi] / 15))
    emit(x, y, 'boulder', sc, rng.uniform(0, 2 * math.pi, len(x)), rng.normal(0, 0.15, len(x)), 0.35 * sc)
    fan = np.exp(-drock / 30) * below * smoothstep(0.35, 0.8, slope) * (1 - smoothstep(0.6, 0.9, rexp)) * np.clip(0.5 + 0.7 * n12, 0, 1)
    x, y, xi, yi = jitter(1.3, rng, np.clip(fan * 0.45, 0, 1).astype(np.float32))
    ok = clear(x, y, 0.5); x, y, xi, yi = x[ok], y[ok], xi[ok], yi[ok]
    sc = np.exp(rng.normal(0, 0.35, len(x)))
    emit(x, y, 'scree', sc, rng.uniform(0, 2 * math.pi, len(x)), rng.normal(0, 0.3, len(x)), 0.12 * sc)
    # ---- road rock cuts (N's walls): stacked ledge blocks along the cut, faces toward the road
    rdict = {f['properties']['id']: np.asarray(f['geometry']['coordinates'])[:, :2] for f in roads}
    ncut = 0
    wp = path('data/roads/walls.geojson')
    for f in (load_json(wp)['features'] if os.path.exists(wp) else []):
        p = f['properties']
        if p.get('kind') != 'rock_cut':
            continue
        c = np.asarray(f['geometry']['coordinates'], float)[:, :2]
        top = np.asarray(p.get('top_z_m') or [], float)
        base = float(p.get('base_z_m') or 0)
        rc = rdict.get(p.get('road'))
        seg = np.hypot(*np.diff(c, axis=0).T); L = np.r_[0, np.cumsum(seg)]
        if L[-1] < 1:
            continue
        step_px = 4.2 / MPP
        for s in np.arange(step_px * 0.5, L[-1], step_px * rng.uniform(0.85, 1.1)):
            k = min(np.searchsorted(L, s) - 1, len(c) - 2); k = max(k, 0)
            t = (s - L[k]) / max(seg[k], 1e-6)
            px = c[k] + (c[k + 1] - c[k]) * t
            tang = (c[k + 1] - c[k]) / max(seg[k], 1e-6); nrm = np.array([-tang[1], tang[0]])
            if rc is not None:  # face toward the road centreline
                q = rc[np.argmin(np.hypot(*(rc - px).T))]
                if np.dot(q - px, nrm) < 0:
                    nrm = -nrm
            topz = np.interp(s, L, top) if len(top) == len(c) else base + p['height_m']
            hgt = max(topz - base, 1.5)
            nst = max(1, int(math.ceil(hgt / 3.4)))
            for j in range(nst):
                sc_ = rng.uniform(0.8, 1.05)
                back = 1.5 * sc_ - 0.4 + 0.35 * j   # front face ~0.4 m proud of the cut line; courses step back
                qx, qy = px - nrm * back / MPP
                if s2(cz_d, np.array([qx]), np.array([qy]))[0] <= 0:
                    continue
                z = base - 0.6 + j * (hgt / nst)
                yaw = math.atan2(nrm[0], nrm[1]) + rng.normal(0, 0.08)
                out.append(np.array([[qx, qy, z, sc_, KINDS.index('cliff_block'), rng.integers(0, VARIANTS), yaw, rng.normal(0.08, 0.04)]]))
                ncut += 1
    # ---- waterfall ledges (waterways.json 'falls'; terrain.py shape_falls cut the step): broken cliff faces
    # across both banks at the lip and every cascade tier, faces downstream, tops ragged around the upper pool
    # level, stacked down to the lower water; kept off the water (N's sheets / pool). variant + 10 = wet rock.
    WLv = np.fromfile(path('data/terrain/water_level_f32.bin'), np.float32).reshape(H, W)
    nfall = 0
    for rv in load_json(path('data/manual/waterways.json'))['rivers']:
        for fl in rv.get('falls', []):
            lip, toe = np.asarray(fl['lip'], float), np.asarray(fl['toe'], float)
            flow = (toe - lip) / max(np.hypot(*(toe - lip)), 1e-6); nrm = np.array([-flow[1], flow[0]])
            nt = max(1, int(fl.get('tiers', 1)))
            half = fl.get('lip_width_m', 10.0) / 2
            def wl(p):
                xi, yi = int(np.clip(p[0], 0, W - 1)), int(np.clip(p[1], 0, H - 1))
                v = WLv[max(yi - 3, 0):yi + 4, max(xi - 3, 0):xi + 4]
                return float(np.nanmax(v)) if np.isfinite(v).any() else float(T[yi, xi])
            for ti in range(nt):
                c = lip + (toe - lip) * (ti / max(nt, 1) if nt > 1 else 0.0)
                up = wl(c - flow * 2.5); dn = wl(c + flow * (2.5 if nt == 1 else 1.5))
                for sd in (-1, 1):
                    for o in np.arange(half + 3.2, half + 16.0, 3.3):
                        q = c + nrm * sd * o / MPP + flow * rng.normal(0, 0.25)
                        qi = (int(np.clip(q[1], 0, H - 1)), int(np.clip(q[0], 0, W - 1)))
                        if dwater[qi] < 2.0 or wet[qi]:
                            continue
                        rise = 1 - (o - half) / 16.0
                        top = up + 0.4 + rng.uniform(-0.9, 1.3) * (0.4 + 0.6 * rise)
                        low = min(dn - 0.8, float(T[qi]) - 0.5)
                        sc = rng.uniform(0.7, 1.0)
                        hgt = max(top - low, 1.0); ncs = max(1, int(math.ceil(hgt / (3.4 * sc))))
                        yaw = math.atan2(flow[0], flow[1]) + rng.normal(0, 0.12)
                        for j in range(ncs):
                            z = low + j * hgt / ncs
                            var = rng.integers(0, VARIANTS) + (10 if o < half + 6 else 0)
                            out.append(np.array([[q[0], q[1], z, sc, KINDS.index('cliff_block'), var, yaw, rng.normal(0.05, 0.05)]]))
                            ncut += 1; nfall += 1
    R = np.concatenate(out).astype(np.float32) if out else np.zeros((0, STRIDE), np.float32)
    # ---- QA: no floating bases, no rock on pavement / shoulders
    #  base (local z = 0) must sit at or below the ground at 8 points around the footprint: lower it
    #  where the downhill side would show (the block is then simply set deeper into the slope)
    dims = np.array([DIMS[k] for k in KINDS], np.float32)
    kx = R[:, 4].astype(int)
    hx = dims[kx, 0] * R[:, 3] * 0.42; hz = dims[kx, 1] * R[:, 3] * 0.42
    gmin = bilinear(T, R[:, 0], R[:, 1])
    cy, sy = np.cos(R[:, 6]), np.sin(R[:, 6])
    for ax, az in ((1, 0), (-1, 0), (0, 1), (0, -1), (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)):
        lx, lz = ax * hx, az * hz                      # local (x along the face, z out of the face), metres
        wx, wz = lx * cy + lz * sy, -lx * sy + lz * cy  # yaw about +Y (web X east, Z south)
        gmin = np.minimum(gmin, bilinear(T, R[:, 0] + wx / MPP, R[:, 1] + wz / MPP))
    # road-cut courses (the last ncut rows) stand against the cut face on purpose: exempt from lowering
    cut = np.zeros(len(R), bool); cut[len(R) - ncut:] = ncut > 0
    low = (R[:, 2] > gmin - 0.05) & ~cut
    lowered = int(low.sum())
    R[low, 2] = gmin[low] - 0.05
    rd = s2(road_d, R[:, 0], R[:, 1])
    on_road = np.where(cut, rd < -0.5, rd < 0.5)
    R = R[~on_road]
    print(f'  rocks QA: {lowered} bases lowered to the ground, {int(on_road.sum())} dropped on pavement/shoulder; '
          f'min road-edge clearance {float(s2(road_d, R[:, 0], R[:, 1]).min()):.2f} m')
    os.makedirs(path('public/world'), exist_ok=True)
    R.astype('<f4').tofile(path('public/world/rocks_f32.bin'))
    cnt = {k: int((R[:, 4] == i).sum()) for i, k in enumerate(KINDS)}
    json.dump({'file': 'rocks_f32.bin', 'stride': STRIDE, 'fields': ['x_px', 'y_px', 'z_m', 'scale', 'kind', 'variant', 'yaw', 'pitch'],
               'kinds': KINDS, 'count': int(len(R)), 'counts': cnt, 'road_cut_blocks': ncut}, open(path('public/world/rocks.json'), 'w'), indent=1)
    print(f'rocks: {len(R)} ({cnt}, road-cut + falls blocks {ncut} (falls {nfall}), cliff-band blocks {ncliff})')


if __name__ == '__main__':
    main()
