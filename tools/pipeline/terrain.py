"""Terrain reconstruction v4: designed landforms (heightfield on the source-pixel grid, 2.5 m).

The large forms are DESIGNED, not noise:
  valley floor : harmonic interpolation of monotonic (rapids-aware) river-surface profiles
  ridges       : hand-authored ridge skeleton (data/manual/ridges.json): crest polylines with
                 absolute crest heights, per-ridge base width/asymmetry, automatic side spurs,
                 knobs and saddles/passes. Ridges are unioned with a smooth max.
  lowland      : water, farmland, towns (from the map) cap the relief with a concave toe slope
  corridors    : roads and rail get natural gaps (grade-envelope cap) so grading needs little cut
  detail       : subtle warped noise, crest crags, then drainage-driven erosion
                 (stream-power incision on a priority-flood drainage tree -> dendritic hollows,
                 droplet erosion for gullies/fans, thermal relaxation)
  channels     : asymmetric river cross-sections (cut banks on outside bends, slip-off slopes and
                 gravel bars inside), pools at bends, riffles at crossings, rocky rapids

Outputs (data/terrain/):
  height_f32.bin       float32 [667][2000] metres (row = image y)
  height_u16.png       16-bit normalized (see terrain.json for min/max)
  water_level_f32.bin  float32 water surface (NaN where dry)
  water_fx_u8.bin      uint8 [5][667][2000]: depth, surface slope, flow direction, bank type, whitewater (see terrain.json)
  rock_exposure_u8.png terrain-derived rock exposure (crest crags, cut banks, cliffs) for materials/landuse
  ridge_skeleton.geojson  ridges + generated spurs actually used (debug / editing aid)
  terrain.json         metadata
  assets/maps/debug/hillshade.png
"""
import sys, os, math, heapq
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from shapely.geometry import Polygon
from PIL import Image

from tools.lib.common import path, load_json, save_json, W, H, rnd
from tools.lib.features import hsv, rock_density
from tools.lib.geom import rasterize_polys, geojson_line, fc
from tools.lib.trace import resample, catmull_rom

try:
    from numba import njit
except ImportError:  # slow fallback
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

MPP = 2.5
# deepest natural gap the terrain itself provides per route type; anything beyond is an engineered cut
# (grading) or a sign the alignment should change (switchbacks) - reported by grading, not hidden here
MAX_CUT = {'freeway': 40.0, 'highway': 40.0, 'ramp': 16.0, 'arterial': 20.0, 'main_street': 10.0, 'collector': 16.0,
           'urban_street': 6.0, 'residential': 6.0, 'rural': 14.0, 'gravel': 8.0, 'dirt': 5.0, 'driveway': 3.0, 'rail': 30.0}
MAX_GRADE = {'freeway': 0.06, 'highway': 0.08, 'ramp': 0.07, 'arterial': 0.08, 'main_street': 0.08, 'collector': 0.10,
             'urban_street': 0.12, 'residential': 0.12, 'rural': 0.12, 'gravel': 0.15, 'dirt': 0.18, 'driveway': 0.20, 'rail': 0.022}


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def bilinear(F, x, y):
    x = np.clip(np.asarray(x, float) - 0.5, 0, F.shape[1] - 1.001)
    y = np.clip(np.asarray(y, float) - 0.5, 0, F.shape[0] - 1.001)
    x0 = x.astype(int); y0 = y.astype(int); fx = x - x0; fy = y - y0
    return (F[y0, x0] * (1 - fx) * (1 - fy) + F[y0, x0 + 1] * fx * (1 - fy) + F[y0 + 1, x0] * (1 - fx) * fy + F[y0 + 1, x0 + 1] * fx * fy)


def noise1d(n, wavelength, seed):
    """Smooth 1-D noise in [-1, 1] sampled at n unit steps."""
    rng = np.random.default_rng(seed)
    k = max(4, int(n / max(wavelength, 1)) + 4)
    v = rng.uniform(-1, 1, k)
    x = np.linspace(1, k - 3, n)
    i = np.clip(x.astype(int), 1, k - 3); f = x - i
    p0, p1, p2, p3 = v[i - 1], v[i], v[i + 1], v[i + 2]
    return 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)


def fbm(seed, scales=(90, 45, 22, 11), amps=(1.0, 0.5, 0.25, 0.12), shape=(H, W)):
    rng = np.random.default_rng(seed)
    hh, ww = shape
    out = np.zeros(shape, np.float32)
    for s, a in zip(scales, amps):
        h, w = int(hh / s) + 3, int(ww / s) + 3
        n = rng.standard_normal((h, w)).astype(np.float32)
        out += a * cv2.resize(n, (ww + int(s * 2), hh + int(s * 2)), interpolation=cv2.INTER_CUBIC)[int(s):int(s) + hh, int(s):int(s) + ww]
    return out / sum(amps)


def warped_fbm(seed, scales, amps, warp_px=25.0, shape=(H, W)):
    wx = fbm(seed + 101, scales=(120, 60), amps=(1, 0.5), shape=shape) * warp_px
    wy = fbm(seed + 202, scales=(120, 60), amps=(1, 0.5), shape=shape) * warp_px
    base = fbm(seed, scales, amps, shape=shape)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    return cv2.remap(base, xx + wx, yy + wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ------------------------------------------------------------------ rivers
def river_profiles(cfg):
    """Monotonic water-surface profiles per channel. The drop is distributed by local 'steepness':
    painted white water (rapids) and authored steps concentrate the fall into riffles and rapids,
    bends stay flatter (pools)."""
    ww = load_json(path('data/water/waterways.geojson'))['features']
    man = {r['id']: r for r in load_json(path('data/manual/waterways.json'))['rivers']}
    feats = {f['properties']['id']: f for f in ww}
    lines = {k: resample(np.asarray(f['geometry']['coordinates'], float), 1.0) for k, f in feats.items()}
    rap = np.load(path('tools/.cache/rapids_mask.npy')).astype(np.float32)
    rapd = cv2.GaussianBlur(rap, (0, 0), 3.0)
    wcfg = cfg.get('water', {})
    rapid_gain = wcfg.get('rapid_gain', 7.0)
    prof, steep = {}, {}

    def elev_at(k, pt):
        L = lines[k]
        i = int(np.argmin(np.hypot(*(L - pt).T)))
        return prof[k][i]

    def curvature(L):
        if len(L) < 7:
            return np.zeros(len(L))
        Ls = np.stack([ndi.gaussian_filter1d(L[:, 0], 4, mode='nearest'), ndi.gaussian_filter1d(L[:, 1], 4, mode='nearest')], 1)
        d1 = np.gradient(Ls, axis=0); d2 = np.gradient(d1, axis=0)
        return (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(np.hypot(*d1.T) ** 3, 1e-6)

    pending = set(feats)
    for _ in range(30):
        for k in list(pending):
            p = feats[k]['properties']
            e0, e1 = p['elev_spec_m'] or [None, None]
            L = lines[k]
            parent = p['flows_into']
            if e1 is None:
                if parent is None or parent not in prof:
                    continue
                e1 = elev_at(parent, L[-1])
            if e0 is None:
                # a side channel leaving a parent inherits the level of the nearest channel at its head -
                # wait until that channel (whichever it is) has been resolved
                best = min((q for q in lines if q != k), key=lambda q: np.hypot(*(lines[q] - L[0]).T).min())
                if best not in prof:
                    continue
                e0 = elev_at(best, L[0])
                if p.get('backwater'):
                    e0 = max(e0, e1)
            e0 = max(e0, e1)
            n = len(L)
            t = np.linspace(0, 1, n)
            xi = np.clip(L[:, 0].astype(int), 0, W - 1); yi = np.clip(L[:, 1].astype(int), 0, H - 1)
            r = ndi.gaussian_filter1d(np.clip(rapd[yi, xi] * 3, 0, 1), 3)
            kap = np.abs(curvature(L))
            wgt = (1.0 + 0.9 * (1 - t)) * (1.0 + rapid_gain * r) * (1.25 - 0.6 * np.clip(kap / 0.03, 0, 1))
            for st in man.get(k, {}).get('steps', []):  # authored falls / rapids: [x, y, weight, radius_px]
                d = np.hypot(L[:, 0] - st[0], L[:, 1] - st[1])
                wgt += st[2] * np.exp(-(d / st[3]) ** 2)
            if e0 - e1 < 0.05:
                z = np.full(n, e1)
            else:
                # per-sample drops proportional to the steepness weights, capped (no water 'walls':
                # the steepest cascades still fall at most ~max_drop per 2.5 m) and redistributed
                total = e0 - e1
                wm = 0.5 * (wgt[1:] + wgt[:-1])
                drop = total * wm / wm.sum()
                cap = max(wcfg.get('max_drop_per_px', 0.45), 1.6 * total / max(n - 1, 1))
                for _ in range(20):
                    over = drop > cap
                    if not over.any():
                        break
                    ex = (drop[over] - cap).sum()
                    drop[over] = cap
                    room = ~over
                    drop[room] += ex * wm[room] / wm[room].sum()
                z = e0 - np.r_[0, np.cumsum(drop)]
            prof[k] = z
            steep[k] = np.abs(np.gradient(z)) / MPP  # m/m
            pending.discard(k)
        if not pending:
            break
    if pending:
        raise RuntimeError(f'unresolved river profiles: {pending}')
    return lines, prof, steep, feats


def channel_fields(lines, prof, steep, feats, wmask):
    """Per-pixel river frame from the nearest centreline sample: surface level, signed lateral offset,
    local curvature (bend side), steepness, flow direction, and channel id."""
    ids, xs, ys, zs, ss, kap, tx, ty, cls = [], [], [], [], [], [], [], [], []
    for n, (k, L) in enumerate(lines.items()):
        Ls = np.stack([ndi.gaussian_filter1d(L[:, 0], 5, mode='nearest'), ndi.gaussian_filter1d(L[:, 1], 5, mode='nearest')], 1) if len(L) > 8 else L
        d1 = np.gradient(Ls, axis=0); d2 = np.gradient(d1, axis=0)
        sp = np.maximum(np.hypot(*d1.T), 1e-6)
        kk = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / sp ** 3
        kk = ndi.gaussian_filter1d(kk, 6, mode='nearest')
        xs.append(L[:, 0]); ys.append(L[:, 1]); zs.append(prof[k]); ss.append(steep[k]); kap.append(kk)
        tx.append(d1[:, 0] / sp); ty.append(d1[:, 1] / sp); ids.append(np.full(len(L), n))
        cls.append(np.full(len(L), {'river': 0, 'creek': 1, 'slough': 2}.get(feats[k]['properties']['class'], 1)))
    X, Y, Z, S_, K, TX, TY, I, C = [np.concatenate(a) for a in (xs, ys, zs, ss, kap, tx, ty, ids, cls)]
    # lower channel wins where channels overlap (confluences): sort so that lower surfaces are preferred
    tree = cKDTree(np.stack([X, Y], 1))
    return dict(tree=tree, X=X, Y=Y, Z=Z, S=S_, K=K, TX=TX, TY=TY, I=I, C=C)


def water_surface(CF, wmask, lines=None, feats=None):
    """Continuous water surface: the centreline profiles are fixed and the level inside each water body is
    the harmonic interpolation between them (Laplace on the water pixels, no-flux at the banks). No
    Voronoi steps across bends or confluences; the fall is spread smoothly over riffles and rapids."""
    yy, xx = np.nonzero(wmask)
    n = len(yy)
    idx = -np.ones((H, W), np.int64)
    idx[yy, xx] = np.arange(n)
    # Dirichlet: pixels on a centreline (lowest profile wins at confluences)
    fixed_v = np.full(n, np.nan)
    ci = np.clip(CF['X'].astype(int), 0, W - 1); cj = np.clip(CF['Y'].astype(int), 0, H - 1)
    on = idx[cj, ci] >= 0
    if lines is not None:
        # inside a parent's channel a tributary's centreline is not a constraint (the mouth blends into the
        # parent surface instead of imposing its own levels across the confluence)
        dwi = ndi.distance_transform_edt(wmask)
        keys = list(lines.keys())
        start = np.r_[0, np.cumsum([len(lines[k]) for k in keys])]
        for n_, k in enumerate(keys):
            par = feats[k]['properties'].get('flows_into')
            if not par or par not in lines:
                continue
            Lp = lines[par]
            dd, jj = cKDTree(Lp).query(lines[k], k=1)
            hwp = dwi[np.clip(Lp[jj, 1].astype(int), 0, H - 1), np.clip(Lp[jj, 0].astype(int), 0, W - 1)]
            near = dd < hwp + 3.0
            on[start[n_]:start[n_ + 1]] &= ~near
    order = np.argsort(-CF['Z'][on])  # write high first so the lowest ends up stored
    k = idx[cj[on], ci[on]][order]
    fixed_v[k] = CF['Z'][on][order]
    fixed = np.isfinite(fixed_v)
    # initial guess / fallback: nearest centreline sample
    d, nn = CF['tree'].query(np.stack([xx + 0.5, yy + 0.5], 1), k=1)
    guess = CF['Z'][nn]
    rows, cols, vals = [], [], []
    b = np.zeros(n)
    deg = np.zeros(n)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        y2, x2 = yy + dy, xx + dx
        ok = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
        j = np.full(n, -1)
        j[ok] = idx[y2[ok], x2[ok]]
        ok = (j >= 0) & ~fixed
        deg += ok
        rows.append(np.nonzero(ok)[0]); cols.append(j[ok]); vals.append(-np.ones(ok.sum()))
    # rows for free cells: deg*u_i - sum(u_j) = 0 ; fixed cells: u_i = v
    free = ~fixed & (deg > 0)
    lone = ~fixed & (deg == 0)
    diag_v = np.where(free, deg, 1.0)
    rows = np.concatenate(rows + [np.arange(n)]); cols = np.concatenate(cols + [np.arange(n)]); vals = np.concatenate(vals + [diag_v])
    b[fixed] = fixed_v[fixed]
    b[lone] = guess[lone]
    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    # water bodies without any centreline pixel keep the nearest-profile level (flat ponds)
    lab, nl = ndi.label(wmask)
    has = np.zeros(nl + 1, bool)
    has[lab[yy[fixed], xx[fixed]]] = True
    orphan = ~has[lab[yy, xx]]
    if orphan.any():
        A = A.tolil()
        for i in np.nonzero(orphan)[0]:
            A.rows[i] = [i]; A.data[i] = [1.0]
        A = A.tocsr()
        b[orphan] = guess[orphan]
    u = spsolve(A, b)
    surf = np.full((H, W), np.nan, np.float32)
    surf[yy, xx] = u
    return surf


def harmonic_fill(values, known, scale=4):
    """Solve Laplace(u)=0 with Dirichlet at known cells, on a downscaled grid."""
    h, w = H // scale + 1, W // scale + 1
    v = cv2.resize(np.where(known, values, 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    k = cv2.resize(known.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    fixed = k > 0.2
    vv = np.where(fixed, v / np.maximum(k, 1e-6), 0)
    N = h * w
    idx = np.arange(N).reshape(h, w)
    rows, cols, vals = [], [], []
    b = np.zeros(N)
    fi = fixed.ravel()
    b[fi] = vv.ravel()[fi]
    rows += list(idx.ravel()[fi]); cols += list(idx.ravel()[fi]); vals += [1.0] * int(fi.sum())
    deg = np.zeros((h, w))
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ys, xs = np.mgrid[0:h, 0:w]
        yn, xn = ys + dy, xs + dx
        ok = (yn >= 0) & (yn < h) & (xn >= 0) & (xn < w) & ~fixed
        deg += ok
        rows += list(idx[ok]); cols += list(idx[yn[ok], xn[ok]]); vals += [-1.0] * int(ok.sum())
    nf = ~fixed
    rows += list(idx[nf]); cols += list(idx[nf]); vals += list(deg[nf])
    A = csr_matrix((vals, (rows, cols)), shape=(N, N))
    u = spsolve(A, b).reshape(h, w).astype(np.float32)
    return cv2.resize(u, (W, H), interpolation=cv2.INTER_CUBIC)


# ------------------------------------------------------------------ land
def lowland_mask(wmask):
    """Lowland = water + open farmland + developed land (dense road network) + towns, from the map.
    Smoothed so landforms are coherent. Road corridors are NOT lowland any more: roads get designed
    gaps/passes instead (see corridor_cap)."""
    Hh, S, V = hsv()
    forest = ((Hh >= 60) & (Hh <= 170) & (V < 0.42)) | (V < 0.25)
    fd = cv2.GaussianBlur(forest.astype(np.float32), (0, 0), 7)
    rock = rock_density()
    rr = np.zeros((H, W), np.uint8)
    for f in load_json(path('data/roads/roads.geojson'))['features']:
        if f['properties']['type'] in ('freeway', 'ramp'):
            continue
        cv2.polylines(rr, [np.asarray(f['geometry']['coordinates'])[:, :2].round().astype(np.int32)], False, 1, 3)
    dev = cv2.GaussianBlur(rr.astype(np.float32), (0, 0), 9)
    openland = (fd < 0.45) & (rock < 0.10)
    zones = [Polygon(z['pts']) for z in load_json(path('data/manual/zones.json'))['zones'] if z['kind'] not in ('city',)]
    zm = rasterize_polys(zones, (H, W)) > 0
    low = wmask | zm | openland | ((dev > 0.16) & (rock < 0.15))
    low = cv2.morphologyEx(low.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    low = cv2.morphologyEx(low, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    lab, n = ndi.label(low)
    sz = ndi.sum(low, lab, range(1, n + 1))
    low = np.isin(lab, 1 + np.where(sz >= 600)[0])
    lab, n = ndi.label(~low)
    sz = ndi.sum(~low, lab, range(1, n + 1))
    low |= np.isin(lab, 1 + np.where(sz < 600)[0])
    low |= wmask
    return low, fd, rock


def feather_field(spec, default):
    num = np.zeros((H, W), np.float32)
    den = np.zeros((H, W), np.float32)
    for rg in spec:
        m = rasterize_polys([Polygon(rg['pts'])], (H, W), value=255, ups=2).astype(np.float32) / 255
        f = rg.get('feather_px', 40)
        m = cv2.GaussianBlur(m, (0, 0), f / 2.5)
        num += m * rg['amplitude_m']
        den += m
    base = np.clip(1 - den, 0, 1)
    return (num + base * default) / np.maximum(den + base, 1e-6)


# ------------------------------------------------------------------ ridges
def ridge_profile(t, p=2.0, q=2.1):
    """Cross profile 1 (crest) -> 0 (valley floor at t=1): rounded crest, steep mid-slope, concave foot."""
    t = np.clip(t, 0, 1)
    return (1 - t ** p) ** q


def smax(a, b, k):
    """Smooth max that is exact where either field is ~0 (no seams at ridge feet)."""
    kk = k * np.clip(np.minimum(a, b) / (2 * k), 0, 1)
    h = np.maximum(kk - np.abs(a - b), 0) / np.maximum(kk, 1e-6)
    return np.maximum(a, b) + h * h * kk * 0.25


def smin(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - h * h * k * 0.25


class Ridge:
    def __init__(self, rid, dense, z, hw_l, hw_r, rock, kind):
        self.id, self.P, self.z, self.hw_l, self.hw_r, self.rock, self.kind = rid, dense, z, hw_l, hw_r, rock, kind
        d1 = np.gradient(dense, axis=0)
        self.T = d1 / np.maximum(np.hypot(*d1.T), 1e-9)[:, None]


def build_ridges(cfg, V, rng):
    """Ridges from data/manual/ridges.json: dense crest polylines with crest height (relief-scaled over the
    valley floor), knobs/sags, saddles, per-side base width and flank lobes (spurs & hollows)."""
    R = load_json(path('data/manual/ridges.json'))
    D = R['defaults']
    rcfg = cfg['ridges']
    rs, hs = rcfg.get('relief_scale', 1.0), rcfg.get('hw_scale', 1.0)
    saddles = R.get('saddles', [])
    out = []
    for n, r in enumerate(R['ridges']):
        pts = np.asarray(r['pts'], float)
        dense = catmull_rom(pts[:, :2], spacing=1.0)
        L = np.r_[0, np.cumsum(np.hypot(*np.diff(dense, axis=0).T))]
        vs = [L[int(np.argmin(np.hypot(*(dense - p[:2]).T)))] for p in pts]
        vs = np.maximum.accumulate(np.asarray(vs))
        zc = np.interp(L, vs, pts[:, 2])
        vb = bilinear(V, dense[:, 0], dense[:, 1])
        zc = vb + (zc - vb) * rs
        kn = r.get('knob_m', D['knob_m']) * noise1d(len(L), r.get('knob_wavelength_px', D['knob_wavelength_px']), 1000 + n)
        endf = np.clip(np.minimum(L, L[-1] - L) / 30.0, 0, 1)
        zc = zc + kn * endf
        for sd in saddles:
            d = np.hypot(dense[:, 0] - sd['at'][0], dense[:, 1] - sd['at'][1])
            f = np.clip(d / sd['radius_px'], 0, 1)
            f = f * f * (3 - 2 * f)
            zc = np.where(zc > sd['elev_m'], sd['elev_m'] + (zc - sd['elev_m']) * f, zc)
        hw = r.get('hw', D['hw']) * hs
        asym = r.get('asym', D['asym'])
        wv = 1 + 0.15 * noise1d(len(L), 60, 2000 + n)
        rid = Ridge(r['id'], dense, zc, hw * (1 + asym) * wv, hw * (1 - asym) * wv, r.get('rock', D['rock']), r.get('kind', 'main'))
        # flank lobes: spurs (wider flank) alternate with hollows (narrower flank) along each side
        sp = r.get('spurs') or D.get('spurs') or {}
        wl = sp.get('spacing', 40) * rcfg.get('spur_spacing_scale', 1.0)
        amp = rcfg.get('lobe_amp', 0.38) * sp.get('len', 0.6 if r.get('kind') == 'hill' else 0.9)
        side = sp.get('side', 'both')
        def lobes(seed):
            x = noise1d(len(L), wl / 2.2, seed)
            return np.sign(x) * np.abs(x) ** 0.7
        rid.sw_l = lobes(3000 + n) * (amp if side in ('both', 'left') else amp * 0.4)
        rid.sw_r = lobes(4000 + n) * (amp if side in ('both', 'right') else amp * 0.4)
        rid.shear_l, rid.shear_r = rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35)
        out.append(rid)
    return out


def ridge_field(ridges, V, k=10.0, warp=None):
    """Relief above the valley floor from the ridge skeleton (smooth union) + crest-proximity (rock) map."""
    F = np.zeros((H, W), np.float32)
    crest = np.zeros((H, W), np.float32)
    for rd in ridges:
        hwmax = float(max(rd.hw_l.max(), rd.hw_r.max())) * (1 + max(np.abs(rd.sw_l).max(), np.abs(rd.sw_r).max()))
        pad = hwmax + 30  # + domain warp + attribute blur margins: the field must reach 0 inside the box
        x0 = int(max(0, np.floor(rd.P[:, 0].min() - pad))); x1 = int(min(W, np.ceil(rd.P[:, 0].max() + pad)))
        y0 = int(max(0, np.floor(rd.P[:, 1].min() - pad))); y1 = int(min(H, np.ceil(rd.P[:, 1].max() + pad)))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        q = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], 1)
        if warp is not None:
            q = q + np.stack([warp[0][yy, xx].ravel(), warp[1][yy, xx].ravel()], 1)
        d, i = cKDTree(rd.P).query(q, k=1, distance_upper_bound=hwmax * 1.15)
        ok = np.isfinite(d)
        i = np.where(ok, i, 0)
        d = np.where(ok, d, hwmax * 2)
        rel_v = q - rd.P[i]
        cross = rd.T[i, 0] * rel_v[:, 1] - rd.T[i, 1] * rel_v[:, 0]
        left = cross < 0  # left of travel (image y down)
        hw = np.where(left, rd.hw_l[i], rd.hw_r[i])
        n = len(rd.P)
        ish = np.clip(i + np.where(left, rd.shear_l, rd.shear_r) * d, 0, n - 1).astype(int)
        sw = np.where(left, rd.sw_l[ish], rd.sw_r[ish])
        # attribute fields extended smoothly off the crest (the nearest-sample index jumps on the concave
        # side of bends and past the ends; blurring avoids crease lines)
        sh = yy.shape
        okm = ok.reshape(sh).astype(np.float32)
        wsum = cv2.GaussianBlur(okm, (0, 0), 5.0) + 1e-6
        blur = lambda a, sg=5.0: cv2.GaussianBlur((a * ok).reshape(sh).astype(np.float32), (0, 0), sg) / (cv2.GaussianBlur(okm, (0, 0), sg) + 1e-6)
        Zm, HWm, SWm = blur(rd.z[i]), blur(hw), blur(sw, 3.0)
        dm = d.reshape(sh)
        dn = dm / np.maximum(HWm, 1.0)
        t = dm / np.maximum(HWm * (1 + SWm * smoothstep(0.08, 0.55, dn)), 1.0)
        t = np.where(okm > 0, t, 2.0)
        rel = np.maximum(Zm - V[y0:y1, x0:x1], 0) * ridge_profile(t)
        rel = np.where(okm > 0, rel, 0).astype(np.float32)
        F[y0:y1, x0:x1] = smax(F[y0:y1, x0:x1], rel, k)
        if rd.rock > 0:
            cr = rd.rock * np.clip(1 - t / 0.45, 0, 1) * np.clip(rel / 60.0, 0, 1)
            crest[y0:y1, x0:x1] = np.maximum(crest[y0:y1, x0:x1], cr.astype(np.float32))
    return F, crest


def ridges_geojson(ridges):
    feats = [geojson_line(r.P[::4].tolist() + [r.P[-1].tolist()], {'id': r.id, 'kind': r.kind, 'crest_m': [rnd(r.z.max(), 1), rnd(r.z.min(), 1)],
                                                               'rock': r.rock}) for r in ridges]
    return fc(feats, 'ridge_skeleton')


# ------------------------------------------------------------------ corridors
def corridor_samples(wmask):
    """Road/rail centreline samples at 1 px with their design grade (per edge)."""
    out = []
    for f in load_json(path('data/roads/roads.geojson'))['features']:
        t = f['properties']['type']
        if t in ('driveway', 'dirt') or f['properties'].get('virtual'):
            continue
        c = np.asarray(f['geometry']['coordinates'], float)[:, :2]
        if len(c) < 2:
            continue
        out.append((resample(c, 1.0), MAX_GRADE.get(t, 0.12), f['properties'].get('width_m', 8) / 2 / MPP, MAX_CUT.get(t, 6.0)))
    for f in load_json(path('data/railways/railways.geojson'))['features']:
        c = np.asarray(f['geometry']['coordinates'], float)[:, :2]
        out.append((resample(c, 1.0), MAX_GRADE['rail'], 3.0, MAX_CUT['rail']))
    return out


def corridor_cap(elev, wmask, cfg, surf=None):
    """Natural gaps for roads/rail: along each route the terrain is capped by its cut-only grade envelope
    (the lowest profile that keeps the design grade without fill), and beside it by a widening V/U-shaped
    allowance. Bridge spans (water) impose nothing. Grading (roads agent) does the engineered cut/fill."""
    cc = cfg['corridors']
    s1, s2, reach = cc['side_m_per_px'], cc['side_curve'], cc['reach_px']
    gscale = cc.get('grade_frac', 0.8)
    wbuf = cv2.dilate(wmask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    XY, P, HW = [], [], []
    for pts, g, hw, mcut in corridor_samples(wmask):
        z0 = bilinear(elev, pts[:, 0], pts[:, 1]).astype(np.float64)
        xi = np.clip(pts[:, 0].astype(int), 0, W - 1); yi = np.clip(pts[:, 1].astype(int), 0, H - 1)
        wet_s = wbuf[yi, xi]
        z0 = np.where(wet_s, 1e5, z0)
        ds = np.r_[0, np.hypot(*np.diff(pts, axis=0).T)] * MPP * g * gscale
        p = z0.copy()
        for i in range(1, len(p)):
            p[i] = min(p[i], p[i - 1] + ds[i])
        for i in range(len(p) - 2, -1, -1):
            p[i] = min(p[i], p[i + 1] + ds[i + 1])
        ok = p < 1e4
        if ok.sum() > 3:
            zs_ = ndi.gaussian_filter1d(np.where(ok, z0, np.nan if False else z0), 3, mode='nearest')
            p = np.where(ok, np.maximum(p, zs_ - mcut), p)
        XY.append(pts[ok]); P.append(p[ok]); HW.append(np.full(ok.sum(), hw))
    XY = np.vstack(XY); P = np.concatenate(P); HW = np.concatenate(HW)
    tree = cKDTree(XY)
    m = np.zeros((H, W), np.uint8)
    for (x, y) in XY[::3]:
        cv2.circle(m, (int(x), int(y)), int(reach), 1, -1)
    ys, xs = np.nonzero(m & ~wbuf)
    d, i = tree.query(np.stack([xs + 0.5, ys + 0.5], 1), k=6)
    dd = np.maximum(d - HW[i], 0)
    allow = (P[i] + cc.get('tolerance_m', 2.0) + s1 * dd + s2 * dd * dd).min(1)
    fade = 1 - smoothstep(0.3 * reach, 0.95 * reach, d[:, 0])  # the cap's influence ends smoothly
    if surf is not None:
        # never cut a bank below the water beside it (roads along rapids/creeks)
        wl_hi = cv2.dilate(np.where(wmask, surf, -1e9).astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        allow = np.maximum(allow, wl_hi[ys, xs] + 1.0)
    cur = elev[ys, xs]
    # only where the natural ground stands clearly above the grade envelope (no grooves along flat streets)
    new = cur - np.maximum(cur - allow, 0) * fade
    out = elev.copy()
    out[ys, xs] = new
    return out


# ------------------------------------------------------------------ hydrology (numba)
_DY = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
_DX = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
_DD = np.array([1.41421356, 1.0, 1.41421356, 1.0, 1.0, 1.41421356, 1.0, 1.41421356])


@njit(cache=True)
def _flood(z, outlet, eps):
    Hh, Ww = z.shape
    N = Hh * Ww
    zf = z.ravel().copy()
    ol = outlet.ravel()
    closed = np.zeros(N, np.bool_)
    order = np.empty(N, np.int64)
    no = 0
    heap = [(0.0, np.int64(0))]
    heap.pop()
    for i in range(N):
        if ol[i]:
            heapq.heappush(heap, (zf[i], np.int64(i)))
            closed[i] = True
    while len(heap) > 0:
        zc, c = heapq.heappop(heap)
        order[no] = c
        no += 1
        cy = c // Ww
        cx = c - cy * Ww
        for k in range(8):
            ny = cy + _DY[k]
            nx = cx + _DX[k]
            if ny < 0 or ny >= Hh or nx < 0 or nx >= Ww:
                continue
            n = ny * Ww + nx
            if closed[n]:
                continue
            closed[n] = True
            if zf[n] <= zc + eps:
                zf[n] = zc + eps
            heapq.heappush(heap, (zf[n], np.int64(n)))
    return zf.reshape(Hh, Ww), order[:no]


@njit(cache=True)
def _receivers(zf, outlet, cell):
    Hh, Ww = zf.shape
    N = Hh * Ww
    rec = np.arange(N)
    dist = np.ones(N) * cell
    for y in range(Hh):
        for x in range(Ww):
            c = y * Ww + x
            if outlet[y, x]:
                continue
            best = 0.0
            for k in range(8):
                ny = y + _DY[k]
                nx = x + _DX[k]
                if ny < 0 or ny >= Hh or nx < 0 or nx >= Ww:
                    continue
                s = (zf[y, x] - zf[ny, nx]) / _DD[k]
                if s > best:
                    best = s
                    rec[c] = ny * Ww + nx
                    dist[c] = _DD[k] * cell
    return rec, dist


@njit(cache=True)
def _area(rec, order, cellarea):
    N = rec.shape[0]
    A = np.ones(N) * cellarea
    for j in range(order.shape[0] - 1, -1, -1):
        c = order[j]
        r = rec[c]
        if r != c:
            A[r] += A[c]
    return A


@njit(cache=True)
def _spl(z, rec, dist, A, order, K, m, dt, fixed):
    zr = z.ravel()
    fx = fixed.ravel()
    for j in range(order.shape[0]):
        c = order[j]
        r = rec[c]
        if r == c or fx[c]:
            continue
        if zr[c] <= zr[r]:
            continue
        f = K * dt * A[c] ** m / dist[c]
        zn = (zr[c] + f * zr[r]) / (1.0 + f)
        if zn < zr[c]:
            zr[c] = zn
    return zr.reshape(z.shape)


def drainage(z, outlet, cell=MPP, eps=1e-3):
    zf, order = _flood(z.astype(np.float64), outlet, eps)
    rec, dist = _receivers(zf, outlet, cell)
    A = _area(rec, order, cell * cell)
    return zf, order, rec, dist, A


def stream_power(z, fixed, cell, K, m, dt, steps, diff):
    """Implicit stream-power incision (Braun & Willett 2013) + linear hillslope diffusion. Carves
    dendritic hollows and creek valleys into the designed landforms down to the rivers' base level."""
    z = z.astype(np.float64).copy()
    outlet = fixed.copy()
    outlet[0, :] = outlet[-1, :] = outlet[:, 0] = outlet[:, -1] = True
    for _ in range(steps):
        zf, order, rec, dist, A = drainage(z, outlet, cell)
        z = _spl(z, rec, dist, A, order, K, m, dt, fixed)
        if diff > 0:
            lap = cv2.Laplacian(z, cv2.CV_64F) / (cell * cell)
            z = np.where(fixed, z, z + diff * dt * lap)
    return z


def evolve_core(base, rel, fixed, cell, ev, seed=5):
    """Uplift + implicit stream power + diffusion grown on a designed relief (any grid). base: floor
    elevation (fixed cells stay on it), rel: designed relief used as the uplift map. Returns the evolved
    relief rescaled (at rescale_sigma cells) back to the designed heights."""
    hh, ww = rel.shape
    U = np.clip(rel / max(np.percentile(rel, 99.5), 1.0), 0, 1.2) ** ev.get('uplift_exp', 0.4) * ev.get('uplift', 1.0)
    rng = np.random.default_rng(seed)
    z = base + rel * 0.05 + rng.normal(0, ev.get('noise_m', 0.5), (hh, ww))
    outlet = fixed.copy()
    outlet[0, :] = outlet[-1, :] = outlet[:, 0] = outlet[:, -1] = True
    K, m, D = ev.get('K', 0.02), ev.get('m', 0.5), ev.get('diffusion', 1.5)
    for _ in range(ev.get('steps', 150)):
        z = z + U
        zf, order, rec, dist, A = drainage(z, outlet, cell)
        z = _spl(z, rec, dist, A, order, K, m, 1.0, fixed)
        lap = cv2.Laplacian(z, cv2.CV_64F) / cell ** 2
        z = np.where(fixed, base, z + D * lap)
    rs = np.maximum(z - base, 0)
    sg = ev.get('rescale_sigma_px', 12)
    bd = cv2.GaussianBlur(rel, (0, 0), sg); bs = cv2.GaussianBlur(rs, (0, 0), sg)
    ratio = np.clip(bd / np.maximum(bs, 1.0), 0.05, 3.0)
    return (rs * ratio).astype(np.float32)


def evolve_landscape(relief, V, wmask, ev, seed=5):
    """Landscape evolution on the designed layout: the designed relief is used as a tectonic uplift map
    and the land is grown under uplift + implicit stream-power incision + hillslope diffusion (half res).
    The result has real dendritic drainage (coves, branching hollows, spurs, divides along the designed
    crests); it is then rescaled at the ~150 m scale back to the designed heights, so the art direction
    (where ridges are and how high) is kept while the forms become erosional."""
    h, w = H // 2, W // 2
    r2 = cv2.resize(relief, (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)
    V2 = cv2.resize(V, (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)
    low = r2 < ev.get('fixed_below_m', 1.0)
    lab, n = ndi.label(low)
    if n:
        sz = ndi.sum(low, lab, np.arange(1, n + 1))
        low = np.isin(lab, 1 + np.nonzero(sz >= ev.get('min_lowland_cells', 400))[0])  # pockets inside mountains are not outlets
    fixed = (cv2.resize(wmask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0) | low
    out = evolve_core(V2, r2, fixed, MPP * 2, ev, seed)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)


def hydraulic_erosion(z, fixed, n=260000, batch=65536, steps=40, seed=3, inertia=0.12, capacity=3.0, deposit=0.2,
                      erode=0.25, evap=0.035, gravity=2.0, min_slope=0.01, cell=MPP, max_sed=1.2):
    """Vectorised droplet erosion: fine gullies and small alluvial fans; fixed cells untouched."""
    rng = np.random.default_rng(seed)
    z = z.astype(np.float64).copy()
    Hh, Ww = z.shape
    zs = z / cell

    def grad(px, py):
        x0 = np.clip(px.astype(int), 0, Ww - 2); y0 = np.clip(py.astype(int), 0, Hh - 2)
        fx = px - x0; fy = py - y0
        h00 = zs[y0, x0]; h10 = zs[y0, x0 + 1]; h01 = zs[y0 + 1, x0]; h11 = zs[y0 + 1, x0 + 1]
        gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
        gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
        h = h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy) + h01 * (1 - fx) * fy + h11 * fx * fy
        return gx, gy, h, x0, y0
    done = 0
    while done < n:
        m = min(batch, n - done); done += m
        px = rng.uniform(1, Ww - 2, m); py = rng.uniform(1, Hh - 2, m)
        dx = np.zeros(m); dy = np.zeros(m); sp = np.ones(m); wat = np.ones(m); sed = np.zeros(m)
        alive = ~fixed[py.astype(int), px.astype(int)]
        for _ in range(steps):
            gx, gy, h, x0, y0 = grad(px, py)
            dx = dx * inertia - gx * (1 - inertia); dy = dy * inertia - gy * (1 - inertia)
            l = np.hypot(dx, dy); l[l == 0] = 1
            dx /= l; dy /= l
            nx = px + dx; ny = py + dy
            alive &= (nx > 1) & (nx < Ww - 2) & (ny > 1) & (ny < Hh - 2)
            nx = np.clip(nx, 1, Ww - 2.001); ny = np.clip(ny, 1, Hh - 2.001)
            alive &= ~fixed[ny.astype(int), nx.astype(int)]
            _, _, nh, *_ = grad(nx, ny)
            dh = nh - h
            cap = np.maximum(-dh, min_slope) * sp * wat * capacity
            dep = np.where((sed > cap) | (dh > 0), np.where(dh > 0, np.minimum(dh, sed), (sed - cap) * deposit), 0.0)
            ero = np.where((sed <= cap) & (dh <= 0), np.minimum((cap - sed) * erode, -dh * 0.35), 0.0)
            amt = (dep - ero) * alive
            sed = np.minimum(sed - amt, max_sed)
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    w = (0.25 if oy == 0 and ox == 0 else (0.125 if oy == 0 or ox == 0 else 0.0625))
                    np.add.at(zs, (np.clip(y0 + oy, 0, Hh - 1), np.clip(x0 + ox, 0, Ww - 1)), amt * w)
            sp = np.minimum(np.sqrt(np.maximum(sp * sp - dh * gravity, 0.0)), 6.0)
            wat *= (1 - evap)
            px, py = nx, ny
            if not alive.any():
                break
    out = zs * cell
    out = z + np.clip(out - z, -12.0, 6.0)
    out[fixed] = z[fixed]
    return (0.7 * out + 0.3 * cv2.GaussianBlur(out, (0, 0), 0.8)).astype(np.float32)


def thermal_erosion(z, fixed, talus=0.85, iters=60, rate=0.35):
    """Relax slopes steeper than `talus` (rise/run) toward the steepest neighbour. Talus aprons,
    rounded crests, no vertical walls. Fixed cells stay put."""
    z = z.astype(np.float64).copy()
    lim = talus * MPP
    offs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    free = ~fixed
    for _ in range(iters):
        delta = np.zeros_like(z)
        pad = np.pad(z, 1, mode='edge')
        best = np.zeros_like(z); bdx = np.zeros(z.shape, np.int8); bdy = np.zeros(z.shape, np.int8)
        for dy, dx, dist in offs:
            nb = pad[1 + dy:1 + dy + z.shape[0], 1 + dx:1 + dx + z.shape[1]]
            ex = (z - nb) / dist - lim
            sel = ex > best
            best = np.where(sel, ex, best); bdx = np.where(sel, dx, bdx); bdy = np.where(sel, dy, bdy)
        mv = best * rate * 0.5 * free
        delta -= mv
        for dy, dx, _ in offs:
            m = (bdx == dx) & (bdy == dy)
            add = np.zeros_like(z)
            add[max(0, dy):z.shape[0] + min(0, dy), max(0, dx):z.shape[1] + min(0, dx)] = (mv * m)[max(0, -dy):z.shape[0] - max(0, dy), max(0, -dx):z.shape[1] - max(0, dx)]
            delta += add * free
        z += delta
    return z.astype(np.float32)


# ------------------------------------------------------------------ channels & banks
def _nblur(vals, ys, xs, mask, sig):
    A = np.zeros((H, W), np.float32)
    A[ys, xs] = vals
    M = mask.astype(np.float32)
    return cv2.GaussianBlur(A * M, (0, 0), sig) / np.maximum(cv2.GaussianBlur(M, (0, 0), sig), 1e-4)


def surf_local_max(surf, wmask, r=2):
    s = np.where(wmask, surf, -1e9).astype(np.float32)
    return cv2.dilate(s, np.ones((2 * r + 1, 2 * r + 1), np.uint8))


def carve_channels(elev, V, surf, wmask, CF, cfg):
    """River cross-sections and banks.
    bed: thalweg shifted to the outside of bends; pools at bends, riffles at crossings, shallow rocky rapids.
    banks: cut banks (steep, rocky) on outside bends and rapids, slip-off slopes + gravel bars inside bends.
    All reference levels are continuous fields (valley floor V), never nearest-pixel (Voronoi) levels."""
    wc = cfg['water']
    reach = wc['bank_reach_px']
    dws = ndi.distance_transform_edt(~wmask).astype(np.float32)   # land: px to water
    dwi = ndi.distance_transform_edt(wmask).astype(np.float32)    # water: px to bank
    near = (dws < reach + 6) | wmask
    ys, xs = np.nonzero(near)
    q = np.stack([xs + 0.5, ys + 0.5], 1)
    d, i = CF['tree'].query(q, k=1)
    rel = q - np.stack([CF['X'][i], CF['Y'][i]], 1)
    lat = CF['TX'][i] * rel[:, 1] - CF['TY'][i] * rel[:, 0]   # >0: right of flow (image y down)
    kap = CF['K'][i]                                           # >0: turning right (clockwise on screen)
    outside = -np.sign(lat) * np.sign(kap)                     # +1 outside of the bend (cut bank side)
    bend = np.clip(np.abs(kap) / wc['bend_kappa'], 0, 1)
    rapid = np.clip((CF['S'][i] - wc['rapid_slope'][0]) / (wc['rapid_slope'][1] - wc['rapid_slope'][0]), 0, 1)
    BETA = _nblur(outside * bend, ys, xs, near, 3.0)            # -1 point bar .. +1 cut bank
    BEND = _nblur(bend, ys, xs, near, 3.0)
    RAP = _nblur(rapid, ys, xs, near, 3.0)
    OFF = _nblur(-np.sign(kap) * bend * 0.5, ys, xs, near, 3.0)  # thalweg offset toward the outside
    CLS = _nblur((CF['C'][i] == 1).astype(np.float32), ys, xs, near, 3.0)
    STEEP = _nblur(CF['S'][i], ys, xs, near, 2.0)
    fx = np.zeros((5, H, W), np.float32)

    # ---- bed (water pixels)
    inw = wmask[ys, xs]
    yw, xw = ys[inw], xs[inw]
    half = np.maximum(dwi[yw, xw] + np.abs(lat[inw]), 1.0)     # local half-width estimate
    un = np.clip(lat[inw] / half, -1, 1)
    o = OFF[yw, xw]
    a = np.where(un > o, (un - o) / np.maximum(1 - o, 0.2), (un - o) / np.maximum(1 + o, 0.2))
    steep_side = np.sign(un - o) == np.sign(o)
    pw = np.where(steep_side, 3.5, 1.8)
    shape_ = np.clip(1 - np.abs(a) ** pw, 0, 1)
    width_px = 2 * half
    dmax = np.clip(wc['depth_base'] + wc['depth_per_px'] * width_px, wc['depth_min'], wc['depth_max'])
    dmax = dmax * (1 - 0.25 * CLS[yw, xw])
    bnd, rap = BEND[yw, xw], RAP[yw, xw]
    pool = 1 + wc['pool_gain'] * bnd - wc['riffle_cut'] * (1 - bnd)
    dmax = dmax * (pool * (1 - rap) + (0.55) * rap)
    irr = 1 + 0.18 * fbm(77, scales=(14, 6), amps=(1, 0.5))[yw, xw]
    depth = np.maximum(dmax * (0.08 + 0.92 * shape_) * irr, 0.12)
    # point bars: inside of strong bends the shallow margin rises to just above the water
    beta_w = BETA[yw, xw]
    barf = np.clip((-beta_w - 0.3) / 0.6, 0, 1) * np.clip((np.abs(un) - 0.4) / 0.5, 0, 1) * (np.sign(un) != np.sign(o)) * (1 - rap)
    depth = depth - wc['bar_rise'] * barf
    elev[yw, xw] = surf[yw, xw] - depth
    fx[0, yw, xw] = depth
    fx[1, yw, xw] = STEEP[yw, xw]
    fx[2, yw, xw] = CF['TX'][i][inw]
    fx[3, yw, xw] = CF['TY'][i][inw]
    fx[4, yw, xw] = np.where(barf > 0.3, -barf, rap)

    # ---- banks (land pixels near water)
    lnd = ~inw
    yl, xl = ys[lnd], xs[lnd]
    dl = dws[yl, xl]
    b = BETA[yl, xl]
    rp = RAP[yl, xl]
    vw = V[yl, xl]
    hb = wc['bank_h'] * (1 + 0.9 * np.clip(b, 0, 1) + 0.6 * rp) * (1 - 0.65 * np.clip(-b, 0, 1))
    ramp = wc['bank_ramp_px'] * (1 - 0.6 * np.clip(b, 0, 1) - 0.3 * rp) * (1 + 2.2 * np.clip(-b, 0, 1))
    ramp = np.maximum(ramp, 1.2)
    tb = np.clip(dl / ramp, 0, 1)
    cut = smoothstep(-0.1, 0.5, b)
    bank_curve = cut * (1 - (1 - tb) ** 3) + (1 - cut) * tb ** 1.25  # cut bank: steep then flat; slip-off: gradual
    cur = elev[yl, xl]
    fade = np.clip((dl - ramp * 0.5) / (ramp * 1.5 + 4), 0, 1) ** 1.2
    shaped = vw + wc.get('bank_min_m', 0.3) + np.maximum(hb * bank_curve, (cur - vw - wc.get('bank_min_m', 0.3)) * fade)
    wgt = 1 - smoothstep(reach * 0.55, reach, dl)
    elev[yl, xl] = cur * (1 - wgt) + shaped * wgt
    gravel = np.clip((-b - 0.3) / 0.5, 0, 1) * np.clip(1 - dl / (ramp * 1.2), 0, 1)
    rocky = np.maximum(rp, np.clip((b - 0.3) / 0.5, 0, 1)) * np.clip(1 - dl / (ramp * 1.5 + 2), 0, 1)
    fx[4, yl, xl] = np.where(gravel > rocky, -gravel, rocky)
    # banks never below the adjacent water (local max of the neighbouring surface, no Voronoi levels)
    sl = surf_local_max(surf, wmask, 2)
    edge = (dws > 0) & (dws <= 2.5)
    elev = np.where(edge, np.maximum(elev, sl + wc.get('bank_min_m', 0.3)), elev)
    return elev, fx


def shape_falls(elev, surf, fx, wmask, lines, prof):
    """Authored waterfalls (data/manual/waterways.json 'falls'), applied LOCALLY after the whole terrain is
    built (a change in the channel profile before erosion would re-shape the entire map):
      * water surface: a step of drop_m at the lip (plunge) or equal tiers between lip and toe (cascade);
        the reach above is raised by drop/2 and the reach below lowered by drop/2, both easing back to the
        original profile over ~60 px, so levels far up/downstream are unchanged
      * terrain: the raised upper reach lifts its floor and banks with it (lateral fade ~25 px); below the
        lip only the channel is lowered, so the untouched banks become gorge walls (old bank + drop/2 +
        the lowering); a plunge pool ~2 m deeper at the toe
      * rock: ledge across the channel at the lip / tiers and the gorge walls -> rock_exposure (E2 rock kit)."""
    man = {r['id']: r for r in load_json(path('data/manual/waterways.json'))['rivers']}
    rock = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dwe = ndi.distance_transform_edt(~wmask).astype(np.float32)        # px to water
    for k, r in man.items():
        if k not in lines or not r.get('falls'):
            continue
        L = lines[k]
        tree = cKDTree(L)
        for fl in r['falls']:
            il = int(np.argmin(np.hypot(*(L - fl['lip']).T))); it = int(np.argmin(np.hypot(*(L - fl['toe']).T)))
            il, it = min(il, it), max(il, it)
            D, nt = float(fl['drop_m']), max(1, int(fl.get('tiers', 1)))
            n = len(L); i = np.arange(n)
            tiers = [il] if nt == 1 else np.linspace(il, max(il, it - 1), nt).round().astype(int).tolist()
            # delta along the channel: +D/2 above the first tier, stepping down D/nt at each tier, -D/2 below the last
            delta = np.full(n, D / 2)
            for t_ in tiers:
                delta[i > t_] -= D / nt
            up = 1 - smoothstep(0, 60, tiers[0] - i)                      # ease above
            dn = 1 - smoothstep(0, 60, i - max(tiers[-1], it))             # ease below
            delta = np.where(i <= tiers[0], delta * up, np.where(i > max(tiers[-1], it), delta * dn, delta))
            # per-pixel nearest sample of this channel within reach
            near = (np.hypot(xx - L[il, 0], yy - L[il, 1]) < 110)
            ys, xs = np.nonzero(near)
            dd, jj = tree.query(np.stack([xs, ys], 1).astype(np.float64), k=1)
            dl = delta[jj]
            on_ch = dd < 40
            ys, xs, dl, jj, dd = ys[on_ch], xs[on_ch], dl[on_ch], jj[on_ch], dd[on_ch]
            wet = wmask[ys, xs]
            # water surface + bed follow the delta
            surf[ys[wet], xs[wet]] += dl[wet]
            lat_up = 1 - smoothstep(6, 25, dwe[ys, xs])                   # raised reach: floor + banks
            lat_dn = (1 - smoothstep(0.5, 2.0, dwe[ys, xs]))              # lowered reach: channel only
            fac = np.where(dl > 0, np.where(wet, 1.0, lat_up), np.where(wet, 1.0, lat_dn))
            elev[ys, xs] += dl * fac
            # rock: ledges across the channel + the gorge walls below the lip
            for t_ in tiers:
                p = L[t_]
                rock = np.maximum(rock, np.clip(1 - np.hypot(xx - p[0], yy - p[1]) / 6.0, 0, 1) * 0.95)
            wall = (~wet) & (dl < -0.5) & (dwe[ys, xs] <= 7) & (jj <= max(tiers[-1], it) + 25)
            rock[ys[wall], xs[wall]] = np.maximum(rock[ys[wall], xs[wall]], 0.9 * np.clip(-dl[wall] / (D / 2), 0, 1))
            # plunge pool below the toe
            pr = fl.get('pool_radius_m', 8.0) / MPP
            pt = L[it]
            pool = (np.clip(1 - np.hypot(xx - pt[0], yy - pt[1]) / pr, 0, 1) ** 0.7 * wmask).astype(np.float32)
            elev -= 2.0 * pool
            fx[0] = fx[0] + 2.0 * pool
            print(f'  falls {fl["id"]}: drop {D} m in {len(tiers)} step(s), upper pool {float(prof[k][max(tiers[0] - 2, 0)] + D / 2):.1f} m, pool r {pr:.1f} px')
    return elev.astype(np.float32), surf, rock


@njit(cache=True)
def _breach(z, order, rec, eps):
    """Lower downstream cells so every cell drains (least-effort carving along the flood tree)."""
    zr = z.ravel().copy()
    for j in range(order.shape[0] - 1, -1, -1):
        c = order[j]
        r = rec[c]
        if r != c and zr[r] > zr[c] - eps:
            zr[r] = zr[c] - eps
    return zr.reshape(z.shape)


def condition_drainage(z, wmask, max_breach=3.0, eps=2e-3):
    """Every land cell drains to water or the map edge: shallow pits are breached (a small gully through
    the rim), deeper ones are filled with a gentle gradient (flats drain toward their outlet)."""
    outlet = wmask.copy()
    outlet[0, :] = outlet[-1, :] = outlet[:, 0] = outlet[:, -1] = True
    z = z.astype(np.float64)
    zf, order, rec, dist, A = drainage(z, outlet, MPP, eps)
    zb = _breach(z, order, rec, eps)
    cutd = z - zb
    # only accept shallow carving; smooth it a little so gullies are not 1-px trenches
    ok = cutd <= max_breach
    zb2 = np.where(ok, zb, z)
    zb2 = np.where(wmask, z, zb2)
    zf2, *_ = _flood(zb2, outlet, eps)
    return zf2.astype(np.float32)


# ------------------------------------------------------------------ main
def _dbg(name, a):
    d = os.environ.get('TERRAIN_DEBUG')
    if d:
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, name + '.npy'), np.asarray(a, np.float32))


def main():
    cfg = load_json(path('data/manual/terrain.json'))
    rng = np.random.default_rng(cfg.get('seed', 11))
    wmask = np.load(path('tools/.cache/water_mask.npy'))
    lines, prof, steep, feats = river_profiles(cfg)
    CF = channel_fields(lines, prof, steep, feats, wmask)
    surf = water_surface(CF, wmask, lines, feats)

    # valley floor: harmonic interpolation between the water surfaces
    valley = harmonic_fill(np.nan_to_num(surf), wmask)
    valley = np.where(wmask, surf, valley)

    low, fd, rock = lowland_mask(wmask)
    core = cv2.erode(low.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg['lowland']['core_erode_px'] + 1,) * 2)) > 0
    core |= wmask
    dlow = ndi.distance_transform_edt(~core).astype(np.float32)
    dlow = cv2.GaussianBlur(dlow, (0, 0), 2.0)

    # ---- designed landforms
    ridges = build_ridges(cfg, valley, rng)
    wa = cfg['ridges'].get('warp_px', 5.0)
    warp = (fbm(51, scales=(45, 22), amps=(1, 0.5)) * wa, fbm(52, scales=(45, 22), amps=(1, 0.5)) * wa)
    F, crest = ridge_field(ridges, valley, k=cfg['ridges'].get('union_k', 10.0), warp=warp)
    # background hills where the map shows woodland but no designed ridge (low, soft)
    bg = cfg['background']
    A = feather_field(cfg.get('amplitude_regions', []), bg['amplitude_m'])
    B = A * (1 - np.exp(-(dlow * bg['slope_m_per_px'] / np.maximum(A, 1)) ** 1.1))
    B = cv2.GaussianBlur(B, (0, 0), 6) * (1 + 0.35 * fbm(21, scales=(80, 40), amps=(1, 0.5)))
    _dbg('F', F)
    relief = smax(F, np.maximum(B, 0), 8.0)
    # lowland cap: concave toe, max toe slope
    lc = cfg['lowland']
    cap = lc['toe_slope_m_per_px'] * (np.sqrt(dlow ** 2 + lc['toe_px'] ** 2) - lc['toe_px'])
    relief = smin(relief, cap, 12.0)
    # lowland is not perfectly flat: gentle rolling, river terraces
    dl_in = ndi.distance_transform_edt(low).astype(np.float32)
    roll = (fbm(7, scales=(110, 55, 28), amps=(1, 0.6, 0.3)) * 0.5 + 0.5) * np.clip(dl_in / 25, 0, 1) * lc['roll_m']
    dws = ndi.distance_transform_edt(~wmask).astype(np.float32)
    terr = lc['terrace_m'] * smoothstep(lc['terrace_px'][0], lc['terrace_px'][1], dws + 12 * fbm(9, scales=(60, 30), amps=(1, .5)))
    relief = relief + (roll + terr) * low * ~wmask
    # flatten developed zones
    flat = np.ones((H, W), np.float32)
    for z in load_json(path('data/manual/zones.json'))['zones']:
        f = cfg['flatten_zone_kinds'].get(z['kind'])
        if f is None:
            continue
        m = rasterize_polys([Polygon(z['pts'])], (H, W), value=255, ups=2).astype(np.float32) / 255
        m = cv2.GaussianBlur(m, (0, 0), 12)
        flat = np.minimum(flat, 1 - m * (1 - f))
    _dbg('relief_flat', relief)
    relief *= flat

    # ---- secondary detail (subtle): warped noise scaled by relief, crest crags
    dt = cfg['detail']
    mt = np.clip(relief / 80.0, 0, 1)
    relief = relief * (1 + dt['macro_var'] * warped_fbm(31, (90, 45), (1, 0.5)))
    relief += dt['meso_m'] * warped_fbm(33, (28, 14, 7), (1, 0.5, 0.25), warp_px=12) * mt
    crag = np.abs(warped_fbm(35, (9, 4.5), (1, 0.5), warp_px=6))
    crag = (np.round(crag * 4) / 4) * 0.6 + crag * 0.4  # stepped ledges
    relief += dt['crag_m'] * crest * crag

    # landscape evolution on the designed layout (dendritic drainage); low ground keeps the designed relief
    ev = cfg.get('evolve')
    if ev and ev.get('steps', 0) > 0:
        rel_ev = evolve_landscape(relief, valley, wmask, ev)
        # erosion adds form, not mass: never much above the design (keeps gaps, passes and lowland edges)
        rel_ev = np.minimum(np.maximum(rel_ev, 0), relief * ev.get('max_over', 1.2) + ev.get('max_over_m', 6.0))
        wv = smoothstep(ev.get('blend_m', [15, 45])[0], ev.get('blend_m', [15, 45])[1], cv2.GaussianBlur(relief, (0, 0), 4))
        relief = relief * (1 - wv) + rel_ev * wv
        _dbg('relief_evolved', relief)

    # banks: relief -> 0 toward the water
    dws_s = cv2.GaussianBlur(dws, (0, 0), 1.5)
    bank = np.clip(dws_s / cfg['water']['relief_fade_px'], 0, 1) ** 1.2
    elev = valley + relief * bank
    sl = surf_local_max(surf, wmask, 3)
    nearw = (dws > 0) & (dws <= 3.5)
    elev = np.where(nearw, np.maximum(elev, sl + 0.2), elev)

    # ---- road / rail gaps
    _dbg('elev_pre_cap', elev)
    elev = corridor_cap(elev, wmask, cfg, surf)

    # ---- drainage-driven erosion (on a drained surface so flow paths are coherent)
    elev = np.where(wmask, elev, condition_drainage(elev, wmask, cfg['erosion'].get('max_breach_m', 3.0))).astype(np.float32)
    fixed = cv2.dilate(wmask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    er = cfg['erosion']
    if er.get('spl_steps', 0) > 0:
        lo = cv2.resize(elev, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        flo = cv2.resize(fixed.astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST) > 0
        elo = stream_power(lo, flo, MPP * 2, er['spl_K'], er['spl_m'], 1.0, er['spl_steps'], er['spl_diffusion'])
        dz = cv2.resize((elo - lo).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        dz[fixed] = 0
        sculpt = np.clip((relief * bank - 4.0) / 20.0, 0, 1)
        elev = (elev + dz * cv2.GaussianBlur(sculpt.astype(np.float32), (0, 0), 2)).astype(np.float32)
    if er.get('droplets', 0) > 0:
        lo = cv2.resize(elev, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        flo = cv2.resize(fixed.astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST) > 0
        elo = hydraulic_erosion(lo, flo, n=er['droplets'], cell=MPP * 2)
        dz = cv2.resize((elo - lo).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        dz = cv2.GaussianBlur(dz, (0, 0), 1.0)
        dz[fixed] = 0
        wr = np.clip((relief * bank - 12.0) / 35.0, 0, 1)
        elev = (elev + dz * cv2.GaussianBlur(wr.astype(np.float32), (0, 0), 3)).astype(np.float32)
    _dbg('elev_pre_thermal', elev)
    if er.get('fine_steps', 0) > 0:
        # second, fine-scale incision at full resolution: ravines and gullies inside the big hollows
        ef = stream_power(elev, fixed, MPP, er['fine_K'], er.get('spl_m', 0.5), 1.0, er['fine_steps'], er.get('fine_diffusion', 1.0))
        dz = (ef - elev).astype(np.float32)
        dz[fixed] = 0
        sculpt = np.clip((relief * bank - 8.0) / 25.0, 0, 1)
        elev = (elev + np.maximum(dz, -er.get('fine_max_m', 12.0)) * cv2.GaussianBlur(sculpt.astype(np.float32), (0, 0), 2)).astype(np.float32)
    elev = thermal_erosion(elev, fixed, talus=er.get('talus', 0.9), iters=er.get('thermal_iters', 60), rate=0.9)
    # crest rounding: Appalachian summits and spur tops are broad and rounded, never pyramids
    rmask = cv2.GaussianBlur(np.clip((relief * bank - 10.0) / 30.0, 0, 1).astype(np.float32), (0, 0), 3) * ~fixed
    for _ in range(er.get('round_iters', 0)):
        b_ = cv2.GaussianBlur(elev, (0, 0), er.get('round_sigma', 3.0))
        convex = np.clip((elev - b_) / er.get('round_thr_m', 1.5), 0, 1)
        elev = (elev + (b_ - elev) * convex * rmask).astype(np.float32)
    elev = corridor_cap(elev, wmask, cfg, surf)  # erosion must not re-block the gaps

    # ---- river channels and banks
    _dbg('elev_pre_channels', elev)
    elev, fx = carve_channels(elev.astype(np.float32), valley, surf, wmask, CF, cfg)

    _dbg('elev_channels', elev)
    # ---- hydrology guarantee: every land cell drains (shallow pits breached, deeper ones filled)
    elev = np.where(wmask, elev, condition_drainage(elev, wmask, cfg['erosion'].get('max_breach_m', 3.0))).astype(np.float32)
    # authored waterfalls last: a local edit (anything earlier re-routes the global drainage / erosion)
    elev, surf, falls_rock = shape_falls(elev, surf, fx, wmask, lines, prof)

    # rock exposure (terrain-derived): steep crests/crags, cut banks, rapids banks
    gy, gx = np.gradient(cv2.GaussianBlur(elev, (0, 0), 1.2), MPP)
    slope = np.hypot(gx, gy)
    rex = np.clip((slope - 0.55) / 0.5, 0, 1) * (0.35 + 0.65 * np.clip(crest * 1.6, 0, 1))
    rex = np.maximum(rex, np.clip(fx[4], 0, 1) * (~wmask) * 0.8)
    rex = np.maximum(rex, falls_rock)   # waterfall ledges + gorge walls: bare rock for the rock kit
    rex = cv2.GaussianBlur(rex.astype(np.float32), (0, 0), 0.8)

    os.makedirs(path('data/terrain'), exist_ok=True)
    elev.tofile(path('data/terrain/height_f32.bin'))
    surf.astype(np.float32).tofile(path('data/terrain/water_level_f32.bin'))
    # surface slope from the (continuous) water surface, and whitewater: where the painted map shows white
    # water, or where the surface is genuinely steep (riffle lips, rapids, cascades); never in pools
    sgy, sgx = np.gradient(np.where(wmask, surf, np.nan), MPP)
    sslope = np.nan_to_num(np.hypot(sgx, sgy))
    sslope = np.where(wmask, cv2.GaussianBlur(sslope.astype(np.float32), (0, 0), 1.0), 0)
    rapm = np.load(path('tools/.cache/rapids_mask.npy')).astype(np.float32)
    rap_map = cv2.GaussianBlur(rapm, (0, 0), 1.5)
    foam = np.maximum(np.clip(rap_map * 2.2, 0, 1) * smoothstep(0.004, 0.02, sslope + 0.01 * rap_map), smoothstep(0.03, 0.11, sslope))
    foam = (foam * wmask).astype(np.float32)
    fx[1] = sslope * wmask
    # compact water attributes for renderers: u8 [5][H][W]
    ang = (np.arctan2(fx[3], fx[2]) % (2 * np.pi)) / (2 * np.pi)
    fxu = np.stack([np.clip(fx[0] / 6.0, 0, 1), np.clip(fx[1] / 0.25, 0, 1), ang, np.clip(fx[4] * 0.5 + 0.5, 0, 1), np.clip(foam, 0, 1)])
    fxu = (fxu * 255).round().astype(np.uint8)
    fxu[:, ~cv2.dilate((fx[0] > 0).astype(np.uint8) | (np.abs(fx[4]) > 0).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = 0
    fxu[3][(fx[0] <= 0) & (np.abs(fx[4]) == 0)] = 128
    fxu.tofile(path('data/terrain/water_fx_u8.bin'))
    if os.path.exists(path('data/terrain/water_fx_f16.bin')):
        os.remove(path('data/terrain/water_fx_f16.bin'))
    Image.fromarray((np.clip(rex, 0, 1) * 255).astype(np.uint8)).save(path('data/terrain/rock_exposure_u8.png'))
    save_json(path('data/terrain/ridge_skeleton.geojson'), ridges_geojson(ridges))
    lo_, hi_ = float(np.floor(elev.min())), float(np.ceil(elev.max()))
    u16 = ((elev - lo_) / (hi_ - lo_) * 65535).round().astype(np.uint16)
    Image.fromarray(u16).save(path('data/terrain/height_u16.png'))
    meta = {
        'grid': {'width': W, 'height': H, 'spacing_m': MPP, 'origin': 'sample (i,j) is at source px (i+0.5, j+0.5)'},
        'files': {'height_f32': 'data/terrain/height_f32.bin', 'height_u16': 'data/terrain/height_u16.png',
                  'water_level_f32': 'data/terrain/water_level_f32.bin', 'water_fx_u8': 'data/terrain/water_fx_u8.bin',
                  'rock_exposure_u8': 'data/terrain/rock_exposure_u8.png', 'ridge_skeleton': 'data/terrain/ridge_skeleton.geojson'},
        'water_fx_u8': {'layout': 'uint8 [5][667][2000]', 'channels': ['depth: v/255*6 m', 'surface slope: v/255*0.25 m/m', 'flow direction: v/255*2pi (atan2(dy,dx), image axes, y down)', 'bank/bed type: v/255*2-1 (-1 gravel bar .. 0 plain .. +1 rock/rapids)', 'whitewater: v/255 (0..1)']},
        'u16_decode': {'min_m': lo_, 'max_m': hi_, 'formula': 'h = min + v/65535*(max-min)'},
        'stats': {'min_m': rnd(elev.min(), 1), 'max_m': rnd(elev.max(), 1), 'mean_m': rnd(elev.mean(), 1)},
        'river_profiles_m': {k: [rnd(v[0], 1), rnd(v[-1], 1)] for k, v in prof.items()},
        'note': 'Designed from the map geography + ridge skeleton (data/manual/ridges.json), not survey data. Rebuild with tools/pipeline/terrain.py; road cut/fill is applied by tools/pipeline/grading.py.'
    }
    save_json(path('data/terrain/terrain.json'), meta, indent=1)
    hillshade(elev, path('assets/maps/debug/hillshade.png'))
    print('terrain', meta['stats'], 'ridges+spurs', len(ridges))


def hillshade(elev, out, az=315, alt=40, tint=True):
    gy, gx = np.gradient(elev, MPP)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = math.radians(az), math.radians(alt)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    hs = np.clip(hs, 0, 1)
    if tint:
        t = (elev - elev.min()) / (elev.max() - elev.min())
        ramp = np.array([[70, 110, 60], [140, 150, 90], [170, 150, 110], [220, 220, 220]], np.float32)
        idx = np.clip(t * 3, 0, 2.999)
        i0 = idx.astype(int)
        f = (idx - i0)[..., None]
        col = ramp[i0] * (1 - f) + ramp[i0 + 1] * f
        img = col * (0.35 + 0.75 * hs[..., None])
    else:
        img = np.repeat(hs[..., None] * 255, 3, 2)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out)


if __name__ == '__main__':
    main()
