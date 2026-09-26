"""Terrain reconstruction (heightfield on the source-pixel grid, 2.5 m spacing).

  valley floor : harmonic interpolation of monotonic river-surface profiles
  relief       : rises with distance from lowland (water, farmland, towns),
                 capped slope, steeper/higher where the map shows exposed rock,
                 amplitude art-directed per region (data/manual/terrain.json)
  flattening   : towns/cities smoothed toward their valley floor
  channels     : rivers carved below banks; water surface stored separately

Outputs (data/terrain/):
  height_f32.bin       float32 [667][2000] metres (row = image y)
  height_u16.png       16-bit normalized (see terrain.json for min/max)
  water_level_f32.bin  float32 water surface (NaN where dry)
  terrain.json         metadata
  assets/maps/debug/hillshade.png
"""
import sys, os, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from shapely.geometry import Polygon, LineString, Point
from PIL import Image

from tools.lib.common import path, load_json, save_json, W, H, rnd
from tools.lib.features import hsv, load_rgb, rock_density
from tools.lib.geom import rasterize_polys
from tools.lib.trace import resample

MPP = 2.5


# ------------------------------------------------------------------ rivers
def river_profiles():
    ww = load_json(path('data/water/waterways.geojson'))['features']
    feats = {f['properties']['id']: f for f in ww}
    lines = {k: resample(np.asarray(f['geometry']['coordinates'], float), 1.0) for k, f in feats.items()}
    prof = {}

    def elev_at(k, pt):
        L = lines[k]
        i = int(np.argmin(np.hypot(*(L - pt).T)))
        return prof[k][i]

    pending = set(feats)
    for _ in range(20):
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
                # side channel leaving a parent: inherit at its head from the nearest resolved channel
                cands = [q for q in prof if q != k]
                if not cands:
                    continue
                best = min(cands, key=lambda q: np.hypot(*(lines[q] - L[0]).T).min())
                e0 = elev_at(best, L[0])
                if p.get('backwater'):
                    e0 = max(e0, e1)
            e0 = max(e0, e1)
            n = len(L)
            t = np.linspace(0, 1, n)
            prof[k] = e1 + (e0 - e1) * (1 - t) ** 1.35
            pending.discard(k)
        if not pending:
            break
    if pending:
        raise RuntimeError(f'unresolved river profiles: {pending}')
    return lines, prof, feats


def water_surface(lines, prof, wmask):
    cl = np.full((H, W), np.nan, np.float32)
    for k, L in lines.items():
        xi = np.clip(L[:, 0].astype(int), 0, W - 1)
        yi = np.clip(L[:, 1].astype(int), 0, H - 1)
        # lower wins where channels overlap
        cur = cl[yi, xi]
        cl[yi, xi] = np.where(np.isnan(cur), prof[k], np.minimum(cur, prof[k]))
    known = ~np.isnan(cl)
    _, (iy, ix) = ndi.distance_transform_edt(~known, return_indices=True)
    near = cl[iy, ix]
    surf = np.where(wmask, near, np.nan).astype(np.float32)
    # smooth the surface inside water so confluences/lakes are level-ish
    s = np.where(wmask, surf, 0)
    wgt = cv2.GaussianBlur(wmask.astype(np.float32), (0, 0), 3)
    s = cv2.GaussianBlur(s.astype(np.float32), (0, 0), 3) / np.maximum(wgt, 1e-6)
    surf = np.where(wmask, np.minimum(s, near + 0.3), np.nan)
    return surf.astype(np.float32), near


def harmonic_fill(values, known, scale=4, iters=0):
    """Solve Laplace(u)=0 with Dirichlet at known cells, on a downscaled grid."""
    h, w = H // scale + 1, W // scale + 1
    v = cv2.resize(np.where(known, values, 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    k = cv2.resize(known.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    fixed = k > 0.2
    vv = np.where(fixed, v / np.maximum(k, 1e-6), 0)
    N = h * w
    idx = np.arange(N).reshape(h, w)
    A = lil_matrix((N, N))
    b = np.zeros(N)
    for y in range(h):
        for x in range(w):
            i = idx[y, x]
            if fixed[y, x]:
                A[i, i] = 1
                b[i] = vv[y, x]
                continue
            nb = [(y + dy, x + dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= y + dy < h and 0 <= x + dx < w]
            A[i, i] = len(nb)
            for (yy, xx) in nb:
                A[i, idx[yy, xx]] = -1
    u = spsolve(csr_matrix(A), b).reshape(h, w).astype(np.float32)
    return cv2.resize(u, (W, H), interpolation=cv2.INTER_CUBIC)


# ------------------------------------------------------------------ land
def lowland_mask(wmask):
    """Lowland = water + open farmland + developed land (dense road network) + towns.
    Built from smoothed fields so the resulting landforms are coherent, not bubbly."""
    Hh, S, V = hsv()
    forest = ((Hh >= 60) & (Hh <= 170) & (V < 0.42)) | (V < 0.25)
    fd = cv2.GaussianBlur(forest.astype(np.float32), (0, 0), 7)
    rock = rock_density()
    types = load_json(path('data/roads/road_types.json'))['types']
    rr = np.zeros((H, W), np.uint8)
    for f in load_json(path('data/roads/roads.geojson'))['features']:
        if f['properties']['type'] in ('freeway', 'ramp'):
            continue
        cv2.polylines(rr, [np.asarray(f['geometry']['coordinates'])[:, :2].round().astype(np.int32)], False, 1, 3)
    dev = cv2.GaussianBlur(rr.astype(np.float32), (0, 0), 9)
    # major routes follow valleys and gaps: their corridors are valley floor
    corr = np.zeros((H, W), np.uint8)
    CORR_W = {'freeway': 16, 'highway': 14, 'ramp': 10, 'arterial': 10, 'main_street': 10, 'collector': 9, 'rural': 8}
    for f in load_json(path('data/roads/roads.geojson'))['features']:
        wpx_ = CORR_W.get(f['properties']['type'])
        if wpx_:
            cv2.polylines(corr, [np.asarray(f['geometry']['coordinates'])[:, :2].round().astype(np.int32)], False, 1, wpx_)
    for f in load_json(path('data/railways/railways.geojson'))['features']:
        cv2.polylines(corr, [np.asarray(f['geometry']['coordinates'])[:, :2].round().astype(np.int32)], False, 1, 10)
    openland = (fd < 0.45) & (rock < 0.10)
    zones = [Polygon(z['pts']) for z in load_json(path('data/manual/zones.json'))['zones'] if z['kind'] not in ('city',)]
    zm = rasterize_polys(zones, (H, W)) > 0
    low = wmask | zm | openland | ((dev > 0.16) & (rock < 0.15))
    low = cv2.morphologyEx(low.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    low = cv2.morphologyEx(low, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    # remove small islands / holes (< 600 px) so landforms are coherent
    lab, n = ndi.label(low)
    sz = ndi.sum(low, lab, range(1, n + 1))
    low = np.isin(lab, 1 + np.where(sz >= 600)[0])
    lab, n = ndi.label(~low)
    sz = ndi.sum(~low, lab, range(1, n + 1))
    low |= np.isin(lab, 1 + np.where(sz < 600)[0])
    low |= wmask | (corr > 0)
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


def fbm(seed, scales=(90, 45, 22, 11), amps=(1.0, 0.5, 0.25, 0.12)):
    rng = np.random.default_rng(seed)
    out = np.zeros((H, W), np.float32)
    for s, a in zip(scales, amps):
        h, w = int(H / s) + 3, int(W / s) + 3
        n = rng.standard_normal((h, w)).astype(np.float32)
        out += a * cv2.resize(n, (W + int(s * 2), H + int(s * 2)), interpolation=cv2.INTER_CUBIC)[int(s):int(s) + H, int(s):int(s) + W]
    return out / sum(amps)


def sfs_detail(decay_px=35.0, light_deg=135.0):
    """Shape-from-shading detail: the base map is lit from the NW, so image
    brightness ~ slope along the light direction. Integrating the high-passed
    luminance along that direction (leaky, so it only adds local detail)
    recovers the painted ridge/spur/ravine structure."""
    a = load_rgb().astype(np.float32)
    L = 0.3 * a[..., 0] + 0.59 * a[..., 1] + 0.11 * a[..., 2]
    L = cv2.GaussianBlur(L, (0, 0), 3.0)
    hp = L - cv2.GaussianBlur(L, (0, 0), 18)
    hp = np.clip(hp, -40, 40)
    D = int(math.hypot(W, H)) + 4
    M = cv2.getRotationMatrix2D((W / 2, H / 2), -light_deg + 180, 1.0)  # rotate so light->shadow runs along +x
    M[0, 2] += D / 2 - W / 2
    M[1, 2] += D / 2 - H / 2
    r = cv2.warpAffine(hp, M, (D, D), flags=cv2.INTER_LINEAR, borderValue=0)
    a_ = math.exp(-1 / decay_px)
    out = np.zeros_like(r)
    acc = np.zeros(D, np.float32)
    for x in range(D):
        acc = acc * a_ + r[:, x]
        out[:, x] = acc
    # symmetric pass (reduces directional streaking)
    acc = np.zeros(D, np.float32)
    out2 = np.zeros_like(r)
    for x in range(D - 1, -1, -1):
        acc = acc * a_ - r[:, x]
        out2[:, x] = acc
    o = 0.5 * (out + out2)
    Mi = cv2.invertAffineTransform(M)
    back = cv2.warpAffine(o, Mi, (W, H), flags=cv2.INTER_LINEAR)
    back = back - cv2.GaussianBlur(back, (0, 0), 30)
    return back / (np.percentile(np.abs(back), 99) + 1e-6)


def main():
    cfg = load_json(path('data/manual/terrain.json'))
    wmask = np.load(path('tools/.cache/water_mask.npy'))
    lines, prof, feats = river_profiles()
    surf, near_cl = water_surface(lines, prof, wmask)

    # valley floor: harmonic interpolation from water surface (+ gentle rise away from water)
    known = wmask.copy()
    valley = harmonic_fill(np.nan_to_num(surf), known)
    valley = np.where(wmask, surf, valley)

    low, fd, rock = lowland_mask(wmask)
    d = ndi.distance_transform_edt(~low).astype(np.float32)
    d = cv2.GaussianBlur(d, (0, 0), 3.0)
    A = feather_field(cfg['amplitude_regions'], cfg['relief']['default_amplitude_m'])
    rk = np.clip(cv2.GaussianBlur(rock, (0, 0), 3) / 0.35, 0, 1)
    rel = cfg['relief']
    Aeff = A + rel['rock_amplitude_bonus_m'] * rk * (A / 200)
    s0 = rel['base_slope_m_per_px'] + rel['rock_slope_bonus'] * rk
    relief = Aeff * (1 - np.exp(-d * s0 / np.maximum(Aeff, 1)))
    # round off apexes (distance fields make conical tops): scale-aware smoothing
    r1 = cv2.GaussianBlur(relief, (0, 0), 4)
    r2 = cv2.GaussianBlur(relief, (0, 0), 10)
    wgt = np.clip(d / 40.0, 0, 1)  # the further from lowland (higher), the rounder
    relief = (relief * (1 - wgt) * 0.3 + r1 * (0.7 - 0.4 * wgt) + r2 * (0.3 + 0.7 * wgt)) / 1.3
    # lowland itself is not perfectly flat: gentle rolling (farmland/town hills)
    dl = ndi.distance_transform_edt(low).astype(np.float32)
    roll = (fbm(7) * 0.5 + 0.5) * np.clip(dl / 25, 0, 1) * 10
    roll *= ~wmask
    relief = relief + roll
    relief = cv2.GaussianBlur(relief, (0, 0), 2.2)
    mt = np.clip(relief / 60.0, 0, 1)
    relief += cfg['relief'].get('sfs_amplitude_m', 12) * cv2.GaussianBlur(sfs_detail(), (0, 0), 1.5) * mt
    # natural variation, scaled by relief (never overrides the large-scale geography)
    relief *= (1 + 0.10 * fbm(11, scales=(70, 35), amps=(1, 0.5)))
    relief += 2.0 * fbm(13, scales=(10, 5), amps=(1, 0.5)) * np.clip(relief / 60, 0, 1)

    # flatten developed zones
    flat = np.ones((H, W), np.float32)
    for z in load_json(path('data/manual/zones.json'))['zones']:
        f = cfg['flatten_zone_kinds'].get(z['kind'])
        if f is None:
            continue
        m = rasterize_polys([Polygon(z['pts'])], (H, W), value=255, ups=2).astype(np.float32) / 255
        m = cv2.GaussianBlur(m, (0, 0), 12)
        flat = np.minimum(flat, 1 - m * (1 - f))
    relief *= flat

    # banks: relief -> 0 at water with a smooth floodplain
    dw = ndi.distance_transform_edt(~wmask).astype(np.float32)
    dws = cv2.GaussianBlur(dw, (0, 0), 1.5)
    bank = np.clip(dws / 12.0, 0, 1) ** 1.3
    elev = valley + relief * bank + 1.0 * np.clip(dws / 4, 0, 1)
    # channel: bed below surface by depth from local width
    wdt = ndi.distance_transform_edt(wmask).astype(np.float32)
    depth = np.clip(0.6 + 0.35 * wdt, 0.6, 4.5)
    elev = np.where(wmask, surf - depth, elev)
    elev = elev.astype(np.float32)

    # hydrology guarantee: along every channel the bed never rises downstream
    for k, L in lines.items():
        xi = np.clip(L[:, 0].astype(int), 0, W - 1)
        yi = np.clip(L[:, 1].astype(int), 0, H - 1)
        bed = elev[yi, xi]
        mono = np.minimum.accumulate(bed)
        elev[yi, xi] = mono

    os.makedirs(path('data/terrain'), exist_ok=True)
    elev.tofile(path('data/terrain/height_f32.bin'))
    np.nan_to_num(surf, nan=np.nan).astype(np.float32).tofile(path('data/terrain/water_level_f32.bin'))
    lo, hi = float(np.floor(elev.min())), float(np.ceil(elev.max()))
    u16 = ((elev - lo) / (hi - lo) * 65535).round().astype(np.uint16)
    Image.fromarray(u16).save(path('data/terrain/height_u16.png'))
    meta = {
        'grid': {'width': W, 'height': H, 'spacing_m': MPP, 'origin': 'sample (i,j) is at source px (i+0.5, j+0.5)'},
        'files': {'height_f32': 'data/terrain/height_f32.bin', 'height_u16': 'data/terrain/height_u16.png',
                  'water_level_f32': 'data/terrain/water_level_f32.bin'},
        'u16_decode': {'min_m': lo, 'max_m': hi, 'formula': 'h = min + v/65535*(max-min)'},
        'stats': {'min_m': rnd(elev.min(), 1), 'max_m': rnd(elev.max(), 1), 'mean_m': rnd(elev.mean(), 1)},
        'river_profiles_m': {k: [rnd(v[0], 1), rnd(v[-1], 1)] for k, v in prof.items()},
        'note': 'Reconstructed from the map geography (not survey data). Rebuild with tools/pipeline/terrain.py; roads grade/terrain cut-fill is applied by tools/pipeline/grading.py.'
    }
    save_json(path('data/terrain/terrain.json'), meta, indent=1)
    hillshade(elev, path('assets/maps/debug/hillshade.png'))
    print('terrain', meta['stats'])


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
