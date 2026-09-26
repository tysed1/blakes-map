"""Backdrop terrain surrounding the playable map (art: graphics ref.png - layered, hazy blue
Appalachian ridges to the horizon).

No mirroring. Around the map the landscape is built from:
  * edge continuation: the map's own edge profile carried outward and relaxed (so ridges and
    valleys that leave the map carry on for a while instead of ending at a wall),
  * river valleys: every river/creek that crosses the map border continues as a valley floor
    (downstream falling, upstream rising gently) with its water surface, meandering slightly,
  * layered ranges: NW-SE grained ridged ranges (the map's own grain) rising with distance into
    a horizon of higher mountains (highest to the north, the Blue Ridge side),
  * drainage erosion (stream power on a priority-flood tree) so the ranges have real hollows,
    spurs and dendritic valleys at every distance.

Output: data/terrain/backdrop_u16.bin (+ backdrop.json), 10 m cells, covering the map with PAD
metres on every side; data/terrain/backdrop_water_u8.bin (river water outside the map).
"""
import sys, os, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import cv2
from scipy import ndimage as ndi
from PIL import Image
from tools.lib.common import path, load_json, save_json, W, H
from tools.pipeline.terrain import stream_power, fbm, smoothstep, _flood, _receivers, _area

CELL = 4       # source px per backdrop cell (10 m)
PAD_PX = 2800  # 7 km on each side
GRAIN_DEG = 28.0  # ridge grain: NW-SE in image space (x right, y down), like the map's ridges


def aniso_noise(shape, seed, along, across, angle_deg, octaves=4, warp=0.0):
    """Anisotropic value-noise fbm, long axis rotated to angle_deg (image axes), in [-1, 1]."""
    hh, ww = shape
    big = int(math.hypot(hh, ww)) + 8
    out = np.zeros((big, big), np.float32)
    amp, tot = 1.0, 0.0
    rng = np.random.default_rng(seed)
    for o in range(octaves):
        sa, sc = along / 2 ** o, across / 2 ** o
        n = rng.standard_normal((int(big / sc) + 4, int(big / sa) + 4)).astype(np.float32)
        n = cv2.resize(n, (big + int(2 * sa), big + int(2 * sc)), interpolation=cv2.INTER_CUBIC)[int(sc):int(sc) + big, int(sa):int(sa) + big]
        out += amp * n
        tot += amp
        amp *= 0.5
    out /= tot
    M = cv2.getRotationMatrix2D((big / 2, big / 2), -angle_deg, 1.0)
    out = cv2.warpAffine(out, M, (big, big), borderMode=cv2.BORDER_REFLECT)
    y0, x0 = (big - hh) // 2, (big - ww) // 2
    out = out[y0:y0 + hh, x0:x0 + ww]
    if warp > 0:
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        wx = fbm(seed + 7, scales=(60, 30), amps=(1, .5), shape=shape) * warp
        wy = fbm(seed + 8, scales=(60, 30), amps=(1, .5), shape=shape) * warp
        out = cv2.remap(out, xx + wx, yy + wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out / (np.abs(out).max() + 1e-6)


def river_exits(pad):
    """Rivers crossing the map border: start point (cells), outward direction, level, and whether
    the river leaves (downstream) or enters (upstream) the map there."""
    WL = np.fromfile(path('data/terrain/water_level_f32.bin'), np.float32).reshape(H, W)
    out = []
    for f in load_json(path('data/water/waterways.geojson'))['features']:
        c = np.asarray(f['geometry']['coordinates'], float)
        wpx = f['properties'].get('width_px') or [6]
        for end, sgn in ((0, -1), (-1, 1)):
            p = c[end]
            if min(p[0], W - p[0], p[1], H - p[1]) > 4:
                continue
            q = c[max(0, len(c) - 25)] if end == -1 else c[min(len(c) - 1, 24)]
            dvec = p - q
            dvec /= max(np.hypot(*dvec), 1e-6)
            xi, yi = int(np.clip(p[0], 0, W - 1)), int(np.clip(p[1], 0, H - 1))
            win = WL[max(0, yi - 4):yi + 5, max(0, xi - 4):xi + 5]
            lvl = float(np.nanmin(win)) if np.isfinite(win).any() else float(np.nanmin(WL))
            width = float(wpx[end]) if isinstance(wpx, list) else 6.0
            out.append(dict(id=f['properties']['id'], p=((p[0] + PAD_PX) / CELL, (p[1] + PAD_PX) / CELL), d=dvec, level=lvl,
                            downstream=(sgn == 1), width_cells=max(width / CELL, 1.0), cls=f['properties']['class']))
    return out


def main():
    T = np.fromfile(path('data/terrain/height_graded_f32.bin'), np.float32).reshape(H, W)
    small = cv2.resize(T, (W // CELL, H // CELL), interpolation=cv2.INTER_AREA)
    pad = PAD_PX // CELL
    sh, sw = small.shape
    hh, ww = sh + 2 * pad, sw + 2 * pad
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    inside = np.zeros((hh, ww), bool)
    inside[pad:pad + sh, pad:pad + sw] = True
    dx = np.maximum(0, np.maximum(pad - xx, xx - (pad + sw - 1)))
    dy = np.maximum(0, np.maximum(pad - yy, yy - (pad + sh - 1)))
    d = np.hypot(dx, dy)                  # cells outside the map rect
    dm = d * CELL * 2.5                   # metres
    base = float(np.median(small))

    # 1) edge continuation along the terrain grain: every outside cell looks back along the NW-SE grain
    #    to where that line meets the map, so ridges and valleys crossing the border carry on in their
    #    own direction and relax into a smooth multi-scale extrapolation (no mirroring, no streaks)
    ext = push_pull(np.where(inside, small_full(small, pad, hh, ww), 0).astype(np.float32), inside)
    gv, gt = grain_continuation(small, pad, hh, ww)
    wg = np.exp(-np.clip(gt, 0, None) * CELL * 2.5 / 1300.0)
    wg = np.where(np.isfinite(gt), wg, 0.0)
    wg = cv2.GaussianBlur(wg.astype(np.float32), (0, 0), 3)
    gv = np.where(np.isfinite(gv), gv, ext)
    # continued forms meander and widen with distance instead of extruding in straight lines
    wamp = np.clip(np.where(np.isfinite(gt), gt, 0) * 0.08, 0, 14).astype(np.float32)
    yy_, xx_ = np.mgrid[0:hh, 0:ww].astype(np.float32)
    gv = cv2.remap(gv.astype(np.float32), xx_ + fbm(61, scales=(50, 25), amps=(1, .5), shape=(hh, ww)) * wamp,
                   yy_ + fbm(62, scales=(50, 25), amps=(1, .5), shape=(hh, ww)) * wamp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gv = gv * 0.5 + cv2.GaussianBlur(gv, (0, 0), 3) * 0.5
    edge = ext + (gv - ext) * wg
    # right at the rim the border profile itself continues (perpendicular), turning to the grain within ~300 m
    rep = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    a = smoothstep(0, 300, dm)
    edge = rep * (1 - a) + edge * a
    edge = np.where(inside, small_full(small, pad, hh, ww), edge)
    edge_low = cv2.GaussianBlur(ext, (0, 0), 25)

    # 2) layered ranges, NW-SE grain, rising with distance (higher to the north)
    r1 = 1 - np.abs(aniso_noise((hh, ww), 1, 520, 115, GRAIN_DEG, 3, warp=22))
    r2 = 1 - np.abs(aniso_noise((hh, ww), 2, 190, 48, GRAIN_DEG + 10, 2, warp=10))
    rid = 0.74 * r1 ** 1.8 + 0.26 * r2 ** 1.5
    north = np.clip((pad - yy) / pad, 0, 1)      # 0 at map top edge -> 1 at the far north edge
    south = np.clip((yy - (pad + sh)) / pad, 0, 1)
    side = np.clip(np.maximum(pad - xx, xx - (pad + sw)) / pad, 0, 1)
    rise = smoothstep(150, 6000, dm)
    amp = 120 + 700 * rise ** 0.8 * (0.8 + 0.4 * north - 0.05 * south + 0.1 * side)
    lift = 420 * rise ** 1.2 * (0.85 + 0.4 * north)
    rng_h = edge_low + lift + amp * (rid - 0.22)
    w = smoothstep(0, 1300, dm)
    z = edge * (1 - w) + np.maximum(rng_h, edge_low - 25) * w

    # 3) river valleys continue beyond the map, routed by the landscape's own drainage:
    #    rivers leaving the map follow the flow tree to the backdrop border (they merge at natural
    #    confluences); rivers entering the map come down the main valley that drains into them.
    water = np.zeros((hh, ww), bool)
    valley_cap = np.full((hh, ww), np.inf, np.float32)
    # least-cost routes (low ground, gentle curvature) from each border crossing to a regional outlet
    # (downstream) or source (upstream); routes sharing an outlet converge into common trunks
    from skimage.graph import MCP_Geometric
    z2 = cv2.resize(z, (ww // 2, hh // 2), interpolation=cv2.INTER_AREA)
    ins2 = cv2.resize(inside.astype(np.uint8), (ww // 2, hh // 2), interpolation=cv2.INTER_NEAREST) > 0
    zb = cv2.GaussianBlur(z2, (0, 0), 4)
    cost = 1.0 + np.clip((zb - np.percentile(zb[~ins2], 5)) / 60.0, 0, None) ** 1.5
    cost *= 1 + 0.9 * np.clip(fbm(71, scales=(40, 20), amps=(1, .5), shape=cost.shape) + 0.3, 0, None)  # meanders
    rim = ndi.distance_transform_edt(~ins2)
    cost[ins2] = 1e6
    cost[(rim < 40) & ~ins2] *= 6.0   # leave the map rim instead of hugging it
    exits = river_exits(pad)
    for rv in exits:
        q = np.array(rv['p']) + rv['d'] * 60 + np.array([-rv['d'][1], rv['d'][0]]) * 0.0
        tgt = region_target(rv, pad, sw, sh, ww, hh)
        s0 = (int(np.clip(q[1] / 2, 0, hh // 2 - 1)), int(np.clip(q[0] / 2, 0, ww // 2 - 1)))
        t0 = (int(np.clip(tgt[1] / 2, 0, hh // 2 - 1)), int(np.clip(tgt[0] / 2, 0, ww // 2 - 1)))
        m = MCP_Geometric(cost)
        m.find_costs([s0], [t0])
        tr = np.asarray(m.traceback(t0), float)  # (row, col) from s0 to t0
        path_ = np.stack([tr[:, 1] * 2 + 1, tr[:, 0] * 2 + 1], 1)
        if not rv['downstream']:
            maxlen = 520 if rv['cls'] == 'river' else 200
            L_ = np.r_[0, np.cumsum(np.hypot(*np.diff(path_, axis=0).T))]
            path_ = path_[L_ <= maxlen]
        lead = np.array(rv['p']) + rv['d'][None, :] * np.linspace(0, 58, 30)[:, None]
        path_ = np.vstack([lead, path_])
        pts = np.asarray(path_, float)
        if len(pts) > 12:
            sm = np.stack([ndi.gaussian_filter1d(pts[:, 0], 5, mode='nearest'), ndi.gaussian_filter1d(pts[:, 1], 5, mode='nearest')], 1)
            sm[0] = pts[0]
            pts = sm
        seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        slope = (0.0025 if rv['cls'] == 'river' else 0.015) * CELL * 2.5
        lv = rv['level'] + (-1 if rv['downstream'] else 1) * slope * seg
        mk = np.zeros((hh, ww), np.uint8)
        cv2.polylines(mk, [pts.round().astype(np.int32)], False, 1, 1)
        dist, (iy, ix) = ndi.distance_transform_edt(mk == 0, return_indices=True)
        lvl_map = np.full((hh, ww), np.nan, np.float32)
        pi = pts.round().astype(int)
        okp = (pi[:, 0] >= 0) & (pi[:, 0] < ww) & (pi[:, 1] >= 0) & (pi[:, 1] < hh)
        lvl_map[pi[okp, 1], pi[okp, 0]] = lv[okp]
        L = lvl_map[iy, ix]
        L = np.where(np.isfinite(L), L, rv['level'])
        big = rv['cls'] == 'river'
        hw = rv['width_cells'] * 0.5 + 0.5
        if not rv['downstream']:
            # upstream: the channel narrows toward its source
            frac = np.interp(np.arange(len(pts)), [0, len(pts) - 1], [1.0, 0.3])
            hw = hw * frac[np.clip(np.searchsorted(seg, seg), 0, len(pts) - 1)].mean()
        floor_w = (7 if big else 2.5)
        ex = np.maximum(dist - floor_w, 0)
        cap = L + 0.6 + np.maximum(dist - hw, 0) * 0.15 + ex * (1.1 if big else 2.2) + ex * ex * (0.03 if big else 0.06)
        cap = cap + 18 * fbm(90 + len(path_) % 50, scales=(40, 20), amps=(1, 0.5), shape=(hh, ww)) * np.clip(ex / 25, 0, 1)
        # near the rim the map's own valley (continued) rules; the designed valley takes over outward
        cap = cap + (1 - smoothstep(0, 450, dm)) * 400
        valley_cap = np.minimum(valley_cap, np.where(inside, np.inf, cap).astype(np.float32))
        wet = (dist <= hw) & ~inside
        water |= wet
        z = np.where(wet, np.minimum(z, L - 1.5), z)
    z = np.where(np.isfinite(valley_cap), np.minimum(z, valley_cap), z)

    # 4) drainage erosion on the outside (20 m cells); the map and river water stay fixed
    z = z + (6 + 18 * rise) * fbm(81, scales=(24, 12, 6), amps=(1, .5, .25), shape=(hh, ww)) * smoothstep(60, 600, dm) * ~water
    zc = cv2.resize(z, (ww // 2, hh // 2), interpolation=cv2.INTER_AREA)
    fixed = cv2.resize((inside | water).astype(np.uint8), (ww // 2, hh // 2), interpolation=cv2.INTER_NEAREST) > 0
    ze = stream_power(zc, fixed, 20.0, 0.03, 0.5, 1.0, 30, 12.0)
    dz = cv2.resize((ze - zc).astype(np.float32), (ww, hh), interpolation=cv2.INTER_CUBIC)
    z = z + dz * smoothstep(20, 400, dm)
    z = np.where(water, np.minimum(z, cv2.GaussianBlur(z, (0, 0), 1)), z)
    # near the rim, keep continuity with the map edge exactly
    z = np.where(inside, small_full(small, pad, hh, ww), z)
    blend = smoothstep(0, 60, dm)
    z = z * blend + edge * (1 - blend)

    out = z.astype(np.float32)
    lo, hi = float(np.floor(out.min())), float(np.ceil(out.max()))
    ((out - lo) / (hi - lo) * 65535).round().astype('<u2').tofile(path('data/terrain/backdrop_u16.bin'))
    water.astype(np.uint8).tofile(path('data/terrain/backdrop_water_u8.bin'))
    meta = {'cell_px': CELL, 'cell_m': CELL * 2.5, 'w': ww, 'h': hh, 'origin_px': [-PAD_PX, -PAD_PX], 'min_m': lo, 'max_m': hi,
            'map_rect_cells': [pad, pad, pad + sw, pad + sh],
            'note': 'cell (i,j) centre at source px (origin + (i+0.5)*cell). Inside map_rect the real terrain is used by the viewer instead.'}
    save_json(path('data/terrain/backdrop.json'), meta, indent=1)
    Image.fromarray(((out - lo) / (hi - lo) * 255).astype(np.uint8)).save(path('assets/maps/debug/backdrop.png'))
    print('backdrop', out.shape, lo, hi)


def region_target(rv, pad, sw, sh, ww, hh):
    """Regional drainage (cells): the south-flowing rivers join and leave to the south-west, the Tanner
    system leaves east, Mill Creek west; inflows come down from sources in the northern/eastern ranges."""
    px, py = rv['p']
    if rv['downstream']:
        if py >= pad + sh - 2:
            return (pad + sw * 0.30, hh - 1)
        if px >= pad + sw - 2:
            return (ww - 1, pad + sh * 0.85)
        if px <= pad + 2:
            return (0, pad + sh * 0.2)
        return (px, 0)
    if py <= pad + 2:
        return (px + (py - pad) * 0 + (px - (pad + sw / 2)) * 0.35, 0)
    if px >= pad + sw - 2:
        return (ww - 1, py - pad * 0.3)
    return (0, py)


def grain_continuation(small, pad, hh, ww, inset=0.5):
    """For each cell, the map value where the line through it along the grain (GRAIN_DEG) first meets the
    map rectangle, and the distance (cells) to that point. inf where the line misses the map."""
    th = math.radians(GRAIN_DEG)
    u = np.array([math.cos(th), math.sin(th)])
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float64)
    x0, x1 = pad + inset, pad + small.shape[1] - 1 - inset
    y0, y1 = pad + inset, pad + small.shape[0] - 1 - inset
    best_t = np.full((hh, ww), np.inf)
    for sgn in (1, -1):
        ux, uy = u * sgn
        tx0, tx1 = (x0 - xx) / ux, (x1 - xx) / ux
        ty0, ty1 = (y0 - yy) / uy, (y1 - yy) / uy
        tmin = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
        tmax = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
        hit = (tmax >= tmin) & (tmax >= 0)
        t = np.where(hit, np.maximum(tmin, 0), np.inf)
        best_t = np.minimum(best_t, t)
        if sgn == 1:
            tp = t
        else:
            tn = t
    use_pos = tp <= tn
    t = np.where(use_pos, tp, tn)
    ux = np.where(use_pos, u[0], -u[0]); uy = np.where(use_pos, u[1], -u[1])
    hx = xx + ux * np.where(np.isfinite(t), t, 0) - pad
    hy = yy + uy * np.where(np.isfinite(t), t, 0) - pad
    hx = np.clip(hx, 0, small.shape[1] - 1.001); hy = np.clip(hy, 0, small.shape[0] - 1.001)
    val = cv2.remap(small.astype(np.float32), hx.astype(np.float32), hy.astype(np.float32), cv2.INTER_LINEAR)
    val = np.where(np.isfinite(t), val, np.nan)
    return val.astype(np.float32), t.astype(np.float32)


def push_pull(vals, known, levels=9):
    """Smooth extrapolation of known values into unknown cells (multi-scale pull-push)."""
    v = [vals * known]; k = [known.astype(np.float32)]
    for _ in range(levels):
        v.append(cv2.resize(v[-1], (max(1, v[-1].shape[1] // 2), max(1, v[-1].shape[0] // 2)), interpolation=cv2.INTER_AREA))
        k.append(cv2.resize(k[-1], (max(1, k[-1].shape[1] // 2), max(1, k[-1].shape[0] // 2)), interpolation=cv2.INTER_AREA))
    est = v[-1] / np.maximum(k[-1], 1e-6)
    for lv in range(levels - 1, -1, -1):
        up = cv2.resize(est, (v[lv].shape[1], v[lv].shape[0]), interpolation=cv2.INTER_LINEAR)
        up = cv2.GaussianBlur(up, (0, 0), 1.0)
        kk = np.clip(k[lv], 0, 1)
        est = (v[lv] / np.maximum(k[lv], 1e-6)) * kk + up * (1 - kk)
    return est.astype(np.float32)


def small_full(small, pad, hh, ww):
    o = np.zeros((hh, ww), np.float32)
    o[pad:pad + small.shape[0], pad:pad + small.shape[1]] = small
    return o


if __name__ == '__main__':
    main()
