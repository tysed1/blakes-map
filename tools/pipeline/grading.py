"""Vertical road engineering + terrain grading.

1. Node elevations (junction-consistent) from smoothed terrain, relaxed so every
   edge is feasible under its max grade.
2. Edge profiles: terrain-following (smoothed per class) with fixed node ends,
   lower bounds for bridge decks over water and for overpasses at grade
   separations, then max-grade enforcement.
3. Crossing classification: bridge / culvert / ford by water width + road class.
4. Terrain grading: road bed flattened to the profile, cut/fill side slopes
   blended into natural terrain (never inside river channels or under spans).
Outputs:
  data/roads/roads.geojson        coordinates become [x, y, z_m]; props gain max_grade_pct, earthwork
  data/roads/bridges.geojson      spans with deck elevation, clearance, kind, structure
  data/railways/railways.geojson  z added
  data/terrain/height_graded_f32.bin (+ u16 png, hillshade) - the terrain used by all viewers
  data/qa/grade_report.json
"""
import sys, os, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import cv2
import networkx as nx
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union, substring
from shapely.strtree import STRtree
from PIL import Image

from tools.lib.common import path, load_json, save_json, W, H, rnd
from tools.lib.trace import resample
from tools.lib.geom import geojson_line, geojson_point, fc

MPP = 2.5
MAX_GRADE = {'freeway': 0.06, 'highway': 0.08, 'ramp': 0.07, 'arterial': 0.08, 'main_street': 0.08, 'collector': 0.10,
             'urban_street': 0.12, 'residential': 0.12, 'rural': 0.12, 'gravel': 0.15, 'dirt': 0.18, 'driveway': 0.20, 'rail': 0.022}
SMOOTH_PX = {'freeway': 40, 'highway': 30, 'ramp': 12, 'arterial': 16, 'main_street': 10, 'collector': 10, 'urban_street': 6,
             'residential': 6, 'rural': 8, 'gravel': 6, 'dirt': 4, 'driveway': 3, 'rail': 50}
CLEAR_WATER = {'river': 6.0, 'creek': 3.5, 'slough': 3.0}
OVERPASS_CLEAR = 7.5  # deck-to-road incl. structure depth
RAIL_OVER_CLEAR = 8.0
MOAT_PX = 1.2  # terrain kept below road level this far beyond the road bed (+ ditch)


def bilinear(F, x, y):
    x = np.clip(np.asarray(x) - 0.5, 0, F.shape[1] - 1.001)
    y = np.clip(np.asarray(y) - 0.5, 0, F.shape[0] - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    fx = x - x0; fy = y - y0
    return (F[y0, x0] * (1 - fx) * (1 - fy) + F[y0, x0 + 1] * fx * (1 - fy) + F[y0 + 1, x0] * (1 - fx) * fy + F[y0 + 1, x0 + 1] * fx * fy)


def enforce_grade(z, g, zmin=None, fixed_ends=True, iters=6):
    z = z.copy()
    n = len(z)
    for _ in range(iters):
        if zmin is not None:
            z = np.maximum(z, zmin)
        # raise-only passes (build approach ramps up to bridges/overpasses)
        for i in range(1, n):
            z[i] = max(z[i], z[i - 1] - g)
        for i in range(n - 2, -1, -1):
            z[i] = max(z[i], z[i + 1] - g)
        # symmetric smoothing passes limiting grade both ways
        for i in range(1, n):
            z[i] = min(max(z[i], z[i - 1] - g), z[i - 1] + g)
        for i in range(n - 2, -1, -1):
            z[i] = min(max(z[i], z[i + 1] - g), z[i + 1] + g)
    return z


def relax_minor_nodes(roads, routes, nz, nodes_xy, Ts, minor_from, iters=200):
    """Joint elevations for the local-street network: stay close to terrain, but
    every street between two junctions must be drivable under its max grade.
    Junctions already fixed by major routes are immovable."""
    edges = []
    for imp, order in routes:
        if imp < minor_from:
            continue
        for (i, a, b) in order:
            p = roads[i]['properties']
            if p['type'] == 'ramp':
                continue
            L = LineString(np.asarray(roads[i]['geometry']['coordinates'])[:, :2]).length * MPP
            edges.append((p['from'], p['to'], MAX_GRADE[p['type']] * 0.9 * max(L, 1.0)))
    free = set()
    for u, v, _ in edges:
        for n in (u, v):
            if n not in nz:
                x, y = nodes_xy[n]
                nz[n] = float(bilinear(Ts, x, y))
                free.add(n)
    for _ in range(iters):
        moved = 0
        for u, v, lim in edges:
            dz = nz[u] - nz[v]
            if abs(dz) <= lim + 1e-6:
                continue
            ex = abs(dz) - lim
            s = 1 if dz > 0 else -1
            fu, fv = u in free, v in free
            if fu and fv:
                nz[u] -= s * ex / 2; nz[v] += s * ex / 2
            elif fu:
                nz[u] -= s * ex
            elif fv:
                nz[v] += s * ex
            else:
                continue
            moved += 1
        if not moved:
            break


def span_runs(XY, span_geoms, tol=0.6):
    if not span_geoms:
        return []
    on = np.zeros(len(XY), bool)
    for g in span_geoms:
        if g.is_empty:
            continue
        x0, y0, x1, y1 = g.bounds
        cand = np.nonzero((XY[:, 0] >= x0 - 1) & (XY[:, 0] <= x1 + 1) & (XY[:, 1] >= y0 - 1) & (XY[:, 1] <= y1 + 1))[0]
        for k in cand:
            if g.distance(Point(XY[k])) < tol:
                on[k] = True
    runs, k = [], 0
    while k < len(on):
        if on[k]:
            j = k
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            if j > k:
                runs.append((k, j))
            k = j + 1
        else:
            k += 1
    return runs


def span_info(XY, ia, ib, t, WL, wlines):
    seg = XY[ia:ib + 1]
    wl = WL[np.clip(seg[:, 1].astype(int), 0, H - 1), np.clip(seg[:, 0].astype(int), 0, W - 1)]
    if not np.isfinite(wl).any():
        return None
    wl = float(np.nanmax(wl))
    mid = Point(XY[(ia + ib) // 2])
    wcls = min(wlines, key=lambda w: w[0].distance(mid))[1]['class']
    wid = float(np.hypot(*np.diff(seg, axis=0).T).sum()) * MPP
    if t in ('dirt', 'driveway') and wcls == 'creek' and wid < 8:
        clear = None
    elif wcls == 'creek' and wid < 7 and t not in ('freeway', 'highway'):
        clear = 1.8
    else:
        clear = CLEAR_WATER.get(wcls, 4.0) + (1.5 if t in ('freeway', 'highway') else 0)
        if t in ('gravel', 'dirt', 'driveway'):
            clear = min(clear, 1.3)  # low-water bridge (concrete slab a metre above normal flow)
        elif t == 'residential':
            clear = min(clear, 2.5)
        elif t in ('rural', 'urban_street'):
            clear = min(clear, 4.5)
    return wl, wcls, wid, clear


def runs_of(mask, min_len=1):
    out, k = [], 0
    while k < len(mask):
        if mask[k]:
            j = k
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            if j - k + 1 >= min_len:
                out.append((k, j))
            k = j + 1
        else:
            k += 1
    return out


CURB_ZONES = {'downtown', 'industrial', 'town_center', 'city', 'town'}
WALL_ZONES = {'downtown', 'industrial', 'town_center'}


def section_for(p, types, zone_kind):
    """Engineered cross-section of an edge (metres) from its type + zone.
    bed_half_m: centreline to the outer edge of the road bed (pavement + shoulders or
    curb + sidewalk). ditch_w_m/ditch_d_m: side ditch beyond the bed (0 when curbed)."""
    t = p['type']
    sec = dict(types[t].get('section', {}))
    width = p.get('width_m') or types[t]['width_m']
    zk = zone_kind(p)
    curb = bool(sec.get('curb_in_zones')) and zk in sec.get('curb_zones', CURB_ZONES) or bool(sec.get('curb'))
    if t in ('freeway', 'ramp'):
        curb = False
    sidewalk = sec.get('sidewalk_m', 0.0) if curb and ('sidewalk_zones' not in sec or zk in sec['sidewalk_zones']) else 0.0
    gravel = 0.0 if curb else sec.get('shoulder_gravel_m', 0.0)
    ditch_w = 0.0 if curb else sec.get('ditch_w_m', 0.0)
    ditch_d = 0.0 if curb else sec.get('ditch_d_m', 0.0)
    return {'curb': curb, 'sidewalk_m': sidewalk, 'shoulder_gravel_m': gravel, 'ditch_w_m': ditch_w, 'ditch_d_m': ditch_d,
            'bed_half_m': width / 2 + gravel + (0.15 + sidewalk if curb else 0.0)}


def _fwd(z, g, pinned):
    z = z.copy()
    for i in range(1, len(z)):
        if not pinned[i]:
            z[i] = min(max(z[i], z[i - 1] - g), z[i - 1] + g)
    return z


def _bwd(z, g, pinned):
    z = z.copy()
    for i in range(len(z) - 2, -1, -1):
        if not pinned[i]:
            z[i] = min(max(z[i], z[i + 1] - g), z[i + 1] + g)
    return z


def enforce_grade_pinned(z, g, zmin, pinned, iters=40):
    """Grade-limited profile close to the target: balanced cut/fill (average of
    forward and backward slope clamps, iterated), bridge/overpass minimums with
    approach ramps, pinned junction elevations untouched."""
    target = z.copy()
    fixed = z.copy()
    has_min = zmin > -1e8
    z = z.copy()
    for it in range(iters):
        z = 0.5 * (_fwd(z, g, pinned) + _bwd(z, g, pinned))
        if has_min.any():
            z = np.maximum(z, zmin)
            # approach ramps up to raised decks
            for i in range(1, len(z)):
                if not pinned[i]:
                    z[i] = max(z[i], z[i - 1] - g)
            for i in range(len(z) - 2, -1, -1):
                if not pinned[i]:
                    z[i] = max(z[i], z[i + 1] - g)
        z[pinned] = fixed[pinned]
    # final strict pass so the limit is never exceeded (except at infeasible pins)
    z = _bwd(_fwd(z, g, pinned), g, pinned)
    if has_min.any():
        z = np.maximum(z, zmin)
        # raise-only approach ramps so a deck minimum never creates a step
        for i in range(1, len(z)):
            if not pinned[i]:
                z[i] = max(z[i], z[i - 1] - g)
        for i in range(len(z) - 2, -1, -1):
            if not pinned[i]:
                z[i] = max(z[i], z[i + 1] - g)
    z[pinned] = fixed[pinned]
    return z


def main():
    T = np.fromfile(path('data/terrain/height_f32.bin'), np.float32).reshape(H, W)
    WL = np.fromfile(path('data/terrain/water_level_f32.bin'), np.float32).reshape(H, W)
    wmask = ~np.isnan(WL)
    Ts = cv2.GaussianBlur(T, (0, 0), 2.5)
    types = load_json(path('data/roads/road_types.json'))['types']
    global load_json_types
    load_json_types = types
    rfc = load_json(path('data/roads/roads.geojson'))
    roads = rfc['features']
    water_poly = unary_union([shape(f['geometry']) for f in load_json(path('data/water/water_bodies.geojson'))['features']])
    for f in roads:  # water spans are always recomputed from geometry (never read back from a previous run)
        g2 = LineString(np.asarray(f['geometry']['coordinates'])[:, :2])
        spans = []
        if g2.intersects(water_poly):
            x = g2.intersection(water_poly)
            for s_ in ([x] if x.geom_type == 'LineString' else [q for q in getattr(x, 'geoms', []) if q.geom_type == 'LineString']):
                if s_.length < 1.0:
                    continue
                a0, a1 = sorted([g2.project(Point(s_.coords[0])), g2.project(Point(s_.coords[-1]))])
                pad = 2.0 + types[f['properties']['type']]['width_m'] / MPP * 0.3
                a0, a1 = max(0.0, a0 - pad), min(g2.length, a1 + pad)
                # extend over low banks / pools that sit below the water surface
                wlv = WL[np.clip(int(s_.coords[0][1]), 0, H - 1), np.clip(int(s_.coords[0][0]), 0, W - 1)]
                if np.isfinite(wlv) and f['properties']['type'] in ('gravel', 'dirt', 'driveway', 'rural', 'residential'):
                    for _ in range(14):
                        q = g2.interpolate(max(0.0, a0 - 1))
                        if a0 <= 0 or bilinear(T, q.x, q.y) > wlv + 0.4:
                            break
                        a0 = max(0.0, a0 - 1)
                    for _ in range(14):
                        q = g2.interpolate(min(g2.length, a1 + 1))
                        if a1 >= g2.length or bilinear(T, q.x, q.y) > wlv + 0.4:
                            break
                        a1 = min(g2.length, a1 + 1)
                spans.append([a0, a1])
        f['properties']['bridge_spans'] = spans
    nodes = {f['properties']['id']: f['geometry']['coordinates'][:2] for f in load_json(path('data/roads/road_nodes.geojson'))['features']}
    ww = load_json(path('data/water/waterways.geojson'))['features']
    wlines = [(LineString(f['geometry']['coordinates']), f['properties']) for f in ww]
    water = unary_union([shape(f['geometry']) for f in load_json(path('data/water/water_bodies.geojson'))['features']])

    # ---------------- rails first (the least flexible alignments; roads adapt to them)
    rails = load_json(path('data/railways/railways.geojson'))
    rail_geoms = [LineString(np.asarray(f['geometry']['coordinates'])[:, :2]) for f in rails['features']]
    rail_prof = []
    for rf, rg in zip(rails['features'], rail_geoms):
        n = max(2, int(math.ceil(rg.length)) + 1)
        S = np.linspace(0, rg.length, n)
        XY = np.array([rg.interpolate(s_).coords[0] for s_ in S])
        zt = bilinear(T, XY[:, 0], XY[:, 1])
        pad = min(n - 1, 150)
        ext = np.r_[2 * zt[0] - zt[1:pad + 1][::-1], zt, 2 * zt[-1] - zt[-pad - 1:-1][::-1]]
        zs = ndi.gaussian_filter1d(ext, 50)[pad:pad + n]
        zmin = np.full(n, -1e9)
        for a_, b_ in rf['properties']['bridge_spans']:
            seg = XY[int(a_):int(b_) + 1]
            wl = np.nanmax(WL[seg[:, 1].astype(int), seg[:, 0].astype(int)])
            if np.isfinite(wl):
                zmin[int(a_):int(b_) + 1] = wl + 5.0
        z = enforce_grade(zs, MAX_GRADE['rail'] * MPP, zmin=zmin)
        rail_prof.append((rf, rg, S, XY, z, zt))
    rail_crossings = []
    import glob
    rail_mode = {r['id']: r['rail_crossing'] for f_ in glob.glob(path('data/manual/roads/*.json')) for r in load_json(f_)['roads'] if r.get('rail_crossing')}
    ic_spec = load_json(path('data/manual/interchanges.json'))
    fw_over = {(ic['freeway'], x) for ic in ic_spec['interchanges'] if ic.get('upper') == 'freeway' for x in ic['crossroad']}

    # ---------------- route-based profiles
    geoms = [LineString(np.asarray(f['geometry']['coordinates'])[:, :2]) for f in roads]
    tree = STRtree(geoms)
    IMPORTANCE = ['freeway', 'highway', 'arterial', 'main_street', 'collector', 'rural', 'urban_street', 'residential', 'gravel', 'dirt', 'driveway', 'ramp']
    groups = {}
    for i, f in enumerate(roads):
        p = f['properties']
        key = p['def_id'] if p.get('def_id') else f"__{i}"
        groups.setdefault(key, []).append(i)
    routes = []
    for key, idxs in groups.items():
        # chain edges into paths
        Gr = nx.MultiGraph()
        for i in idxs:
            Gr.add_edge(roads[i]['properties']['from'], roads[i]['properties']['to'], idx=i)
        for comp in nx.connected_components(Gr):
            sub = Gr.subgraph(comp)
            ends = [n for n in sub.nodes if sub.degree(n) == 1] or [next(iter(sub.nodes))]
            start_n = ends[0]
            order, seen, cur = [], set(), start_n
            while True:
                nxt = [(u, v, d['idx']) for u, v, d in sub.edges(cur, data=True) if d['idx'] not in seen]
                if not nxt:
                    break
                u, v, i = nxt[0]
                seen.add(i)
                other = v if u == cur else u
                order.append((i, cur, other))
                cur = other
            for u, v, d in sub.edges(data=True):
                if d['idx'] not in seen:
                    order.append((d['idx'], u, v))
                    seen.add(d['idx'])
            t = roads[order[0][0]]['properties']['type']
            routes.append((IMPORTANCE.index(t) if t in IMPORTANCE else 99, order))
    routes.sort(key=lambda r: r[0])

    nz = {}
    # junctions right next to an at-grade railway crossing take the rail-head elevation up
    # front (otherwise the crossing pin and the junction pin fight and create a step)
    for i_, f_ in enumerate(roads):
        t_ = f_['properties']['type']
        if t_ in ('freeway', 'highway', 'ramp') or f_['properties'].get('virtual'):
            continue
        for rf, rg, Sr, XYr, zr_, ztr in rail_prof:
            if rf['properties'].get('tracks', 1) >= 3 or not geoms[i_].intersects(rg):
                continue
            x = geoms[i_].intersection(rg)
            for q in ([x] if x.geom_type == 'Point' else [q_ for q_ in getattr(x, 'geoms', []) if q_.geom_type == 'Point']):
                zrail = float(zr_[min(int(round(rg.project(q))), len(zr_) - 1)])
                for nd in (f_['properties']['from'], f_['properties']['to']):
                    if nd in nodes and Point(nodes[nd]).distance(q) < 8.0:
                        nz[nd] = zrail + 0.46
    profiles = {}
    seps = []
    nodes_xy = {f['properties']['id']: f['geometry']['coordinates'][:2] for f in load_json(path('data/roads/road_nodes.geojson'))['features']}

    def edge_samples(i, a_node):
        f = roads[i]
        P = np.asarray(f['geometry']['coordinates'])[:, :2]
        if f['properties']['from'] != a_node:
            P = P[::-1]
        ls = LineString(P)
        n = max(2, int(math.ceil(ls.length)) + 1)
        S = np.linspace(0, ls.length, n)
        return np.array([ls.interpolate(s).coords[0] for s in S]), S

    MINOR_FROM = IMPORTANCE.index('urban_street')
    relaxed = False
    for imp, order in routes:
        if imp >= MINOR_FROM and not relaxed and imp != IMPORTANCE.index('ramp'):
            relax_minor_nodes(roads, routes, nz, nodes_xy, Ts, MINOR_FROM)
            relaxed = True
        XYs, owner, idx_in_edge, node_at = [], [], [], {}
        for (i, a, b) in order:
            XY, S = edge_samples(i, a)
            start = len(XYs[0]) if False else sum(len(x) for x in XYs)
            if XYs and np.hypot(*(XYs[-1][-1] - XY[0])) < 0.5:
                XY = XY[1:]
                off = -1
            else:
                off = 0
            node_at.setdefault(a, start + off if off == -1 else start)
            XYs.append(XY)
            owner += [i] * len(XY)
            node_at[b] = sum(len(x) for x in XYs) - 1
        XY = np.vstack(XYs)
        t = roads[order[0][0]]['properties']['type']
        n = len(XY)
        zt = bilinear(T, XY[:, 0], XY[:, 1])
        sig = SMOOTH_PX[t]
        pad = min(n - 1, int(3 * sig))
        if n > 3 and pad > 0:
            ext = np.r_[2 * zt[0] - zt[1:pad + 1][::-1], zt, 2 * zt[-1] - zt[-pad - 1:-1][::-1]]
            zs = ndi.gaussian_filter1d(ext, sig)[pad:pad + n]
        else:
            zs = zt.copy()
        pins = {k: nz[nd] for nd, k in node_at.items() if nd in nz}
        if pins:
            anchors = {k: pins[k] - zs[k] for k in pins}
            # unpinned route ends keep their own (terrain) elevation: zero correction there
            anchors.setdefault(0, 0.0)
            anchors.setdefault(n - 1, 0.0)
            ks = np.array(sorted(anchors))
            corr = np.interp(np.arange(n), ks, [anchors[k] for k in ks])
            zs = zs + corr
        zmin = np.full(n, -1e9)
        spans_by_edge = {}
        # bridges: route samples lying on any bridge span of the route's edges
        span_geoms = []
        for (i, _, _) in order:
            for sa, sb in roads[i]['properties']['bridge_spans']:
                span_geoms.append(substring(geoms[i], sa, sb))
        runs = span_runs(XY, span_geoms)
        spans_route = []
        for ia, ib in runs:
            info = span_info(XY, ia, ib, t, WL, wlines)
            if info is None:
                continue
            wl, wcls, wid, clear = info
            spans_route.append((ia, ib))
            if clear is not None:
                zmin[ia:ib + 1] = np.maximum(zmin[ia:ib + 1], wl + clear)
        # grade separations against already-profiled freeways / ramps
        if t != 'freeway':
            rl = LineString(XY)
            for j, (Sj, XYj, zj) in [(j, profiles[j][:3]) for j in profiles]:
                tj = roads[j]['properties']['type']
                if not ((tj == 'freeway' and t != 'ramp') or (tj == 'ramp' and t not in ('ramp', 'freeway'))):
                    continue
                if (roads[j]['properties'].get('def_id'), roads[order[0][0]]['properties'].get('def_id')) in fw_over:
                    continue  # the freeway goes over this crossroad (handled in the freeway's own profile)
                if not rl.intersects(geoms[j]):
                    continue
                x = rl.intersection(geoms[j])
                pts = [x] if x.geom_type == 'Point' else [q for q in getattr(x, 'geoms', []) if q.geom_type == 'Point']
                ends_j = [Point(XYj[0]), Point(XYj[-1])]
                ends_i = [Point(roads[i_]['geometry']['coordinates'][e][:2]) for (i_, _, _) in order for e in (0, -1)]
                for q in pts:
                    if min(e.distance(q) for e in ends_j + ends_i) < 1.5:
                        continue  # touching at a shared end (ramp terminal / freeway terminus), not a crossing
                    kj = int(round(LineString(XYj).project(q)))
                    zl = float(zj[min(kj, len(zj) - 1)])
                    k = int(round(rl.project(q)))
                    half = int(load_json_types[t]['width_m'] / MPP / 2) + 2
                    zmin[max(0, k - half):k + half + 1] = np.maximum(zmin[max(0, k - half):k + half + 1], zl + OVERPASS_CLEAR)
                    seps.append({'upper': roads[owner[min(k, n - 1)]]['properties']['id'], 'lower': roads[j]['properties']['id'], 'at': [rnd(q.x), rnd(q.y)], 'lower_z': rnd(zl, 2)})
        # railway crossings: the road goes over the railway (major roads, yards, or when
        # the road is already high) or crosses at grade (pinned to the rail head)
        rl = LineString(XY)
        dfid = roads[order[0][0]]['properties'].get('def_id')
        for rf, rg, Sr, XYr, zr_, ztr in rail_prof:
            if not rl.intersects(rg):
                continue
            x = rl.intersection(rg)
            for q in ([x] if x.geom_type == 'Point' else [q_ for q_ in getattr(x, 'geoms', []) if q_.geom_type == 'Point']):
                k = int(round(rl.project(q)))
                k = min(max(k, 0), n - 1)
                kr = min(int(round(rg.project(q))), len(zr_) - 1)
                zrail = float(zr_[kr])
                tracks = rf['properties'].get('tracks', 1)
                over = t in ('freeway', 'highway', 'ramp') or tracks >= 3
                if dfid in rail_mode:
                    over = rail_mode[dfid] == 'over'
                if not over and zs[k] > zrail + RAIL_OVER_CLEAR * 0.6:
                    over = True
                a_ = np.asarray(rg.interpolate(max(0, rg.project(q) - 3)).coords[0]); b_ = np.asarray(rg.interpolate(rg.project(q) + 3).coords[0])
                c_ = XY[max(0, k - 3)]; d_ = XY[min(n - 1, k + 3)]
                ang = math.degrees(math.acos(min(1, abs(np.dot(b_ - a_, d_ - c_)) / max(np.hypot(*(b_ - a_)) * np.hypot(*(d_ - c_)), 1e-9))))
                ow = owner[min(k, n - 1)]
                if over:
                    half = int((rf['properties']['width_m'] / 2 + 6) / MPP / max(math.sin(math.radians(max(ang, 20))), 0.3)) + 2
                    zmin[max(0, k - half):k + half + 1] = np.maximum(zmin[max(0, k - half):k + half + 1], zrail + RAIL_OVER_CLEAR)
                    seps.append({'upper': roads[ow]['properties']['id'], 'lower': rf['properties']['id'], 'at': [rnd(q.x), rnd(q.y)], 'lower_z': rnd(zrail, 2), 'kind': 'rail', 'angle_deg': rnd(ang, 1)})
                else:
                    pins[k] = zrail + 0.46  # road surface ~7 cm below the rail heads (flangeways; timber crossing panels)
                    rail_crossings.append({'road': roads[ow]['properties']['id'], 'rail': rf['properties']['id'], 'at': [rnd(q.x), rnd(q.y)], 'z': rnd(zrail + 0.46, 2),
                                           'angle_deg': rnd(ang, 1), 'road_type': t, 'tracks': tracks})
        # freeway carried over a crossroad (interchanges.json upper='freeway')
        if t == 'freeway':
            for fw, xr in fw_over:
                if fw != dfid:
                    continue
                for j_, f_ in enumerate(roads):
                    if f_['properties'].get('def_id') != xr or not rl.intersects(geoms[j_]):
                        continue
                    x = rl.intersection(geoms[j_])
                    for q in ([x] if x.geom_type == 'Point' else [q_ for q_ in getattr(x, 'geoms', []) if q_.geom_type == 'Point']):
                        k = min(int(round(rl.project(q))), n - 1)
                        zl = float(bilinear(Ts, q.x, q.y))
                        half = int(load_json_types['arterial']['width_m'] / MPP / 2) + 3
                        zmin[max(0, k - half):k + half + 1] = np.maximum(zmin[max(0, k - half):k + half + 1], zl + OVERPASS_CLEAR)
                        seps.append({'upper': roads[owner[k]]['properties']['id'], 'lower': f_['properties']['id'], 'at': [rnd(q.x), rnd(q.y)], 'lower_z': rnd(zl, 2), 'kind': 'road'})
        if pins:
            anchors = {k: pins[k] - zs[k] for k in pins}  # node pins are already met (0); rail pins pull the profile
            anchors.setdefault(0, 0.0)
            anchors.setdefault(n - 1, 0.0)
            ks = np.array(sorted(anchors))
            zs = zs + np.interp(np.arange(n), ks, [anchors[k] for k in ks])
        g = MAX_GRADE[t] * MPP
        pinned = np.zeros(n, bool)
        for k in pins:
            pinned[k] = True
        z = enforce_grade_pinned(zs, g, zmin, pinned)
        # a pinned junction too close to a raised deck makes the approach infeasible: lower
        # the deck minimum towards the pin (a low-water bridge beats a 100 % ramp), max 2 tries
        zmin0 = zmin.copy()
        for _try in range(2):
            gr_ = np.abs(np.diff(z)) / MPP
            if len(gr_) == 0 or gr_.max() <= MAX_GRADE[t] * 1.05 or not (zmin > -1e8).any():
                break
            zmin = np.where(zmin > -1e8, np.maximum(zmin - min(2.0, (gr_.max() - MAX_GRADE[t]) * MPP * 4), zmin0 - 0.8), zmin)
            z = enforce_grade_pinned(zs, g, zmin, pinned)
        if os.environ.get('DEBUG_ROUTE') and roads[order[0][0]]['properties'].get('def_id') == os.environ['DEBUG_ROUTE']:
            np.savez('/tmp/route_debug.npz', XY=XY, zt=zt, zs=zs, zmin=zmin, z=z, pinned=pinned)
        for nd, k in node_at.items():
            if nd not in nz:
                nz[nd] = float(z[k])
        # split back into edges
        for (i, a, b) in order:
            e_idx = [k for k in range(n) if owner[k] == i]
            k0 = e_idx[0] - 1 if e_idx[0] > 0 and owner[e_idx[0] - 1] != i and np.hypot(*(XY[e_idx[0] - 1] - edge_samples(i, a)[0][0])) < 0.6 else e_idx[0]
            ks = list(range(k0, e_idx[-1] + 1))
            XYe, ze, zte = XY[ks], z[ks], zt[ks]
            Se = np.r_[0, np.cumsum(np.hypot(*np.diff(XYe, axis=0).T))]
            if roads[i]['properties']['from'] != a:
                XYe, ze, zte = XYe[::-1], ze[::-1], zte[::-1]
                Se = Se[-1] - Se[::-1]
            sg = [substring(geoms[i], sa, sb) for sa, sb in roads[i]['properties']['bridge_spans']]
            sp = []
            for ia, ib in span_runs(XYe, sg):
                info = span_info(XYe, ia, ib, roads[i]['properties']['type'], WL, wlines)
                if info is not None:
                    sp.append((ia, ib, info[0], info[1], info[2]))
            profiles[i] = (Se, XYe, ze, zte, sp)

    # ---------------- write roads with z + stats
    bridges = []
    report = []
    zroad_samples = []
    sample_ref = []
    walls = []
    op_spans = {}
    zones_ = {z_['id']: z_['kind'] for z_ in load_json(path('data/manual/zones.json'))['zones']}
    zone_kind = lambda p: zones_.get(p.get('zone'))
    node_deg = {}
    for f in roads:
        if f['properties'].get('virtual'):
            continue
        for nd in (f['properties']['from'], f['properties']['to']):
            node_deg[nd] = node_deg.get(nd, 0) + 1

    def prof_sz(f):
        c = np.asarray(f['geometry']['coordinates'])
        return np.r_[0, np.cumsum(np.hypot(*np.diff(c[:, :2], axis=0).T))], c[:, 2]
    for i, f in enumerate(roads):
        S, XY, z, zt, spans = profiles[i]
        p = f['properties']
        t = p['type']
        P = np.asarray(f['geometry']['coordinates'])[:, :2]
        ls = LineString(P)
        sv = np.array([ls.project(Point(q)) for q in P])
        zv = np.interp(sv, S, z)
        f['geometry']['coordinates'] = [[rnd(x), rnd(y), rnd(zz, 2)] for (x, y), zz in zip(P, zv)]
        gr = np.abs(np.diff(z)) / MPP if len(z) > 1 else np.array([0.0])
        mg = float(gr.max()) if len(gr) else 0.0
        on_bridge = np.zeros(len(z), bool)
        for ia, ib, wl, wcls, wid in spans:
            on_bridge[ia:ib + 1] = True
        spans = list(spans)
        cutfill = (z - zt)
        # engineering: high fills become viaducts (a 25 m embankment is not how anyone builds a road)
        high = (cutfill > 10.0) & (t in ('freeway', 'highway', 'ramp', 'arterial', 'collector', 'main_street', 'rural'))
        k = 0
        while k < len(high):
            if high[k]:
                j = k
                while j + 1 < len(high) and high[j + 1]:
                    j += 1
                if j - k >= 5 and not on_bridge[k:j + 1].all():
                    a_, b_ = max(0, k - 2), min(len(z) - 1, j + 2)
                    spans.append((a_, b_, float('nan'), 'valley', (S[b_] - S[a_]) * MPP))
                    on_bridge[a_:b_ + 1] = True
                k = j + 1
            else:
                k += 1
        # one structure per continuous elevated stretch: merge overlapping / nearly touching spans
        spans.sort(key=lambda q: q[0])
        merged = []
        for sp_ in spans:
            if merged and sp_[0] <= merged[-1][1] + 3:
                a0, b0, wl0, c0, w0 = merged[-1]
                water_ = c0 if c0 != 'valley' else sp_[3]
                wl_ = wl0 if np.isfinite(wl0) else sp_[2]
                merged[-1] = (a0, max(b0, sp_[1]), wl_, water_, (S[max(b0, sp_[1])] - S[a0]) * MPP)
            else:
                merged.append(tuple(sp_))
        spans = merged
        p['max_grade_pct'] = rnd(mg * 100, 1)
        p['z_range_m'] = [rnd(z.min(), 1), rnd(z.max(), 1)]
        dev = (z - zt)[~on_bridge] if (~on_bridge).any() else np.array([0.0])
        p['max_fill_m'] = rnd(max(0.0, dev.max()), 1)
        p['max_cut_m'] = rnd(max(0.0, -dev.min()), 1)
        if mg > MAX_GRADE[t] * 1.05:
            report.append({'kind': 'grade', 'id': p['id'], 'type': t, 'max_grade_pct': rnd(mg * 100, 1), 'limit_pct': MAX_GRADE[t] * 100})
        if p['max_fill_m'] > 12 or p['max_cut_m'] > 12:
            report.append({'kind': 'earthwork', 'id': p['id'], 'type': t, 'cut_m': p['max_cut_m'], 'fill_m': p['max_fill_m']})
        new_spans = []
        for ia, ib, wl, wcls, wid in spans:
            if wcls == 'valley':
                kind, struct = 'viaduct', 'concrete girder viaduct' if t in ('freeway', 'highway', 'ramp', 'arterial') else 'steel trestle'
            elif t in ('dirt', 'driveway') and wcls == 'creek' and wid < 8:
                kind, struct = 'ford', 'ford'
            elif wcls == 'creek' and wid < 7 and t not in ('freeway', 'highway'):
                kind, struct = 'culvert', 'box culvert' if t in ('arterial', 'collector', 'main_street') else 'pipe culvert'
            else:
                kind = 'bridge'
                L = wid
                if t in ('freeway', 'highway', 'ramp', 'arterial'):
                    struct = 'concrete girder' if L < 120 else 'steel plate girder'
                elif t in ('rural', 'gravel', 'dirt') and L > 25:
                    struct = 'steel truss'
                else:
                    struct = 'concrete slab' if L < 20 else 'concrete girder'
            deck = z[ia:ib + 1]
            new_spans.append([rnd(S[ia]), rnd(S[ib])])
            bridges.append(geojson_line(XY[ia:ib + 1], {
                'id': f"{p['id']}_X{len(new_spans)}", 'road': p['id'], 'road_type': t, 'name': p.get('name'), 's_px': [rnd(S[ia]), rnd(S[ib])],
                'kind': kind, 'structure': struct, 'length_m': rnd((S[ib] - S[ia]) * MPP, 1), 'deck_width_m': p.get('width_m', types[t]['width_m']),
                'deck_z_m': [rnd(deck[0], 2), rnd(deck.max(), 2), rnd(deck[-1], 2)], 'water_level_m': None if not np.isfinite(wl) else rnd(wl, 2),
                'clearance_m': rnd(float((deck - zt[ia:ib + 1]).min()), 2) if not np.isfinite(wl) else rnd(float(deck.min() - wl), 2), 'waterway_class': None if wcls == 'valley' else wcls,
                'piers': int(max(0, (S[ib] - S[ia]) * MPP // 30)) if kind == 'bridge' else 0}))
        p['bridge_spans'] = new_spans
        p['crossings'] = [b['properties']['id'] for b in bridges if b['properties']['road'] == p['id']]
        # samples for grading (skip bridge decks, keep culverts/fords as ground)
        ground = ~on_bridge
        for ia, ib, wl, wcls, wid in spans:
            if wcls == 'creek' and wid < 7 and t not in ('freeway', 'highway'):
                ground[ia:ib + 1] = True
        if p.get('virtual'):
            continue
        sec = section_for(p, types, zone_kind)
        hw = sec['bed_half_m'] / MPP
        pri = {'freeway': 5, 'ramp': 4, 'highway': 4}.get(t, 2)
        gk = np.nonzero(ground)[0]
        if len(gk):
            # side terrain (natural) just beyond the road bed -> rock cuts / retaining walls
            tan = np.gradient(XY, axis=0)
            tan /= np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-9)[:, None]
            nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
            off = hw + sec['ditch_w_m'] / MPP + 2.0
            side = {}
            for sgn in (-1, 1):
                side[sgn] = bilinear(T, XY[:, 0] + nrm[:, 0] * off * sgn, XY[:, 1] + nrm[:, 1] * off * sgn) - z
            for sgn in (-1, 1):
                dz = side[sgn]
                kind_ = None
                if sec['curb'] and zone_kind(p) in WALL_ZONES:
                    wall = (np.abs(dz) > 2.5) & ground
                    kind_ = 'retaining_wall'
                elif sec['curb']:
                    continue
                else:
                    wall = (dz > 5.0) & ground & (t not in ('driveway', 'dirt'))
                    kind_ = 'rock_cut'
                for a_, b_ in runs_of(wall, 6):
                    hgt = float(np.abs(dz[a_:b_ + 1]).max())
                    oo = (hw + (0.3 if sec['curb'] else sec['ditch_w_m'] / MPP)) * sgn
                    line = XY[a_:b_ + 1] + nrm[a_:b_ + 1] * oo
                    walls.append(geojson_line(line, {'id': f"{p['id']}_{'W' if kind_ == 'retaining_wall' else 'C'}{len(walls) + 1}", 'road': p['id'], 'kind': kind_,
                                                     'side': 'right' if sgn > 0 else 'left', 'height_m': rnd(min(hgt, 14.0), 1),
                                                     'retains': 'cut' if float(np.median(dz[a_:b_ + 1])) > 0 else 'fill',
                                                     'top_z_m': [rnd(float(z[q] + max(dz[q], 0)), 2) for q in range(a_, b_ + 1, max(1, (b_ - a_) // 12))],
                                                     'base_z_m': rnd(float(np.median(z[a_:b_ + 1])), 2)}))
            steep = np.maximum(side[-1], side[1])
            # no side ditch inside junctions (the road mesh has curb returns / aprons there)
            jr = np.zeros(len(S), bool)
            widen = np.zeros(len(S))
            rng_ = hw * 2.2 + 6.0
            for nd, st_ in ((p['from'], S - S[0]), (p['to'], S[-1] - S)):
                if node_deg.get(nd, 0) >= 3:
                    jr |= st_ < rng_
                    # junction corners (curb-return fillets): widen the bed near the node
                    widen = np.maximum(widen, np.clip(rng_ - st_, 0, None) * 0.6)
            for k in gk:
                s_cut = 0.85
                if sec['curb'] and zone_kind(p) in WALL_ZONES and max(abs(side[-1][k]), abs(side[1][k])) > 2.5:
                    s_cut = s_fill = 12.0
                else:
                    s_fill = 0.5
                    if steep[k] > 5.0 and t not in ('driveway', 'dirt'):
                        s_cut = 3.0  # rock cut (Appalachian road cuts are near-vertical rock)
                dwk = 0.0 if jr[k] else sec['ditch_w_m'] / MPP
                zroad_samples.append((XY[k, 0], XY[k, 1], z[k], hw + widen[k], pri, dwk, sec['ditch_d_m'] if dwk else 0.0, s_cut, s_fill))
                sample_ref.append((p['id'], S[k]))
    # grade-separated crossings: the upper road over the lower one is a bridge (overpass)
    for s in seps:
        up = next(f for f in roads if f['properties']['id'] == s['upper'])
        ls = LineString(np.asarray(up['geometry']['coordinates'])[:, :2])
        a0 = ls.project(Point(s['at']))
        low = next((f for f in roads if f['properties']['id'] == s['lower']), None)
        if low is not None:
            lw = low['properties'].get('width_m', types[low['properties']['type']]['width_m']) / MPP
        else:
            lw = next(f['properties']['width_m'] for f in rails['features'] if f['properties']['id'] == s['lower']) / MPP
        lw = lw / max(math.sin(math.radians(max(s.get('angle_deg', 90), 25))), 0.4)
        a, b = max(0, a0 - lw / 2 - 3), min(ls.length, a0 + lw / 2 + 3)
        cover = [q for q in up['properties']['bridge_spans'] if q[0] <= a + 1 and q[1] >= b - 1]
        if cover:  # already on a bridge/viaduct: that structure also spans the lower road/railway
            for bf in bridges:
                bp = bf['properties']
                if bp['road'] == s['upper'] and bp.get('kind') in ('bridge', 'viaduct') and bp['s_px'][0] <= a + 1 and bp['s_px'][1] >= b - 1:
                    bp.setdefault('over', []).append(s['lower'])
            continue
        up['properties']['bridge_spans'].append([rnd(a), rnd(b)])
        seg = substring(ls, a, b)
        bridges.append(geojson_line(np.asarray(seg.coords), {'id': f"{s['upper']}_OP{len(bridges)}", 'road': s['upper'], 'road_type': up['properties']['type'],
                                    'kind': 'overpass', 'structure': 'concrete girder overpass', 'over': s['lower'], 'length_m': rnd((b - a) * MPP, 1),
                                    'deck_width_m': up['properties'].get('width_m', types[up['properties']['type']]['width_m']),
                                    'clearance_m': (RAIL_OVER_CLEAR if s.get('kind') == 'rail' else OVERPASS_CLEAR) - 1.5, 'lower_z_m': s['lower_z'],
                                    'over_kind': s.get('kind', 'road'), 'deck_z_m': [rnd(float(np.interp(a, *prof_sz(up))), 2), rnd(float(np.interp((a + b) / 2, *prof_sz(up))), 2), rnd(float(np.interp(b, *prof_sz(up))), 2)]}))
        op_spans.setdefault(s['upper'], []).append((a - 1.0, b + 1.0))
    # the terrain is not graded under overpass decks (the lower road/railway passes there)
    keep = [i for i, (rid, st) in enumerate(sample_ref) if not any(a <= st <= b for a, b in op_spans.get(rid, []))]
    zroad_samples = [zroad_samples[i] for i in keep]
    save_json(path('data/roads/roads.geojson'), rfc)
    save_json(path('data/roads/walls.geojson'), fc(walls, 'walls'))
    save_json(path('data/roads/rail_crossings.geojson'), fc([geojson_point(c['at'], c) for c in rail_crossings], 'rail_crossings'))
    save_json(path('data/roads/bridges.geojson'), fc(bridges, 'bridges'))

    # ---------------- rails (profiled first, written here)
    rail_samples = []
    for rf, rg, S, XY, z, zt in rail_prof:
        n = len(S)
        rf['geometry']['coordinates'] = [[rnd(x), rnd(y), rnd(zz, 2)] for (x, y), zz in zip(XY[::4], z[::4])] + [[rnd(XY[-1, 0]), rnd(XY[-1, 1]), rnd(z[-1], 2)]]
        rf['properties']['max_grade_pct'] = rnd(float(np.abs(np.diff(z)).max() / MPP * 100), 2)
        hw = rf['properties']['width_m'] / 2 / MPP + 0.5
        on_br = np.zeros(n, bool)
        for a_, b_ in rf['properties']['bridge_spans']:
            on_br[int(a_):int(b_) + 1] = True
        for k in range(n):
            if not on_br[k]:
                rail_samples.append((XY[k, 0], XY[k, 1], z[k], hw, 3, 0.0, 0.0, 0.85, 0.5))
    save_json(path('data/railways/railways.geojson'), rails)

    # ---------------- terrain grading
    Tg = grade_terrain(T, wmask, zroad_samples + rail_samples)
    Tg.astype(np.float32).tofile(path('data/terrain/height_graded_f32.bin'))
    lo, hi = float(np.floor(Tg.min())), float(np.ceil(Tg.max()))
    Image.fromarray(((Tg - lo) / (hi - lo) * 65535).round().astype(np.uint16)).save(path('data/terrain/height_graded_u16.png'))
    meta = load_json(path('data/terrain/terrain.json'))
    meta['files']['height_graded_f32'] = 'data/terrain/height_graded_f32.bin'
    meta['files']['height_graded_u16'] = 'data/terrain/height_graded_u16.png'
    meta['u16_graded_decode'] = {'min_m': lo, 'max_m': hi}
    save_json(path('data/terrain/terrain.json'), meta, indent=1)
    from tools.pipeline.terrain import hillshade
    hillshade(Tg, path('assets/maps/debug/hillshade_graded.png'))
    os.makedirs(path('data/qa'), exist_ok=True)
    save_json(path('data/qa/grade_report.json'), {'grade_separations': seps, 'issues': report}, indent=1)
    from collections import Counter
    print('bridges/crossings:', Counter(b['properties']['kind'] for b in bridges), 'grade issues:', Counter(r['kind'] for r in report))


def grade_terrain(T, wmask, samples, reach=24.0):
    """Cut/fill the terrain to the road beds. samples rows:
    (x, y, z_road, bed_half_px, priority, ditch_w_px, ditch_d_m, s_cut, s_fill)
      - under the bed: road level - 0.15 m (the road mesh sits on it)
      - cut side (terrain above the road): a trench under the side ditch (the road mesh
        draws the real ditch above it), then a backslope at s_cut (m/m; 0.85 = earth,
        3 = rock cut, 12 = retaining wall)
      - fill side: embankment at s_fill (0.5 = 1:2), no ditch at the shoulder
    Higher-priority classes are graded last (win where corridors overlap)."""
    S = np.asarray(samples, float)
    Tg = T.copy()
    for pri in sorted(set(S[:, 4].astype(int))):
        sub = S[S[:, 4] == pri]
        tree = cKDTree(sub[:, :2])
        mask = np.zeros((H, W), np.uint8)
        for row in sub[::2]:
            cv2.circle(mask, (int(row[0]), int(row[1])), int(row[3] + row[5] + reach), 1, -1)
        ys, xs = np.nonzero(mask)
        pts = np.stack([xs + 0.5, ys + 0.5], 1)
        dist, idx = tree.query(pts, k=1)
        zr, hw, dw, dd, sc, sf = (sub[idx, c] for c in (2, 3, 5, 6, 7, 8))
        d = dist - hw
        cur = Tg[ys, xs]
        cutside = (cur - zr) > -0.3
        # a 2.5 m terrain grid cannot hold a 0.6 m ditch next to a road edge, so the terrain
        # is kept below the road's own side geometry: a 'moat' (ditch trench or a 3 m strip)
        # beyond the bed, then the backslope. The road mesh draws the real ditch / sidewalk
        # and its backslope meets the terrain beyond the moat (no terrain poking through).
        moat = MOAT_PX
        dwe = np.where(cutside, dw + moat, 0.0)
        dde = np.where(cutside, dd + 0.35, 0.0)
        d2 = np.maximum(d - dwe, 0.0)
        up = zr - dde + sc * d2 * MPP
        lo = np.where(cutside, -1e9, zr - sf * np.maximum(d, 0.0) * MPP)
        new = np.clip(cur, lo, np.maximum(up, lo))
        new = np.where(d <= 0.35, zr - 0.4, new)  # below crown / superelevated edges
        w = np.clip((hw + dwe + reach - dist) / 4.0, 0, 1)
        out = cur * (1 - w) + new * w
        keep = ~wmask[ys, xs]
        Tg[ys[keep], xs[keep]] = out[keep]
    return Tg


if __name__ == '__main__':
    main()
