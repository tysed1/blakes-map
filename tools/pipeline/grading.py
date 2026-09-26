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
from tools.lib.geom import geojson_line, fc

MPP = 2.5
MAX_GRADE = {'freeway': 0.06, 'highway': 0.08, 'ramp': 0.07, 'arterial': 0.08, 'main_street': 0.08, 'collector': 0.10,
             'urban_street': 0.12, 'residential': 0.12, 'rural': 0.12, 'gravel': 0.15, 'dirt': 0.18, 'driveway': 0.20, 'rail': 0.022}
SMOOTH_PX = {'freeway': 40, 'highway': 30, 'ramp': 12, 'arterial': 16, 'main_street': 10, 'collector': 10, 'urban_street': 6,
             'residential': 6, 'rural': 8, 'gravel': 6, 'dirt': 4, 'driveway': 3, 'rail': 50}
CLEAR_WATER = {'river': 6.0, 'creek': 3.5, 'slough': 3.0}
OVERPASS_CLEAR = 7.5  # deck-to-road incl. structure depth
RAIL_OVER_CLEAR = 8.0


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
    return wl, wcls, wid, clear


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
                spans.append([max(0.0, a0 - pad), min(g2.length, a1 + pad)])
        f['properties']['bridge_spans'] = spans
    nodes = {f['properties']['id']: f['geometry']['coordinates'][:2] for f in load_json(path('data/roads/road_nodes.geojson'))['features']}
    ww = load_json(path('data/water/waterways.geojson'))['features']
    wlines = [(LineString(f['geometry']['coordinates']), f['properties']) for f in ww]
    water = unary_union([shape(f['geometry']) for f in load_json(path('data/water/water_bodies.geojson'))['features']])

    # ---------------- route-based profiles
    geoms = [LineString(np.asarray(f['geometry']['coordinates'])[:, :2]) for f in roads]
    tree = STRtree(geoms)
    rails = load_json(path('data/railways/railways.geojson'))
    rail_geoms = [LineString(np.asarray(f['geometry']['coordinates'])[:, :2]) for f in rails['features']]
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
        g = MAX_GRADE[t] * MPP
        pinned = np.zeros(n, bool)
        for k in pins:
            pinned[k] = True
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
                'id': f"{p['id']}_X{len(new_spans)}", 'road': p['id'], 'road_type': t, 'name': p.get('name'),
                'kind': kind, 'structure': struct, 'length_m': rnd((S[ib] - S[ia]) * MPP, 1), 'deck_width_m': types[t]['width_m'],
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
        hw = types[t]['width_m'] / 2 / MPP + (types[t]['shoulder_m'] / MPP if t in ('freeway', 'highway') else 0.3)
        if p.get('virtual'):
            continue
        pri = {'freeway': 5, 'ramp': 4, 'highway': 4}.get(t, 2)
        for k in np.nonzero(ground)[0]:
            zroad_samples.append((XY[k, 0], XY[k, 1], z[k], hw, pri))
    # grade-separated crossings: the upper road over the lower one is a bridge (overpass)
    for s in seps:
        up = next(f for f in roads if f['properties']['id'] == s['upper'])
        ls = LineString(np.asarray(up['geometry']['coordinates'])[:, :2])
        a0 = ls.project(Point(s['at']))
        lw = types[next(f for f in roads if f['properties']['id'] == s['lower'])['properties']['type']]['width_m'] / MPP
        a, b = max(0, a0 - lw / 2 - 3), min(ls.length, a0 + lw / 2 + 3)
        up['properties']['bridge_spans'].append([rnd(a), rnd(b)])
        seg = substring(ls, a, b)
        bridges.append(geojson_line(np.asarray(seg.coords), {'id': f"{s['upper']}_OP{len(bridges)}", 'road': s['upper'], 'road_type': up['properties']['type'],
                                    'kind': 'overpass', 'structure': 'concrete girder overpass', 'over': s['lower'], 'length_m': rnd((b - a) * MPP, 1),
                                    'deck_width_m': types[up['properties']['type']]['width_m'], 'clearance_m': OVERPASS_CLEAR - 1.5, 'lower_z_m': s['lower_z']}))
    save_json(path('data/roads/roads.geojson'), rfc)
    save_json(path('data/roads/bridges.geojson'), fc(bridges, 'bridges'))

    # ---------------- rails
    rail_samples = []
    for rf, rg in zip(rails['features'], rail_geoms):
        n = max(2, int(math.ceil(rg.length)) + 1)
        S = np.linspace(0, rg.length, n)
        XY = np.array([rg.interpolate(s).coords[0] for s in S])
        zt = bilinear(T, XY[:, 0], XY[:, 1])
        pad = min(n - 1, 150)
        ext = np.r_[2 * zt[0] - zt[1:pad + 1][::-1], zt, 2 * zt[-1] - zt[-pad - 1:-1][::-1]]
        zs = ndi.gaussian_filter1d(ext, 50)[pad:pad + n]
        zmin = np.full(n, -1e9)
        for a, b in rf['properties']['bridge_spans']:
            seg = XY[int(a):int(b) + 1]
            wl = np.nanmax(WL[seg[:, 1].astype(int), seg[:, 0].astype(int)])
            if np.isfinite(wl):
                zmin[int(a):int(b) + 1] = wl + 5.0
        # rails pass under highways/freeways where they cross
        z = enforce_grade(zs, MAX_GRADE['rail'] * MPP, zmin=zmin)
        rf['geometry']['coordinates'] = [[rnd(x), rnd(y), rnd(zz, 2)] for (x, y), zz in zip(XY[::4], z[::4])] + [[rnd(XY[-1, 0]), rnd(XY[-1, 1]), rnd(z[-1], 2)]]
        rf['properties']['max_grade_pct'] = rnd(float(np.abs(np.diff(z)).max() / MPP * 100), 2)
        hw = rf['properties']['width_m'] / 2 / MPP + 0.5
        tunnel_until = None
        for k in range(n):
            rail_samples.append((XY[k, 0], XY[k, 1], z[k], hw, 3))
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


def grade_terrain(T, wmask, samples, s_fill=0.5, s_cut=0.85, reach=24.0):
    S = np.asarray(samples, float)
    Tg = T.copy()
    # process in priority order: higher priority wins where corridors overlap
    for pri in sorted(set(S[:, 4].astype(int))):
        sub = S[S[:, 4] == pri]
        tree = cKDTree(sub[:, :2])
        # candidate pixels near this class of roads
        mask = np.zeros((H, W), np.uint8)
        for x, y, z, hw, _ in sub[::2]:
            cv2.circle(mask, (int(x), int(y)), int(hw + reach), 1, -1)
        ys, xs = np.nonzero(mask)
        pts = np.stack([xs + 0.5, ys + 0.5], 1)
        dist, idx = tree.query(pts, k=1)
        zr = sub[idx, 2]
        hw = sub[idx, 3]
        d = dist - hw
        cur = Tg[ys, xs]
        dz = cur - zr
        lim = np.where(dz > 0, s_cut, s_fill) * np.maximum(d, 0) * MPP
        new = zr + np.clip(dz, -lim, lim)
        new = np.where(d <= 0.5, zr - 0.12, new)
        # feather the outer edge
        w = np.clip((hw + reach - dist) / 4.0, 0, 1)
        out = cur * (1 - w) + new * w
        keep = ~wmask[ys, xs]
        Tg[ys[keep], xs[keep]] = out[keep]
    return Tg


if __name__ == '__main__':
    main()
