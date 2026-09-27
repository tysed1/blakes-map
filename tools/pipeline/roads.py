"""Road network assembly.

Inputs
  tools/.cache/roads_auto.json      automatic skeleton graph (roads_auto.py)
  data/manual/roads/*.json          hand-authored alignments (authoritative)
  data/manual/road_edits.json       deletions / forced connections / overrides
  data/manual/zones.json            development zones (classification)
  data/roads/road_types.json        type catalogue
Outputs
  data/roads/roads.geojson          one LineString per edge (junction to junction)
  data/roads/road_nodes.geojson     junctions / dead ends with kind + degree
  data/roads/bridges.geojson        road spans over water
  data/roads/id_registry.json       stable-ID registry (reused across runs)
"""
import sys, os, math, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import cv2
from shapely.geometry import LineString, Point, Polygon, MultiLineString, shape
from shapely.strtree import STRtree
from shapely.ops import substring, unary_union

from tools.lib.common import path, load_json, save_json, W, H, rnd
from tools.lib.features import road_prob, rock_density
from tools.lib.trace import robust_smooth, rdp, resample, route
from tools.lib.manual_roads import all_traced
from tools.lib.geom import geojson_line, geojson_point, fc, polyline_length

TYPES = load_json(path('data/roads/road_types.json'))['types']
GRADE_SEP = {'freeway', 'ramp'}
ZONE_BASE = {'downtown': 'urban_street', 'industrial': 'urban_street', 'town_center': 'urban_street',
             'town': 'residential', 'city': 'residential'}
REGION_PREFIX = {'laurel_city': 'LC', 'hollow_ridge': 'HR', 'tannersville': 'TV', None: 'RU'}


def wpx(t):
    return TYPES[t]['width_m'] / 2.5


def width_for(t, lanes=None):
    """Finished width (m) of a road of type t with an explicit lane count."""
    spec = TYPES[t]
    if not lanes or lanes == spec['lanes'] or 'section' not in spec:
        return spec['width_m']
    sec = spec['section']
    side = sec.get('parking_m', sec.get('shoulder_paved_m', 0.0))
    return round(lanes * sec['lane_w'] + 2 * side + sec.get('median_m', 0.0), 1)


# ------------------------------------------------------------------ helpers
def sample_along(pts, step=1.0):
    return resample(np.asarray(pts, float), step)


def field_at(F, pts):
    xi = np.clip(pts[:, 0].astype(int), 0, W - 1)
    yi = np.clip(pts[:, 1].astype(int), 0, H - 1)
    return F[yi, xi]


def load_zones():
    z = load_json(path('data/manual/zones.json'))['zones']
    return [(d, Polygon(d['pts'])) for d in z]


def zone_of(pt, zones):
    p = Point(pt)
    for d, poly in zones:
        if poly.contains(p):
            return d
    return None


# ------------------------------------------------------------------ auto cleanup
def auto_lines(edits, zones):
    g = load_json(path('tools/.cache/roads_auto.json'))
    P = road_prob()
    R = rock_density()
    lines = []
    dels = [Polygon(z) for z in edits.get('delete_zones', [])]
    for e in g['edges']:
        pts = np.array(e['pts'], float)
        if len(pts) < 2:
            continue
        s = sample_along(pts, 1.0)
        L = polyline_length(s)
        mp = float(field_at(P, s).mean())
        rk = float(field_at(R, s).mean())
        mid = s[len(s) // 2]
        z = zone_of(mid, zones)
        if z is not None and z['kind'] in ('town_center', 'downtown', 'industrial'):
            continue  # cores are hand-authored / grid-detected
        if z is None and rk > 0.22 and mp < 0.5:
            continue  # rocky-slope texture, not a road
        if z is None and mp < 0.28 and L < 25:
            continue
        ls = LineString(s)
        if any(d.intersects(ls) and d.intersection(ls).length > 0.5 * ls.length for d in dels):
            continue
        lines.append({'pts': s, 'src': 'auto', 'zone': z})
    return lines


def dangling_ends(lines):
    """Endpoints not touching any other line (within 1.5px)."""
    geoms = [LineString(l['pts']) for l in lines]
    tree = STRtree(geoms)
    out = []
    for i, l in enumerate(lines):
        for end in (0, -1):
            p = Point(l['pts'][end])
            near = [j for j in tree.query(p.buffer(1.5)) if j != i and geoms[j].distance(p) < 1.5]
            if not near:
                out.append((i, end))
    return out, geoms, tree


def bridge_gaps(lines, max_gap=22.0):
    P = road_prob()
    ends, geoms, tree = dangling_ends(lines)
    added = []
    for i, end in ends:
        pts = lines[i]['pts']
        if polyline_length(pts) < 6:
            continue
        e = pts[end]
        back = pts[min(6, len(pts) - 1)] if end == 0 else pts[max(-7, -len(pts))]
        d = e - back
        n = np.hypot(*d)
        if n < 1e-6:
            continue
        d /= n
        best = None
        for j in tree.query(Point(e).buffer(max_gap)):
            if j == i:
                continue
            g = geoms[j]
            # candidate: nearest point on g within a forward cone
            cand = []
            q = np.asarray(g.coords)
            qs = sample_along(q, 2.0)
            v = qs - e
            dist = np.hypot(v[:, 0], v[:, 1])
            ok = (dist > 2) & (dist < max_gap)
            if not ok.any():
                continue
            cos = (v[:, 0] * d[0] + v[:, 1] * d[1]) / np.maximum(dist, 1e-6)
            ok &= cos > math.cos(math.radians(50))
            if not ok.any():
                continue
            k = np.argmin(np.where(ok, dist - 4 * cos, 1e9))
            c = qs[k]
            try:
                r = np.array(route(tuple(e), tuple(c), 'local', margin=6))
            except Exception:
                continue
            Lr = polyline_length(r)
            if Lr > 1.6 * dist[k] + 3:
                continue
            mp = float(field_at(P, r).mean())
            if mp < 0.13:
                continue
            score = Lr / (mp + 0.05)
            if best is None or score < best[0]:
                best = (score, r, c)
        if best is not None:
            r = best[1]
            r[0] = e
            r[-1] = best[2]
            added.append({'pts': r, 'src': 'auto_gap', 'zone': lines[i]['zone']})
    return lines + added


# ------------------------------------------------------------------ manual precedence
def corridor_mask(manual):
    m = np.zeros((H * 2, W * 2), np.uint8)
    for r, pts in manual:
        w = wpx(r['type'])
        th = max(3, int(round((w + 4.0) * 2)))
        cv2.polylines(m, [(np.asarray(pts) * 2).round().astype(np.int32)], False, 1, th)
    return m


def suppress_covered(autos, manual):
    m = corridor_mask(manual)
    out = []
    for l in autos:
        s = l['pts']
        inside = m[np.clip((s[:, 1] * 2).astype(int), 0, H * 2 - 1), np.clip((s[:, 0] * 2).astype(int), 0, W * 2 - 1)] > 0
        if inside.mean() > 0.5:
            continue
        # trim covered ends
        a, b = 0, len(s)
        while a < b and inside[a]:
            a += 1
        while b > a and inside[b - 1]:
            b -= 1
        if b - a < 3:
            continue
        # keep one covered sample at each trimmed end so the snap can reach the manual road
        a2, b2 = max(0, a - 1), min(len(s), b + 1)
        l = dict(l, pts=s[a2:b2], trimmed=(a2 > 0 or b2 < len(s)))
        out.append(l)
    return out


# ------------------------------------------------------------------ snapping + noding
def snap_endpoints(lines, tol_auto=4.5, tol_manual=3.5):
    geoms = [LineString(l['pts']) for l in lines]
    tree = STRtree(geoms)
    for i, l in enumerate(lines):
        for end in (0, -1):
            p = Point(l['pts'][end])
            tol = tol_manual if l['src'] == 'manual' else tol_auto
            if l.get('trimmed'):
                tol = max(tol, wpx('freeway') / 2 + 3)
            best = None
            for j in tree.query(p.buffer(tol)):
                if j == i:
                    continue
                o = lines[j]
                # auto roads never connect to grade-separated roads
                if l['src'] != 'manual' and o.get('type') in GRADE_SEP:
                    continue
                if l.get('type') in GRADE_SEP and o['src'] != 'manual':
                    continue
                if l.get('no_snap') or o.get('no_snap_target'):
                    continue
                d = geoms[j].distance(p)
                if d < tol and (best is None or d < best[0]):
                    best = (d, j)
            if best is None or best[0] < 1e-3:
                continue
            g = geoms[best[1]]
            # prefer an existing endpoint of the target if close
            ends = [np.asarray(g.coords[0]), np.asarray(g.coords[-1])]
            q = None
            for ep in ends:
                if np.hypot(*(ep - l['pts'][end])) < tol:
                    q = ep
            if q is None:
                q = np.asarray(g.interpolate(g.project(p)).coords[0])
            pts = l['pts'].copy()
            pts[end] = q
            l['pts'] = pts
        geoms[i] = LineString(l['pts'])
    return lines


def regularize_t_junctions(lines, min_deg=40.0):
    """A road ending on another road should meet it at a real junction angle,
    not slide tangentially onto it. Back the approach off and come in straight."""
    geoms = [LineString(l['pts']) for l in lines]
    tree = STRtree(geoms)
    fixed = 0
    for i, l in enumerate(lines):
        if l['type'] in GRADE_SEP:
            continue
        for end in (0, -1):
            pts = l['pts'] if end == -1 else l['pts'][::-1]
            e = Point(pts[-1])
            host = None
            for j in tree.query(e.buffer(0.2)):
                if j == i or lines[j]['type'] in GRADE_SEP:
                    continue
                g = geoms[j]
                if g.distance(e) < 0.1:
                    s = g.project(e)
                    if 1.0 < s < g.length - 1.0:
                        host = j
            if host is None:
                continue
            g = geoms[host]
            s = resample(pts, 1.0)
            if len(s) < 12:
                continue
            def angle_at(tail_pt, q):
                v = np.asarray(q) - np.asarray(tail_pt)
                sq = g.project(Point(q))
                a = np.asarray(g.interpolate(max(0, sq - 2)).coords[0]); b = np.asarray(g.interpolate(min(g.length, sq + 2)).coords[0])
                t = b - a
                c = abs(np.dot(v, t)) / max(np.hypot(*v) * np.hypot(*t), 1e-9)
                return math.degrees(math.acos(min(1, c)))
            if angle_at(s[-8], s[-1]) >= min_deg:
                continue
            best = None
            for back in range(6, min(30, len(s) - 3)):
                b = s[-1 - back]
                q = np.asarray(g.interpolate(g.project(Point(b))).coords[0])
                if np.hypot(*(q - b)) < 2.5:
                    continue
                ang = angle_at(b, q)
                if ang >= min_deg:
                    best = (back, q)
                    break
            if best is None:
                continue
            back, q = best
            nt = resample(np.vstack([s[-1 - back], q]), 1.0)
            new = np.vstack([s[:-1 - back], nt])
            l['pts'] = new if end == -1 else new[::-1]
            geoms[i] = LineString(l['pts'])
            fixed += 1
    print('  T-junctions regularised:', fixed)


def node_network(lines):
    """Split lines at shared endpoints and at surface-surface crossings."""
    geoms = [LineString(l['pts']) for l in lines]
    tree = STRtree(geoms)
    splits = [[] for _ in lines]
    for i, g in enumerate(geoms):
        # endpoints of any line lying on g
        for end in (0, -1):
            p = Point(lines[i]['pts'][end])
            for j in tree.query(p.buffer(0.05)):
                if j != i and geoms[j].distance(p) < 0.05:
                    splits[j].append(geoms[j].project(p))
        if lines[i].get('type') in GRADE_SEP or lines[i].get('layer', 0) != 0:
            continue
        for j in tree.query(g):
            if j <= i or lines[j].get('type') in GRADE_SEP or lines[j].get('layer', 0) != 0:
                continue
            if lines[i].get('railway') != lines[j].get('railway'):
                continue
            x = g.intersection(geoms[j])
            if x.is_empty:
                continue
            pts = [x] if x.geom_type == 'Point' else [q for q in getattr(x, 'geoms', []) if q.geom_type == 'Point']
            for q in pts:
                splits[i].append(g.project(q))
                splits[j].append(geoms[j].project(q))
    pieces = []
    for i, g in enumerate(geoms):
        cuts = sorted(set([0.0, g.length] + [round(s, 4) for s in splits[i] if 0.3 < s < g.length - 0.3]))
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < 0.2:
                continue
            seg = substring(g, a, b)
            if seg.length < 0.2:
                continue
            pieces.append((i, np.asarray(seg.coords)))
    return pieces


def load_grid(zones):
    p = path('tools/.cache/grid_streets.json')
    if not os.path.exists(p):
        return []
    zmap = {d['id']: d for d, _ in zones}
    out = []
    for g in load_json(p):
        pts = resample(np.asarray(g['pts'], float), 1.0)
        out.append({'pts': pts, 'src': 'grid', 'zone': zmap.get(g['zone']), 'type': g['type']})
    return out


def clear_freeway_corridors(autos, fw):
    """Local streets may pass under/over a freeway at a clean angle, but never
    touch ramps or run inside the freeway right-of-way."""
    ramps = unary_union([LineString(l['pts']).buffer(wpx('ramp') / 2 + 2.5) for l in fw if l['type'] == 'ramp'])
    row = unary_union([LineString(l['pts']).buffer(wpx('freeway') / 2 + 3.0) for l in fw if l['type'] == 'freeway'])
    out = []
    for l in autos:
        g = LineString(l['pts'])
        if g.intersects(ramps):
            continue
        if g.intersection(row).length > 14:
            continue
        out.append(l)
    return out


def clip_rail_corridors(lines, rails, zones):
    """Local streets never run through rail yards. A street crossing a multi-track
    yard, or crossing any rail at a shallow angle, or crossing inside an industrial
    zone, is cut back to the corridor edge (it becomes a dead end at the yard fence).
    Single-track mainline crossings at a clean angle stay (at-grade crossings)."""
    if not rails:
        return lines
    out = []
    cor = []
    for r, pts in rails:
        tr = r.get('tracks', 1)
        cor.append((LineString(pts), tr, LineString(pts).buffer((4.5 + 4.0 * (tr - 1)) / 5.0 + 1.5)))
    for l in lines:
        g = LineString(l['pts'])
        cut = None
        for rl, tr, poly in cor:
            if not g.intersects(poly):
                continue
            x = g.intersection(rl)
            ok = False
            if tr == 1 and not x.is_empty and x.geom_type == 'Point':
                z = zone_of((x.x, x.y), zones)
                sr = rl.project(x); sg = g.project(x)
                a = np.asarray(rl.interpolate(sr + 3).coords[0]) - np.asarray(rl.interpolate(sr - 3).coords[0])
                b = np.asarray(g.interpolate(sg + 3).coords[0]) - np.asarray(g.interpolate(sg - 3).coords[0])
                ang = math.degrees(math.acos(min(1, abs(np.dot(a, b)) / max(np.hypot(*a) * np.hypot(*b), 1e-9))))
                ok = ang > 55 and not (z and z['kind'] == 'industrial')
            if not ok:
                cut = poly if cut is None else cut.union(poly)
        if cut is None:
            out.append(l)
            continue
        rest = g.difference(cut)
        parts = [rest] if rest.geom_type == 'LineString' else [q for q in getattr(rest, 'geoms', []) if q.geom_type == 'LineString']
        for q in parts:
            if q.length >= 8:
                out.append(dict(l, pts=resample(np.asarray(q.coords), 1.0)))
    return out


def interchange_ramps(manual):
    from tools.lib.interchange import diamond
    spec = load_json(path('data/manual/interchanges.json'))
    byid = {r['id']: (r, np.asarray(p)) for r, p in manual}
    out, seps = [], []
    for ic in spec['interchanges']:
        f = byid[ic['freeway']][1]
        xs = [byid[i][1] for i in ic['crossroad']]
        x = xs[0]
        for nxt in xs[1:]:
            if np.hypot(*(x[-1] - nxt[0])) < 1:
                x = np.vstack([x, nxt[1:]])
            elif np.hypot(*(x[0] - nxt[-1])) < 1:
                x = np.vstack([nxt, x[1:]])
            elif np.hypot(*(x[0] - nxt[0])) < 1:
                x = np.vstack([nxt[::-1], x[1:]])
            else:
                x = np.vstack([x, nxt[::-1][1:]])
        ramps, sep = diamond(ic, f, x, wpx(byid[ic['freeway']][0]['type']) / 2, wpx('ramp') / 2, spec['defaults'])
        if ic.get('kind') == 'overpass':
            ramps = []  # grade separation only
        if ic.get('kind') == 'half_diamond':
            cy = sep['point'][1]
            side = ic.get('keep_gore_side', 'north')
            ramps = [r for r in ramps if (r['gore'][1] < cy) == (side == 'north')]
        sep.update(id=ic['id'], lower=ic['freeway'], upper_roads=ic['crossroad'])
        seps.append(sep)
        fname = byid[ic['freeway']][0].get('name', ic['freeway'])
        xname = byid[ic['crossroad'][0]][0].get('name', '')
        for k, r in enumerate(ramps):
            nm = f"{fname} {r['direction']} {'exit to' if r['kind'] == 'exit' else 'entrance from'} {xname}"
            out.append({'pts': r['pts'], 'src': 'manual', 'type': 'ramp', 'name': nm, 'route': None,
                        'def_id': f"{ic['id']}_R{k + 1}", 'lanes': 1, 'layer': 0, 'zone': zone_of(r['pts'][len(r['pts']) // 2], load_zones()),
                        'oneway': True, 'no_snap': True, 'interchange': ic['id'], 'min_radius_m': rnd(r['min_radius_px'] * 2.5, 1)})
            if r['min_radius_px'] * 2.5 < 45:
                print(f"  WARN {ic['id']} ramp {k + 1} min radius {r['min_radius_px'] * 2.5:.0f} m")
    save_json(path('tools/.cache/grade_separations.json'), seps)
    return out


# ------------------------------------------------------------------ main
def classify_auto(l, P, mask_dt):
    z = l['zone']
    s = l['pts']
    L = polyline_length(s)
    wv = float(np.median(field_at(mask_dt, s))) * 2
    if z is not None:
        t = ZONE_BASE[z['kind']]
        if t == 'residential' and wv >= 4.5 and L >= 60:
            t = 'collector'
        return t
    return 'rural'


def main():
    zones = load_zones()
    edits = load_json(path('data/manual/road_edits.json')) if os.path.exists(path('data/manual/road_edits.json')) else {}
    from tools.lib.engineer import design_alignments
    manual = [(r, pts) for r, pts in design_alignments(all_traced()) if r['type'] != 'rail']
    mlines = []
    for r, pts in manual:
        mlines.append({'pts': np.asarray(pts, float), 'src': 'manual', 'type': r['type'], 'name': r.get('name'),
                       'route': r.get('route', r.get('name')), 'def_id': r['id'], 'lanes': r.get('lanes'),
                       'layer': r.get('layer', 0), 'zone': zone_of(pts[len(pts) // 2], zones),
                       'oneway': r.get('oneway', False), 'no_snap': r.get('no_snap', False)})
    mlines += interchange_ramps(manual)
    autos = auto_lines(edits, zones)
    autos = bridge_gaps(autos)
    grid = load_grid(zones)
    gridman = manual + [({'type': 'urban_street'}, l['pts']) for l in grid]
    autos = suppress_covered(autos, gridman)
    autos = clear_freeway_corridors(autos, [l for l in mlines if l['type'] in ('freeway', 'ramp')])
    grid = suppress_covered(grid, manual)
    grid = clear_freeway_corridors(grid, [l for l in mlines if l['type'] in ('freeway', 'ramp')])
    rails = [(r, pts) for r, pts in design_alignments(all_traced()) if r['type'] == 'rail']
    grid = clip_rail_corridors(grid, rails, zones)
    autos = clip_rail_corridors(autos, rails, zones)
    for l in grid:
        l['src'] = 'grid'
    # smooth / regularise auto geometry
    for l in autos:
        s = robust_smooth(l['pts'], 2.8 if polyline_length(l['pts']) > 12 else 1.5, iters=3)
        z = l['zone']
        eps = {'industrial': 1.8, 'downtown': 1.2, 'city': 0.9, 'town_center': 0.9}.get(z['kind'] if z else None, 0.35)
        s2 = rdp(s, eps)
        s2[0], s2[-1] = l['pts'][0], l['pts'][-1]
        l['pts'] = s2
    P = road_prob()
    rm = np.load(path('tools/.cache/road_mask.npy'))
    mask_dt = cv2.distanceTransform(rm.astype(np.uint8), cv2.DIST_L2, 5)
    for l in autos:
        l['type'] = classify_auto(l, P, mask_dt)
    for l in grid:
        l['pts'] = rdp(l['pts'], 0.3)
        l['name'] = None
        l['route'] = None
    for l in grid:
        l['name'] = None
        l['route'] = None
    lines = mlines + grid + autos
    lines = snap_endpoints(lines)
    regularize_t_junctions(lines)
    pieces = node_network(lines)
    build_outputs(lines, pieces, zones)


def build_outputs(lines, pieces, zones):
    import networkx as nx
    key = lambda p: (round(float(p[0]) * 10), round(float(p[1]) * 10))
    G = nx.MultiGraph()
    nodes = {}

    def nid(p):
        k = key(p)
        # merge with an existing node within 0.3px
        for dx in (-3, -2, -1, 0, 1, 2, 3):
            for dy in (-3, -2, -1, 0, 1, 2, 3):
                kk = (k[0] + dx, k[1] + dy)
                if kk in nodes:
                    return nodes[kk]
        nodes[k] = len(nodes)
        G.add_node(nodes[k], xy=(float(p[0]), float(p[1])))
        return nodes[k]

    for i, pts in pieces:
        a, b = nid(pts[0]), nid(pts[-1])
        pts = pts.copy()
        pts[0] = G.nodes[a]['xy']
        pts[-1] = G.nodes[b]['xy']
        l = lines[i]
        G.add_edge(a, b, pts=pts, attrs={k: l.get(k) for k in ('src', 'type', 'name', 'route', 'def_id', 'lanes', 'layer', 'oneway', 'interchange', 'min_radius_m', 'virtual')},
                   zone=l.get('zone'))
    # drop tiny auto stubs left by trimming/snapping
    for u, v, k, d in list(G.edges(keys=True, data=True)):
        if d['attrs']['src'] != 'manual' and polyline_length(d['pts']) < 2.0 and (G.degree(u) == 1 or G.degree(v) == 1):
            G.remove_edge(u, v, k)
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    merge_chains(G)
    trim_traced_at_rivers(G)
    clean_faces(G)
    engineer_pass(G)
    link_ramp_gores(G)
    prune_steep_spurs(G)
    from tools.lib import engineer as E
    merge_chains(G)
    E.remove_duplicates(G)
    for u, v in {(min(a, b), max(a, b)) for a, b in G.edges() if G.number_of_edges(a, b) > 1}:
        es = sorted(G.get_edge_data(u, v).items(), key=lambda kv: polyline_length(kv[1]['pts']))
        short = LineString(es[0][1]['pts'])
        if short.hausdorff_distance(LineString(es[1][1]['pts'])) < 8.0 or short.buffer(4).contains(LineString(es[1][1]['pts'])) \
                or LineString(es[1][1]['pts']).buffer(4).intersection(short).length > 0.6 * short.length:
            if es[0][1]['attrs']['src'] != 'manual':
                G.remove_edge(u, v, es[0][0])
            elif es[1][1]['attrs']['src'] != 'manual':
                G.remove_edge(u, v, es[1][0])  # a traced loop shadowing an authored road between the same junctions
        elif es[0][1]['attrs']['src'] == 'manual' and es[1][1]['attrs']['src'] != 'manual' \
                and LineString(es[1][1]['pts']).hausdorff_distance(short) < 10.0:
            G.remove_edge(u, v, es[1][0])
    E.prune_dead_ends(G)
    merge_chains(G)
    for _ in range(3):
        if not E.remove_duplicates(G):
            break
        E.prune_dead_ends(G)
        merge_chains(G)
    classify_rural(G)
    write(G, zones)


def trim_traced_at_rivers(G, max_wet=3.0):
    """Only authored roads bridge rivers. Traced/grid streets that run into or
    across a river are cut back to the bank (creeks < max_wet px stay as culverts)."""
    from tools.lib import engineer as E
    wb = load_json(path('data/water/water_bodies.geojson'))
    water = unary_union([shape(f['geometry']) for f in wb['features']])
    wbuf = water.buffer(1.5)
    n_cut = 0
    for u, v, k, d in list(G.edges(keys=True, data=True)):
        if d['attrs']['src'] == 'manual':
            continue
        g = LineString(d['pts'])
        if not g.intersects(water) or g.intersection(water).length <= max_wet:
            continue
        pts = E.oriented(G, u, v, d)
        g = LineString(pts)
        rest = g.difference(wbuf)
        pieces = [rest] if rest.geom_type == 'LineString' else [q for q in getattr(rest, 'geoms', []) if q.geom_type == 'LineString']
        G.remove_edge(u, v, k)
        for q in pieces:
            c = np.asarray(q.coords)
            if q.length < 2:
                continue
            for node, end in ((u, 0), (v, -1)):
                nxy = np.asarray(G.nodes[node]['xy'])
                if np.hypot(*(c[0] - nxy)) < 0.3:
                    nn = E.new_node(G, c[-1]); G.add_edge(node, nn, pts=c, attrs=d['attrs'], zone=d.get('zone'))
                elif np.hypot(*(c[-1] - nxy)) < 0.3:
                    nn = E.new_node(G, c[0]); G.add_edge(nn, node, pts=c, attrs=d['attrs'], zone=d.get('zone'))
        n_cut += 1
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    print('  traced streets cut back at rivers:', n_cut)


def link_ramp_gores(G):
    """Connect each ramp gore (on the freeway edge) to the freeway centreline with a
    short virtual merge link, so the network graph is connected through interchanges."""
    from tools.lib import engineer as E
    for n in list(G.nodes):
        if G.degree(n) != 1:
            continue
        (u, v, k, d), = list(G.edges(n, keys=True, data=True))
        if d['attrs']['type'] != 'ramp':
            continue
        xy = np.asarray(G.nodes[n]['xy'])
        best = None
        for a, b, kk, dd in G.edges(keys=True, data=True):
            if dd['attrs']['type'] != 'freeway':
                continue
            g = LineString(dd['pts'])
            dist = g.distance(Point(xy))
            if dist < 8 and (best is None or dist < best[0]):
                best = (dist, a, b, kk)
        if best is None:
            continue
        _, a, b, kk = best
        # two gores on opposite carriageways at (nearly) the same station share one merge
        # node on the centreline (no 3 m freeway stubs between junctions)
        g = LineString(G.edges[a, b, kk]['pts'])
        q = g.interpolate(g.project(Point(xy)))
        t = None
        for m in (a, b):
            if G.degree(m) >= 3 and Point(G.nodes[m]['xy']).distance(q) < 4.0:
                t = m
        if t is None:
            t = E.split_edge(G, a, b, kk, xy)
        attrs = dict(d['attrs']); attrs['virtual'] = True; attrs['name'] = 'merge link'
        G.add_edge(n, t, pts=np.array([xy, G.nodes[t]['xy']]), attrs=attrs, zone=d.get('zone'))


def engineer_pass(G, rounds=4):
    from tools.lib import engineer as E
    from tools.lib.engineer import design_alignments
    wb = load_json(path('data/water/water_bodies.geojson'))
    water = unary_union([shape(f['geometry']) for f in wb['features']])
    bar = [water]
    for rr, pts in design_alignments(all_traced()):
        if rr['type'] == 'rail':
            bar.append(LineString(pts).buffer((4.5 + 4.0 * (rr.get('tracks', 1) - 1)) / 5.0 + 1.0))
        elif rr['type'] == 'freeway':
            bar.append(LineString(pts).buffer(wpx('freeway') / 2 + 1.0))
    barriers = unary_union(bar)
    for r in range(rounds):
        st = dict(loops=E.fix_loops(G), near=E.connect_near_misses(G, water=barriers), short=E.contract_short(G, max_len=4.2),
                  sharp=E.fix_sharp_angles(G), dup=E.remove_duplicates(G), dead=E.prune_dead_ends(G),
                  comp=E.prune_components(G))
        st['conn'] = E.connect_components(G, barriers=barriers)
        merge_chains(G)
        print('  engineer pass', r + 1, st)
        if not any(st.values()):
            break


def edge_prob(pts, P):
    s = resample(np.asarray(pts), 1.0)
    return float(field_at(P, s).mean())


def clean_faces(G, min_area=110.0, spur=9.0, rounds=6):
    """Remove loops around yards/roofs (tiny faces) and short spurs from
    auto-extracted streets; manual/grid edges are never removed."""
    from shapely.ops import polygonize
    P = road_prob()
    for _ in range(rounds):
        changed = 0
        edges = list(G.edges(keys=True, data=True))
        geoms = [LineString(d['pts']) for _, _, _, d in edges]
        tree = STRtree(geoms)
        faces = [f for f in polygonize(geoms) if f.area < min_area]
        faces.sort(key=lambda f: f.area)
        removed = set()
        for f in faces:
            bd = f.boundary.buffer(0.15)
            cand = []
            for j in tree.query(bd):
                if j in removed:
                    continue
                u, v, k, d = edges[j]
                if d['attrs']['src'] in ('manual', 'grid'):
                    continue
                if geoms[j].within(bd):
                    cand.append(j)
            if not cand:
                continue
            j = min(cand, key=lambda j: edge_prob(edges[j][3]['pts'], P) - 0.002 * geoms[j].length)
            u, v, k, d = edges[j]
            if G.has_edge(u, v, k):
                G.remove_edge(u, v, k)
                removed.add(j)
                changed += 1
        # short spurs
        for u, v, k, d in list(G.edges(keys=True, data=True)):
            if d['attrs']['src'] in ('manual', 'grid'):
                continue
            if (G.degree(u) == 1 or G.degree(v) == 1) and polyline_length(d['pts']) < spur:
                G.remove_edge(u, v, k)
                changed += 1
        G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
        merge_chains(G)
        if not changed:
            break
    # drop small disconnected auto fragments
    import networkx as nx
    for comp in list(nx.connected_components(G)):
        sub = G.subgraph(comp)
        if all(d['attrs']['src'] not in ('manual', 'grid') for _, _, d in sub.edges(data=True)):
            if sum(polyline_length(d['pts']) for _, _, d in sub.edges(data=True)) < 40:
                G.remove_nodes_from(list(comp))


def prune_steep_spurs(G, max_grade=0.22):
    """Dead-end driveways / tracks that would climb straight up a slope (a real
    driveway follows the contour) are dropped. Uses the current terrain if present."""
    p = path('data/terrain/height_f32.bin')
    if not os.path.exists(p):
        return 0
    T = np.fromfile(p, np.float32).reshape(H, W)
    n = 0
    for u, v, k, d in list(G.edges(keys=True, data=True)):
        a = d['attrs']
        if a['src'] == 'manual' or d['zone'] is not None or not (G.degree(u) == 1 or G.degree(v) == 1):
            continue
        pts = resample(np.asarray(d['pts']), 1.0)
        L = polyline_length(pts)
        if L < 4:
            continue
        z = field_at(T, pts)
        if abs(float(z[-1] - z[0])) / (L * 2.5) > max_grade or (np.abs(np.diff(z)) / 2.5).mean() > max_grade:
            G.remove_edge(u, v, k)
            n += 1
    G.remove_nodes_from([q for q in list(G.nodes) if G.degree(q) == 0])
    print('  steep dead-end spurs removed:', n)
    return n


def classify_rural(G):
    for u, v, k, d in G.edges(keys=True, data=True):
        a = d['attrs']
        if a['src'] == 'manual' or d['zone'] is not None:
            continue
        L = polyline_length(d['pts'])
        dead = G.degree(u) == 1 or G.degree(v) == 1
        a['type'] = 'driveway' if (dead and L < 35) else 'gravel'


def same(a, b):
    return all(a['attrs'].get(k) == b['attrs'].get(k) for k in ('type', 'name', 'route', 'def_id', 'layer', 'oneway'))


def merge_chains(G):
    changed = True
    while changed:
        changed = False
        for n in list(G.nodes):
            if n not in G or G.degree(n) != 2:
                continue
            es = list(G.edges(n, keys=True, data=True))
            if len(es) != 2:
                continue
            (u1, v1, k1, d1), (u2, v2, k2, d2) = es
            o1 = v1 if u1 == n else u1
            o2 = v2 if u2 == n else u2
            if o1 == n or o2 == n or not same(d1, d2):
                continue
            xy = np.asarray(G.nodes[n]['xy'])
            p1 = d1['pts'] if np.allclose(d1['pts'][-1], xy) else d1['pts'][::-1]
            p2 = d2['pts'] if np.allclose(d2['pts'][0], xy) else d2['pts'][::-1]
            G.remove_edge(u1, v1, k1)
            G.remove_edge(u2, v2, k2)
            G.remove_node(n)
            G.add_edge(o1, o2, pts=np.vstack([p1, p2[1:]]), attrs=d1['attrs'], zone=d1['zone'])
            changed = True


def bearing_at(pts, xy):
    pts = np.asarray(pts)
    if np.allclose(pts[0], xy):
        q = pts[min(len(pts) - 1, 1)]
        s = resample(pts, 1.0)
        q = s[min(len(s) - 1, 6)]
    else:
        s = resample(pts[::-1], 1.0)
        q = s[min(len(s) - 1, 6)]
    return math.degrees(math.atan2(q[1] - xy[1], q[0] - xy[0])) % 360


def junction_kind(G, n):
    deg = G.degree(n)
    xy = G.nodes[n]['xy']
    es = list(G.edges(n, data=True))
    types = {d['attrs']['type'] for _, _, d in es}
    if deg == 1:
        return 'end'
    if deg == 2:
        return 'type_change'
    if types & GRADE_SEP:
        return 'merge'
    bs = sorted(bearing_at(d['pts'], xy) for _, _, d in es)
    gaps = [(bs[(i + 1) % len(bs)] - bs[i]) % 360 for i in range(len(bs))]
    zone_kinds = {d['zone']['kind'] if d['zone'] else None for _, _, d in es}
    urban = bool(zone_kinds & {'downtown', 'industrial', 'town_center'})
    if deg == 3:
        mx = max(gaps)
        if min(gaps) < 40:
            return 'fork'
        if abs(mx - 180) < 30 or any(abs(g - 180) < 25 for g in gaps):
            return 'city' if urban else 'T'
        return 'fork' if mx > 200 else 'angled'
    if deg == 4:
        right = all(abs(g - 90) < 30 for g in gaps)
        if urban:
            return 'city'
        return 'cross' if right else 'angled'
    return 'multi'


def load_registry():
    p = path('data/roads/id_registry.json')
    return load_json(p) if os.path.exists(p) else {'edges': {}, 'nodes': {}, 'next': {}}


def reg_id(reg, kind, prefix, sig):
    table = reg[kind]
    # match by signature proximity (midpoint within 2px, similar length)
    for k, v in table.items():
        if v['prefix'] == prefix and abs(v['sig'][0] - sig[0]) < 2.0 and abs(v['sig'][1] - sig[1]) < 2.0 and abs(v['sig'][2] - sig[2]) < max(3.0, 0.15 * sig[2]):
            if not v.get('_used'):
                v['_used'] = True
                v['sig'] = sig
                return k
    n = reg['next'].get(f'{kind}:{prefix}', 1)
    reg['next'][f'{kind}:{prefix}'] = n + 1
    k = f'{prefix}_RD_{n:04d}' if kind == 'edges' else f'{prefix}_JCT_{n:04d}'
    table[k] = {'prefix': prefix, 'sig': sig, '_used': True}
    return k


def region_prefix(pt, zones):
    z = zone_of(pt, zones)
    return REGION_PREFIX[z['settlement'] if z else None]


def chaikin(p, n=2):
    """Corner-cutting smoothing with fixed endpoints (curves for traced local streets)."""
    p = np.asarray(p, float)
    for _ in range(n):
        q = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        q.append(p[-1])
        p = np.asarray(q)
    return rdp(p, 0.08)


MIN_RADIUS_M = {'residential': 12, 'urban_street': 12, 'rural': 20, 'gravel': 12, 'dirt': 8, 'driveway': 5, 'collector': 35}


def local_radius_px(P, h=6):
    """Circumradius (px) through P[i-h], P[i], P[i+h] (1 px samples); inf near the ends."""
    R = np.full(len(P), np.inf)
    if len(P) < 2 * h + 1:
        return R
    a, b, c = P[:-2 * h], P[h:-h], P[2 * h:]
    ab, bc, ca = np.hypot(*(b - a).T), np.hypot(*(c - b).T), np.hypot(*(a - c).T)
    area2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    R[h:-h] = ab * bc * ca / np.maximum(2 * area2, 1e-9)
    return R


def ease_min_radius(pts, rmin_m, max_shift_px=4.0, free_end=None):
    """Traced / grid local streets: kinks and elbows tighter than the class minimum
    (AASHO local street 12 m centreline radius) become curves. The turning angle is kept
    but spread along the street until the curvature is <= 1/R (a spiral-ish fillet, no
    shrinkage of large loops); the ends stay on their junctions. Streets whose fix would
    move them more than max_shift_px off the trace (true hairpins) are left alone."""
    if free_end == 0:  # dead end at the start: work from the junction end
        q = ease_min_radius(np.asarray(pts)[::-1], rmin_m, max_shift_px, 1)
        return pts if q is not None and len(q) == len(pts) and np.allclose(q, np.asarray(pts)[::-1]) else q[::-1]
    P = resample(np.asarray(pts, float), 1.0)
    if len(P) < 15:
        return pts
    if not (local_radius_px(P) < rmin_m * 1.05 / 2.5).any():
        return pts
    target = rmin_m * 1.3 / 2.5
    d = np.diff(P, axis=0)
    step = np.hypot(d[:, 0], d[:, 1])
    phi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    k = np.diff(phi)                     # turning per sample (rad / px)
    kmax = 1.0 / target * float(np.mean(step))
    for _ in range(4000):
        over = np.abs(k) > kmax * 1.0001
        if not over.any():
            break
        ex = np.where(over, k - np.sign(k) * kmax, 0.0)
        k = k - ex
        k[1:] += 0.5 * ex[:-1]
        k[:-1] += 0.5 * ex[1:]
        # turning pushed off either end is folded back (the end tangents stay put)
        k[0] += 0.5 * ex[0]
        k[-1] += 0.5 * ex[-1]
    phi2 = phi[0] + np.r_[0.0, np.cumsum(k)]
    Q = np.vstack([P[:1], P[0] + np.cumsum(np.c_[np.cos(phi2), np.sin(phi2)] * step[:, None], axis=0)])
    s = np.r_[0, np.cumsum(step)]
    if free_end != 1:
        Q = Q - (Q[-1] - P[-1]) * (s / s[-1])[:, None]      # close onto the far junction
    if float(np.hypot(*(Q - P).T).max()) > max_shift_px:
        return pts
    Q[0] = P[0]
    if free_end != 1:
        Q[-1] = P[-1]
    return rdp(Q, 0.04)


def finalize_geometry(G):
    """Final plan shape of every edge: chaikin curves for traced streets, then minimum-
    radius easing for non-authored local streets. An eased street may move at most a few
    px, and never onto (duplicate) or across (unjunctioned crossing) another street."""
    items = list(G.edges(keys=True, data=True))
    for u, v, k, d in items:
        pts = np.asarray(d['pts'])
        if d['attrs']['src'] in ('auto', 'auto_gap') and len(pts) > 2:
            pts = chaikin(pts, 3)
        d['pts'] = pts
    geoms = [LineString(d['pts']) for *_, d in items]
    tree = STRtree(geoms)
    n_ok = n_skip = 0
    for i, (u, v, k, d) in enumerate(items):
        if d['attrs']['src'] == 'manual' or d['attrs']['type'] not in MIN_RADIUS_M or len(d['pts']) < 2:
            continue
        g0 = geoms[i]
        for shift in (4.0, 6.0, 9.0):
            fe = 0 if G.degree(u) == 1 and G.degree(v) > 1 else (1 if G.degree(v) == 1 and G.degree(u) > 1 else None)
            if fe is not None and np.hypot(*(np.asarray(d['pts'][0]) - np.asarray(G.nodes[u]['xy']))) > 0.01:
                fe = 1 - fe  # pts run v -> u
            q = ease_min_radius(d['pts'], MIN_RADIUS_M[d['attrs']['type']], shift, fe)
            if q is d['pts']:
                continue
            g = LineString(q)
            bad = False
            if fe is not None:  # a moved dead end must not become a near miss
                e_new = Point(q[-1] if fe == 1 else q[0])
                e_old = Point(d['pts'][-1] if fe == 1 else d['pts'][0])
                for j in tree.query(e_new.buffer(7.0)):
                    if j != i and geoms[j].distance(e_new) < 6.3 and geoms[j].distance(e_new) < geoms[j].distance(e_old):
                        bad = True
            for j in tree.query(g.buffer(7.0)):  # nor move next to someone else's dead end
                if j == i:
                    continue
                uj, vj = items[j][0], items[j][1]
                for nd in (uj, vj):
                    if G.degree(nd) == 1 and nd not in (u, v):
                        pe = Point(G.nodes[nd]['xy'])
                        if g.distance(pe) < 6.3 and g.distance(pe) < g0.distance(pe):
                            bad = True
            for j in tree.query(g.buffer(4.0)):
                if j == i:
                    continue
                h = geoms[j]
                if g.crosses(h) and not g0.crosses(h):
                    bad = True
                    break
                short = min(g.length, h.length)
                ov = min(g.intersection(h.buffer(3.5)).length, h.intersection(g.buffer(3.5)).length)
                ov0 = min(g0.intersection(h.buffer(3.5)).length, h.intersection(g0.buffer(3.5)).length)
                if short >= 6 and ov > 0.5 * short and ov > ov0 + 0.5:
                    bad = True
                    break
            if not bad:
                d['pts'] = q
                geoms[i] = g
                for nd in (u, v):  # a dead end may move with its eased street
                    if G.degree(nd) == 1:
                        e0 = q[0] if np.hypot(*(np.asarray(q[0]) - np.asarray(G.nodes[nd]['xy']))) < np.hypot(*(np.asarray(q[-1]) - np.asarray(G.nodes[nd]['xy']))) else q[-1]
                        G.nodes[nd]['xy'] = (float(e0[0]), float(e0[1]))
                n_ok += 1
                break
        else:
            if (local_radius_px(resample(np.asarray(d['pts']), 1.0)) < MIN_RADIUS_M[d['attrs']['type']] * 1.05 / 2.5).any():
                n_skip += 1
    print(f'  min-radius easing: {n_ok} streets eased, {n_skip} left (would move > 9 px or collide)')


def write_clear_zones(feats):
    """Maintained grass (no trees / brush) for E's vegetation.py: interchange infields and loops
    (faces enclosed by ramps, the freeway and the crossroad) and the freeway wye / flyover
    triangles, minus the pavements. data/roads/clear_zones.geojson (source px)."""
    from shapely.ops import polygonize
    from shapely.geometry import mapping
    geo = [(f['properties'], LineString(np.asarray(f['geometry']['coordinates'])[:, :2])) for f in feats]
    grade_sep = [(p, g) for p, g in geo if p['type'] in ('ramp', 'freeway')]
    ramps = [(p, g) for p, g in grade_sep if p['type'] == 'ramp' and not p.get('virtual')]
    out = []
    groups = {}
    for p, g in ramps:
        groups.setdefault(p.get('interchange') or p.get('def_id') or 'ramp', []).append(g)
    for key, rg in groups.items():
        area = unary_union(rg).buffer(70)
        lines = [g for p, g in geo if g.intersects(area)]
        faces = [f for f in polygonize(unary_union(lines)) if 30 < f.area < 25000]
        rset = unary_union(rg)
        pav = unary_union([g.buffer(wpx(p['type']) / 2 + 1.0) for p, g in geo if g.intersects(area)])
        keep = []
        for f in faces:
            if f.boundary.intersection(rset.buffer(0.6)).length < 8:
                continue  # not bounded by a ramp: an ordinary block
            q = f.difference(pav)
            if q.area > 20:
                keep.append(q)
        if keep:
            u = unary_union(keep)
            out.append({'type': 'Feature', 'geometry': mapping(u), 'properties': {'kind': 'interchange_infield', 'interchange': key,
                                                                                   'margin_m': 4.0, 'area_m2': round(u.area * 6.25)}})
    fc_ = {'type': 'FeatureCollection', 'name': 'clear_zones', 'features': out}
    save_json(path('data/roads/clear_zones.geojson'), fc_)
    print(f'  clear zones: {len(out)} ({sum(f["properties"]["area_m2"] for f in out) / 1e4:.1f} ha)')


def write(G, zones):
    wb = load_json(path('data/water/water_bodies.geojson'))
    water = unary_union([shape(f['geometry']) for f in wb['features']])
    reg = load_registry()
    finalize_geometry(G)
    # nodes
    node_ids = {}
    node_feats = []
    for n in sorted(G.nodes, key=lambda n: (round(G.nodes[n]['xy'][1] / 50), G.nodes[n]['xy'][0])):
        xy = G.nodes[n]['xy']
        pre = region_prefix(xy, zones)
        k = reg_id(reg, 'nodes', pre, [rnd(xy[0], 1), rnd(xy[1], 1), 0.0])
        node_ids[n] = k
        node_feats.append((n, k, xy))
    feats = []
    bridges = []
    edge_ids = {}
    for u, v, d in sorted(G.edges(data=True), key=lambda e: tuple(np.round(e[2]['pts'][len(e[2]['pts']) // 2]))):
        pts = np.asarray(d['pts'])
        a_xy = np.asarray(G.nodes[u]['xy'])
        if np.hypot(*(pts[-1] - a_xy)) < np.hypot(*(pts[0] - a_xy)):
            pts = pts[::-1]  # geometry always runs from 'from' to 'to'
        L = polyline_length(pts)
        mid = resample(pts, 1.0)
        mid = mid[len(mid) // 2]
        pre = region_prefix(mid, zones)
        rid = reg_id(reg, 'edges', pre, [rnd(mid[0], 1), rnd(mid[1], 1), rnd(L, 1)])
        a = d['attrs']
        t = a['type']
        spec = TYPES[t]
        ls = LineString(pts)
        br = []
        if ls.intersects(water):
            x = ls.intersection(water)
            segs = [x] if x.geom_type == 'LineString' else [g for g in getattr(x, 'geoms', []) if g.geom_type == 'LineString']
            for s_ in segs:
                if s_.length < 1.0:
                    continue
                a0, a1 = ls.project(Point(s_.coords[0])), ls.project(Point(s_.coords[-1]))
                a0, a1 = min(a0, a1), max(a0, a1)
                pad = 2.0 + spec['width_m'] / 2.5 * 0.3
                b0, b1 = max(0, a0 - pad), min(L, a1 + pad)
                br.append([rnd(b0, 2), rnd(b1, 2)])
        z = d['zone']
        props = {
            'id': rid, 'type': t, 'name': a.get('name'), 'route': a.get('route'),
            'from': node_ids[u], 'to': node_ids[v],
            'width_m': width_for(t, a.get('lanes')), 'lanes': a.get('lanes') or spec['lanes'], 'surface': spec['surface'],
            'material': spec['material'], 'shoulder_m': spec['shoulder_m'], 'speed_mph': spec['speed_mph'],
            'oneway': bool(a.get('oneway')), 'layer': a.get('layer') or 0,
            'interchange': a.get('interchange'), 'min_radius_m': a.get('min_radius_m'), 'virtual': bool(a.get('virtual')),
            'grade_separated': t in GRADE_SEP,
            'zone': z['id'] if z else None, 'settlement': z['settlement'] if z else None,
            'source': a['src'], 'def_id': a.get('def_id'),
            'length_m': rnd(L * 2.5, 1),
            'bridge_spans': br,
            'status': {'manual': 'traced', 'auto': 'extracted', 'grid': 'grid_detected'}.get(a['src'], 'inferred_gap'),
        }
        feats.append(geojson_line(pts, props))
        edge_ids[(u, v)] = rid
        for i, (b0, b1) in enumerate(br):
            seg = substring(ls, b0, b1)
            bridges.append(geojson_line(np.asarray(seg.coords), {
                'id': f'{rid}_BR{i + 1}', 'road': rid, 'road_type': t, 'name': a.get('name'),
                'start_along_px': b0, 'end_along_px': b1, 'length_m': rnd((b1 - b0) * 2.5, 1),
                'deck_width_m': spec['width_m'], 'structure': 'steel truss' if t in ('rural', 'gravel') and (b1 - b0) * 2.5 > 30 else ('concrete girder' if t in ('freeway', 'highway', 'ramp', 'arterial') else 'concrete slab')}))
    nfeats = []
    for n, k, xy in node_feats:
        es = list(G.edges(n, data=True))
        nfeats.append(geojson_point(xy, {'id': k, 'kind': junction_kind(G, n), 'degree': G.degree(n),
                                         'road_types': sorted({d['attrs']['type'] for _, _, d in es})}))
    for t in (reg['edges'], reg['nodes']):
        for k in list(t):
            if not t[k].pop('_used', False):
                t[k]['retired'] = True
    save_json(path('data/roads/roads.geojson'), fc(feats, 'roads'))
    save_json(path('data/roads/road_nodes.geojson'), fc(nfeats, 'road_nodes'))
    save_json(path('data/roads/bridges.geojson'), fc(bridges, 'bridges'))
    save_json(path('data/roads/id_registry.json'), reg)
    write_clear_zones(feats)
    from collections import Counter
    c = Counter(f['properties']['type'] for f in feats)
    tot = Counter()
    for f in feats:
        tot[f['properties']['type']] += f['properties']['length_m']
    print(f'roads: {len(feats)} edges, {len(nfeats)} nodes, {len(bridges)} bridges')
    for t in TYPES:
        if c[t]:
            print(f'  {t:13s} {c[t]:5d} edges {tot[t] / 1000:7.2f} km')


if __name__ == '__main__':
    main()
