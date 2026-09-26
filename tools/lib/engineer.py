"""Graph-level road engineering fixes (operate on the roads.py MultiGraph).

Edges carry d['pts'] (Nx2, from node u to node v in insertion order, but
orientation is resolved by endpoint matching) and d['attrs'] (src, type, ...).
Rank: manual (authored) > grid > auto. Only lower-ranked geometry is modified
unless stated.
"""
import math
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.ops import substring
from shapely.strtree import STRtree
from .trace import resample

RANK = {'manual': 3, 'grid': 2, 'auto': 1, 'auto_gap': 1}
NO_TOUCH = {'freeway', 'ramp'}


def rank(d):
    return RANK.get(d['attrs']['src'], 1)


def length(p):
    p = np.asarray(p)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def oriented(G, u, v, d):
    """pts oriented from u to v."""
    p = np.asarray(d['pts'])
    a = np.asarray(G.nodes[u]['xy'])
    return p if np.hypot(*(p[0] - a)) <= np.hypot(*(p[-1] - a)) else p[::-1]


def new_node(G, xy):
    n = max(G.nodes) + 1 if len(G) else 0
    G.add_node(n, xy=(float(xy[0]), float(xy[1])))
    return n


def split_edge(G, u, v, k, at_xy):
    """Split edge (u,v,k) at the point nearest at_xy; returns the new node."""
    d = G.edges[u, v, k]
    p = oriented(G, u, v, d)
    ls = LineString(p)
    s = ls.project(Point(at_xy))
    if s < 0.3:
        return u
    if s > ls.length - 0.3:
        return v
    q = np.asarray(ls.interpolate(s).coords[0])
    a = np.asarray(substring(ls, 0, s).coords)
    b = np.asarray(substring(ls, s, ls.length).coords)
    n = new_node(G, q)
    attrs, zone = d['attrs'], d.get('zone')
    G.remove_edge(u, v, k)
    G.add_edge(u, n, pts=a, attrs=dict(attrs), zone=zone)
    G.add_edge(n, v, pts=b, attrs=dict(attrs), zone=zone)
    return n


def edge_index(G):
    items = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True)]
    geoms = [LineString(d['pts']) for *_, d in items]
    return items, geoms, STRtree(geoms)


def connect_near_misses(G, tol=7.0, water=None, log=None, tol_by_rank={3: 16.0, 2: 12.0, 1: 7.0}):
    fixed = 0
    for n in list(G.nodes):
        if n not in G or G.degree(n) != 1:
            continue
        (u, v, k, d), = list(G.edges(n, keys=True, data=True))
        if d['attrs']['type'] in NO_TOUCH:
            continue
        xy = np.asarray(G.nodes[n]['xy'])
        if xy[0] < 1.5 or xy[1] < 1.5 or xy[0] > 1998.5 or xy[1] > 665.5:
            continue
        tol = tol_by_rank.get(rank(d), 7.0)
        items, geoms, tree = edge_index(G)
        p = Point(xy)
        own = oriented(G, n, v if u == n else u, d)
        tan = own[0] - resample(own, 1.0)[min(4, len(resample(own, 1.0)) - 1)]
        tan = tan / max(np.hypot(*tan), 1e-9)
        best = None
        for j in tree.query(p.buffer(tol)):
            uu, vv, kk, dd = items[j]
            if (uu, vv, kk) == (u, v, k) or dd['attrs']['type'] in NO_TOUCH:
                continue
            if (v if u == n else u) in (uu, vv):
                continue  # would close a tiny loop back onto the spur's own junction
            g = geoms[j]
            dist = g.distance(p)
            if dist >= tol or dist < 1e-6:
                continue
            q = np.asarray(g.interpolate(g.project(p)).coords[0])
            dirv = (q - xy) / max(dist, 1e-9)
            # prefer targets ahead of the dead end
            score = dist - 3.0 * float(np.dot(dirv, tan))
            if best is None or score < best[0]:
                best = (score, j, q)
        if best is None:
            continue
        _, j, q = best
        link = LineString([xy, q])
        if water is not None and link.intersects(water):
            continue
        cross = [i for i in tree.query(link) if i != j and items[i][:3] != (u, v, k) and geoms[i].crosses(link)]
        if cross:
            continue
        uu, vv, kk, _ = items[j]
        # snap to an existing node if close
        tgt = None
        for cand in (uu, vv):
            if np.hypot(*(np.asarray(G.nodes[cand]['xy']) - q)) < 2.0:
                tgt = cand
        if tgt is None:
            tgt = split_edge(G, uu, vv, kk, q)
        other = v if u == n else u
        pts = oriented(G, other, n, d)
        pts = np.vstack([pts, [G.nodes[tgt]['xy']]])
        G.remove_edge(u, v, k)
        G.remove_node(n)
        G.add_edge(other, tgt, pts=pts, attrs=d['attrs'], zone=d.get('zone'))
        fixed += 1
    return fixed


def contract_short(G, max_len=3.0):
    fixed = 0
    changed = True
    while changed:
        changed = False
        for u, v, k, d in list(G.edges(keys=True, data=True)):
            if u == v or not G.has_edge(u, v, k):
                continue
            if length(d['pts']) >= max_len or G.degree(u) < 3 or G.degree(v) < 3:
                continue
            if d['attrs']['type'] in NO_TOUCH:
                continue
            # merge into the junction carrying the higher-ranked (authored) roads
            ru = max(rank(dd) for *_, dd in G.edges(u, keys=True, data=True))
            rv = max(rank(dd) for *_, dd in G.edges(v, keys=True, data=True))
            keep, drop = (u, v) if ru >= rv else (v, u)
            mid = np.asarray(G.nodes[keep]['xy'])
            G.remove_edge(u, v, k)
            for a, b, kk, dd in list(G.edges(drop, keys=True, data=True)):
                o = b if a == drop else a
                p = oriented(G, o, drop, dd).copy()
                p[-1] = mid
                G.remove_edge(a, b, kk)
                if o == drop:
                    continue
                G.add_edge(o, keep, pts=p, attrs=dd['attrs'], zone=dd.get('zone'))
            G.remove_node(drop)
            fixed += 1
            changed = True
            break
    return fixed


def leg_bearing(G, n, u, v, d, dist=6):
    other = v if u == n else u
    p = resample(oriented(G, n, other, d), 1.0)
    q = p[min(len(p) - 1, dist)]
    xy = G.nodes[n]['xy']
    return math.degrees(math.atan2(q[1] - xy[1], q[0] - xy[0])) % 360


def fix_sharp_angles(G, min_deg=25.0, sep=4.0):
    """Where two legs leave a junction almost together, re-attach the lower-ranked
    leg to the other road at the point where they actually diverge, giving a clean
    T/fork. A leg that never diverges (a duplicate) is removed if not authored."""
    fixed = 0
    for n in list(G.nodes):
        if n not in G or G.degree(n) < 3:
            continue
        legs = [(u, v, k, d) for u, v, k, d in G.edges(n, keys=True, data=True)]
        bs = [leg_bearing(G, n, *l[:2], l[3]) for l in legs]
        pair = None
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                a = abs((bs[i] - bs[j] + 180) % 360 - 180)
                if a < min_deg and (pair is None or a < pair[0]):
                    pair = (a, i, j)
        if pair is None:
            continue
        _, i, j = pair
        li, lj = legs[i], legs[j]
        if li[3]['attrs']['type'] in NO_TOUCH or lj[3]['attrs']['type'] in NO_TOUCH:
            continue
        # minor leg = lower rank, then lower type importance, then shorter
        key = lambda l: (rank(l[3]), -['driveway', 'dirt', 'gravel', 'residential', 'urban_street', 'rural', 'collector', 'main_street', 'arterial', 'highway', 'freeway', 'ramp'].index(l[3]['attrs']['type']) * -1, length(l[3]['pts']))
        major, minor = (li, lj) if key(li) >= key(lj) else (lj, li)
        both_authored = rank(minor[3]) == 3 and rank(major[3]) == 3
        mu, mv, mk, md = minor
        mo = mv if mu == n else mu
        mp = resample(oriented(G, n, mo, md), 1.0)
        Mu, Mv, Mk, Md = major
        Mo = Mv if Mu == n else Mu
        Mls = LineString(oriented(G, n, Mo, Md))
        idx = None
        for t in range(len(mp)):
            if Mls.distance(Point(mp[t])) > sep:
                idx = t
                break
        if both_authored and (idx is None or idx > 20):
            continue  # long shared run between authored roads: fix by hand
        G.remove_edge(mu, mv, mk)
        if idx is None or idx >= len(mp) - 2:
            # minor leg is a duplicate of the major one
            if G.degree(mo) == 0:
                G.remove_node(mo)
            fixed += 1
            continue
        attach = np.asarray(Mls.interpolate(Mls.project(Point(mp[idx]))).coords[0])
        v1 = mp[min(idx + 6, len(mp) - 1)] - attach
        tq = np.asarray(Mls.interpolate(min(Mls.length, Mls.project(Point(attach)) + 2)).coords[0]) - np.asarray(Mls.interpolate(max(0, Mls.project(Point(attach)) - 2)).coords[0])
        cosang = abs(np.dot(v1, tq)) / max(np.hypot(*v1) * np.hypot(*tq), 1e-9)
        if cosang > math.cos(math.radians(25)) and rank(minor[3]) == 1 and length(mp) < 40:
            # still a shallow merge: a short traced spur -> drop it
            if G.degree(mo) == 0:
                G.remove_node(mo)
            fixed += 1
            continue
        tgt = split_edge(G, *[(a, b, c) for a, b, c in G.edges(n, keys=True) if (a, b, c) in ((Mu, Mv, Mk), (Mv, Mu, Mk))][0], attach) if G.has_edge(Mu, Mv, Mk) else n
        newp = np.vstack([[G.nodes[tgt]['xy']], mp[idx:]])
        G.add_edge(tgt, mo, pts=newp, attrs=md['attrs'], zone=md.get('zone'))
        fixed += 1
    return fixed


def remove_duplicates(G, tol=3.5, frac=0.6):
    fixed = 0
    items, geoms, tree = edge_index(G)
    dead = set()
    for i, (u, v, k, d) in enumerate(items):
        if i in dead or d['attrs']['type'] in NO_TOUCH:
            continue
        g = geoms[i]
        for j in tree.query(g.buffer(tol)):
            if j == i or j in dead:
                continue
            h = geoms[j]
            short = min(g.length, h.length)
            if short < 4:
                continue
            if min(g.intersection(h.buffer(tol)).length, h.intersection(g.buffer(tol)).length) > frac * short:
                a, b = items[i], items[j]
                if b[3]['attrs']['type'] in NO_TOUCH:
                    loser = i
                else:
                    loser = i if (rank(a[3]), g.length) < (rank(b[3]), h.length) else j
                if rank(items[loser][3]) == 3:
                    continue
                dead.add(loser)
                if loser == i:
                    break
    for i in dead:
        u, v, k, _ = items[i]
        if G.has_edge(u, v, k):
            G.remove_edge(u, v, k)
            fixed += 1
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    return fixed


def prune_components(G, keep_len_px=60):
    """Remove isolated auto-only fragments; keep pieces containing authored roads."""
    fixed = 0
    comps = sorted(nx.connected_components(G), key=lambda c: -sum(length(d['pts']) for *_, d in G.subgraph(c).edges(data=True)))
    for c in comps[1:]:
        sub = G.subgraph(c)
        authored = any(rank(d) >= 2 for *_, d in sub.edges(data=True))
        if authored:
            continue
        G.remove_nodes_from(list(c))
        fixed += 1
    return fixed


def prune_dead_ends(G, min_len=6.0, min_len_driveway=12.0):
    fixed = 0
    for n in list(G.nodes):
        if n not in G or G.degree(n) != 1:
            continue
        (u, v, k, d), = list(G.edges(n, keys=True, data=True))
        if rank(d) == 3 or d['attrs']['type'] in NO_TOUCH:
            continue
        xy = G.nodes[n]['xy']
        if xy[0] < 1.5 or xy[1] < 1.5 or xy[0] > 1998.5 or xy[1] > 665.5:
            continue
        other = v if u == n else u
        hairpin = any(LineString(dd['pts']).distance(Point(xy)) < 6.5
                      for a, b, kk, dd in G.edges(other, keys=True, data=True) if (a, b, kk) != (u, v, k) and {a, b} != {u, v})
        if hairpin or length(d['pts']) < (min_len_driveway if d['attrs']['type'] in ('driveway',) else min_len):
            G.remove_edge(u, v, k)
            G.remove_node(n)
            fixed += 1
    return fixed


def fix_loops(G, min_area_ratio=1.5):
    """Self-loop edges (a street leaving a junction and returning to it).
    * zero-length / tiny loops (snapping artefacts) are removed;
    * degenerate out-and-back loops (thin: area small relative to length) become a
      single dead-end spur to the loop's farthest point (or are removed if short);
    * real loop streets (P-loops, cul-de-sac rings) are kept."""
    from shapely.geometry import Polygon
    fixed = 0
    for u, v, k, d in list(G.edges(keys=True, data=True)):
        if u != v or not G.has_edge(u, v, k):
            continue
        p = np.asarray(d['pts'])
        L = length(p)
        if L < 6.0:
            G.remove_edge(u, v, k)
            fixed += 1
            continue
        try:
            area = abs(Polygon(p).area) if len(p) >= 4 else 0.0
        except Exception:
            area = 0.0
        # a real loop encloses roughly (L/4)^2 (square) .. (L/2pi)^2*pi (circle)
        if area > min_area_ratio * L:
            continue
        G.remove_edge(u, v, k)
        fixed += 1
        if d['attrs']['type'] in NO_TOUCH:
            continue
        xy = np.asarray(G.nodes[u]['xy'])
        far = int(np.argmax(np.hypot(*(p - xy).T)))
        spur = p[:far + 1]
        near_own = any(LineString(dd['pts']).distance(Point(spur[-1])) < 7.0
                       for *_, dd in G.edges(u, keys=True, data=True))
        if length(spur) >= 8.0 and not near_own:
            n = new_node(G, spur[-1])
            G.add_edge(u, n, pts=spur, attrs=d['attrs'], zone=d.get('zone'))
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    return fixed


MIN_RADIUS_M = {'freeway': 350, 'highway': 150, 'ramp': 40, 'arterial': 60, 'main_street': 40, 'collector': 35,
                'urban_street': 12, 'residential': 12, 'rural': 20, 'gravel': 12, 'dirt': 8, 'driveway': 5, 'rail': 160}


def radii_px(P, h=3):
    """3-point circumradius at every sample (P uniformly spaced); ends = inf."""
    n = len(P)
    R = np.full(n, np.inf)
    if n < 2 * h + 1:
        return R
    a, b, c = P[:-2 * h], P[h:-h], P[2 * h:]
    ab = np.hypot(*(b - a).T); bc = np.hypot(*(c - b).T); ca = np.hypot(*(a - c).T)
    area2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    R[h:-h] = ab * bc * ca / np.maximum(2 * area2, 1e-9)
    return R


def limit_curvature(pts, r_min_px, max_dev_px, pin_px=4.0, iters=60, step=2.0):
    """Geometric design of a traced alignment: a Whittaker (penalised least squares)
    smoother whose curvature penalty is raised locally, iteratively, wherever the
    radius is below the class minimum. Kink-free by construction; the trace is kept
    where it is already drivable. Ends (junction approaches) are pinned. Stops early
    (keeping the best result) if the deviation from the trace would exceed max_dev_px."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    from scipy.ndimage import gaussian_filter1d
    P0 = resample(np.asarray(pts, float), step)
    n = len(P0)
    if n < 10:
        return resample(P0, 1.0)
    w = np.ones(n)
    k = max(1, int(round(pin_px / step)))
    w[0] = 1e6; w[-1] = 1e6  # positions pinned, tangents free (a pinned tangent forces a kink)
    D = sp.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
    lam = np.full(n - 2, 0.05)
    best = P0
    Rmin = lambda Q: np.minimum(radii_px(Q, max(1, int(3 / step * 2))), radii_px(Q, max(2, int(6 / step * 2))))
    for it in range(iters):
        A = sp.diags(w) + D.T @ sp.diags(lam) @ D
        P = np.stack([spsolve(A.tocsc(), w * P0[:, 0]), spsolve(A.tocsc(), w * P0[:, 1])], 1)
        dev = np.hypot(*(P - P0).T).max()
        if dev > max_dev_px:
            break
        best = P
        R = Rmin(P)
        R[:k + 2] = np.inf; R[-k - 2:] = np.inf
        bad = R < r_min_px
        if not bad.any():
            break
        f = gaussian_filter1d(bad.astype(float), 4.0)[1:-1]
        lam = lam * (1.0 + 4.0 * np.clip(f * 3, 0, 1))
        # stiffness must vary slowly along the road, else the curvature just
        # concentrates where stiff and soft sections meet
        lam = np.exp(gaussian_filter1d(np.log(lam), max(3.0, 0.6 * r_min_px / step), mode='nearest'))
    return resample(best, 1.0)


DESIGN_DEV_PX = {'freeway': 16, 'highway': 8, 'ramp': 3, 'arterial': 4, 'main_street': 2.5, 'collector': 5, 'rural': 4,
                 'gravel': 3, 'dirt': 2, 'driveway': 2, 'urban_street': 2, 'residential': 2, 'rail': 12}
DESIGN_ORDER = ['freeway', 'highway', 'rail', 'arterial', 'main_street', 'collector', 'rural', 'urban_street',
                'residential', 'gravel', 'dirt', 'driveway']


def _shift_end(P, end, dxy, taper=25.0):
    P = P.copy()
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    d = s if end == 0 else s[-1] - s
    w = np.clip(1 - d / taper, 0, 1)
    w = w * w * (3 - 2 * w)
    return P + w[:, None] * np.asarray(dxy)[None]


def design_alignments(items):
    """items: list of (def, pts). Returns the same list with horizontally designed
    alignments (limit_curvature per class). Roads are designed in importance order;
    an end that joined another road in the trace is re-attached (with a smooth taper)
    to that road's designed alignment before the road itself is designed."""
    from shapely.geometry import LineString, Point
    orig = [np.asarray(p, float) for _, p in items]
    out = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: DESIGN_ORDER.index(items[i][0]['type']) if items[i][0]['type'] in DESIGN_ORDER else 99)
    done = []
    for i in order:
        r, _ = items[i]
        P = resample(orig[i], 1.0)
        t = r['type']
        for end in (0, -1):
            e = P[end]
            best = None
            for j in done:
                if (items[j][0]['type'] == 'rail') != (t == 'rail'):
                    continue
                g0 = LineString(orig[j])
                dd = g0.distance(Point(e))
                if dd < 2.0 and (best is None or dd < best[0]):
                    best = (dd, j)
            if best is not None:
                j = best[1]
                g1 = LineString(out[j])
                # corresponding point on the designed host: same station as on the trace
                s0 = LineString(orig[j]).project(Point(e), normalized=True)
                q = np.asarray(g1.interpolate(s0, normalized=True).coords[0])
                q = np.asarray(g1.interpolate(g1.project(Point(q))).coords[0])
                if np.hypot(*(q - e)) > 0.05:
                    P = _shift_end(P, 0 if end == 0 else -1, q - e)
        if r.get('design', True) and t in MIN_RADIUS_M:
            rm = r.get('min_radius_m', MIN_RADIUS_M[t]) / 2.5
            P = limit_curvature(P, rm, r.get('max_dev_px', DESIGN_DEV_PX.get(t, 3)))
        out[i] = P
        done.append(i)
    return [(items[i][0], out[i]) for i in range(len(items))]


def connect_components(G, barriers=None, max_link=25.0, drop_len=160.0):
    """Every piece of the network must connect. For each component other than the
    main one: add the shortest believable link (<= max_link px, from one of its
    nodes to a surface road of the main network, not crossing barriers such as water,
    rail yards or freeways); failing that, drop it if it holds no authored road."""
    fixed = 0
    comps = sorted(nx.connected_components(G), key=lambda c: -sum(length(d['pts']) for *_, d in G.subgraph(c).edges(data=True)))
    if len(comps) < 2:
        return 0
    main = comps[0]
    items = [(u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True) if u in main and d['attrs']['type'] not in NO_TOUCH and not d['attrs'].get('virtual')]
    geoms = [LineString(d['pts']) for *_, d in items]
    tree = STRtree(geoms)
    for c in comps[1:]:
        sub = G.subgraph(c)
        best = None
        for n in c:
            xy = np.asarray(G.nodes[n]['xy'])
            p = Point(xy)
            w = 3.0 if G.degree(n) == 1 else 1.0  # prefer extending a dead end
            for j in tree.query(p.buffer(max_link)):
                g = geoms[j]
                dist = g.distance(p)
                if dist > max_link or dist < 1e-6:
                    continue
                q = np.asarray(g.interpolate(g.project(p)).coords[0])
                link = LineString([xy, q])
                if barriers is not None and link.intersects(barriers):
                    continue
                score = dist / w
                if best is None or score < best[0]:
                    best = (score, n, j, q)
        if best is not None:
            _, n, j, q = best
            u, v, k, d = items[j]
            if not G.has_edge(u, v, k):
                continue
            tgt = None
            for cand in (u, v):
                if np.hypot(*(np.asarray(G.nodes[cand]['xy']) - q)) < 2.0:
                    tgt = cand
            if tgt is None:
                tgt = split_edge(G, u, v, k, q)
            e = next(iter(sub.edges(n, data=True)), None)
            attrs = dict(e[2]['attrs']) if e else {'src': 'auto_gap', 'type': 'urban_street'}
            attrs['src'] = 'auto_gap'
            G.add_edge(n, tgt, pts=resample(np.array([G.nodes[n]['xy'], G.nodes[tgt]['xy']]), 1.0), attrs=attrs, zone=e[2].get('zone') if e else None)
            fixed += 1
            # the new edges changed the main-network index
            items = [(a, b, kk, dd) for a, b, kk, dd in G.edges(keys=True, data=True) if (a in main or a in c) and dd['attrs']['type'] not in NO_TOUCH and not dd['attrs'].get('virtual')]
            main = main | set(c) | {tgt}
            geoms = [LineString(dd['pts']) for *_, dd in items]
            tree = STRtree(geoms)
        else:
            if not any(rank(d) == 3 for *_, d in sub.edges(data=True)) and sum(length(d['pts']) for *_, d in sub.edges(data=True)) < drop_len:
                G.remove_nodes_from(list(c))
                fixed += 1
    return fixed
