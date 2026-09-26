"""Road-network engineering validator.

Checks (see docs/WORLD_DESIGN_PRINCIPLES.md):
  components     network pieces not connected to the main network
  near_miss      dead end within 6 px of another road it does not join
  dead_end       dead ends without a justification (type/zone/length rules)
  sharp_angle    junction legs meeting at < 25 deg
  short_edge     junction-to-junction edges < 3 px (junction clusters)
  duplicate      parallel roads within 3.5 px for > 60 % of the shorter length
  ramp_conflict  surface road touching a ramp away from a ramp terminal
  radius         horizontal curve radius below the minimum for the road type
  water          road over water without a bridge/culvert span
Writes data/qa/road_report.json and assets/maps/debug/road_qa.png
"""
import sys, os, math, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point, shape
from shapely.strtree import STRtree
from shapely.ops import unary_union
from PIL import Image, ImageDraw
from tools.lib.common import path, load_json, save_json, W, H
from tools.lib.trace import resample

MIN_RADIUS_M = {'freeway': 350, 'highway': 150, 'ramp': 40, 'arterial': 60, 'main_street': 40, 'collector': 35,
                'urban_street': 12, 'residential': 12, 'rural': 20, 'gravel': 12, 'dirt': 8, 'driveway': 5}
JUSTIFIED_DEAD_END = {'driveway', 'dirt', 'gravel'}


def radius_profile(pts):
    P = resample(np.asarray(pts), 2.0)
    if len(P) < 7:
        return []
    out = []
    for i in range(3, len(P) - 3):
        a, b, c = P[i - 3], P[i], P[i + 3]
        ab, bc, ca = np.hypot(*(b - a)), np.hypot(*(c - b)), np.hypot(*(a - c))
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        R = ab * bc * ca / max(2 * area2, 1e-9)
        out.append((R * 2.5, b))
    return out


def main():
    roads = load_json(path('data/roads/roads.geojson'))['features']
    nodes = {f['properties']['id']: f for f in load_json(path('data/roads/road_nodes.geojson'))['features']}
    water = unary_union([shape(f['geometry']) for f in load_json(path('data/water/water_bodies.geojson'))['features']])
    G = nx.MultiGraph()
    geoms = []
    for i, f in enumerate(roads):
        p = f['properties']
        G.add_edge(p['from'], p['to'], idx=i)
        geoms.append(LineString(f['geometry']['coordinates']))
    tree = STRtree(geoms)
    issues = []

    def add(kind, xy, msg, sev='error', ref=None):
        issues.append({'kind': kind, 'severity': sev, 'at': [round(xy[0], 1), round(xy[1], 1)], 'msg': msg, 'ref': ref})

    comps = sorted(nx.connected_components(G), key=lambda c: -sum(geoms[d['idx']].length for _, _, d in G.subgraph(c).edges(data=True)))
    for c in comps[1:]:
        L = sum(geoms[d['idx']].length for _, _, d in G.subgraph(c).edges(data=True))
        n = next(iter(c))
        add('components', nodes[n]['geometry']['coordinates'], f'isolated network piece ({L * 2.5:.0f} m, {len(c)} nodes)', 'error', n)
    for n, nf in nodes.items():
        if n not in G:
            continue
        deg = G.degree(n)
        xy = nf['geometry']['coordinates']
        es = [roads[d['idx']] for _, _, d in G.edges(n, data=True)]
        if deg == 1:
            f = es[0]
            t = f['properties']['type']
            if t == 'ramp':
                continue  # ramp gore on the freeway edge
            p = Point(xy)
            own = next(d['idx'] for _, _, d in G.edges(n, data=True))
            near = [j for j in tree.query(p.buffer(6)) if j != own and geoms[j].distance(p) < 6 and roads[j]['properties']['type'] not in ('freeway', 'ramp')]
            near = [j for j in near if not LineString([xy, geoms[j].interpolate(geoms[j].project(p)).coords[0]]).intersects(water)]
            if near:
                add('near_miss', xy, f"{f['properties']['id']} ({t}) ends {min(geoms[j].distance(p) for j in near) * 2.5:.1f} m from {roads[near[0]]['properties']['id']}", 'error', f['properties']['id'])
            elif xy[0] < 1.6 or xy[0] > W - 1.6 or xy[1] < 1.6 or xy[1] > H - 1.6:
                pass  # continues off-map
            elif t not in JUSTIFIED_DEAD_END and not (t in ('residential', 'urban_street') and f['properties']['length_m'] > 15):
                add('dead_end', xy, f"{f['properties']['id']} ({t}, {f['properties']['length_m']:.0f} m) ends without reason", 'warn', f['properties']['id'])
        if deg >= 3:
            bs = []
            for f in es:
                c = np.asarray(f['geometry']['coordinates'])
                s = resample(c if np.hypot(*(c[0] - xy)) < 0.1 else c[::-1], 1.0)
                q = s[min(len(s) - 1, 6)]
                bs.append(math.degrees(math.atan2(q[1] - xy[1], q[0] - xy[0])) % 360)
            bs.sort()
            gaps = [(bs[(i + 1) % len(bs)] - bs[i]) % 360 for i in range(len(bs))]
            types = {f['properties']['type'] for f in es}
            if min(gaps) < 25 and not (types & {'ramp'}) and types != {'freeway'}:
                add('sharp_angle', xy, f'legs meet at {min(gaps):.0f} deg ({", ".join(sorted(types))})', 'error', n)
    for i, f in enumerate(roads):
        p = f['properties']
        g = geoms[i]
        a, b = G.degree(p['from']), G.degree(p['to'])
        if g.length < 3 and a >= 3 and b >= 3:
            add('short_edge', g.interpolate(0.5, normalized=True).coords[0], f"{p['id']} only {g.length * 2.5:.1f} m between junctions", 'warn', p['id'])
        for j in tree.query(g.buffer(3.5)):
            if j <= i:
                continue
            h = geoms[j]
            if roads[j]['properties']['from'] in (p['from'], p['to']) and roads[j]['properties']['to'] in (p['from'], p['to']) and False:
                continue
            short = min(g.length, h.length)
            if short < 6:
                continue
            ov = g.intersection(h.buffer(3.5)).length
            ov2 = h.intersection(g.buffer(3.5)).length
            if min(ov, ov2) > 0.6 * short:
                add('duplicate', g.interpolate(0.5, normalized=True).coords[0], f"{p['id']} runs on top of {roads[j]['properties']['id']}", 'error', p['id'])
        if p['type'] != 'ramp' and p['type'] != 'freeway':
            for j in tree.query(g):
                q = roads[j]['properties']
                if q['type'] == 'ramp' and g.intersects(geoms[j]):
                    x = g.intersection(geoms[j])
                    pts = [x] if x.geom_type == 'Point' else list(getattr(x, 'geoms', []))
                    for pt in pts:
                        if pt.geom_type != 'Point':
                            continue
                        ends = [Point(geoms[j].coords[0]), Point(geoms[j].coords[-1])]
                        if min(e.distance(pt) for e in ends) > 1.0:
                            add('ramp_conflict', pt.coords[0], f"{p['id']} crosses ramp {q['id']} at grade", 'error', p['id'])
        mr = MIN_RADIUS_M.get(p['type'], 10)
        prof = radius_profile(f['geometry']['coordinates'])
        L = g.length
        for k, (R, at) in enumerate(prof):
            # ignore the first/last 6 m (curb returns at junctions)
            if R < mr * 0.85 and 3 < k < len(prof) - 4:
                add('radius', at, f"{p['id']} ({p['type']}) curve radius {R:.0f} m < {mr} m", 'warn', p['id'])
                break
        if g.intersects(water):
            wet = g.intersection(water).length
            spans = sum(b - a for a, b in p['bridge_spans'])
            if wet > 1.0 and spans < wet * 0.8:
                add('water', g.interpolate(0.5, normalized=True).coords[0], f"{p['id']} crosses {wet * 2.5:.0f} m of water without a span", 'error', p['id'])
    from collections import Counter
    c = Counter((i['kind'], i['severity']) for i in issues)
    rep = {'summary': {f'{k}:{s}': v for (k, s), v in sorted(c.items())}, 'components': len(comps),
           'edges': len(roads), 'nodes': len(nodes), 'issues': issues}
    os.makedirs(path('data/qa'), exist_ok=True)
    save_json(path('data/qa/road_report.json'), rep, indent=1)
    draw(roads, issues)
    print(json.dumps(rep['summary'], indent=0), 'components', len(comps))


COLORS = {'components': (255, 0, 255), 'near_miss': (255, 0, 0), 'dead_end': (255, 160, 0), 'sharp_angle': (255, 255, 0),
          'short_edge': (0, 255, 255), 'duplicate': (255, 80, 160), 'ramp_conflict': (255, 0, 0), 'radius': (160, 120, 255), 'water': (0, 120, 255)}


def draw(roads, issues, scale=2):
    im = Image.open(path('assets/maps/processed/base_map.png')).convert('RGB').resize((W * scale, H * scale))
    im = Image.fromarray((np.asarray(im) * 0.45).astype(np.uint8))
    d = ImageDraw.Draw(im)
    types = load_json(path('data/roads/road_types.json'))['types']
    for f in roads:
        c = types[f['properties']['type']]['color'].lstrip('#')
        col = tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
        d.line([(x * scale, y * scale) for x, y in f['geometry']['coordinates']], fill=col, width=max(1, int(types[f['properties']['type']]['width_m'] / 2.5 * scale * 0.4)))
    for i in issues:
        x, y = i['at']
        r = 6 if i['severity'] == 'error' else 4
        d.ellipse([x * scale - r, y * scale - r, x * scale + r, y * scale + r], outline=COLORS[i['kind']], width=2)
    os.makedirs(path('assets/maps/debug'), exist_ok=True)
    im.save(path('assets/maps/debug/road_qa.png'))


if __name__ == '__main__':
    main()
