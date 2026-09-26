"""QA render: final road network (coloured by type) + junctions over the base map.
usage: python3 tools/qa/render_roads.py x0 y0 x1 y1 scale out.png [--ids] [--nodes] [--water]"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tools.lib.viz import render
from tools.lib.common import load_json, path


def hexrgba(h, a=255):
    h = h.lstrip('#')
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


def main(argv):
    box = tuple(map(int, argv[:4])); s = int(argv[4]); out = argv[5]
    x0, y0, x1, y1 = box
    types = load_json(path('data/roads/road_types.json'))['types']
    roads = load_json(path('data/roads/roads.geojson'))['features']
    lines, pts, polys = [], [], []
    if '--water' in argv:
        for f in load_json(path('data/water/water_bodies.geojson'))['features']:
            g = f['geometry']
            rings = g['coordinates'] if g['type'] == 'Polygon' else [r for p in g['coordinates'] for r in p]
            for r in rings:
                polys.append((r, (0, 200, 255, 200), (0, 120, 255, 60)))
    for f in sorted(roads, key=lambda f: types[f['properties']['type']]['z']):
        c = f['geometry']['coordinates']
        xs = [p[0] for p in c]; ys = [p[1] for p in c]
        if max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1:
            continue
        p = f['properties']
        w = max(1, int(round(types[p['type']]['width_m'] / 2.5 * s * 0.35)))
        lines.append((c, hexrgba(types[p['type']]['color']), p['id'] if '--ids' in argv else None, w))
        for b0, b1 in p['bridge_spans']:
            pass
    if '--nodes' in argv:
        col = {'end': (255, 60, 60, 255), 'T': (255, 255, 0, 255), 'cross': (0, 255, 0, 255), 'fork': (0, 255, 255, 255),
               'angled': (255, 140, 0, 255), 'merge': (255, 0, 255, 255), 'city': (255, 255, 255, 255), 'multi': (0, 0, 255, 255)}
        for f in load_json(path('data/roads/road_nodes.geojson'))['features']:
            x, y = f['geometry']['coordinates']
            k = f['properties']['kind']
            if x0 <= x <= x1 and y0 <= y <= y1 and k in col:
                pts.append((x, y, col[k]))
    render(box, s, lines=lines, points=pts, polys=polys, out=out, grid=50 if s < 4 else 10, dim=0.6, width=2)


if __name__ == '__main__':
    main(sys.argv[1:])
