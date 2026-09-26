"""Railway centrelines from data/manual/roads/40_railways.json -> data/railways/railways.geojson"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from shapely.geometry import LineString, shape, Point
from shapely.ops import unary_union
from tools.lib.common import path, load_json, save_json, rnd
from tools.lib.manual_roads import all_traced
from tools.lib.trace import rdp
from tools.lib.geom import geojson_line, fc, polyline_length

wb = load_json(path('data/water/water_bodies.geojson'))
water = unary_union([shape(f['geometry']) for f in wb['features']])
feats = []
from tools.lib.engineer import design_alignments
for r, pts in design_alignments(all_traced()):
    if r['type'] != 'rail':
        continue
    simp = rdp(pts, 0.25)
    ls = LineString(simp)
    br = []
    x = ls.intersection(water)
    for g in ([x] if x.geom_type == 'LineString' else getattr(x, 'geoms', [])):
        if g.geom_type == 'LineString' and g.length > 1:
            a, b = sorted([ls.project(Point(g.coords[0])), ls.project(Point(g.coords[-1]))])
            br.append([rnd(max(0, a - 3)), rnd(min(ls.length, b + 3))])
    feats.append(geojson_line(simp, {'id': r['id'], 'name': r['name'], 'type': 'rail', 'gauge_mm': 1435,
                                     'tracks': r.get('tracks', 1), 'length_m': rnd(ls.length * 2.5, 1),
                                     'bridge_spans': br, 'width_m': 4.5 + 4.0 * (r.get('tracks', 1) - 1),
                                     'material': 'RAIL_BALLAST'}))
save_json(path('data/railways/railways.geojson'), fc(feats, 'railways'))
print('railways:', len(feats))
