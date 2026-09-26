"""Render traced manual roads (+ optional auto graph) over a crop for review.
usage: python3 tools/lib/review.py x0 y0 x1 y1 scale out.png [--auto] [--labels]"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tools.lib.viz import render
from tools.lib.manual_roads import all_traced
from tools.lib.common import load_json, path

COL = {'freeway': (255, 60, 60, 255), 'highway': (255, 120, 0, 255), 'ramp': (255, 200, 0, 255), 'arterial': (255, 255, 0, 255),
       'main_street': (255, 0, 255, 255), 'collector': (0, 255, 120, 255), 'urban_street': (0, 220, 255, 255), 'residential': (120, 200, 255, 255),
       'rural': (255, 255, 255, 255), 'gravel': (200, 200, 160, 255), 'dirt': (220, 150, 90, 255), 'driveway': (180, 120, 255, 255), 'rail': (60, 30, 0, 255)}
a = sys.argv[1:]
box = tuple(map(int, a[:4])); s = int(a[4]); out = a[5]
lines = []
if '--auto' in a:
    g = load_json(path('tools/.cache/roads_auto.json'))
    lines += [(e['pts'], (255, 0, 200, 160), None, 1) for e in g['edges']]
x0, y0, x1, y1 = box
for r, pts in all_traced():
    if pts[:, 0].max() < x0 or pts[:, 0].min() > x1 or pts[:, 1].max() < y0 or pts[:, 1].min() > y1:
        continue
    lines.append((pts.tolist(), COL.get(r['type'], (255, 255, 255, 255)), r['id'] if '--labels' in a else None, 2))
render(box, s, lines=lines, out=out, grid=10 if s >= 4 else 50, dim=0.8, width=2)
