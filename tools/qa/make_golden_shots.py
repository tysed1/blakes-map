"""Golden QA shot list (tools/qa/golden_shots.json): existing Blender cams + new locations placed from data.
Camera format = public/world/cams.json (three.js world: pos, dir, fov deg)."""
import json, math, os
import numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
H, W = 667, 2000
T = np.fromfile(os.path.join(R, 'data/terrain/height_graded_f32.bin'), np.float32).reshape(H, W)
def h(x, y):
    x = min(max(x, 0), W - 1.001); y = min(max(y, 0), H - 1.001); x0, y0 = int(x), int(y); fx, fy = x - x0, y - y0
    return float(T[y0, x0] * (1 - fx) * (1 - fy) + T[y0, x0 + 1] * fx * (1 - fy) + T[y0 + 1, x0] * (1 - fx) * fy + T[y0 + 1, x0 + 1] * fx * fy)
def w(x, y, agl=None, z=None):
    return [(x - 1000) * 2.5, (z if z is not None else h(x, y)) + (agl or 0), (y - 333.5) * 2.5]
def cam(p, t, fov=42.5):
    d = np.subtract(t, p); d = d / np.linalg.norm(d)
    return dict(pos=[round(v, 2) for v in p], dir=[round(float(v), 4) for v in d], fov=fov)
cams = json.load(open(os.path.join(R, 'public/world/cams.json')))
shots = {}
for k in ['CAM_Ref_Match', 'CAM_HollowRidge_Valley', 'CAM_Interchange_SR400', 'CAM_LaurelRiver_Bridges', 'CAM_US19_LaurelGap',
          'TC_road_driver', 'TC_hwy_low', 'TC_river_low', 'TC_field_ped', 'TC_forest_ped', 'TC_aerial_100', 'TC_high_1000']:
    shots[k] = cams[k]
shots['GS_HollowFalls'] = cam(w(1112, 280, 22), w(1128, 262, 4))
shots['GS_FallsCreekCascades'] = cam(w(1025, 84, 30), w(1043, 60, 4))
shots['GS_SouthForkGorge'] = cam(w(925, 515, 90), w(975, 565, 0))
rc = json.load(open(os.path.join(R, 'data/roads/rail_crossings.geojson')))['features'][0]['properties']
x, y = rc['at']; shots['GS_RailCrossing'] = cam(w(x - 14, y - 9, 8), w(x, y, 1))
walls = [f for f in json.load(open(os.path.join(R, 'data/roads/walls.geojson')))['features'] if f['properties'].get('kind') == 'rock_cut']
wl = sorted(walls, key=lambda f: -f['properties']['height_m'])[1]  # 2nd tallest: clear of the regraded Railside Lane
c = np.asarray(wl['geometry']['coordinates'], float)[:, :2]
a, b = c[0], c[-1]; d = (b - a) / max(np.linalg.norm(b - a), 1e-6); m = (a + b) / 2
p0 = m - d * 14; zb = wl['properties']['base_z_m']
shots['GS_RockCut'] = cam(w(p0[0], p0[1], z=zb + 1.6), w(m[0] + d[0] * 6, m[1] + d[1] * 6, z=zb + 3))
lu = np.fromfile(os.path.join(R, 'public/world/landuse_u8.bin'), np.uint8).reshape(H, W)
best = None
for yy in range(260, 440, 2):
    for xx in range(960, 1160, 2):
        if lu[yy, xx] in (4, 5) and (lu[yy - 3:yy + 4, xx - 3:xx + 4] == 2).sum() > 12:
            fy, fx = np.nonzero(lu[yy - 3:yy + 4, xx - 3:xx + 4] == 2); fx = fx.mean() - 3; fy = fy.mean() - 3
            best = (xx, yy, fx, fy); break
    if best: break
xx, yy, fx, fy = best; n = math.hypot(fx, fy) or 1
shots['GS_ForestEdge'] = cam(w(xx - fx / n * 12, yy - fy / n * 12, 1.7), w(xx + fx / n * 8, yy + fy / n * 8, 5))
shots['GS_BaldRidge'] = cam(w(812, 296, 28), w(1050, 330, 40))
shots['GS_US76_SR60'] = cam(w(1830, 215, 140), w(1872, 158, 0))
json.dump(shots, open(os.path.join(R, 'tools/qa/golden_shots.json'), 'w'), indent=1)
print(len(shots), 'shots')
