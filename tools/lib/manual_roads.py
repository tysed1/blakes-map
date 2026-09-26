"""Load hand-authored road traces (data/manual/roads/*.json) and snap them to
the base map with the livewire tracer. Results are cached per road keyed by a
hash of its definition, so editing one road only re-traces that road."""
import glob, hashlib, json, os
import numpy as np
from .common import path, load_json, save_json
from .trace import trace, center_trace

CACHE = path('tools/.cache/traces')
os.makedirs(CACHE, exist_ok=True)

KIND_BY_TYPE = {
    'freeway': 'highway', 'highway': 'highway', 'ramp': 'major', 'arterial': 'major', 'main_street': 'major',
    'collector': 'major', 'urban_street': 'local', 'residential': 'local', 'rural': 'local', 'gravel': 'local',
    'dirt': 'local', 'driveway': 'local', 'rail': 'major', 'parking': 'local',
}
SIGMA_BY_TYPE = {'freeway': 9.0, 'highway': 8.0, 'ramp': 5.0, 'arterial': 5.0, 'main_street': 3.0, 'collector': 3.0,
                 'urban_street': 2.0, 'residential': 2.0, 'rural': 2.5, 'gravel': 2.5, 'dirt': 2.0, 'driveway': 1.5,
                 'rail': 6.0, 'parking': 1.5}


CENTER_TYPES = {'freeway', 'highway', 'ramp', 'arterial', 'main_street', 'rail', 'collector', 'urban_street', 'residential', 'rural', 'gravel', 'dirt', 'driveway'}
WIDTH_PX = {'freeway': 9.0, 'highway': 8.0, 'ramp': 4.0, 'arterial': 5.0, 'main_street': 6.0, 'rail': 3.0, 'collector': 3.6,
            'urban_street': 4.0, 'residential': 3.0, 'rural': 2.6, 'gravel': 2.2, 'dirt': 1.8, 'driveway': 1.6}
SMOOTH = {'freeway': 10.0, 'highway': 10.0, 'ramp': 6.0, 'arterial': 8.0, 'main_street': 8.0, 'rail': 12.0}


def load_defs(pattern='data/manual/roads/*.json'):
    defs = []
    for f in sorted(glob.glob(path(pattern))):
        d = load_json(f)
        for r in d.get('roads', []):
            r['_file'] = os.path.basename(f)
            defs.append(r)
    ids = [r['id'] for r in defs]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f'duplicate road ids: {sorted(dup)}')
    return defs


def traced(r):
    kind = r.get('kind') or KIND_BY_TYPE.get(r['type'], 'local')
    sigma = r.get('smooth', SIGMA_BY_TYPE.get(r['type'], 2.5))
    snap = r.get('snap', True)
    mode = r.get('mode') or ('center' if r['type'] in CENTER_TYPES else 'livewire')
    width = r.get('width_px', WIDTH_PX.get(r['type'], 5.0))
    smooth = r.get('smooth', SMOOTH.get(r['type'], 5.0))
    key = hashlib.sha1(json.dumps([r['wp'], kind, sigma, snap, mode, width, smooth, 7], sort_keys=True).encode()).hexdigest()[:12]
    cp = os.path.join(CACHE, f"{r['id']}_{key}.json")
    if os.path.exists(cp):
        return np.array(load_json(cp))
    if mode == 'center':
        pts = center_trace(r['wp'], width=width, smooth=smooth)
    elif mode == 'spline':
        from .trace import catmull_rom
        pts = catmull_rom(r['wp'], 1.0)
    else:
        pts = trace(r['wp'], kind=kind, sigma=sigma, snap=snap, margin=r.get('margin', 14))
    save_json(cp, [[round(float(x), 3), round(float(y), 3)] for x, y in pts])
    return pts


def all_traced(pattern='data/manual/roads/*.json'):
    return [(r, traced(r)) for r in load_defs(pattern)]
