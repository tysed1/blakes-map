"""Export the world dataset to compact web assets (public/world/).

  manifest.json         versions, extents, file list, road/landuse types, terrain decode
  roads.json            [{id,type,name,route,...,c:[x,y,z,...]}]  (px, px, m)
  nodes.json, bridges.json, rail.json, water.json, landuse.json, regions.json, settlements.json, landmarks.json, qa.json
  terrain_u16.bin       graded terrain, 2000x667 uint16 (h = min + v/65535*(max-min))
  water_u16.bin         water-surface level (nearest-filled) same encoding, + water mask in landuse
  landuse_u8.bin        class raster
  trees.bin             Float32 [x_px, y_px, z_m, scale, kind] per tree (kind 0 hardwood, 1 conifer)
  base_map.png, hillshade.png, landuse.png (overlays)
"""
import sys, os, json, shutil, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import cv2
from scipy import ndimage as ndi
from PIL import Image
from tools.lib.common import path, load_json, save_json, W, H, rnd

OUT = path('public/world')


def r2(v):
    return round(float(v), 2)


def flat(coords, nd=2):
    out = []
    for c in coords:
        out += [round(float(v), nd) for v in c]
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    world = load_json(path('data/world/world.json'))
    rt = load_json(path('data/roads/road_types.json'))
    lu = load_json(path('data/landuse/landuse.json'))
    tmeta = load_json(path('data/terrain/terrain.json'))
    # roads
    roads = []
    for f in load_json(path('data/roads/roads.geojson'))['features']:
        p = dict(f['properties'])
        p['c'] = flat(f['geometry']['coordinates'])
        roads.append(p)
    save_json(os.path.join(OUT, 'roads.json'), roads)
    nodes = [dict(f['properties'], c=[r2(v) for v in f['geometry']['coordinates']]) for f in load_json(path('data/roads/road_nodes.geojson'))['features']]
    save_json(os.path.join(OUT, 'nodes.json'), nodes)
    save_json(os.path.join(OUT, 'bridges.json'), [dict(f['properties'], c=flat(f['geometry']['coordinates'])) for f in load_json(path('data/roads/bridges.geojson'))['features']])
    save_json(os.path.join(OUT, 'rail.json'), [dict(f['properties'], c=flat(f['geometry']['coordinates'])) for f in load_json(path('data/railways/railways.geojson'))['features']])
    ww = [dict(f['properties'], c=flat(f['geometry']['coordinates'])) for f in load_json(path('data/water/waterways.geojson'))['features']]
    wb = [dict(f['properties'], rings=[flat(r) for r in (f['geometry']['coordinates'] if f['geometry']['type'] == 'Polygon' else [q for p in f['geometry']['coordinates'] for q in p])],
               polys=[[flat(r) for r in poly] for poly in ([f['geometry']['coordinates']] if f['geometry']['type'] == 'Polygon' else f['geometry']['coordinates'])])
          for f in load_json(path('data/water/water_bodies.geojson'))['features']]
    for b in wb:
        b.pop('rings')
    save_json(os.path.join(OUT, 'water.json'), {'lines': ww, 'bodies': wb})
    lup = []
    for f in load_json(path('data/landuse/landuse.geojson'))['features']:
        g = f['geometry']
        polys = [g['coordinates']] if g['type'] == 'Polygon' else g['coordinates']
        lup.append(dict(f['properties'], polys=[[flat(r, 1) for r in poly] for poly in polys]))
    save_json(os.path.join(OUT, 'landuse.json'), lup)
    for name, src in (('regions', 'data/world/regions.geojson'),):
        save_json(os.path.join(OUT, name + '.json'), [dict(f['properties'], c=flat(f['geometry']['coordinates'][0])) for f in load_json(path(src))['features']])
    save_json(os.path.join(OUT, 'settlements.json'), [dict(f['properties'], c=f['geometry']['coordinates']) for f in load_json(path('data/world/settlements.geojson'))['features']])
    save_json(os.path.join(OUT, 'landmarks.json'), [dict(f['properties'], c=f['geometry']['coordinates']) for f in load_json(path('data/poi/landmarks.geojson'))['features']])
    qa = {'roads': load_json(path('data/qa/road_report.json')) if os.path.exists(path('data/qa/road_report.json')) else None,
          'grades': load_json(path('data/qa/grade_report.json')) if os.path.exists(path('data/qa/grade_report.json')) else None,
          'world': load_json(path('data/qa/world_report.json')) if os.path.exists(path('data/qa/world_report.json')) else None}
    save_json(os.path.join(OUT, 'qa.json'), qa)
    # terrain + water level
    T = np.fromfile(path('data/terrain/height_graded_f32.bin'), np.float32).reshape(H, W)
    WL = np.fromfile(path('data/terrain/water_level_f32.bin'), np.float32).reshape(H, W)
    wet = ~np.isnan(WL)
    _, (iy, ix) = ndi.distance_transform_edt(~wet, return_indices=True)
    WLf = WL[iy, ix]
    lo, hi = float(np.floor(min(T.min(), np.nanmin(WL)) - 1)), float(np.ceil(max(T.max(), np.nanmax(WL)) + 1))
    enc = lambda A: ((A - lo) / (hi - lo) * 65535).round().clip(0, 65535).astype('<u2')
    enc(T).tofile(os.path.join(OUT, 'terrain_u16.bin'))
    enc(WLf).tofile(os.path.join(OUT, 'water_u16.bin'))
    cls = np.array(Image.open(path('data/landuse/landuse_classes.png')))
    cls.astype(np.uint8).tofile(os.path.join(OUT, 'landuse_u8.bin'))
    trees = scatter_trees(T, cls)
    trees.astype('<f4').tofile(os.path.join(OUT, 'trees.bin'))
    for f in ('backdrop_u16.bin', 'backdrop_water_u8.bin', 'backdrop.json'):
        shutil.copy(path('data/terrain/' + f), os.path.join(OUT, f))
    # images
    shutil.copy(path('assets/maps/processed/base_map.png'), os.path.join(OUT, 'base_map.png'))
    shutil.copy(path('data/terrain/albedo_2x.jpg'), os.path.join(OUT, 'albedo.jpg'))
    shutil.copy(path('assets/maps/debug/hillshade_graded.png'), os.path.join(OUT, 'hillshade.png'))
    pal = np.array([[int(c['color'][i:i + 2], 16) for i in (1, 3, 5)] for c in lu['classes']], np.uint8)
    Image.fromarray(pal[cls]).save(os.path.join(OUT, 'landuse.png'))
    if os.path.exists(path('assets/maps/debug/road_qa.png')):
        Image.open(path('assets/maps/debug/road_qa.png')).convert('RGB').save(os.path.join(OUT, 'road_qa.jpg'), quality=82)
    manifest = {
        'name': world['name'], 'image': {'w': W, 'h': H}, 'coords': world['coordinate_system'],
        'terrain': {'w': W, 'h': H, 'min_m': lo, 'max_m': hi, 'file': 'terrain_u16.bin', 'water': 'water_u16.bin'},
        'landuse': lu['classes'], 'road_types': rt['types'], 'junction_kinds': rt['junction_kinds'],
        'trees': {'file': 'trees.bin', 'count': int(len(trees)), 'stride': 5},
        'counts': {'roads': len(roads), 'nodes': len(nodes), 'waterways': len(ww)},
    }
    save_json(os.path.join(OUT, 'manifest.json'), manifest, indent=1)
    tot = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f'web export: {len(roads)} roads, {len(trees)} trees, {tot / 1e6:.1f} MB')


def scatter_trees(T, cls, cell=1.6, seed=7):
    """Ecosystem vegetation scatter (tools/pipeline/vegetation.py): writes vegetation_f32.bin,
    props_f32.bin and eco_u8.bin for Blender and returns the web tree array
    [x_px, y_px, z_m, scale, kind(0 hardwood / 1 conifer)] (trees only; shrubs / snags excluded)."""
    from tools.pipeline import vegetation
    return vegetation.run(T, cls)


if __name__ == '__main__':
    main()
