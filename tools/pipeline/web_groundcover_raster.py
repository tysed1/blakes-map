"""Ground-cover weight raster for the web viewer (src/components/world3d/groundcover.ts).

    python3 tools/pipeline/web_groundcover_raster.py

Re-derives, per source pixel, the same layer weights that the Blender ground-cover scatter
(tools/blender/lib_groundcover.scatter_modifier) computes from the terrain attributes
(lu_a / lu_b from build_world.landuse_weights, eco_a..eco_d from lib_materials.add_terrain_attributes),
but WITHOUT the fine road / rail / water exclusion masks: those stay as distance fields so the GPU
can evaluate them per blade with sub-pixel accuracy (bilinear filtering of a distance field).

Writes public/world/groundcover/:
  gc_u8.bin   16 channels x 667 x 2000 u8, channel-major (groups of 4 = one RGBA texture on the GPU)
    plane 0 (distances, 0.0125 m / step, clamped at 3 m): road edge, rail/bridge edge, water, verge (0..1)
    plane 1 (grass kinds, GPU): pasture, broomsedge, stubble, short
    plane 2 (grass kinds, GPU): rush (wet), forest grass, canopy-under, moisture
    plane 3 (CPU decor):        flowers, weeds, fern, fallow (goldenrod / broomsedge bias)
  raster.json  layout + channel names
"""
import json, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
W, H = 2000, 667
OUT = os.path.join(ROOT, 'public/world/groundcover'); os.makedirs(OUT, exist_ok=True)


def blur(a, r):
    """build_world._blur: separable box blur x3 (~gaussian)."""
    for _ in range(3):
        k = 2 * r + 1
        c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r)), mode='edge'), 1); a = (c[:, k:] - c[:, :-k]) / k
        c = np.cumsum(np.pad(a, ((r + 1, r), (0, 0)), mode='edge'), 0); a = (c[k:] - c[:-k]) / k
    return a


def S(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def main():
    lu = np.fromfile(os.path.join(ROOT, 'public/world/landuse_u8.bin'), np.uint8).reshape(H, W)
    meta = json.load(open(os.path.join(ROOT, 'public/world/eco.json')))
    ch = {n: i for i, n in enumerate(meta['channels'])}
    raw = np.fromfile(os.path.join(ROOT, 'public/world/eco_u8.bin'), np.uint8).reshape(H, W, -1)
    E = raw.astype(np.float32) / 255
    ft = raw[..., ch['field_type']]
    # lu_a / lu_b (build_world.landuse_weights)
    field = blur((lu == 4).astype(np.float32), 1)
    meadow = blur((lu == 5).astype(np.float32), 1)
    dev = blur(np.isin(lu, [7, 8, 9, 10, 11]).astype(np.float32), 1)
    wet = (lu == 1).astype(np.float32)
    bank = np.clip(blur(wet, 2) * 2.2, 0, 1) * (1 - wet)
    forest = blur(np.isin(lu, [2, 3]).astype(np.float32), 1)
    # eco (one-hot field types are blurred by the bilinear vertex sampling in Blender: ~ r=1 blur)
    oh = {k: blur((ft == v).astype(np.float32), 1) for k, v in meta['field_types'].items() if v}
    moist, disturbed, canopy, hedge = E[..., ch['moisture']], E[..., ch['disturbed']], E[..., ch['canopy']], E[..., ch['hedge']]
    rock = E[..., ch['rock_exposure']] if 'rock_exposure' in ch else np.zeros((H, W), np.float32)
    road_m = E[..., ch['road_edge_m/25.5']] * 25.5
    rail_m = E[..., ch['rail_bridge_edge_m/25.5']] * 25.5
    water_m = E[..., ch['water_dist_m/25.5']] * 25.5 if 'water_dist_m/25.5' in ch else np.full((H, W), 25.5, np.float32)

    under = S(canopy, 0.35, 0.75)
    op = (1 - under) * (1 - S(rock, 0.35, 0.75))          # open ground (x offroad masks on the GPU)
    verge = 1 - S(road_m, 5.0, 9.0)                       # roadside verge band (x offroad_tall on the GPU)
    pasture, hay, plowed, fallow, lawn = oh['pasture'], oh['hay'], oh['plowed'], oh['fallow'], oh['lawn']

    tall = np.maximum.reduce([pasture, fallow * 0.6, meadow * 0.8]) * op
    tall = np.maximum(tall, np.maximum(bank * 0.9, disturbed * 0.55) * op * (1 - dev))
    tall = np.maximum.reduce([tall, verge * 0.8 * (1 - S(rock, 0.35, 0.75)), hedge * 0.24])
    # autumn hay fields are aftermath: green regrowth blades through the cut stubble (E3), lawns replace it
    tall = np.maximum(tall, hay * 0.6 * op)
    w_pasture = tall * (1 - 0.75 * lawn)
    w_broom = fallow * op
    w_stubble = hay * op
    w_short = np.maximum(np.maximum(lawn, dev * 0.7) * op, plowed * 0.12 * op)
    w_rush = np.maximum(bank, S(moist, 0.7, 0.95)) * (1 - S(rock, 0.35, 0.75))
    w_forest = forest * np.clip(1.1 - canopy, 0, 1)
    w_fern = under * np.clip(0.35 + moist, 0, 1)
    w_flowers = np.maximum.reduce([fallow * 0.8, pasture * 0.25, hay * 0.15, verge, hedge]) * op
    w_weeds = np.maximum.reduce([verge, disturbed, pasture * 0.3]) * op

    # weights are quantised to 32 levels (x 8.2) and small values zeroed: 4x smaller gzip, invisible on screen
    q = lambda a: (np.round(np.where(a < 0.03, 0, np.clip(a, 0, 1)) * 31) * (255 / 31) + 0.5).astype(np.uint8)
    qd = lambda d: (np.clip(d, 0, 3.0) * 80 + 0.5).astype(np.uint8)  # only the first ~3 m matter
    planes = [
        np.stack([qd(road_m), qd(rail_m), qd(water_m), q(verge)], -1),
        np.stack([q(w_pasture), q(w_broom), q(w_stubble), q(w_short)], -1),
        np.stack([q(w_rush), q(w_forest), q(under), q(moist)], -1),
        np.stack([q(w_flowers), q(w_weeds), q(w_fern), q(fallow)], -1),
    ]
    out = np.ascontiguousarray(np.stack(planes).transpose(0, 3, 1, 2))  # channel-major: compresses ~25 % better
    out.tofile(os.path.join(OUT, 'gc_u8.bin'))
    json.dump({'w': W, 'h': H, 'planes': 4, 'layout': 'channel-major [plane][rgba][y][x]', 'dist_scale_m': 1 / 80,
               'channels': [['road_m', 'rail_m', 'water_m', 'verge'], ['pasture', 'broomsedge', 'stubble', 'short'],
                            ['rush', 'forest_grass', 'under', 'moisture'], ['flowers', 'weeds', 'fern', 'fallow']]},
              open(os.path.join(OUT, 'raster.json'), 'w'), indent=1)
    import gzip
    print('gc_u8.bin', out.nbytes / 1e6, 'MB raw,', len(gzip.compress(out.tobytes(), 6)) / 1e6, 'MB gz')
    for i, p in enumerate(planes):
        print(i, [round(float((p[..., c] > 20).mean()), 3) for c in range(4)])


if __name__ == '__main__':
    main()
