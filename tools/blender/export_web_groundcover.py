"""Export the A3 ground cover + props prototypes for the web viewer (src/components/world3d/groundcover.ts).

    blender -b --python tools/blender/export_web_groundcover.py
    python3 tools/pipeline/web_groundcover_textures.py      # atlas / fern textures (needs PIL + cv2)
    python3 tools/pipeline/web_groundcover_raster.py        # per-pixel layer weights

Small headless job: builds only the prototypes (no world). Writes public/world/groundcover/:
  geo.bin   every mesh LOD concatenated: positions f32x3 (Y-up: x, z, -y), normals i8x4,
            uv u16x2 (normalised; props are remapped into the atlas), colour u8x4 (linear vertex
            colour, forbs only; white elsewhere), indices u16
  geo.json  grass kinds (the lib_groundcover.build_protos clump specs, rendered as GPU blades),
            decor prototypes (lib_groundcover.forb + Poly Haven fern_02), props (lib_props.KINDS
            subset, decimated to 2 LODs) + atlas layout for web_groundcover_textures.py

Grass itself is not exported as meshes: groundcover.ts grows the same clumps procedurally on the
GPU (blade height / lean / width / root->tip colours / seed-head share from GRASS below, which
mirrors build_protos) because a camera-following blade field is far cheaper than instanced
55-blade clumps at open-world density.
"""
import bpy, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_groundcover as GC

ROOT = GC.ROOT
PH = GC.PH
OUT = os.path.join(ROOT, 'public/world/groundcover')
os.makedirs(OUT, exist_ok=True)

# lib_groundcover.build_protos specs: (height range, blade width, clump radius, lean range, root colour,
# tip colours, seed-head share) + relative blade density (Blender clumps/m2 x blades/clump, pasture = 1)
GRASS = [
    dict(name='pasture', h=(0.35, 0.7), w=0.011, lean=(0.15, 0.5), col0=(0.045, 0.065, 0.02),
         col1=[(0.26, 0.24, 0.075), (0.20, 0.22, 0.065), (0.30, 0.26, 0.085), (0.17, 0.20, 0.055)], seed=0.15, dens=1.0, scale=(0.8, 1.3)),
    dict(name='broomsedge', h=(0.55, 1.0), w=0.009, lean=(0.05, 0.25), col0=(0.12, 0.08, 0.03),
         col1=[(0.42, 0.2, 0.07), (0.38, 0.24, 0.08), (0.45, 0.26, 0.1)], seed=0.2, dens=0.75, scale=(0.8, 1.3)),
    dict(name='stubble', h=(0.08, 0.18), w=0.006, lean=(0.0, 0.2), col0=(0.12, 0.1, 0.04),
         col1=[(0.38, 0.3, 0.13), (0.3, 0.26, 0.1)], seed=0.0, dens=0.9, scale=(0.9, 1.3)),
    dict(name='short', h=(0.06, 0.16), w=0.008, lean=(0.1, 0.6), col0=(0.04, 0.06, 0.02),
         col1=[(0.12, 0.16, 0.05), (0.16, 0.17, 0.06), (0.1, 0.14, 0.04)], seed=0.0, dens=0.9, scale=(0.8, 1.4)),
    dict(name='rush', h=(0.5, 1.0), w=0.007, lean=(0.02, 0.15), col0=(0.03, 0.05, 0.02),
         col1=[(0.07, 0.1, 0.035), (0.1, 0.11, 0.04)], seed=0.0, dens=0.35, scale=(0.8, 1.3)),
    dict(name='forest_grass', h=(0.15, 0.35), w=0.008, lean=(0.3, 0.8), col0=(0.04, 0.05, 0.02),
         col1=[(0.1, 0.12, 0.04), (0.2, 0.15, 0.05)], seed=0.0, dens=0.08, scale=(0.8, 1.2)),
]

# decor: (name, layer, builder) ; layer = raster channel that drives it (flowers / weeds / fern)
DECOR_FORBS = [('goldenrod', 'flowers'), ('goldenrod_b', 'flowers'), ('aster', 'flowers'), ('aster_w', 'flowers'),
               ('qalace', 'flowers'), ('chicory', 'flowers'), ('weed', 'weeds'), ('weed_b', 'weeds')]
FERNS = ['fern_02_b', 'fern_02_c', 'fern_02_a']

# props (lib_props.KINDS order = eco.json 'props'); web subset of prototypes, decimated to 2 LODs
# (asset, object name, base scale, LOD0 tris, LOD1 tris)
PROP_KINDS = [
    ('boulder_large', 900, [('rock_moss_set_01', 'rock_moss_set_01_rock01', 1.0, 1800, 220), ('rock_moss_set_01', 'rock_moss_set_01_rock04', 1.0, 1800, 220),
                            ('boulder_01', 'boulder_01_LOD3', 1.4, 1800, 220)]),
    ('boulder', 600, [('rock_moss_set_02', 'rock_moss_set_02_rock07', 0.8, 1200, 160), ('rock_moss_set_02', 'rock_moss_set_02_rock11', 0.8, 1200, 160),
                      ('namaqualand_boulder_05', 'namaqualand_boulder_05_LOD2', 1.0, 1200, 160)]),
    ('rock_small', 160, [('rock_07', 'rock_07_LOD3', 2.2, 500, 80), ('stone_01', 'stone_01_LOD3', 3.5, 400, 60)]),
    ('log', 400, [('dead_tree_trunk_02', 'dead_tree_trunk_02_LOD3', 1.3, 1600, 200)]),
    ('log_mossy', 400, [('dead_tree_trunk', 'dead_tree_trunk', 1.6, 1400, 160)]),
    ('stump', 300, [('tree_stump_01', 'tree_stump_01', 0.75, 1400, 180), ('tree_stump_02', 'tree_stump_02', 0.75, 1400, 180)]),
    ('branches', 140, [('dry_branches_medium_01', 'dry_branches_medium_01_a', 2.2, 900, 150), ('dry_branches_medium_01', 'dry_branches_medium_01_b', 2.2, 700, 120)]),
    ('roots', 250, [('root_cluster_01', 'root_cluster_01', 0.8, 3000, 400)]),
    ('bark_debris', 90, [('bark_debris_01', 'bark_debris_01_a_LOD3', 2.0, 400, 60), ('bark_debris_01', 'bark_debris_01_c_LOD3', 2.0, 400, 60)]),
]
ROCK_KINDS = ('boulder_large', 'boulder', 'rock_small')
ATLAS_GRID = 4
ATLAS_PAD = 1 / 64  # fraction of a tile left as bleed border

scene = bpy.context.scene


def ph_objects(aid, names):
    p = os.path.join(PH, aid, f'{aid}_1k.blend')
    with bpy.data.libraries.load(p, link=False) as (df, dt):
        dt.objects = [n for n in df.objects if n in names]
    out = {}
    for o in dt.objects:
        if o is None or o.type != 'MESH':
            continue
        o.parent = None; o.location = (0, 0, 0); o.rotation_euler = (0, 0, 0); o.scale = (1, 1, 1)
        scene.collection.objects.link(o)
        out[o.name.split('.')[0]] = o
    return out


def decimated(o, target):
    """Evaluated copy of o reduced to ~target triangles (collapse decimation)."""
    ntri = sum(len(p.vertices) - 2 for p in o.data.polygons)
    m = None
    if ntri > target:
        m = o.modifiers.new('dec', 'DECIMATE'); m.ratio = max(0.001, target / ntri); m.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    if m:
        o.modifiers.remove(m)
    return me


def arrays(me, uv_rect=None, up_bias=0.0, base=None):
    """Indexed, per-corner-split arrays in web (Y-up) coordinates."""
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    loops = np.zeros(nt * 3, np.int32); me.loop_triangles.foreach_get('loops', loops)
    lv = np.zeros(len(me.loops), np.int32); me.loops.foreach_get('vertex_index', lv)
    co = np.zeros(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    cn = np.zeros(len(me.loops) * 3, np.float32); me.corner_normals.foreach_get('vector', cn); cn = cn.reshape(-1, 3)
    if me.uv_layers.active:
        uv = np.zeros(len(me.loops) * 2, np.float32); me.uv_layers.active.data.foreach_get('uv', uv); uv = uv.reshape(-1, 2)
    else:
        uv = np.zeros((len(me.loops), 2), np.float32)
    ca = me.color_attributes.get('col')
    if ca is not None:
        c = np.zeros(len(ca.data) * 4, np.float32); ca.data.foreach_get('color', c); c = c.reshape(-1, 4)
        col = c[lv] if ca.domain == 'POINT' else c
    else:
        col = np.ones((len(me.loops), 4), np.float32)
    if up_bias:
        cn = cn * (1 - up_bias) + np.array([0, 0, 1], np.float32) * up_bias
        cn /= np.maximum(np.linalg.norm(cn, axis=1, keepdims=True), 1e-6)
    uv = np.clip(uv, 0, 1)
    if uv_rect is not None:  # atlas tile: (u0, v0, size) in atlas uv space
        u0, v0, s = uv_rect
        uv = np.c_[u0 + uv[:, 0] * s, v0 + uv[:, 1] * s]
    # split vertices by (vertex, uv, normal, colour)
    key = np.c_[lv, np.round(uv * 4096), np.round(cn * 64), np.round(col[:, :3] * 255)].astype(np.int64)
    uk, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    P = co[lv[first]]; N = cn[first]; UV = uv[first]; C = col[first]
    I = inv[loops]
    P = np.c_[P[:, 0], P[:, 2], -P[:, 1]]; N = np.c_[N[:, 0], N[:, 2], -N[:, 1]]
    assert len(P) < 65536
    return P.astype(np.float32), N, UV, C, I.astype(np.uint16)


BLOB = bytearray()


def put(a):
    while len(BLOB) % 4:
        BLOB.append(0)
    off = len(BLOB); BLOB.extend(np.ascontiguousarray(a).tobytes()); return off


def pack(P, N, UV, C, I):
    n4 = np.zeros((len(N), 4), np.int8); n4[:, :3] = np.clip(np.round(N * 127), -127, 127)
    uv = np.clip(np.round(UV * 65535), 0, 65535).astype(np.uint16)
    # vertex colours are linear (Blender FLOAT_COLOR); stored sqrt-encoded for u8 precision in the darks
    c = np.clip(np.round(np.sqrt(np.clip(C, 0, 1)) * 255), 0, 255).astype(np.uint8)
    return dict(vcount=len(P), icount=len(I), pos=put(P), nrm=put(n4), uv=put(uv), col=put(c), idx=put(I))


def bounds(P):
    return float(np.max(np.linalg.norm(P[:, [0, 2]], axis=1))), float(P[:, 1].max()), float(P[:, 1].min())


def export_decor():
    rng = np.random.default_rng(21)
    out = []
    for name, layer in DECOR_FORBS:
        kind = name.split('_b')[0] if name.endswith('_b') else name
        m = GC.forb(rng, kind)
        me = m.mesh(f'D_{name}')
        o = bpy.data.objects.new(f'D_{name}', me); scene.collection.objects.link(o)
        lods = []
        for li, ratio in enumerate((1.0, 0.3)):
            ntri = sum(len(p.vertices) - 2 for p in me.polygons)
            mm = decimated(o, int(ntri * ratio)) if ratio < 1 else me
            P, N, UV, C, I = arrays(mm, up_bias=0.0 if li == 0 else 0.5)
            lods.append(pack(P, N, UV, C, I))
            if li == 0:
                r, h, _ = bounds(P)
            print(f'decor {name} lod{li}: {len(I) // 3} tris')
        out.append(dict(name=name, layer=layer, material='vcol', radius=r, height=h, lods=lods))
    ferns = ph_objects('fern_02', FERNS)
    for name in FERNS:
        o = ferns[name]
        lods = []
        for li, ratio in enumerate((1.0, 0.35)):
            ntri = sum(len(p.vertices) - 2 for p in o.data.polygons)
            mm = decimated(o, int(ntri * ratio))
            P, N, UV, C, I = arrays(mm)
            # sit on the ground: lowest point slightly buried
            P[:, 1] -= P[:, 1].min() + 0.02
            P[:, [0, 2]] -= P[:, [0, 2]].mean(0)
            lods.append(pack(P, N, UV, C, I))
            if li == 0:
                r, h, _ = bounds(P)
            print(f'decor {name} lod{li}: {len(I) // 3} tris')
        out.append(dict(name=name, layer='fern', material='fern', radius=r, height=h, lods=lods))
    return out


def export_props():
    tiles, kinds, protos = {}, [], []
    for kname, far, sources in PROP_KINDS:
        ids = []
        for aid, oname, s, t0, t1 in sources:
            if aid not in tiles:
                n = len(tiles); tiles[aid] = (n % ATLAS_GRID, n // ATLAS_GRID)
            col, row = tiles[aid]
            size = (1 - 2 * ATLAS_PAD) / ATLAS_GRID
            rect = ((col + ATLAS_PAD) / ATLAS_GRID, 1 - (row + 1 - ATLAS_PAD) / ATLAS_GRID, size)
            o = ph_objects(aid, [oname])[oname]
            # bake scale, centre, sit on base (lib_props.load_protos: rocks 25 % buried, wood 5 %)
            co = np.zeros(len(o.data.vertices) * 3, np.float32); o.data.vertices.foreach_get('co', co); co = co.reshape(-1, 3) * s
            co[:, :2] -= co[:, :2].mean(0)
            bury = 0.25 if kname in ROCK_KINDS else 0.05
            co[:, 2] -= co[:, 2].min() + bury * (co[:, 2].max() - co[:, 2].min())
            o.data.vertices.foreach_set('co', co.ravel()); o.data.update()
            lods = []
            for li, tgt in enumerate((t0, t1)):
                mm = decimated(o, tgt)
                P, N, UV, C, I = arrays(mm, uv_rect=rect)
                lods.append(pack(P, N, UV, C, I))
                if li == 0:
                    r, h, lo = bounds(P)
                print(f'prop {kname} {oname} lod{li}: {len(I) // 3} tris, {len(P)} verts')
            ids.append(len(protos))
            protos.append(dict(name=oname, kind=kname, asset=aid, radius=r, height=h, bottom=lo, lods=lods))
            bpy.data.objects.remove(o)
        kinds.append(dict(name=kname, far=far, protos=ids, rock=kname in ROCK_KINDS))
    atlas = dict(grid=ATLAS_GRID, pad=ATLAS_PAD, tiles={k: list(v) for k, v in tiles.items()}, rock=sorted({a for k, _, src in PROP_KINDS if k in ROCK_KINDS for a, *_ in src}))
    return kinds, protos, atlas


def convert_normals(tiles):
    """Poly Haven normal maps are EXR (system cv2 has no EXR codec): write 512 px PNGs to
    tools/.cache/web_groundcover/ for web_groundcover_textures.py."""
    cache = os.path.join(ROOT, 'tools/.cache/web_groundcover'); os.makedirs(cache, exist_ok=True)
    for aid in tiles:
        src = os.path.join(PH, aid, 'textures', f'{aid}_nor_gl_1k.exr')
        dst = os.path.join(cache, f'{aid}_nor_gl.png')
        if not os.path.exists(src) or os.path.exists(dst):
            continue
        img = bpy.data.images.load(src)
        img.colorspace_settings.name = 'Non-Color'
        img.scale(512, 512)
        img.filepath_raw = dst; img.file_format = 'PNG'
        img.save()
        bpy.data.images.remove(img)


def main():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    decor = export_decor()
    kinds, protos, atlas = export_props()
    convert_normals(atlas['tiles'])
    open(os.path.join(OUT, 'geo.bin'), 'wb').write(bytes(BLOB))
    json.dump(dict(grass=GRASS, decor=decor, prop_kinds=kinds, props=protos, atlas=atlas,
                   note='tools/blender/export_web_groundcover.py'), open(os.path.join(OUT, 'geo.json'), 'w'), indent=0)
    print('geo.bin', round(len(BLOB) / 1e6, 2), 'MB')


if __name__ == '__main__':
    main()
