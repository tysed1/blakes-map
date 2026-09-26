"""Environment props: boulders, rocks, fallen logs, stumps, branches, roots, bark debris.

Placement comes from the ecosystem scatter (tools/pipeline/vegetation.py -> public/world/
props_f32.bin: x_px, y_px, z_m, scale, kind, rot): boulders on river banks / in rapids and as
talus under cliffs, logs and branches on the forest floor (more in moist hollows), creek
debris and exposed roots along banks, stumps at logged edges. Meshes are CC0 Poly Haven
models (assets/external/polyhaven, re-fetch with tools/assets/fetch_polyhaven.py); rocks are
re-tinted toward Appalachian grey-brown sandstone / gneiss with lichen.
Instanced with the same Geometry Nodes instancer as the trees (lib_trees.instancer).
"""
import bpy, os
import numpy as np
import lib_trees as LT

ROOT = LT.ROOT
PH = os.path.join(ROOT, 'assets/external/polyhaven')
H, W = 667, 2000
# kind -> [(asset id, object-name filter, base scale)]  (order must match vegetation.PROPS)
KINDS = [
    ('boulder_large', [('rock_moss_set_01', lambda n: 'rock0' in n, 1.0), ('namaqualand_boulder_02', lambda n: n.endswith('LOD2'), 1.1),
                       ('boulder_01', lambda n: n.endswith('LOD2'), 1.4)]),
    ('boulder', [('rock_moss_set_02', lambda n: 'rock' in n, 0.8), ('namaqualand_boulder_05', lambda n: n.endswith('LOD1'), 1.0)]),
    ('rock_small', [('rock_07', lambda n: n.endswith('LOD2'), 2.2), ('stone_01', lambda n: n.endswith('LOD2'), 3.5)]),
    ('log', [('dead_tree_trunk_02', lambda n: n.endswith('LOD2'), 1.3)]),
    ('log_mossy', [('dead_tree_trunk', lambda n: True, 1.6)]),
    ('stump', [('tree_stump_01', lambda n: True, 0.75), ('tree_stump_02', lambda n: True, 0.75)]),
    ('branches', [('dry_branches_medium_01', lambda n: True, 2.2)]),
    ('roots', [('root_cluster_01', lambda n: True, 0.8)]),
    ('bark_debris', [('bark_debris_01', lambda n: n.endswith('LOD2'), 2.0)]),
]
ROCK_KINDS = ('boulder_large', 'boulder', 'rock_small')


def _retint_rock(mat, tint=(0.95, 0.9, 0.82, 1), sat=0.55):
    """Push the (mossy European) rock albedo toward grey-brown Appalachian sandstone / gneiss."""
    if mat is None or not mat.use_nodes or mat.get('a3_tinted'):
        return
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None or not bsdf.inputs['Base Color'].is_linked:
        return
    src = bsdf.inputs['Base Color'].links[0].from_socket
    hs = nt.nodes.new('ShaderNodeHueSaturation'); hs.inputs['Saturation'].default_value = sat; hs.inputs['Value'].default_value = 1.0
    mul = nt.nodes.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs['Factor'].default_value = 1.0
    mul.inputs['B'].default_value = tint
    nt.links.new(src, hs.inputs['Color']); nt.links.new(hs.outputs['Color'], mul.inputs['A'])
    nt.links.new(mul.outputs['Result'], bsdf.inputs['Base Color'])
    mat['a3_tinted'] = True


def _decimate(o, target):
    """Reduce heavy scanned meshes (root cluster 225k, trunks 100k faces) to ~target faces."""
    n = len(o.data.polygons)
    if n <= target:
        return
    m = o.modifiers.new('dec', 'DECIMATE'); m.ratio = target / n
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
    o.modifiers.remove(m)
    old = o.data; o.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)


def load_protos(coll):
    """Returns (collection, list of objects in instance order, kind -> (start, count))."""
    pc = bpy.data.collections.new('PROP_prototypes'); coll.children.link(pc)
    objs, ranges = [], {}
    for kname, sources in KINDS:
        start = len(objs)
        for aid, keep, s in sources:
            p = os.path.join(PH, aid, f'{aid}_1k.blend')
            if not os.path.exists(p):
                print('  missing prop asset', aid)
                continue
            with bpy.data.libraries.load(p, link=False) as (df, dt):
                dt.objects = [n for n in df.objects if keep(n) and 'geonodes' not in n and 'geometry_nodes' not in n]
            for o in dt.objects:
                if o is None or o.type != 'MESH':
                    continue
                o.parent = None; o.location = (0, 0, 0); o.rotation_euler = (0, 0, 0)
                pc.objects.link(o)
                _decimate(o, 12000)
                pc.objects.unlink(o)
                o.scale = (s, s, s)
                # bake scale and sit the mesh on its base (z min -> slightly buried)
                me = o.data
                co = np.zeros(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3) * s
                co[:, :2] -= co[:, :2].mean(0)
                bury = 0.25 if kname in ROCK_KINDS else 0.05
                co[:, 2] -= co[:, 2].min() + bury * (co[:, 2].max() - co[:, 2].min())
                me.vertices.foreach_set('co', co.ravel()); me.update()
                o.scale = (1, 1, 1)
                o.name = f'P{len(objs):02d}_{kname}'
                pc.objects.link(o)
                objs.append(o)
                if kname in ROCK_KINDS:
                    for m in me.materials:
                        _retint_rock(m)
        ranges[kname] = (start, len(objs) - start)
    pc.hide_render = True; pc.hide_viewport = True
    return pc, objs, ranges


def build(coll, seed=9):
    """Instance all scattered props. Returns the instancer object (or None)."""
    fp = os.path.join(ROOT, 'public/world/props_f32.bin')
    if not os.path.exists(fp):
        print('  no props_f32.bin (run tools/pipeline/export_web.py)')
        return None
    v = np.fromfile(fp, '<f4').reshape(-1, 6)
    pc, objs, ranges = load_protos(coll)
    names = [k for k, _ in KINDS]
    rng = np.random.default_rng(seed)
    kind = v[:, 4].astype(int)
    idx = np.zeros(len(v), np.int32); keep = np.ones(len(v), bool)
    for k, kname in enumerate(names):
        sel = kind == k
        st, n = ranges.get(kname, (0, 0))
        if n == 0:
            keep[sel] = False; continue
        idx[sel] = st + rng.integers(0, n, sel.sum())
    v, idx = v[keep], idx[keep]
    hgt = np.fromfile(os.path.join(ROOT, 'data/terrain/height_graded_f32.bin'), np.float32).reshape(H, W)
    z = LT._bilinear(hgt, v[:, 0], v[:, 1])
    bx, by = (v[:, 0] - 1000.0) * 2.5, -(v[:, 1] - 333.5) * 2.5
    tilt = rng.normal(0, 0.06, (len(v), 2))
    tint = np.ones((len(v), 4), np.float32)
    ob = LT.instancer('PROP_points', coll, np.c_[bx, by, z - 0.05], idx, np.clip(v[:, 3], 0.4, 2.5), v[:, 5], tint, pc, tilt)
    print(f'  props: {len(v)} instances, {len(objs)} prototypes')
    return ob
