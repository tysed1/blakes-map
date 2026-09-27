"""Export the rock kit (tools/blender/lib_rocks.py) for the web viewer (E2).

    blender -b --factory-startup --python tools/blender/export_web_rocks.py

Writes public/world/rocks/:
  geo.bin   per kind x variant x LOD: positions f32x3 (Y-up: x, z, -y), normals i8x4, ao u8, indices u16
  geo.json  kinds (lib_rocks.KINDS order), per-variant LOD offsets / counts
Normals are split at hard edges (> 38 deg: joint faces stay crisp, weathered faces stay soft).
LODs: 0 full (~350-500 tris), 1 decimated to ~30 %, 2 to ~10 % (a silhouette-keeping lump).
Placement: tools/pipeline/rocks.py (public/world/rocks_f32.bin); shading: src/components/world3d/rocks.ts
(triplanar public/world/terrain/detail_rock.webp, the terrain's own rock, so crags and rock terrain match).
"""
import bpy, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_rocks as LR

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'public/world/rocks')
LOD_RATIO = {'cliff_block': (1.0, 0.3, 0.1), 'crag': (1.0, 0.3, 0.1), 'boulder': (0.8, 0.25, 0.08), 'scree': (0.45, 0.15, 0.15)}
HARD = math.radians(38)


def mesh_arrays(ob):
    me = ob.data
    me.calc_loop_triangles()
    if hasattr(me, 'set_sharp_from_angle'):
        me.set_sharp_from_angle(angle=HARD)
    cn = np.array([c.vector[:] for c in me.corner_normals], np.float32) if hasattr(me, 'corner_normals') else None
    ao_attr = me.color_attributes.get('ao')
    P, N, A, I, key = [], [], [], [], {}
    for t in me.loop_triangles:
        tri = []
        for li, vi in zip(t.loops, t.vertices):
            n = cn[li] if cn is not None else np.array(me.vertices[vi].normal[:], np.float32)
            k = (vi, tuple(np.round(n, 2)))
            if k not in key:
                key[k] = len(P)
                P.append(me.vertices[vi].co[:]); N.append(n)
                A.append(ao_attr.data[vi].color[0] if ao_attr else 1.0)
            tri.append(key[k])
        I.append(tri)
    return np.array(P, np.float32), np.array(N, np.float32), np.array(A, np.float32), np.array(I, np.uint32)


def main():
    os.makedirs(OUT, exist_ok=True)
    bpy.ops.wm.read_homefile(use_empty=True)
    coll = bpy.context.scene.collection
    blob = bytearray()

    def put(a):
        while len(blob) % 4:
            blob.append(0)
        off = len(blob); blob.extend(a.tobytes()); return off
    kinds = []
    for kind in LR.KINDS:
        variants = []
        for v in range(LR.VARIANTS):
            V, F, ao = LR.build_rock(kind, v)
            lods = []
            for li, ratio in enumerate(LOD_RATIO[kind]):
                ob = LR.to_object(f'{kind}_{v}_{li}', V, F, ao, coll)
                if ratio < 1.0:
                    m = ob.modifiers.new('dec', 'DECIMATE'); m.ratio = ratio
                    bpy.context.view_layer.objects.active = ob
                    dg = bpy.context.evaluated_depsgraph_get()
                    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
                    ob.modifiers.clear(); ob.data = me
                for p in ob.data.polygons:
                    p.use_smooth = True
                P, N, A, I = mesh_arrays(ob)
                Py = np.c_[P[:, 0], P[:, 2], -P[:, 1]].astype(np.float32)
                Ny = np.c_[N[:, 0], N[:, 2], -N[:, 1]]
                Ny /= np.maximum(np.linalg.norm(Ny, axis=1, keepdims=True), 1e-6)
                n4 = np.zeros((len(Ny), 4), np.int8); n4[:, :3] = np.clip(np.round(Ny * 127), -127, 127)
                lods.append(dict(vcount=len(Py), icount=int(I.size), pos=put(Py), nrm=put(n4),
                                 ao=put(np.clip(np.round(A * 255), 0, 255).astype(np.uint8)), idx=put(I.astype(np.uint16).ravel())))
                bpy.data.objects.remove(ob)
                if li == 0:
                    rad = float(np.max(np.linalg.norm(Py[:, [0, 2]], axis=1))); hgt = float(Py[:, 1].max())
            variants.append(dict(lods=lods, radius=rad, height=hgt))
            print(kind, v, [l['icount'] // 3 for l in lods])
        kinds.append(dict(name=kind, dims=LR.DIMS[kind], variants=variants))
    open(os.path.join(OUT, 'geo.bin'), 'wb').write(bytes(blob))
    json.dump(dict(kinds=kinds, variants=LR.VARIANTS, lods=3), open(os.path.join(OUT, 'geo.json'), 'w'), indent=0)
    print('rocks geo.bin', len(blob) / 1e6, 'MB')


if __name__ == '__main__':
    main()
