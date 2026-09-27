"""Export the lib_trees species (the exact Blender leaf-card trees) for the web viewer, with LODs.

    blender -b --python tools/blender/export_web_trees.py

Writes public/world/trees/:
  geo.bin   all species x LODs, concatenated per mesh:
              positions f32x3 (Y-up: x, z, -y), normals i8x4, uv u8x2, ao/lv u8x2, indices u16
  geo.json  species table (card, bark, family) + per-LOD offsets and material groups
(textures: tools/pipeline/web_tree_textures.py)

LODs rebuild each species with the same seed (lobes, limbs and cores are drawn before the
cards, so every LOD shares one silhouette):
  0  as in Blender (ico-2 cores, full card shells)          near  (< ~45 m at High)
  1  ico-1 cores, 33 % of the cards at 1.8x                 mid   (< ~150 m)
  2  ico-0 cores, sparse 3.2x cards, trunk only              far   (< ~360 m)
  3  the 4 largest ico-0 cores, no wood                      farther (< ~750 m)
  4  one ico-0 crown ellipsoid (20 tris): the web viewer only reads its extents and draws a
     camera-facing crown card (trees.ts billboard LOD)       horizon
"""
import bpy, bmesh, inspect, json, math, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_trees as LT

ROOT = LT.ROOT
OUT = os.path.join(ROOT, 'public/world/trees')
os.makedirs(OUT, exist_ok=True)
MATS = ['bark', 'leaf', 'core']

LODS = [
    dict(ico=2, dens=1.0, card=1.0, min_tube=0.0, sides=None),
    dict(ico=1, dens=0.33, card=1.8, min_tube=0.05, sides=4, core_scale=1.06, keep_lobes=24),
    dict(ico=0, dens=0.045, card=3.2, min_tube=0.25, sides=3, core_scale=1.2, keep_lobes=12),
    dict(ico=0, dens=0.0, card=1.0, min_tube=99.0, sides=3, keep_lobes=4, core_scale=1.32),
    dict(ellipsoid=True),   # web: crown extents for the shared billboard LOD (trees.ts)
]
_orig_blob = LT.blob

_orig_tube = LT.tube


def ico(sub):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=1.0)
    V = [v.co.copy() for v in bm.verts]
    F = [[v.index for v in f.verts] for f in bm.faces]
    bm.free()
    return V, F


def ellipsoid(ref):
    """Horizon LOD: one ico-0 crown ellipsoid fitted to the LOD-0 core volume (+ nothing else)."""
    V = np.asarray(ref.V, np.float32); M = np.asarray(ref.M)
    core = set(v for f, m in zip(ref.F, M) if m == 2 for v in f)
    if not core:  # snag: a thin grey spike
        core = set(range(len(V)))
    C = V[sorted(core)]
    lo, hi = C.min(0), C.max(0)
    c = (lo + hi) / 2; rx = max(hi[0] - lo[0], hi[1] - lo[1]) / 2 * 1.0; rz = (hi[2] - lo[2]) / 2 * 1.0
    g = LT.Geo()
    Vi, Fi = ico(0)
    for v in Vi:
        g.add_vert((c[0] + v.x * rx, c[1] + v.y * rx, c[2] + v.z * rz), v.normalized(), (0.5, 0.5), 0.45 + 0.35 * (v.z * 0.5 + 0.5), 0.5)
    for f in Fi:
        g.face(f, 2)
    return g


def build(i, lod, ref=None):
    name, fam, fn, kw, cardn, bark = LT.SPECIES[i]
    if lod.get('ellipsoid'):
        return ellipsoid(ref)
    LT._ICO = ico(lod['ico'])
    blobs = []

    def blob(g, c, rx, rz, rng, mat, *a, **k):
        v0, f0 = len(g.V), len(g.F)
        # cheaper LODs lose their card shells: inflate the cores to keep the crown's coverage
        f = lod.get('core_scale', 1.0)
        _orig_blob(g, c, rx * f, rz * f, rng, mat, *a, **k)
        blobs.append((rx * rx * rz, f0, len(g.F)))
    LT.blob = blob

    def tube(g, path, r0, r1, sides=6, mat=0, flare=0.0):
        if max(r0, r1) < lod['min_tube']:
            return
        return _orig_tube(g, path, r0, r1, min(sides, lod['sides']) if lod['sides'] else sides, mat, flare)
    LT.tube = tube
    kw2 = dict(kw)
    sig = inspect.signature(fn).parameters
    if 'density' in sig:
        kw2['density'] = kw.get('density', sig['density'].default) * lod['dens']
    if 'card_size' in sig:
        kw2['card_size'] = tuple(c * lod['card'] for c in kw.get('card_size', sig['card_size'].default))
    g = fn(np.random.default_rng(7 * 100 + i), **kw2)   # same seed as lib_trees.build_prototypes
    LT.tube = _orig_tube; LT.blob = _orig_blob
    if lod.get('keep_lobes') and len(blobs) > lod['keep_lobes']:
        drop = set()
        for _, f0, f1 in sorted(blobs, reverse=True)[lod['keep_lobes']:]:
            drop.update(range(f0, f1))
        keep = [k for k in range(len(g.F)) if k not in drop]
        g.F = [g.F[k] for k in keep]; g.M = [g.M[k] for k in keep]
    return g


def pack(g):
    V = np.asarray(g.V, np.float32)
    N = np.array([n if n is not None else (0, 0, 1) for n in g.N], np.float32)
    UV = np.asarray(g.UV, np.float32)
    AO = np.asarray(g.AO, np.float32); LV = np.asarray(g.LV, np.float32)
    tris = {0: [], 1: [], 2: []}
    for f, m in zip(g.F, g.M):
        for k in range(1, len(f) - 1):
            tris[m].append((f[0], f[k], f[k + 1]))
    # Blender Z-up -> web Y-up
    P = np.c_[V[:, 0], V[:, 2], -V[:, 1]]
    Nn = np.c_[N[:, 0], N[:, 2], -N[:, 1]]
    Nn /= np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-6)
    idx, groups, start = [], [], 0
    for m in (0, 2, 1):  # bark, core, leaf (opaque first)
        t = np.asarray(tris[m], np.uint32).reshape(-1, 3)
        if len(t):
            groups.append([start, t.size, m])
            idx.append(t.ravel()); start += t.size
    I = np.concatenate(idx) if idx else np.zeros(0, np.uint32)
    assert len(P) < 65536, len(P)
    return dict(P=P.astype(np.float32), N=np.clip(np.round(Nn * 127), -127, 127).astype(np.int8),
                UV=np.clip(np.round(UV * 255), 0, 255).astype(np.uint8),
                AL=np.clip(np.round(np.c_[AO, LV] * 255), 0, 255).astype(np.uint8),
                I=I.astype(np.uint16), groups=groups,
                r=float(np.max(np.linalg.norm(P[:, [0, 2]], axis=1))), h=float(P[:, 1].max()))


def main():
    blob = bytearray()
    table = []

    def put(a):
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(a.tobytes())
        return off
    for i, (name, fam, fn, kw, cardn, bark) in enumerate(LT.SPECIES):
        lods = []
        ref = None
        for li, lod in enumerate(LODS):
            g = build(i, lod, ref)
            if not g.F:  # e.g. snags: nothing survives the tube cut -> reuse the previous LOD
                g = prev
            ref = ref or g
            prev = g
            m = pack(g)
            n4 = np.zeros((len(m['N']), 4), np.int8); n4[:, :3] = m['N']
            lods.append(dict(vcount=len(m['P']), icount=len(m['I']), groups=m['groups'],
                             pos=put(m['P']), nrm=put(n4), uv=put(m['UV']), al=put(m['AL']), idx=put(m['I'])))
            if li == 0:
                r, h = m['r'], m['h']
            print(f'{name} lod{li}: {len(m["I"]) // 3} tris')
        table.append(dict(name=name, family=fam, card=cardn, bark=bark, radius=r, height=h, lods=lods))
    open(os.path.join(OUT, 'geo.bin'), 'wb').write(bytes(blob))
    json.dump(dict(species=table, lods=len(LODS), palettes=LT.PAL,
                   leaf_k=LT.LEAF_K), open(os.path.join(OUT, 'geo.json'), 'w'), indent=0)
    print('geo.bin', len(blob) / 1e6, 'MB')


if __name__ == '__main__':
    main()
