"""Rock kit (E2): procedural Blue Ridge crags, cliff-band blocks and scree for Blender + the web viewer.

Southern Appalachian rock is metamorphic (gneiss / schist / metasandstone): blocky, jointed, with
layered ledges that step down the slope, rounded by weathering at the arrises, lichen-grey on top.
Every piece is built from a cube / icosphere that is
  * cut by 3-6 random joint planes (flat faces + sharp edges = blocky, not blobby),
  * stepped by 2-4 horizontal bedding ledges (the cliff-band look),
  * displaced by low-amplitude multi-octave noise (weathering), then smoothed at the arrises,
and written with a per-vertex cavity/exposure AO (concave = dark, upward = light).

Kinds (order = KINDS, shared with tools/pipeline/rocks.py and src/components/world3d/rocks.ts):
  0 cliff_block   4-8 m wide ledge block, taller than deep; cliff bands + road rock cuts
  1 crag          3-5 m blocky outcrop / tor on crests and spur noses
  2 boulder       1-2.5 m detached block (talus apron, scree fan toe)
  3 scree         0.3-0.8 m angular stones in fans below the crags
Each kind has VARIANTS variants x 3 LODs (web: tools/blender/export_web_rocks.py).
"""
import math
import bmesh
import numpy as np
from mathutils import Vector, noise

KINDS = ['cliff_block', 'crag', 'boulder', 'scree']
VARIANTS = 4
# base dimensions (x = along the face / contour, y = depth into the hill, z = up) in metres
DIMS = {'cliff_block': (6.0, 3.0, 4.2), 'crag': (4.0, 3.4, 3.2), 'boulder': (1.8, 1.5, 1.2), 'scree': (0.55, 0.45, 0.32)}
# subdivision level of the base cube, joint cuts, bedding ledges, noise amplitude (fraction of size)
SPEC = {'cliff_block': (4, (4, 7), (2, 4), 0.09), 'crag': (4, (5, 8), (1, 3), 0.1),
        'boulder': (3, (5, 8), (0, 1), 0.07), 'scree': (1, (4, 7), (0, 0), 0.05)}


def _cube(bm, subdiv, rng):
    bmesh.ops.create_cube(bm, size=2.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=subdiv, use_grid_fill=True)
    # spherify a little so joint cuts produce irregular polygons, not a box; jitter breaks the grid
    j = 0.5 / (subdiv + 1)
    for v in bm.verts:
        v.co = v.co.lerp(v.co.normalized() * 1.25, 0.35) + Vector(rng.uniform(-j, j, 3))


def build_rock(kind, variant, seed=0):
    """Return (verts Nx3, faces list, ao N) in metres, Z-up, base at z = 0 (sunk parts below)."""
    rng = np.random.default_rng(1000 * KINDS.index(kind) + 17 * variant + seed)
    sub, cuts, ledges, amp = SPEC[kind]
    sx, sy, sz = DIMS[kind]
    sx *= rng.uniform(0.8, 1.2); sy *= rng.uniform(0.8, 1.2); sz *= rng.uniform(0.85, 1.15)
    bm = bmesh.new()
    _cube(bm, sub, rng)
    # joints: planar cuts through the blob (remove the outside) -> flat faces, sharp edges
    for _ in range(int(rng.integers(*cuts)) if cuts[1] > cuts[0] else cuts[0]):
        n = Vector(rng.normal(size=3)); n.z *= 0.6
        if n.length < 1e-3:
            continue
        n.normalize()
        d = rng.uniform(0.5, 0.85)
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        r = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=n * d, plane_no=n, clear_outer=True)
        edges = [e for e in r['geom_cut'] if isinstance(e, bmesh.types.BMEdge)]
        if edges:
            try:
                bmesh.ops.holes_fill(bm, edges=edges, sides=0)
            except Exception:
                pass
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    # a flat base (the part that sits in the ground)
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    r = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(0, 0, -0.75), plane_no=(0, 0, -1), clear_outer=True)
    edges = [e for e in r['geom_cut'] if isinstance(e, bmesh.types.BMEdge)]
    if edges:
        bmesh.ops.holes_fill(bm, edges=edges, sides=0)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    # denser mesh for the weathering noise
    bmesh.ops.subdivide_edges(bm, edges=[e for e in bm.edges if e.calc_length() > 0.45], cuts=1, use_grid_fill=False)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    off = Vector(rng.uniform(-50, 50, 3))
    # bedding ledges: the block steps back in 2-4 horizontal layers (face on -y side = the exposed cliff)
    nl = int(rng.integers(ledges[0], ledges[1] + 1)) if ledges[1] else 0
    lz = sorted(rng.uniform(-0.5, 0.8, nl)) if nl else []
    for v in bm.verts:
        p = v.co
        # anisotropic scale to size
        q = Vector((p.x * sx / 2, p.y * sy / 2, (p.z + 0.75) * sz / 1.75))
        # ledges: above each bedding plane the front face (y < 0) is set back
        if lz and p.y < 0:
            step = sum(1 for z in lz if p.z > z)
            q.y += step * sy * 0.16 * min(1.0, -p.y * 1.5)
            q.z += step * sz * 0.02
        # weathering: fBm (bigger along the bedding: horizontal streaks), stronger on exposed faces
        s = 0.45 / max(sz, 0.3)
        nn = noise.fractal(Vector((q.x * s * 0.8 + off.x, q.y * s * 0.8 + off.y, q.z * s * 2.2 + off.z)), 0.6, 2.1, 4)
        q += p.normalized() * nn * amp * max(sx, sy, sz) if p.length > 1e-4 else Vector()
        v.co = q
    bm.normal_update()
    # soften arrises a touch (one pass of laplacian-ish smoothing keeps the flats)
    bmesh.ops.smooth_vert(bm, verts=bm.verts[:], factor=0.2, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.normal_update()
    V = np.array([v.co[:] for v in bm.verts], np.float32)
    idx = {v: i for i, v in enumerate(bm.verts)}
    F = [[idx[v] for v in f.verts] for f in bm.faces]
    # ao: concavity (vertex normal vs neighbour centroid) + up-facing exposure + lower in the ground = darker
    ao = np.zeros(len(V), np.float32)
    for v in bm.verts:
        nb = [e.other_vert(v).co for e in v.link_edges]
        if nb:
            c = sum(nb, Vector()) / len(nb)
            conc = (c - v.co).dot(v.normal) / max(0.05, (c - v.co).length)
        else:
            conc = 0
        up = v.normal.z
        ao[idx[v]] = np.clip(0.72 - 0.35 * conc + 0.18 * up + 0.1 * min(1.0, v.co.z / max(sz, 0.1)), 0.25, 1.0)
    bm.free()
    return V, F, ao


def to_object(name, V, F, ao, coll=None):
    import bpy
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], F)
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    ca = me.color_attributes.new('ao', 'FLOAT_COLOR', 'POINT')
    for i, a in enumerate(ao):
        ca.data[i].color = (a, a, a, 1)
    ob = bpy.data.objects.new(name, me)
    if coll is not None:
        coll.objects.link(ob)
    return ob
