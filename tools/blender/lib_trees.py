"""Procedural leaf-card trees, shrubs and snags for the Blender world (Appalachian set).

v2 art direction (graphics ref.png: stylized realism, soft clustered crowns):
  * A crown is a union of *lobes* (sub-crowns around branch clusters). Each lobe gets
      - an opaque, noise-displaced CORE mesh (dark lifted green, tinted per instance): the
        crown reads as a solid soft mass at every distance, no sky speckle, no black holes;
      - a SHELL of foliage cluster cards (assets/foliage/card_*.png: hundreds of small
        correctly-scaled leaves per card) on the lobe surface; cards buried inside a
        neighbouring lobe are culled.
  * Card normals are replaced by lobe/crown-volume normals -> clumps shade as soft masses
    (lit tops, shaded undersides) while the cards keep leaf-level silhouette detail.
  * Per-vertex 'ao' (crown depth / height) lifts the interior instead of letting it go
    black; per-card 'lv' drives subtle hue/value variation inside one crown.
  * Leaves are translucent and partially transparent to shadow rays (forgiving shadows).

Prototypes are named 'Vnn_<Species>' (nn = instance index, alphabetical == Collection
Info order). build_vegetation() instances them with Geometry Nodes from the ecosystem
scatter written by tools/pipeline/vegetation.py (public/world/vegetation_f32.bin).
"""
import bpy, math, os
import numpy as np
from mathutils import Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PH = os.path.join(ROOT, 'assets/external/polyhaven')


# ---------------------------------------------------------------- geometry helpers
class Geo:
    """Accumulates verts/faces/material index/per-vertex attrs/custom normals."""
    def __init__(self):
        self.V, self.F, self.M, self.N, self.UV, self.AO, self.LV = [], [], [], [], [], [], []

    def add_vert(self, p, n=None, uv=(0.5, 0.5), ao=1.0, lv=0.5):
        self.V.append(tuple(p)); self.N.append(tuple(n) if n is not None else None)
        self.UV.append(uv); self.AO.append(ao); self.LV.append(lv)
        return len(self.V) - 1

    def face(self, idx, mat):
        self.F.append(tuple(idx)); self.M.append(mat)


def tube(g, path, r0, r1, sides=6, mat=0, flare=0.0):
    n = len(path)
    base = len(g.V)
    L = 0.0
    for i, p in enumerate(path):
        t = (path[min(n - 1, i + 1)] - path[max(0, i - 1)]).normalized()
        a = t.orthogonal().normalized(); b = t.cross(a)
        f = i / (n - 1)
        r = r0 + (r1 - r0) * f
        if flare and i == 0:
            r *= 1 + flare
        if i:
            L += (path[i] - path[i - 1]).length
        for k in range(sides):
            ang = 2 * math.pi * k / sides
            d = a * math.cos(ang) + b * math.sin(ang)
            g.add_vert(p + d * r, d, (k / sides, L / 1.5), 1.0, 0.5)
    for i in range(n - 1):
        for k in range(sides):
            a0 = base + i * sides + k; a1 = base + i * sides + (k + 1) % sides
            g.face((a0, a1, a1 + sides, a0 + sides), mat)


def curve(p0, d, L, bend, rng, segs=6, up=0.0):
    pts = [p0.copy()]
    p = p0.copy(); d = d.normalized()
    for i in range(segs):
        d = (d + Vector((rng.uniform(-bend, bend), rng.uniform(-bend, bend), rng.uniform(-bend, bend) + up))).normalized()
        p = p + d * (L / segs)
        pts.append(p.copy())
    return pts


_ICO = None


def _ico():
    """Unit icosphere (subdiv 2) verts/faces, cached."""
    global _ICO
    if _ICO is None:
        import bmesh
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
        V = [v.co.copy() for v in bm.verts]
        F = [[v.index for v in f.verts] for f in bm.faces]
        bm.free()
        _ICO = (V, F)
    return _ICO


def blob(g, c, rx, rz, rng, mat, ao=0.6, lumpy=0.22, flat_bottom=0.0):
    V, F = _ico()
    ph = rng.uniform(0, 6.28, 3)
    base = len(g.V)
    for v in V:
        f = 1 + lumpy * math.sin(v.x * 3.3 + ph[0]) * math.sin(v.y * 2.9 + ph[1]) * math.sin(v.z * 3.7 + ph[2])
        z = v.z
        if flat_bottom and z < -flat_bottom:
            z = -flat_bottom + (z + flat_bottom) * 0.3
        p = Vector((c.x + v.x * rx * f, c.y + v.y * rx * f, c.z + z * rz * f))
        g.add_vert(p, v.normalized(), (0.5, 0.5), ao, 0.5)
    for f in F:
        g.face([base + i for i in f], mat)


def ell_normal(c, rx, rz, up=0.2, crown=None, w=0.6):
    """Per-vertex shading normal from an ellipsoid (lobe) + optional crown ellipsoid: soft volume shading."""
    def f(v):
        q = v - c
        n = Vector((q.x / rx, q.y / rx, q.z / rz))
        n = n.normalized() if n.length > 1e-6 else Vector((0, 0, 1))
        if crown is not None:
            c2, rx2, rz2 = crown
            q2 = v - c2
            n2 = Vector((q2.x / rx2, q2.y / rx2, q2.z / rz2))
            n2 = n2.normalized() if n2.length > 1e-6 else Vector((0, 0, 1))
            n = n * w + n2 * (1 - w)
        return (n + Vector((0, 0, up))).normalized()
    return f


def card(g, pos, facing, size, shade_n, rng, ao, lv, mat=1, aspect=1.0, spin=None):
    facing = facing.normalized()
    a = facing.orthogonal().normalized(); b = facing.cross(a)
    rot = rng.uniform(0, 2 * math.pi) if spin is None else spin
    a, b = a * math.cos(rot) + b * math.sin(rot), -a * math.sin(rot) + b * math.cos(rot)
    s = size / 2
    idx = []
    for (u, v) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        vp = pos + a * u * s + b * v * s * aspect
        idx.append(g.add_vert(vp, shade_n(vp) if callable(shade_n) else shade_n, ((u + 1) / 2, (v + 1) / 2), ao, lv))
    g.face(idx, mat)


def rand_unit(rng):
    v = Vector((rng.normal(), rng.normal(), rng.normal()))
    return v.normalized() if v.length > 1e-6 else Vector((0, 0, 1))


# ---------------------------------------------------------------- species builders
def broadleaf(rng, h=17.0, crown_base=6.0, crown_r=6.0, lobes=(9, 13), lobe_r=(0.3, 0.5), card_size=(1.1, 1.6),
              density=2.2, lean=0.04, trunk_r=0.38, n_primary=6, flat=0.85, stems=1, core_ao=0.42, spread_low=0.0):
    """Hardwood: trunk + primary limbs to lobes, core blobs + card shells."""
    g = Geo()
    crown_h = h - crown_base
    top = Vector((rng.uniform(-lean, lean) * h, rng.uniform(-lean, lean) * h, crown_base + crown_h * 0.45))
    cen = Vector((top.x, top.y, crown_base + crown_h * 0.5))
    # lobes: distributed in the crown ellipsoid, biased to the outside/top
    nl = int(rng.integers(*lobes))
    L = []
    # asymmetric crown: reaches toward one side (light gaps), irregular lobe radii/distances
    reach = Vector((rng.normal(), rng.normal(), 0)); reach = reach.normalized() * crown_r * rng.uniform(0.0, 0.25) if reach.length > 0 else Vector()
    for k in range(nl):
        u = rand_unit(rng)
        u.z = u.z * 0.8 + 0.15
        rr = rng.uniform(*lobe_r) * crown_r
        d = (crown_r - rr * 0.75) * rng.uniform(0.55, 1.1)
        c = cen + reach * max(0.0, u.dot(reach.normalized()) if reach.length else 0) + Vector((u.x * d, u.y * d, u.z * (crown_h * 0.5 - rr * flat * 0.7) * rng.uniform(0.7, 1.15)))
        if spread_low:
            c.z -= spread_low * (1 - abs(u.z)) * crown_h * 0.15
        L.append((c, rr, rr * flat * rng.uniform(0.85, 1.1)))
    L.append((cen + Vector((0, 0, crown_h * 0.1)), crown_r * 0.62, crown_h * 0.42))
    # trunk(s)
    for s in range(stems):
        off = Vector((0, 0, 0)) if stems == 1 else Vector((rng.normal(0, 0.35), rng.normal(0, 0.35), 0))
        pts = [off.lerp(top, t) + Vector((rng.normal(0, 0.08), rng.normal(0, 0.08), 0)) * (0 < t < 1) for t in np.linspace(0, 1, 7)]
        tube(g, pts, trunk_r * (1 if stems == 1 else 0.7), trunk_r * 0.35, 8, 0, flare=0.45)
    # primary limbs from trunk to lobes (visible below / between lobes)
    for k, (c, rr, rh) in enumerate(L[:-1]):
        t0 = rng.uniform(0.45, 0.95)
        p0 = Vector((0, 0, 0)).lerp(top, t0)
        d = (c - p0)
        path = curve(p0, d, d.length * 0.9, 0.12, rng, 5, 0.02)
        tube(g, path, trunk_r * 0.45, 0.05, 5, 0)
        for j in range(2):  # a couple of visible secondaries
            q = path[int(rng.integers(2, len(path)))]
            sub = curve(q, rand_unit(rng) + Vector((0, 0, 0.3)), rr * rng.uniform(0.5, 0.9), 0.3, rng, 3, 0.05)
            tube(g, sub, 0.05, 0.015, 4, 0)
    # cores
    for c, rr, rh in L:
        blob(g, c, rr * 0.74, rh * 0.74, rng, 2, core_ao, 0.25)
    # card shells
    zlo = min(c.z - rh for c, rr, rh in L); zhi = max(c.z + rh for c, rr, rh in L)
    for i, (c, rr, rh) in enumerate(L):
        area = 4 * math.pi * rr * rh
        n = int(area * density / (np.mean(card_size) ** 2 * 0.4))
        for _ in range(n):
            u = rand_unit(rng)
            sh = rng.uniform(0.82, 1.08) if rng.random() > 0.08 else rng.uniform(1.08, 1.28)  # a few tufts break the silhouette
            p = c + Vector((u.x * rr * sh, u.y * rr * sh, u.z * rh * sh))
            # cull if buried in another lobe
            buried = False
            for j, (c2, r2, h2) in enumerate(L):
                if j != i:
                    q = p - c2
                    if (q.x / r2) ** 2 + (q.y / r2) ** 2 + (q.z / h2) ** 2 < 0.62:
                        buried = True; break
            if buried:
                continue
            ln = Vector((u.x / rr, u.y / rr, u.z / rh)).normalized()
            cn = (p - cen); cn = Vector((cn.x / crown_r, cn.y / crown_r, cn.z / (crown_h * 0.5))).normalized()
            sn = (ln * 0.6 + cn * 0.4 + Vector((0, 0, 0.2))).normalized()
            facing = (ln * 0.7 + rand_unit(rng) * 0.75)
            hf = (p.z - zlo) / max(zhi - zlo, 1e-3)
            outer = max(0.0, min(1.0, (cn.dot(sn) + 0.2)))
            ao = 0.52 + 0.3 * hf + 0.18 * outer * (0.5 + 0.5 * max(0, u.z))
            card(g, p, facing, rng.uniform(*card_size), ell_normal(c, rr, rh, 0.2, (cen, crown_r, crown_h * 0.5)), rng, min(ao, 1.0), rng.random())
    return g


def white_pine(rng, h=24.0, crown_base=6.0, base_r=5.0, tiers=(7, 10), card_size=(1.4, 1.9), density=1.9, scrubby=False):
    """Eastern white pine: tall straight trunk, open irregular crown of horizontal branch pads."""
    g = Geo()
    lean = Vector((rng.normal(0, 0.03), rng.normal(0, 0.03), 1))
    trunk = [Vector((lean.x * h * t, lean.y * h * t, h * t)) for t in np.linspace(0, 1, 9)]
    tube(g, trunk, 0.34 * h / 22, 0.05, 8, 0, flare=0.4)
    nt = int(rng.integers(*tiers))
    pads = []
    for w in range(nt):
        t = min(1.0, max(0.0, (w + rng.uniform(-0.2, 0.2)) / (nt - 1)))
        z = crown_base + (h - crown_base - 1.0) * t
        r = base_r * (1 - t) ** 0.7 * rng.uniform(0.7, 1.15) + 0.8
        nb = int(rng.integers(3, 6)) if not scrubby else int(rng.integers(4, 7))
        az0 = rng.uniform(0, 6.28)
        for k in range(nb):
            if rng.random() < 0.18 and w < nt - 2:
                continue  # gaps: the white pine look
            az = az0 + 2 * math.pi * k / nb + rng.normal(0, 0.3)
            L = r * rng.uniform(0.7, 1.05)
            d = Vector((math.cos(az), math.sin(az), 0.18 + (0.25 if scrubby else 0)))
            p0 = Vector((lean.x * z, lean.y * z, z))
            path = curve(p0, d, L, 0.1, rng, 4, 0.03)
            tube(g, path, 0.09, 0.02, 4, 0)
            tip = path[-1]
            # pad = flattened lobe near the branch tip (+ one mid pad on long limbs)
            for fr, sz in ((1.0, 1.0), (0.55, 0.7)):
                if fr < 1 and L < 3:
                    continue
                c = path[0].lerp(tip, fr) + Vector((0, 0, 0.4))
                pr = (1.3 + 0.35 * L) * sz * rng.uniform(0.85, 1.15)
                pads.append((c, pr, pr * rng.uniform(0.38, 0.5)))
    top = Vector((lean.x * h, lean.y * h, h))
    pads.append((top - Vector((0, 0, 1.2)), 1.3, 1.4))
    cen_z = (crown_base + h) / 2
    for c, rr, rh in pads:
        blob(g, c, rr * 0.72, rh * 0.7, rng, 2, 0.42, 0.3, flat_bottom=0.3)
    for i, (c, rr, rh) in enumerate(pads):
        n = int(4 * math.pi * rr * rh * density / (np.mean(card_size) ** 2 * 0.25))
        for _ in range(n):
            u = rand_unit(rng)
            p = c + Vector((u.x * rr * rng.uniform(0.8, 1.1), u.y * rr * rng.uniform(0.8, 1.1), u.z * rh * rng.uniform(0.7, 1.1)))
            ln = Vector((u.x / rr, u.y / rr, u.z / rh * 0.6 + 0.35)).normalized()
            facing = (Vector((0, 0, 1)) * 0.9 + rand_unit(rng) * 0.7)   # needle tufts mostly horizontal
            hf = (p.z - crown_base) / max(h - crown_base, 1)
            ao = 0.55 + 0.25 * hf + 0.2 * max(0, u.z)
            card(g, p, facing, rng.uniform(*card_size), ell_normal(c, rr, rh * 1.6, 0.35), rng, min(ao, 1), rng.random())
    return g


def hemlock(rng, h=19.0, crown_base=1.5, base_r=4.0, card_size=(1.3, 1.8), density=2.3):
    """Eastern hemlock: dense, broad-conical, soft drooping sprays, droopy leader."""
    g = Geo()
    trunk = [Vector((rng.normal(0, 0.05) * (t > 0), rng.normal(0, 0.05) * (t > 0), h * t)) for t in np.linspace(0, 1, 9)]
    tube(g, trunk, 0.3 * h / 18, 0.04, 8, 0, flare=0.35)
    # stack of overlapping ellipsoid tiers forming a cone (soft scalloped silhouette)
    tiers = []
    nt = int((h - crown_base) / 1.7)
    for w in range(nt):
        t = w / max(nt - 1, 1)
        z = crown_base + 0.8 + (h - crown_base - 1.6) * t
        r = base_r * (1 - t) ** 0.95 * rng.uniform(0.88, 1.1) + 0.45
        for k in range(int(rng.integers(3, 5)) if r > 1.5 else 1):
            az = rng.uniform(0, 6.28)
            off = r * 0.35 if r > 1.5 else 0
            c = Vector((math.cos(az) * off, math.sin(az) * off, z))
            tiers.append((c, r * (0.75 if off else 1.0), min(1.6, r * 0.6) + 0.5))
    for c, rr, rh in tiers:
        blob(g, c, rr * 0.8, rh * 0.75, rng, 2, 0.4, 0.2)
    for c, rr, rh in tiers:
        n = int(4 * math.pi * rr * rh * density / (np.mean(card_size) ** 2 * 0.22) * 0.5)
        for _ in range(n):
            u = rand_unit(rng)
            if u.z > 0.6:
                continue
            p = c + Vector((u.x * rr * rng.uniform(0.85, 1.08), u.y * rr * rng.uniform(0.85, 1.08), u.z * rh))
            ln = Vector((u.x, u.y, 0.35)).normalized()
            # drooping sprays: cards tilt down-and-out
            out = Vector((u.x, u.y, 0)).normalized() if abs(u.x) + abs(u.y) > 1e-3 else Vector((1, 0, 0))
            facing = (Vector((0, 0, 1)) * 0.8 + out * 0.5 + rand_unit(rng) * 0.4)
            hf = p.z / h
            ao = 0.52 + 0.3 * hf + 0.15 * (1 - abs(u.z))
            card(g, p, facing, rng.uniform(*card_size), ell_normal(Vector((0, 0, c.z)), rr, rh * 2.5, 0.3), rng, min(ao, 1), rng.random())
    return g


def shrub(rng, w=3.0, hh=2.2, lobes=(4, 7), card_size=(0.9, 1.3), density=2.0, stems=5):
    """Multi-stem shrub dome (rhododendron / laurel / brush)."""
    g = Geo()
    for s in range(stems):
        a = rng.uniform(0, 6.28); r = rng.uniform(0, w * 0.2)
        p0 = Vector((math.cos(a) * r, math.sin(a) * r, 0))
        path = curve(p0, Vector((math.cos(a) * 0.4, math.sin(a) * 0.4, 1)), hh * 0.7, 0.2, rng, 3, 0.0)
        tube(g, path, 0.05, 0.02, 4, 0)
    L = []
    for k in range(int(rng.integers(*lobes))):
        a = rng.uniform(0, 6.28); r = rng.uniform(0, w * 0.3)
        rr = rng.uniform(0.35, 0.55) * w
        L.append((Vector((math.cos(a) * r, math.sin(a) * r, hh * rng.uniform(0.45, 0.62))), rr, min(rr * 0.75, hh * 0.5)))
    for c, rr, rh in L:
        blob(g, c, rr * 0.8, rh * 0.8, rng, 2, 0.45, 0.25, flat_bottom=0.5)
    for i, (c, rr, rh) in enumerate(L):
        n = int(4 * math.pi * rr * rh * density / (np.mean(card_size) ** 2 * 0.3))
        for _ in range(n):
            u = rand_unit(rng)
            if u.z < -0.55:
                continue
            p = c + Vector((u.x * rr, u.y * rr, u.z * rh)) * rng.uniform(0.85, 1.07)
            if p.z < 0.15:
                continue
            ln = Vector((u.x / rr, u.y / rr, u.z / rh + 0.3)).normalized()
            facing = ln * 0.6 + rand_unit(rng) * 0.7
            ao = 0.5 + 0.35 * min(1, p.z / hh) + 0.15 * max(0, u.z)
            card(g, p, facing, rng.uniform(*card_size), ell_normal(c, rr, rh, 0.3), rng, min(ao, 1), rng.random())
    return g


def snag(rng, h=14.0, broken=True):
    """Dead standing tree: grey trunk, few broken stubs, no foliage."""
    g = Geo()
    hh = h * (rng.uniform(0.5, 0.8) if broken else 1.0)
    trunk = curve(Vector((0, 0, 0)), Vector((rng.normal(0, 0.05), rng.normal(0, 0.05), 1)), hh, 0.03, rng, 8, 0.0)
    tube(g, trunk, 0.3 * h / 16, 0.12 * h / 16 if broken else 0.04, 7, 0, flare=0.4)
    for k in range(int(rng.integers(3, 7))):
        q = trunk[int(rng.integers(3, len(trunk)))]
        d = rand_unit(rng) + Vector((0, 0, 0.6))
        sub = curve(q, d, rng.uniform(1.0, 3.5), 0.2, rng, 3, 0.05)
        tube(g, sub, 0.09, 0.02, 4, 0)
    return g


# ---------------------------------------------------------------- mesh
def to_mesh(name, g):
    me = bpy.data.meshes.new(name)
    me.from_pydata(g.V, [], g.F)
    me.polygons.foreach_set('material_index', np.asarray(g.M, np.int32))
    me.polygons.foreach_set('use_smooth', np.ones(len(g.F), bool))
    uv = me.uv_layers.new(name='UVMap')
    li = np.zeros(len(me.loops), np.int32); me.loops.foreach_get('vertex_index', li)
    uva = np.asarray(g.UV, np.float32)
    uv.data.foreach_set('uv', uva[li].ravel())
    for an, data in (('ao', g.AO), ('lv', g.LV)):
        a = me.attributes.new(an, 'FLOAT', 'POINT'); a.data.foreach_set('value', np.asarray(data, np.float32))
    me.update()
    vn = np.array([n if n is not None else (0, 0, 1) for n in g.N], np.float32)
    me.normals_split_custom_set_from_vertices(vn)
    return me


# ---------------------------------------------------------------- materials
def _img(path, color=True):
    c = os.path.join(PH, '_cache1k', os.path.splitext(os.path.basename(path))[0])
    for ext in ('.jpg', '.png'):
        if os.path.exists(c + ext) and 'polyhaven' in path:
            path = c + ext; break
    im = bpy.data.images.load(path, check_existing=True)
    if not color:
        im.colorspace_settings.name = 'Non-Color'
    return im


LEAF_K = 1.9  # card avg luminance compensation: final albedo ~= tint


def foliage_material(card):
    name = f'MAT_Foliage_{card}'
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; N = nt.nodes; L = nt.links
    for n in list(N):
        N.remove(n)
    out = N.new('ShaderNodeOutputMaterial')
    tex = N.new('ShaderNodeTexImage'); tex.image = _img(os.path.join(ROOT, f'assets/foliage/card_{card}.png'))
    ntex = N.new('ShaderNodeTexImage'); ntex.image = _img(os.path.join(ROOT, f'assets/foliage/card_{card}_nrm.png'), False)
    nm = N.new('ShaderNodeNormalMap'); nm.inputs['Strength'].default_value = 0.55
    L.new(ntex.outputs['Color'], nm.inputs['Color'])
    col = _tinted(nt, tex.outputs['Color'], LEAF_K, True)
    bsdf = N.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.62
    bsdf.inputs['Specular IOR Level'].default_value = 0.35
    L.new(col, bsdf.inputs['Base Color'])
    L.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
    tr = N.new('ShaderNodeBsdfTranslucent')
    warm = N.new('ShaderNodeMix'); warm.data_type = 'RGBA'; warm.blend_type = 'MULTIPLY'; warm.inputs['Factor'].default_value = 1.0
    warm.inputs['B'].default_value = (1.5, 1.35, 0.7, 1)   # backlit leaves glow warm yellow-green
    L.new(col, warm.inputs['A']); L.new(warm.outputs['Result'], tr.inputs['Color'])
    L.new(nm.outputs['Normal'], tr.inputs['Normal'])
    mixs = N.new('ShaderNodeMixShader'); mixs.inputs['Fac'].default_value = 0.32
    L.new(bsdf.outputs[0], mixs.inputs[1]); L.new(tr.outputs[0], mixs.inputs[2])
    _alpha_out(nt, tex.outputs['Alpha'], mixs.outputs[0], out, shadow_pass=0.35)
    if hasattr(m, 'blend_method'):
        m.blend_method = 'CLIP'
    return m


def _tinted(nt, card_col, k, per_card_var):
    """card colour * instance tint * k, with per-card hue/value variation and crown AO lift."""
    N, L = nt.nodes, nt.links
    tint = N.new('ShaderNodeAttribute'); tint.attribute_type = 'INSTANCER'; tint.attribute_name = 'tint'
    ao = N.new('ShaderNodeAttribute'); ao.attribute_type = 'GEOMETRY'; ao.attribute_name = 'ao'
    hs = N.new('ShaderNodeHueSaturation')
    L.new(tint.outputs['Color'], hs.inputs['Color'])
    if per_card_var:
        lv = N.new('ShaderNodeAttribute'); lv.attribute_type = 'GEOMETRY'; lv.attribute_name = 'lv'
        hue = N.new('ShaderNodeMapRange'); hue.inputs['To Min'].default_value = 0.47; hue.inputs['To Max'].default_value = 0.53
        L.new(lv.outputs['Fac'], hue.inputs['Value']); L.new(hue.outputs['Result'], hs.inputs['Hue'])
        val = N.new('ShaderNodeMapRange'); val.inputs['To Min'].default_value = 0.85; val.inputs['To Max'].default_value = 1.15
        L.new(lv.outputs['Fac'], val.inputs['Value']); L.new(val.outputs['Result'], hs.inputs['Value'])
    mul = N.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs['Factor'].default_value = 1.0
    L.new(card_col, mul.inputs['A']); L.new(hs.outputs['Color'], mul.inputs['B'])
    sc = N.new('ShaderNodeVectorMath'); sc.operation = 'SCALE'
    aok = N.new('ShaderNodeMath'); aok.operation = 'MULTIPLY'; aok.inputs[1].default_value = k
    L.new(ao.outputs['Fac'], aok.inputs[0])
    L.new(mul.outputs['Result'], sc.inputs[0]); L.new(aok.outputs[0], sc.inputs['Scale'])
    return sc.outputs['Vector']


def _alpha_out(nt, alpha, shader, out, shadow_pass=0.0):
    N, L = nt.nodes, nt.links
    cut = N.new('ShaderNodeMath'); cut.operation = 'GREATER_THAN'; cut.inputs[1].default_value = 0.5
    L.new(alpha, cut.inputs[0])
    fac = cut.outputs[0]
    if shadow_pass:
        lp = N.new('ShaderNodeLightPath')
        sp = N.new('ShaderNodeMath'); sp.operation = 'MULTIPLY_ADD'
        L.new(lp.outputs['Is Shadow Ray'], sp.inputs[0]); sp.inputs[1].default_value = -shadow_pass; sp.inputs[2].default_value = 1.0
        f2 = N.new('ShaderNodeMath'); f2.operation = 'MULTIPLY'
        L.new(fac, f2.inputs[0]); L.new(sp.outputs[0], f2.inputs[1])
        fac = f2.outputs[0]
    tp = N.new('ShaderNodeBsdfTransparent')
    ma = N.new('ShaderNodeMixShader')
    L.new(fac, ma.inputs['Fac']); L.new(tp.outputs[0], ma.inputs[1]); L.new(shader, ma.inputs[2])
    L.new(ma.outputs[0], out.inputs['Surface'])


def core_material():
    m = bpy.data.materials.get('MAT_Foliage_Core')
    if m:
        return m
    m = bpy.data.materials.new('MAT_Foliage_Core'); m.use_nodes = True
    nt = m.node_tree; N = nt.nodes; L = nt.links
    for n in list(N):
        N.remove(n)
    out = N.new('ShaderNodeOutputMaterial')
    # soft leafy breakup on the core (visible only through card gaps / at distance)
    tc = N.new('ShaderNodeTexCoord')
    vor = N.new('ShaderNodeTexVoronoi'); vor.inputs['Scale'].default_value = 3.0
    L.new(tc.outputs['Object'], vor.inputs['Vector'])
    mr = N.new('ShaderNodeMapRange'); mr.inputs['To Min'].default_value = 0.7; mr.inputs['To Max'].default_value = 1.15
    L.new(vor.outputs['Distance'], mr.inputs['Value'])
    cc = N.new('ShaderNodeCombineColor')
    for i in range(3):
        L.new(mr.outputs['Result'], cc.inputs[i])
    col = _tinted(nt, cc.outputs['Color'], 1.0, False)
    bsdf = N.new('ShaderNodeBsdfPrincipled'); bsdf.inputs['Roughness'].default_value = 0.85
    bsdf.inputs['Specular IOR Level'].default_value = 0.2
    L.new(col, bsdf.inputs['Base Color'])
    bp = N.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.8; bp.inputs['Distance'].default_value = 0.3
    L.new(vor.outputs['Distance'], bp.inputs['Height']); L.new(bp.outputs['Normal'], bsdf.inputs['Normal'])
    tr = N.new('ShaderNodeBsdfTranslucent'); L.new(col, tr.inputs['Color'])
    mixs = N.new('ShaderNodeMixShader'); mixs.inputs['Fac'].default_value = 0.25
    L.new(bsdf.outputs[0], mixs.inputs[1]); L.new(tr.outputs[0], mixs.inputs[2])
    # partially transparent to shadow rays: sun leaks through crowns (lit undersides, soft shade)
    lp = N.new('ShaderNodeLightPath'); tp = N.new('ShaderNodeBsdfTransparent')
    sh = N.new('ShaderNodeMath'); sh.operation = 'MULTIPLY'; sh.inputs[1].default_value = 0.5
    L.new(lp.outputs['Is Shadow Ray'], sh.inputs[0])
    ms = N.new('ShaderNodeMixShader'); L.new(sh.outputs[0], ms.inputs['Fac'])
    L.new(mixs.outputs[0], ms.inputs[1]); L.new(tp.outputs[0], ms.inputs[2])
    L.new(ms.outputs[0], out.inputs['Surface'])
    return m


def bark_material(name='MAT_Bark_PBR', aid='bark_brown_02', tint=(1, 1, 1, 1), scale=1.2):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; N = nt.nodes; L = nt.links
    bsdf = next(n for n in N if n.type == 'BSDF_PRINCIPLED')
    p = os.path.join(PH, aid, 'textures')
    diff = [f for f in os.listdir(p) if '_diff_' in f] if os.path.isdir(p) else []
    if diff:
        tc = N.new('ShaderNodeTexCoord')
        mp = N.new('ShaderNodeMapping'); mp.inputs['Scale'].default_value = (1 / scale, 1 / scale, 0.5 / scale)
        L.new(tc.outputs['Object'], mp.inputs[0])
        t = N.new('ShaderNodeTexImage'); t.image = _img(os.path.join(p, diff[0])); t.projection = 'BOX'; t.projection_blend = 0.4
        L.new(mp.outputs[0], t.inputs[0])
        mul = N.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs['Factor'].default_value = 1
        L.new(t.outputs['Color'], mul.inputs['A']); mul.inputs['B'].default_value = tint
        L.new(mul.outputs['Result'], bsdf.inputs['Base Color'])
        disp = [f for f in os.listdir(p) if '_disp_' in f]
        if disp:
            d = N.new('ShaderNodeTexImage'); d.image = _img(os.path.join(p, disp[0]), False); d.projection = 'BOX'; d.projection_blend = 0.4
            L.new(mp.outputs[0], d.inputs[0])
            bp = N.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 1.0; bp.inputs['Distance'].default_value = 0.04
            L.new(d.outputs['Color'], bp.inputs['Height']); L.new(bp.outputs['Normal'], bsdf.inputs['Normal'])
    else:
        bsdf.inputs['Base Color'].default_value = (0.05, 0.04, 0.03, 1)
    bsdf.inputs['Roughness'].default_value = 0.9
    return m


# ---------------------------------------------------------------- species table
# id: (name, family, builder, kwargs, card, bark)
def _species():
    B, WP, HM, SH, SN = broadleaf, white_pine, hemlock, shrub, snag
    return [
        ('WhiteOak_A', 'oak', B, dict(h=20, crown_base=5.5, crown_r=6.8, lobes=(10, 14), n_primary=7, trunk_r=0.45), 'oak', 'brown'),
        ('WhiteOak_B', 'oak', B, dict(h=17, crown_base=4.5, crown_r=6.0, lobes=(9, 12)), 'oak', 'brown'),
        ('Oak_OpenGrown', 'oak', B, dict(h=15, crown_base=3.2, crown_r=8.0, lobes=(12, 16), lobe_r=(0.28, 0.42), flat=0.75, spread_low=1.0, trunk_r=0.55), 'oak', 'brown'),
        ('RedMaple_A', 'maple', B, dict(h=17, crown_base=4.5, crown_r=5.2, lobes=(8, 12), flat=0.95), 'maple', 'grey'),
        ('RedMaple_B', 'maple', B, dict(h=14, crown_base=3.5, crown_r=4.6, lobes=(7, 10), stems=2), 'maple', 'grey'),
        ('TulipPoplar_A', 'poplar', B, dict(h=28, crown_base=13, crown_r=5.0, lobes=(8, 12), flat=1.1, trunk_r=0.45), 'poplar', 'grey'),
        ('TulipPoplar_B', 'poplar', B, dict(h=23, crown_base=9, crown_r=4.6, lobes=(7, 11), flat=1.1), 'poplar', 'grey'),
        ('Hickory_A', 'hickory', B, dict(h=21, crown_base=8, crown_r=5.2, lobes=(8, 11), flat=1.0), 'hickory', 'grey'),
        ('Dogwood', 'dogwood', B, dict(h=7, crown_base=1.8, crown_r=3.2, lobes=(4, 6), lobe_r=(0.45, 0.6), flat=0.6, trunk_r=0.12, card_size=(1.0, 1.4), stems=2), 'dogwood', 'grey'),
        ('Sapling_HW_A', 'sapling', B, dict(h=7, crown_base=2.5, crown_r=2.2, lobes=(3, 5), trunk_r=0.08, card_size=(0.9, 1.3), core_ao=0.5), 'maple', 'grey'),
        ('Sapling_HW_B', 'sapling', B, dict(h=9, crown_base=3.5, crown_r=2.6, lobes=(3, 6), trunk_r=0.1, card_size=(1.0, 1.4), flat=1.1, core_ao=0.5), 'poplar', 'grey'),
        ('WhitePine_A', 'pine', WP, dict(h=26, crown_base=8, base_r=5.2), 'pine', 'pine'),
        ('WhitePine_B', 'pine', WP, dict(h=21, crown_base=4, base_r=4.6, tiers=(8, 11)), 'pine', 'pine'),
        ('Hemlock_A', 'hemlock', HM, dict(h=21, crown_base=1.5, base_r=4.4), 'hemlock', 'pine'),
        ('Hemlock_B', 'hemlock', HM, dict(h=16, crown_base=1.0, base_r=3.8), 'hemlock', 'pine'),
        ('VirginiaPine', 'pine', WP, dict(h=13, crown_base=3, base_r=3.2, tiers=(5, 8), scrubby=True, card_size=(1.1, 1.5)), 'pine', 'pine'),
        ('Sapling_Pine', 'pine', HM, dict(h=5, crown_base=0.3, base_r=1.6, card_size=(0.8, 1.1)), 'pine', 'pine'),
        ('Snag_A', 'snag', SN, dict(h=15, broken=True), None, 'dead'),
        ('Snag_B', 'snag', SN, dict(h=18, broken=False), None, 'dead'),
        ('Rhododendron_A', 'rhodo', SH, dict(w=4.0, hh=3.0, lobes=(5, 8)), 'rhodo', 'grey'),
        ('Laurel_B', 'rhodo', SH, dict(w=2.8, hh=2.0, lobes=(4, 6), card_size=(0.8, 1.1)), 'rhodo', 'grey'),
        ('Brush_A', 'brush', SH, dict(w=2.6, hh=1.6, lobes=(4, 7), card_size=(0.8, 1.2), stems=6), 'brush', 'grey'),
        ('Brush_B', 'brush', SH, dict(w=1.8, hh=1.1, lobes=(3, 5), card_size=(0.7, 1.0), stems=4), 'brush', 'grey'),
        # extra variants (appended: ids above stay stable; vegetation.py remaps a share of A -> C)
        ('WhiteOak_C', 'oak', B, dict(h=18, crown_base=5.0, crown_r=6.4, lobes=(9, 13), lean=0.07, trunk_r=0.42), 'oak', 'brown'),
        ('RedMaple_C', 'maple', B, dict(h=16, crown_base=4.0, crown_r=4.8, lobes=(7, 11), flat=1.0, lean=0.07), 'maple', 'grey'),
        ('Hemlock_C', 'hemlock', HM, dict(h=24, crown_base=3.0, base_r=4.8), 'hemlock', 'pine'),
        ('WhitePine_C', 'pine', WP, dict(h=28, crown_base=10, base_r=5.6, tiers=(6, 9)), 'pine', 'pine'),
    ]


SPECIES = _species()
SPECIES_ID = {s[0]: i for i, s in enumerate(SPECIES)}
FAMILY_OF = [s[1] for s in SPECIES]

# early-autumn Blue Ridge palettes (linear albedo tints). graphics ref: deep olive greens,
# ~30-40% warm ochre / amber / rust accents in hardwoods, dark blue-green conifers.
PAL = {
    'oak':     [((0.046, 0.074, 0.028), 0.5), ((0.060, 0.080, 0.030), 0.3), ((0.15, 0.10, 0.030), 0.12), ((0.14, 0.062, 0.024), 0.08)],
    'maple':   [((0.050, 0.078, 0.030), 0.6), ((0.25, 0.085, 0.020), 0.14), ((0.22, 0.045, 0.018), 0.1), ((0.24, 0.14, 0.028), 0.16)],
    'poplar':  [((0.058, 0.088, 0.032), 0.7), ((0.20, 0.15, 0.035), 0.2), ((0.12, 0.105, 0.030), 0.1)],
    'hickory': [((0.062, 0.088, 0.03), 0.65), ((0.23, 0.165, 0.035), 0.35)],
    'dogwood': [((0.06, 0.085, 0.035), 0.45), ((0.19, 0.045, 0.030), 0.55)],
    'sapling': [((0.055, 0.090, 0.030), 0.72), ((0.17, 0.115, 0.03), 0.16), ((0.19, 0.06, 0.02), 0.12)],
    'pine':    [((0.034, 0.064, 0.038), 0.6), ((0.042, 0.072, 0.042), 0.4)],
    'hemlock': [((0.026, 0.050, 0.034), 0.7), ((0.032, 0.056, 0.037), 0.3)],
    'rhodo':   [((0.030, 0.058, 0.030), 0.8), ((0.040, 0.066, 0.030), 0.2)],
    'brush':   [((0.07, 0.09, 0.03), 0.5), ((0.19, 0.07, 0.025), 0.18), ((0.16, 0.115, 0.035), 0.32)],
    'snag':    [((0.1, 0.1, 0.1), 1.0)],
}


def build_prototypes(coll, seed=7, only=None):
    """Build all species prototype objects (hidden source meshes for instancing)."""
    rng = np.random.default_rng(seed)
    barks = {
        'brown': bark_material('MAT_Bark_Oak', 'bark_brown_02', (0.34, 0.31, 0.28, 1), 0.6),
        'grey': bark_material('MAT_Bark_Grey', 'bark_brown_02', (0.42, 0.42, 0.4, 1), 0.6),
        'pine': bark_material('MAT_Bark_Pine', 'pine_bark', (0.45, 0.38, 0.33, 1), 0.7),
        'dead': bark_material('MAT_Bark_Dead', 'bark_brown_02', (0.75, 0.73, 0.7, 1), 0.6),
    }
    core = core_material()
    objs = []
    for i, (name, fam, fn, kw, cardn, bark) in enumerate(SPECIES):
        r = np.random.default_rng(seed * 100 + i)
        if only and name not in only:
            continue
        g = fn(r, **kw)
        me = to_mesh(f'V{i:02d}_{name}', g)
        me.materials.append(barks[bark])
        me.materials.append(foliage_material(cardn) if cardn else barks[bark])
        me.materials.append(core)
        ob = bpy.data.objects.new(f'V{i:02d}_{name}', me)
        coll.objects.link(ob)
        objs.append(ob)
        print(f'  proto {ob.name}: {len(me.polygons)} faces')
    return objs


def srgb_jitter(rng, n):
    return 0.85 + 0.3 * rng.random((n, 1))


def instance_tints(species, rng):
    """Per-instance linear tint from the family palette."""
    col = np.ones((len(species), 4), np.float32)
    for i, fam in enumerate(FAMILY_OF):
        sel = species == i
        if not sel.any():
            continue
        pal = PAL[fam]
        cols = np.array([c for c, _ in pal]); w = np.array([w for _, w in pal])
        pick = rng.choice(len(pal), sel.sum(), p=w / w.sum())
        # blend two palette entries sometimes (turning trees: part green, part colour)
        pick2 = rng.choice(len(pal), sel.sum(), p=w / w.sum())
        t = np.where(rng.random(sel.sum()) < 0.25, rng.random(sel.sum()) * 0.5, 0)[:, None]
        c = cols[pick] * (1 - t) + cols[pick2] * t
        col[sel, :3] = c * srgb_jitter(rng, sel.sum())
    return col


def instancer(name, coll, pts, species, scale, rot, tint, proto_coll, tilt=None):
    """Points mesh + GN Instance-on-Points picking prototype by 'kind'."""
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(pts))
    me.vertices.foreach_set('co', np.asarray(pts, np.float32).ravel())
    if tilt is None:
        tilt = np.zeros((len(pts), 2), np.float32)
    rv = np.c_[tilt, rot].astype(np.float32)
    for n, typ, data in (('kind', 'INT', species), ('scale', 'FLOAT', scale)):
        a = me.attributes.new(n, typ, 'POINT'); a.data.foreach_set('value', np.asarray(data, np.int32 if typ == 'INT' else np.float32))
    a = me.attributes.new('rotv', 'FLOAT_VECTOR', 'POINT'); a.data.foreach_set('vector', rv.ravel())
    a = me.attributes.new('tint', 'FLOAT_COLOR', 'POINT'); a.data.foreach_set('color', np.asarray(tint, np.float32).ravel())
    ob = bpy.data.objects.new(name, me); coll.objects.link(ob)
    ng = bpy.data.node_groups.get('GN_Instancer')
    if ng is None:
        ng = bpy.data.node_groups.new('GN_Instancer', 'GeometryNodeTree')
        ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        s = ng.interface.new_socket('Collection', in_out='INPUT', socket_type='NodeSocketCollection')
        ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        N = ng.nodes; L = ng.links
        gi = N.new('NodeGroupInput'); go = N.new('NodeGroupOutput')
        ci = N.new('GeometryNodeCollectionInfo'); ci.inputs['Separate Children'].default_value = True; ci.inputs['Reset Children'].default_value = True
        L.new(gi.outputs[1], ci.inputs['Collection'])
        iop = N.new('GeometryNodeInstanceOnPoints'); iop.inputs['Pick Instance'].default_value = True
        ak = N.new('GeometryNodeInputNamedAttribute'); ak.data_type = 'INT'; ak.inputs['Name'].default_value = 'kind'
        asc = N.new('GeometryNodeInputNamedAttribute'); asc.data_type = 'FLOAT'; asc.inputs['Name'].default_value = 'scale'
        ar = N.new('GeometryNodeInputNamedAttribute'); ar.data_type = 'FLOAT_VECTOR'; ar.inputs['Name'].default_value = 'rotv'
        L.new(gi.outputs[0], iop.inputs['Points']); L.new(ci.outputs[0], iop.inputs['Instance'])
        L.new(ak.outputs['Attribute'], iop.inputs['Instance Index'])
        L.new(asc.outputs['Attribute'], iop.inputs['Scale']); L.new(ar.outputs['Attribute'], iop.inputs['Rotation'])
        L.new(iop.outputs[0], go.inputs[0])
    mod = ob.modifiers.new('Instancer', 'NODES'); mod.node_group = ng
    ident = ng.interface.items_tree['Collection'].identifier
    mod[ident] = proto_coll
    return ob


def build_vegetation(coll, seed=5):
    """World vegetation from public/world/vegetation_f32.bin (x_px, y_px, z_m, scale, species, seed).
    Falls back to trees.bin (kind 0 hardwood / 1 conifer) if the ecosystem scatter is missing."""
    pc = bpy.data.collections.new('VEG_prototypes'); coll.children.link(pc)
    protos = build_prototypes(pc)
    for p in protos:
        p.hide_render = False
    pc.hide_render = True; pc.hide_viewport = True
    rng = np.random.default_rng(seed)
    H, W = 667, 2000
    hgt = np.fromfile(os.path.join(ROOT, 'data/terrain/height_graded_f32.bin'), np.float32).reshape(H, W)
    vp = os.path.join(ROOT, 'public/world/vegetation_f32.bin')
    if os.path.exists(vp):
        v = np.fromfile(vp, '<f4').reshape(-1, 6)
        sp = v[:, 4].astype(np.int32)
    else:
        v = np.fromfile(os.path.join(ROOT, 'public/world/trees.bin'), '<f4').reshape(-1, 5)
        k = v[:, 4].astype(int)
        hw = [SPECIES_ID[n] for n in ('WhiteOak_A', 'WhiteOak_B', 'RedMaple_A', 'RedMaple_B', 'TulipPoplar_A', 'Hickory_A')]
        cf = [SPECIES_ID[n] for n in ('WhitePine_A', 'WhitePine_B', 'Hemlock_A', 'Hemlock_B')]
        sp = np.where(k == 1, rng.choice(cf, len(v)), rng.choice(hw, len(v)))
    x, y = v[:, 0], v[:, 1]
    z = _bilinear(hgt, x, y)
    # sink on slopes so the root flare never floats
    gy, gx = np.gradient(hgt, 2.5)
    sl = np.hypot(_bilinear(gx, x, y), _bilinear(gy, x, y))
    z = z - 0.25 - np.clip(sl, 0, 1.5) * 0.6
    bx, by = (x - 1000.0) * 2.5, -(y - 333.5) * 2.5
    scale = v[:, 3]
    tint = instance_tints(sp, rng)
    rot = rng.random(len(v)) * 6.283
    tilt = rng.normal(0, 0.025, (len(v), 2))
    ob = instancer('VEG_points', coll, np.c_[bx, by, z], sp, scale, rot, tint, pc, tilt)
    print(f'  vegetation: {len(v)} instances, {len(protos)} prototypes')
    return ob


def _bilinear(a, x, y):
    x = np.clip(np.asarray(x, float) - 0.5, 0, a.shape[1] - 1.001)
    y = np.clip(np.asarray(y, float) - 0.5, 0, a.shape[0] - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    fx = x - x0; fy = y - y0
    return a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy) + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy
