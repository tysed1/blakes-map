"""Procedural leaf-card trees for the Blender world (Appalachian species set).

Each prototype = tapered branch skeleton + foliage cluster cards (assets/foliage/card_*.png)
placed around branch tips. Card normals are replaced by custom "spherical" normals pointing
out of the crown volume, so a crown shades like one soft mass (stylized realism) while the
cards keep leaf-level silhouette detail. ~400-900 cards per tree -> a few k tris: cheap to
instance across the whole world.

Species (Blue Ridge / north Georgia mixed forest): white oak, red maple, tulip poplar,
dogwood (understory), eastern white pine, eastern hemlock.
"""
import bpy, bmesh, math, os
import numpy as np
from mathutils import Vector, Matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _tube(verts, faces, path, r0, r1, sides=6):
    """Tapered tube along path (list of Vector)."""
    n = len(path)
    base = len(verts)
    for i, p in enumerate(path):
        t = (path[min(n - 1, i + 1)] - path[max(0, i - 1)]).normalized()
        a = t.orthogonal().normalized(); b = t.cross(a)
        r = r0 + (r1 - r0) * i / (n - 1)
        for k in range(sides):
            ang = 2 * math.pi * k / sides
            verts.append(p + (a * math.cos(ang) + b * math.sin(ang)) * r)
    for i in range(n - 1):
        for k in range(sides):
            a0 = base + i * sides + k; a1 = base + i * sides + (k + 1) % sides
            faces.append((a0, a1, a1 + sides, a0 + sides))


def _curve(p0, d, L, bend, rng, segs=6, up=0.0):
    pts = [p0.copy()]
    p = p0.copy(); d = d.normalized()
    for i in range(segs):
        d = (d + Vector((rng.uniform(-bend, bend), rng.uniform(-bend, bend), rng.uniform(-bend, bend) + up))).normalized()
        p = p + d * (L / segs)
        pts.append(p.copy())
    return pts


def hardwood(name, rng, h=16.0, crown_r=5.5, crown_h=9.0, crown_base=5.0, n_primary=6, cards=520, card_size=(1.7, 2.6), card='oak', lean=0.05):
    V, F = [], []
    trunk_top = Vector((rng.uniform(-lean, lean) * h, rng.uniform(-lean, lean) * h, crown_base + crown_h * 0.45))
    trunk = [Vector((0, 0, 0)).lerp(trunk_top, t) + Vector((rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), 0)) * (t > 0) for t in np.linspace(0, 1, 7)]
    _tube(V, F, trunk, 0.32 * h / 16, 0.12 * h / 16, 7)
    tips = []
    center = Vector((trunk_top.x, trunk_top.y, crown_base + crown_h * 0.5))
    for k in range(n_primary):
        t0 = rng.uniform(0.35, 0.95)
        p0 = Vector((0, 0, 0)).lerp(trunk_top, t0)
        az = 2 * math.pi * (k + rng.uniform(-0.3, 0.3)) / n_primary
        d = Vector((math.cos(az), math.sin(az), rng.uniform(0.5, 1.3)))
        L = crown_r * rng.uniform(0.8, 1.15)
        path = _curve(p0, d, L, 0.25, rng, 5, 0.05)
        _tube(V, F, path, 0.14 * h / 16, 0.035, 5)
        for q in path[2:]:
            for _ in range(2):
                d2 = (q - center).normalized() + Vector((rng.uniform(-.6, .6), rng.uniform(-.6, .6), rng.uniform(-.2, .7)))
                sub = _curve(q, d2, rng.uniform(1.2, 2.6), 0.4, rng, 3, 0.1)
                _tube(V, F, sub, 0.04, 0.012, 3)
                tips.append(sub[-1])
    # foliage: cards clustered at tips + filling the crown shell (ellipsoid)
    cardV, cardF, cardN, cardUV = [], [], [], []
    def add_card(pos, size, lobe_n=None):
        nrm = Vector(((pos.x - center.x) / crown_r, (pos.y - center.y) / crown_r, (pos.z - center.z) / (crown_h * 0.5) + 0.35)).normalized()
        if lobe_n is not None:
            nrm = (nrm * 0.45 + lobe_n * 0.55 + Vector((0, 0, 0.15))).normalized()
        # card plane: random orientation, biased to face outward
        rnd = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))).normalized()
        facing = (nrm * 0.55 + rnd * 0.45).normalized()
        a = facing.orthogonal().normalized(); b = facing.cross(a)
        rot = rng.uniform(0, 2 * math.pi)
        a, b = a * math.cos(rot) + b * math.sin(rot), -a * math.sin(rot) + b * math.cos(rot)
        s = size / 2
        base = len(cardV)
        for (u, v) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            cardV.append(pos + a * u * s + b * v * s)
            cardUV.append(((u + 1) / 2, (v + 1) / 2))
            cardN.append(nrm)
        cardF.append((base, base + 1, base + 2, base + 3))
    # crown = union of lobes (sub-crowns) around branch clusters -> irregular, cloud-like silhouette
    nl = int(rng.integers(5, 9))
    lobes = []
    for k in range(nl):
        u = Vector((rng.normal(), rng.normal(), rng.normal() * 0.7 + 0.25)).normalized()
        c = center + Vector((u.x * crown_r * 0.55, u.y * crown_r * 0.55, u.z * crown_h * 0.3))
        lobes.append((c, rng.uniform(0.45, 0.62) * crown_r, rng.uniform(0.35, 0.5) * crown_h))
    lobes.append((center, crown_r * 0.7, crown_h * 0.45))
    for i in range(cards):
        c, rr, rh = lobes[rng.integers(len(lobes))]
        u = Vector((rng.normal(), rng.normal(), rng.normal())).normalized() * rng.uniform(0.6, 1.0) ** 0.4
        p = c + Vector((u.x * rr, u.y * rr, u.z * rh))
        lobe_n = (p - c).normalized()
        add_card(p, rng.uniform(*card_size), lobe_n)
    return _mesh(name, V, F, cardV, cardF, cardN, cardUV)


def conifer(name, rng, h=22.0, base_r=4.2, crown_base=3.0, whorls=13, card='pine', droop=0.25, irregular=0.35, cards_per_branch=7, card_size=(1.4, 2.2)):
    V, F = [], []
    trunk = [Vector((rng.uniform(-.1, .1) * (t > 0), rng.uniform(-.1, .1) * (t > 0), t * h)) for t in np.linspace(0, 1, 9)]
    _tube(V, F, trunk, 0.3 * h / 20, 0.04, 7)
    cardV, cardF, cardN, cardUV = [], [], [], []
    for w in range(whorls):
        t = w / (whorls - 1)
        z = crown_base + (h - crown_base - 0.5) * t
        r = base_r * (1 - t) ** 0.85 * (1 + rng.uniform(-irregular, irregular)) + 0.6
        nb = max(4, int(rng.integers(6, 9) * (1 - 0.4 * t)))
        for k in range(nb):
            if rng.random() < irregular * 0.35:
                continue
            az = 2 * math.pi * (k + rng.uniform(-.35, .35)) / nb + w * 0.7
            d = Vector((math.cos(az), math.sin(az), 0.15 - droop * (1 - t)))
            path = _curve(Vector((0, 0, z)), d, r, 0.12, rng, 4, -0.02)
            _tube(V, F, path, 0.07, 0.015, 3)
            for c in range(cards_per_branch):
                q = path[0].lerp(path[-1], 0.15 + 0.85 * (c + 0.5) / cards_per_branch)
                q = q + Vector((rng.normal(0, .45), rng.normal(0, .45), rng.normal(0, .3)))
                # needle sprays lie roughly horizontal along the branch, slightly tilted up
                along = (path[-1] - path[0]).normalized()
                side = along.cross(Vector((0, 0, 1))).normalized()
                tilt = rng.uniform(-0.35, 0.5)
                up = (Vector((0, 0, 1)) + side * tilt).normalized()
                a = along; b = up.cross(a).normalized()
                s = rng.uniform(*card_size) * (0.8 + 0.4 * (1 - t)) / 2
                nrm = Vector((q.x, q.y, 0)).normalized() * 0.8 + Vector((0, 0, 0.6 + 0.6 * t))
                nrm.normalize()
                base = len(cardV)
                for (u, v) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    cardV.append(q + a * u * s + b * v * s * 0.8)
                    cardUV.append(((u + 1) / 2, (v + 1) / 2))
                    cardN.append(nrm)
                cardF.append((base, base + 1, base + 2, base + 3))
    # leader tip
    return _mesh(name, V, F, cardV, cardF, cardN, cardUV)


def _mesh(name, V, F, cV, cF, cN, cUV):
    me = bpy.data.meshes.new(name)
    nv = len(V)
    allV = [tuple(v) for v in V] + [tuple(v) for v in cV]
    allF = [tuple(f) for f in F] + [tuple(i + nv for i in f) for f in cF]
    me.from_pydata(allV, [], allF)
    uv = me.uv_layers.new(name='UVMap')
    loop_uv = []
    for p in me.polygons:
        for li in p.loop_indices:
            vi = me.loops[li].vertex_index
            loop_uv.append(cUV[vi - nv] if vi >= nv else (0.5, (vi % 7) / 7))
    uv.data.foreach_set('uv', np.asarray(loop_uv, np.float32).ravel())
    mi = np.zeros(len(allF), np.int32); mi[len(F):] = 1
    me.polygons.foreach_set('material_index', mi)
    me.polygons.foreach_set('use_smooth', np.ones(len(allF), bool))
    # custom normals: bark keeps its own, foliage gets crown-volume normals
    me.update()
    loops_n = []
    for p in me.polygons:
        for li in p.loop_indices:
            vi = me.loops[li].vertex_index
            loops_n.append(tuple(cN[vi - nv]) if vi >= nv else tuple(p.normal))
    me.normals_split_custom_set(loops_n)
    return me


SPECIES = {
    # name: (kind, builder kwargs, card)
    'A_WhiteOak_1': ('hw', dict(h=17, crown_r=6.5, crown_h=9.5, crown_base=5.0, n_primary=7, cards=720), 'oak'),
    'A_WhiteOak_2': ('hw', dict(h=15, crown_r=5.8, crown_h=8.5, crown_base=4.5, n_primary=6, cards=620), 'oak'),
    'B_RedMaple_1': ('hw', dict(h=15, crown_r=4.8, crown_h=9.0, crown_base=4.0, n_primary=6, cards=600), 'maple'),
    'B_RedMaple_2': ('hw', dict(h=13, crown_r=4.4, crown_h=8.0, crown_base=3.5, n_primary=5, cards=520), 'maple'),
    'C_TulipPoplar_1': ('hw', dict(h=24, crown_r=4.4, crown_h=13.0, crown_base=9.0, n_primary=7, cards=680), 'poplar'),
    'C_TulipPoplar_2': ('hw', dict(h=21, crown_r=4.0, crown_h=11.0, crown_base=8.0, n_primary=6, cards=600), 'poplar'),
    'D_Dogwood_1': ('hw', dict(h=7, crown_r=3.2, crown_h=4.0, crown_base=1.8, n_primary=5, cards=260, card_size=(1.1, 1.7)), 'birch'),
    'E_WhitePine_1': ('cf', dict(h=24, base_r=4.8, whorls=14, droop=0.1, irregular=0.45, cards_per_branch=14, card_size=(2.4, 3.4)), 'pine'),
    'E_WhitePine_2': ('cf', dict(h=20, base_r=4.2, whorls=13, droop=0.08, irregular=0.4, cards_per_branch=13, card_size=(2.2, 3.2)), 'pine'),
    'F_Hemlock_1': ('cf', dict(h=19, base_r=4.2, whorls=20, droop=0.35, irregular=0.15, cards_per_branch=14, card_size=(2.0, 2.8)), 'hemlock'),
    'F_Hemlock_2': ('cf', dict(h=16, base_r=3.8, whorls=18, droop=0.3, irregular=0.15, cards_per_branch=13, card_size=(1.9, 2.6)), 'hemlock'),
}
HARDWOOD_IDS = [i for i, k in enumerate(SPECIES) if SPECIES[k][0] == 'hw' and not k.startswith('D_')]
UNDERSTORY_IDS = [i for i, k in enumerate(SPECIES) if k.startswith('D_')]
CONIFER_IDS = [i for i, k in enumerate(SPECIES) if SPECIES[k][0] == 'cf']


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
    img = bpy.data.images.load(os.path.join(ROOT, f'assets/foliage/card_{card}.png'), check_existing=True)
    nimg = bpy.data.images.load(os.path.join(ROOT, f'assets/foliage/card_{card}_nrm.png'), check_existing=True)
    nimg.colorspace_settings.name = 'Non-Color'
    tex = N.new('ShaderNodeTexImage'); tex.image = img
    ntex = N.new('ShaderNodeTexImage'); ntex.image = nimg
    nm = N.new('ShaderNodeNormalMap'); nm.inputs['Strength'].default_value = 0.6
    L.new(ntex.outputs['Color'], nm.inputs['Color'])
    tint = N.new('ShaderNodeAttribute'); tint.attribute_type = 'INSTANCER'; tint.attribute_name = 'tint'
    # card is near-neutral olive: multiply by tint*2.2 so tint (linear) sets hue/value
    sc = N.new('ShaderNodeVectorMath'); sc.operation = 'SCALE'; sc.inputs['Scale'].default_value = 4.6
    L.new(tint.outputs['Color'], sc.inputs[0])
    mul = N.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs['Factor'].default_value = 1.0
    L.new(tex.outputs['Color'], mul.inputs['A']); L.new(sc.outputs[0], mul.inputs['B'])
    bsdf = N.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.7
    L.new(mul.outputs['Result'], bsdf.inputs['Base Color'])
    L.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
    tr = N.new('ShaderNodeBsdfTranslucent')
    L.new(mul.outputs['Result'], tr.inputs['Color'])
    mixs = N.new('ShaderNodeMixShader'); mixs.inputs['Fac'].default_value = 0.28
    L.new(bsdf.outputs[0], mixs.inputs[1]); L.new(tr.outputs[0], mixs.inputs[2])
    tp = N.new('ShaderNodeBsdfTransparent')
    cut = N.new('ShaderNodeMath'); cut.operation = 'GREATER_THAN'; cut.inputs[1].default_value = 0.45
    L.new(tex.outputs['Alpha'], cut.inputs[0])
    ma = N.new('ShaderNodeMixShader')
    L.new(cut.outputs[0], ma.inputs['Fac']); L.new(tp.outputs[0], ma.inputs[1]); L.new(mixs.outputs[0], ma.inputs[2])
    L.new(ma.outputs[0], out.inputs['Surface'])
    m.blend_method = 'CLIP' if hasattr(m, 'blend_method') else None
    return m


def bark_material():
    m = bpy.data.materials.get('MAT_Bark_PBR')
    if m:
        return m
    m = bpy.data.materials.new('MAT_Bark_PBR'); m.use_nodes = True
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    p = os.path.join(ROOT, 'assets/external/polyhaven/bark_brown_02/textures')
    diff = [f for f in os.listdir(p) if 'diff' in f] if os.path.isdir(p) else []
    if diff:
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = bpy.data.images.load(os.path.join(p, diff[0]), check_existing=True)
        uv = nt.nodes.new('ShaderNodeUVMap'); mp = nt.nodes.new('ShaderNodeMapping'); mp.inputs['Scale'].default_value = (1, 4, 1)
        nt.links.new(uv.outputs[0], mp.inputs[0]); nt.links.new(mp.outputs[0], t.inputs[0])
        nt.links.new(t.outputs['Color'], bsdf.inputs['Base Color'])
    else:
        bsdf.inputs['Base Color'].default_value = (0.05, 0.04, 0.03, 1)
    bsdf.inputs['Roughness'].default_value = 0.9
    return m


def build_prototypes(coll, seed=7):
    rng = np.random.default_rng(seed)
    bark = bark_material()
    objs = []
    for name, (kind, kw, card) in SPECIES.items():
        me = hardwood(name, rng, **kw) if kind == 'hw' else conifer(name, rng, **kw)
        me.materials.append(bark); me.materials.append(foliage_material(card))
        ob = bpy.data.objects.new(name, me)
        coll.objects.link(ob)
        objs.append(ob)
    return objs
