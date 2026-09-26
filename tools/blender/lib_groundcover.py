"""Ground cover v2: procedural grasses, weeds and wildflowers + CC0 Poly Haven ferns, instanced
by Geometry Nodes around the ACTIVE CAMERA (works for every camera, no per-camera setup).

Plant set (southern Appalachia, early autumn):
  pasture grass (golden-green, 0.35-0.7 m), broomsedge (orange-tan fallow bunchgrass, 0.6-1 m),
  hay stubble / mown lawn tufts, roadside weeds, goldenrod (yellow plumes), asters (purple /
  white), Queen Anne's lace (white umbels), chicory (blue), rushes / sedges on creek banks,
  ferns + small forest-floor plants under the canopy.
Placement is driven by per-vertex terrain attributes written by lib_materials.add_terrain_attributes:
  lu_a (field, meadow, developed, rock), lu_b (forest, bank, shoulder),
  eco_a (moisture, disturbed, canopy, hedge), eco_b (pasture, hay, plowed, fallow), eco_c (lawn, angle, tpi, talus).
Density is full within ~25 m of the camera and fades out by GC_RADIUS (in front of the camera,
plus a small ring around it); beyond that the terrain material carries the look.
"""
import bpy, os, math
import numpy as np
from mathutils import Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PH = os.path.join(ROOT, 'assets/external/polyhaven')
GC_RADIUS = 150.0


# ---------------------------------------------------------------- procedural plant meshes
class M:
    def __init__(self):
        self.V, self.F, self.C, self.N = [], [], [], []

    def quad_strip(self, pts, widths, side, col0, col1, up_bias=0.65):
        """Blade / stem ribbon along pts with per-point width, colour gradient col0->col1."""
        base = len(self.V); n = len(pts)
        for i, (p, w) in enumerate(zip(pts, widths)):
            t = i / (n - 1)
            c = tuple(col0[k] + (col1[k] - col0[k]) * t for k in range(3))
            nn = (side.cross((pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized())).normalized()
            if nn.z < 0:
                nn = -nn
            nn = (nn * (1 - up_bias) + Vector((0, 0, 1)) * up_bias).normalized()
            for s in (-1, 1):
                self.V.append(tuple(p + side * (w * 0.5 * s))); self.C.append(c + (1,)); self.N.append(tuple(nn))
        for i in range(n - 1):
            a = base + i * 2
            self.F.append((a, a + 1, a + 3, a + 2))

    def blob(self, c, r, col, n=6):
        """small flat disc (flower head / floret cluster)."""
        base = len(self.V)
        self.V.append(tuple(c)); self.C.append(col + (1,)); self.N.append((0, 0, 1))
        for k in range(n):
            a = 2 * math.pi * k / n
            self.V.append((c.x + math.cos(a) * r, c.y + math.sin(a) * r, c.z + (0.15 * r if k % 2 else -0.1 * r)))
            self.C.append(col + (1,)); self.N.append((0, 0, 1))
        for k in range(n):
            self.F.append((base, base + 1 + k, base + 1 + (k + 1) % n))

    def mesh(self, name):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.V, [], self.F)
        me.polygons.foreach_set('use_smooth', np.ones(len(self.F), bool))
        ca = me.color_attributes.new('col', 'FLOAT_COLOR', 'POINT')
        ca.data.foreach_set('color', np.asarray(self.C, np.float32).ravel())
        me.update()
        me.normals_split_custom_set_from_vertices(np.asarray(self.N, np.float32))
        return me


def blade(m, rng, root, h, w, lean, col0, col1, segs=4):
    az = rng.uniform(0, 2 * math.pi)
    d = Vector((math.cos(az), math.sin(az), 0))
    side = Vector((-d.y, d.x, 0)).normalized()
    side = (side + Vector((0, 0, rng.uniform(-0.3, 0.3)))).normalized()
    pts = []
    for i in range(segs + 1):
        t = i / segs
        pts.append(root + d * (lean * h * t * t) + Vector((0, 0, h * t * (1 - 0.25 * lean * t))))
    m.quad_strip(pts, [w * (1 - t / (segs + 0.3)) for t in range(segs + 1)], side, col0, col1)


def grass_clump(rng, n=40, h=(0.35, 0.7), w=0.012, r=0.18, lean=(0.1, 0.5), col0=(0.06, 0.07, 0.025), col1=(0.34, 0.27, 0.09), seed_heads=0.0):
    m = M()
    for i in range(n):
        a = rng.uniform(0, 2 * math.pi); rr = r * math.sqrt(rng.random())
        root = Vector((math.cos(a) * rr, math.sin(a) * rr, -0.02))
        hh = rng.uniform(*h)
        c1 = tuple(v * rng.uniform(0.8, 1.15) for v in col1)
        blade(m, rng, root, hh, w * rng.uniform(0.7, 1.3), rng.uniform(*lean), col0, c1)
        if seed_heads and rng.random() < seed_heads:  # thin seed stalk with a small head
            top = root + Vector((rng.normal(0, 0.05), rng.normal(0, 0.05), hh * 1.25))
            side = Vector((1, 0, 0))
            m.quad_strip([root, (root + top) / 2, top], [0.004, 0.003, 0.002], side, col0, c1)
            m.blob(top, 0.02, tuple(v * 0.9 for v in c1), 5)
    return m


def forb(rng, kind):
    """Wildflowers / weeds: stems with leaves + coloured heads."""
    m = M()
    stem0, stem1 = (0.05, 0.07, 0.025), (0.1, 0.12, 0.04)
    if kind == 'goldenrod':
        for s in range(int(rng.integers(3, 7))):
            a = rng.uniform(0, 6.28); root = Vector((math.cos(a) * 0.08, math.sin(a) * 0.08, -0.02))
            h = rng.uniform(0.6, 1.1); bend = rng.uniform(0.1, 0.3)
            pts = [root + Vector((math.cos(a) * bend * h * t * t, math.sin(a) * bend * h * t * t, h * t)) for t in np.linspace(0, 1, 6)]
            m.quad_strip(pts, [0.012] * 6, Vector((-math.sin(a), math.cos(a), 0)), stem0, stem1)
            # plume: many tiny floret clusters on short side branches of the arching tip (fuzzy panicle)
            for k in range(28):
                t = rng.uniform(0.62, 1.0)
                p = pts[0].lerp(pts[-1], t) + Vector((math.cos(a) * 0.08 * t, math.sin(a) * 0.08 * t, 0))
                off = Vector((rng.normal(0, 0.035), rng.normal(0, 0.035), rng.normal(0, 0.02) - 0.02 * (t - 0.6)))
                m.blob(p + off, rng.uniform(0.008, 0.016), tuple(c * rng.uniform(0.8, 1.1) for c in (0.55, 0.38, 0.03)), 5)
            for t in np.linspace(0.15, 0.6, 5):  # leaves
                p = pts[0].lerp(pts[-1], t); la = rng.uniform(0, 6.28)
                m.quad_strip([p, p + Vector((math.cos(la) * 0.08, math.sin(la) * 0.08, 0.02)), p + Vector((math.cos(la) * 0.14, math.sin(la) * 0.14, 0.0))],
                             [0.004, 0.022, 0.002], Vector((0, 0, 1)).cross(Vector((math.cos(la), math.sin(la), 0))), stem1, stem1)
    elif kind in ('aster', 'aster_w'):
        col = (0.26, 0.14, 0.42) if kind == 'aster' else (0.62, 0.6, 0.55)
        for s in range(int(rng.integers(5, 10))):
            a = rng.uniform(0, 6.28); h = rng.uniform(0.3, 0.6)
            root = Vector((math.cos(a) * 0.05, math.sin(a) * 0.05, -0.02))
            top = root + Vector((math.cos(a) * 0.12, math.sin(a) * 0.12, h))
            m.quad_strip([root, root.lerp(top, 0.5), top], [0.008, 0.006, 0.004], Vector((-math.sin(a), math.cos(a), 0)), stem0, stem1)
            for k in range(int(rng.integers(4, 9))):
                p = top + Vector((rng.normal(0, 0.06), rng.normal(0, 0.06), rng.normal(0, 0.04)))
                m.blob(p, 0.018, col, 6)
                m.blob(p + Vector((0, 0, 0.002)), 0.006, (0.5, 0.35, 0.03), 4)
    elif kind == 'qalace':
        for s in range(int(rng.integers(2, 5))):
            a = rng.uniform(0, 6.28); h = rng.uniform(0.5, 0.95)
            root = Vector((math.cos(a) * 0.05, math.sin(a) * 0.05, -0.02))
            top = root + Vector((rng.normal(0, 0.06), rng.normal(0, 0.06), h))
            m.quad_strip([root, root.lerp(top, 0.5), top], [0.007, 0.006, 0.005], Vector((1, 0, 0)), stem0, stem1)
            m.blob(top, rng.uniform(0.05, 0.075), (0.66, 0.64, 0.58), 10)
    elif kind == 'chicory':
        for s in range(int(rng.integers(2, 5))):
            a = rng.uniform(0, 6.28); h = rng.uniform(0.4, 0.8)
            root = Vector((math.cos(a) * 0.05, math.sin(a) * 0.05, -0.02))
            top = root + Vector((rng.normal(0, 0.12), rng.normal(0, 0.12), h))
            pts = [root.lerp(top, t) for t in np.linspace(0, 1, 5)]
            m.quad_strip(pts, [0.006] * 5, Vector((1, 0, 0)), stem0, stem1)
            for p in pts[2:]:
                m.blob(p + Vector((0.02, 0, 0)), 0.02, (0.18, 0.26, 0.6), 6)
    elif kind == 'weed':  # broadleaf roadside weed (plantain / dock / ironweed rosette)
        for k in range(int(rng.integers(6, 11))):
            la = rng.uniform(0, 6.28); L = rng.uniform(0.12, 0.25)
            d = Vector((math.cos(la), math.sin(la), 0))
            pts = [Vector((0, 0, 0.0)), d * L * 0.5 + Vector((0, 0, 0.08)), d * L + Vector((0, 0, 0.04))]
            m.quad_strip(pts, [0.01, 0.07, 0.01], Vector((0, 0, 1)).cross(d).normalized(), stem0, (0.11, 0.13, 0.04))
        if rng.random() < 0.6:  # seed stalk (dock: rusty brown)
            h = rng.uniform(0.5, 0.9)
            m.quad_strip([Vector((0, 0, 0)), Vector((0.02, 0, h * 0.5)), Vector((0.03, 0, h))], [0.012, 0.01, 0.02], Vector((1, 0, 0)), (0.12, 0.07, 0.03), (0.3, 0.12, 0.04))
    return m


def build_protos(coll):
    """Returns {layer: (collection, [objects])}."""
    rng = np.random.default_rng(21)
    specs = {
        'pasture': [grass_clump(rng, 55, (0.35, 0.7), 0.011, 0.2, (0.15, 0.5), (0.045, 0.065, 0.02), c, 0.15)
                    for c in ((0.26, 0.24, 0.075), (0.20, 0.22, 0.065), (0.30, 0.26, 0.085), (0.17, 0.20, 0.055))],
        'broomsedge': [grass_clump(rng, 45, (0.55, 1.0), 0.009, 0.14, (0.05, 0.25), (0.12, 0.08, 0.03), c, 0.2)
                       for c in ((0.42, 0.2, 0.07), (0.38, 0.24, 0.08), (0.45, 0.26, 0.1))],
        'short': [grass_clump(rng, 40, (0.06, 0.16), 0.008, 0.15, (0.1, 0.6), (0.04, 0.06, 0.02), c)
                  for c in ((0.12, 0.16, 0.05), (0.16, 0.17, 0.06), (0.1, 0.14, 0.04))],
        'stubble': [grass_clump(rng, 50, (0.08, 0.18), 0.006, 0.2, (0.0, 0.2), (0.12, 0.1, 0.04), c)
                    for c in ((0.38, 0.3, 0.13), (0.3, 0.26, 0.1))],
        'rush': [grass_clump(rng, 35, (0.5, 1.0), 0.007, 0.1, (0.02, 0.15), (0.03, 0.05, 0.02), c)
                 for c in ((0.07, 0.1, 0.035), (0.1, 0.11, 0.04))],
        'forest_grass': [grass_clump(rng, 25, (0.15, 0.35), 0.008, 0.15, (0.3, 0.8), (0.04, 0.05, 0.02), c)
                         for c in ((0.1, 0.12, 0.04), (0.2, 0.15, 0.05))],
        'flowers': [forb(rng, k) for k in ('goldenrod', 'goldenrod', 'aster', 'aster_w', 'qalace', 'chicory')],
        'weeds': [forb(rng, 'weed') for _ in range(3)],
    }
    mat = plant_material()
    out = {}
    for key, meshes in specs.items():
        c = bpy.data.collections.new(f'GC_{key}'); coll.children.link(c)
        objs = []
        for i, m in enumerate(meshes):
            me = m.mesh(f'GC_{key}_{i}')
            me.materials.append(mat)
            o = bpy.data.objects.new(f'GC_{key}_{i}', me); c.objects.link(o)
            objs.append(o)
        out[key] = (c, objs)
    # ferns (Poly Haven fern_02)
    p = os.path.join(PH, 'fern_02', 'fern_02_1k.blend')
    if os.path.exists(p):
        with bpy.data.libraries.load(p, link=False) as (df, dt):
            dt.objects = [n for n in df.objects]
        c = bpy.data.collections.new('GC_fern'); coll.children.link(c)
        objs = []
        for o in dt.objects:
            if o is not None and o.type == 'MESH':
                o.parent = None; o.location = (0, 0, 0); c.objects.link(o); objs.append(o)
                for mm in o.data.materials:  # autumn: ferns bronze-tinged
                    _tint_fern(mm)
        out['fern'] = (c, objs)
    for c, _ in out.values():
        c.hide_render = True; c.hide_viewport = True
    try:
        import lib_atmosphere
        if bpy.data.node_groups.get('A3_Ambient'):
            lib_atmosphere.apply_ambient()
    except Exception as e:
        print('  ground cover ambient skipped', e)
    return out


def _tint_fern(mat):
    if mat is None or not mat.use_nodes or mat.get('a3'):
        return
    nt = mat.node_tree
    b = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if b is None or not b.inputs['Base Color'].is_linked:
        return
    src = b.inputs['Base Color'].links[0].from_socket
    oi = nt.nodes.new('ShaderNodeObjectInfo')
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = (0.8, 0.95, 0.7, 1); ramp.color_ramp.elements[1].color = (1.6, 1.0, 0.45, 1)
    nt.links.new(oi.outputs['Random'], ramp.inputs['Fac'])
    mix.inputs['Factor'].default_value = 1.0
    nt.links.new(src, mix.inputs['A']); nt.links.new(ramp.outputs['Color'], mix.inputs['B'])
    nt.links.new(mix.outputs['Result'], b.inputs['Base Color'])
    mat['a3'] = True


def plant_material():
    m = bpy.data.materials.get('MAT_GC_Plant')
    if m:
        return m
    m = bpy.data.materials.new('MAT_GC_Plant'); m.use_nodes = True
    nt = m.node_tree; N, L = nt.nodes, nt.links
    for n in list(N):
        N.remove(n)
    out = N.new('ShaderNodeOutputMaterial')
    col = N.new('ShaderNodeAttribute'); col.attribute_name = 'col'
    oi = N.new('ShaderNodeObjectInfo')
    geo = N.new('ShaderNodeNewGeometry')
    # field-scale colour patchiness (world space) + per-instance jitter
    nz = N.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 0.08; nz.inputs['Detail'].default_value = 2
    L.new(geo.outputs['Position'], nz.inputs['Vector'])
    hs = N.new('ShaderNodeHueSaturation')
    hue = N.new('ShaderNodeMapRange'); hue.inputs['To Min'].default_value = 0.47; hue.inputs['To Max'].default_value = 0.53
    L.new(nz.outputs['Fac'], hue.inputs['Value']); L.new(hue.outputs['Result'], hs.inputs['Hue'])
    val = N.new('ShaderNodeMapRange'); val.inputs['To Min'].default_value = 0.8; val.inputs['To Max'].default_value = 1.2
    L.new(oi.outputs['Random'], val.inputs['Value']); L.new(val.outputs['Result'], hs.inputs['Value'])
    L.new(col.outputs['Color'], hs.inputs['Color'])
    b = N.new('ShaderNodeBsdfPrincipled'); b.inputs['Roughness'].default_value = 0.65; b.inputs['Specular IOR Level'].default_value = 0.3
    L.new(hs.outputs['Color'], b.inputs['Base Color'])
    tr = N.new('ShaderNodeBsdfTranslucent')
    warm = N.new('ShaderNodeMix'); warm.data_type = 'RGBA'; warm.blend_type = 'MULTIPLY'; warm.inputs['Factor'].default_value = 1
    warm.inputs['B'].default_value = (1.4, 1.25, 0.7, 1)
    L.new(hs.outputs['Color'], warm.inputs['A']); L.new(warm.outputs['Result'], tr.inputs['Color'])
    ms = N.new('ShaderNodeMixShader'); ms.inputs['Fac'].default_value = 0.35
    L.new(b.outputs[0], ms.inputs[1]); L.new(tr.outputs[0], ms.inputs[2])
    L.new(ms.outputs[0], out.inputs['Surface'])
    return m


# ---------------------------------------------------------------- GN scatter
def scatter_modifier(terrain_objs, protos, cam_locs=None, grass_radius=GC_RADIUS):
    """One GN tree reused by all terrain chunks; follows the scene's active camera."""
    R = grass_radius
    ng = bpy.data.node_groups.new('GN_GroundCover', 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    N, L = ng.nodes, ng.links
    gi = N.new('NodeGroupInput'); go = N.new('NodeGroupOutput')
    join = N.new('GeometryNodeJoinGeometry')
    L.new(gi.outputs[0], join.inputs[0]); L.new(join.outputs[0], go.inputs[0])

    def math(op, a, b=None):
        m = N.new('ShaderNodeMath'); m.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                L.new(v, m.inputs[i])
        return m.outputs[0]

    def attr(name, comp):
        a = N.new('GeometryNodeInputNamedAttribute'); a.data_type = 'FLOAT_COLOR'; a.inputs['Name'].default_value = name
        sep = N.new('FunctionNodeSeparateColor'); L.new(a.outputs['Attribute'], sep.inputs[0])
        return sep.outputs[comp]

    def smooth(x, lo, hi):
        mr = N.new('ShaderNodeMapRange'); mr.interpolation_type = 'SMOOTHSTEP'
        mr.inputs['From Min'].default_value = lo; mr.inputs['From Max'].default_value = hi
        L.new(x, mr.inputs['Value'])
        return mr.outputs['Result']

    # camera
    ac = N.new('GeometryNodeInputActiveCamera')
    oi = N.new('GeometryNodeObjectInfo'); oi.transform_space = 'ORIGINAL'
    L.new(ac.outputs[0], oi.inputs['Object'])
    vr = N.new('ShaderNodeVectorRotate'); vr.rotation_type = 'EULER_XYZ'
    vr.inputs['Vector'].default_value = (0, 0, -1)
    L.new(oi.outputs['Rotation'], vr.inputs['Rotation'])
    fsep = N.new('ShaderNodeSeparateXYZ'); L.new(vr.outputs[0], fsep.inputs[0])
    fcomb = N.new('ShaderNodeCombineXYZ'); L.new(fsep.outputs['X'], fcomb.inputs['X']); L.new(fsep.outputs['Y'], fcomb.inputs['Y'])
    fnorm = N.new('ShaderNodeVectorMath'); fnorm.operation = 'NORMALIZE'; L.new(fcomb.outputs[0], fnorm.inputs[0])

    def cam_terms():
        """(horizontal distance, frontal factor) fields for the current domain."""
        pos = N.new('GeometryNodeInputPosition')
        d = N.new('ShaderNodeVectorMath'); d.operation = 'SUBTRACT'
        L.new(pos.outputs[0], d.inputs[0]); L.new(oi.outputs['Location'], d.inputs[1])
        ds = N.new('ShaderNodeSeparateXYZ'); L.new(d.outputs[0], ds.inputs[0])
        dc = N.new('ShaderNodeCombineXYZ'); L.new(ds.outputs['X'], dc.inputs['X']); L.new(ds.outputs['Y'], dc.inputs['Y'])
        ln = N.new('ShaderNodeVectorMath'); ln.operation = 'LENGTH'; L.new(dc.outputs[0], ln.inputs[0])
        nd = N.new('ShaderNodeVectorMath'); nd.operation = 'NORMALIZE'; L.new(dc.outputs[0], nd.inputs[0])
        dt = N.new('ShaderNodeVectorMath'); dt.operation = 'DOT_PRODUCT'; L.new(nd.outputs[0], dt.inputs[0]); L.new(fnorm.outputs[0], dt.inputs[1])
        return ln.outputs['Value'], dt.outputs['Value']

    # face selection: within R and in front (or within 12 m all around)
    dist_f, dot_f = cam_terms()
    sel_face = math('MULTIPLY', math('LESS_THAN', dist_f, R),
                    math('MAXIMUM', math('GREATER_THAN', dot_f, 0.25), math('LESS_THAN', dist_f, 14.0)))
    # density falloff for points: 1 near, 0 at R (quadratic)
    dist_p, dot_p = cam_terms()
    fall = math('POWER', math('MAXIMUM', math('SUBTRACT', 1.0, math('DIVIDE', math('MAXIMUM', math('SUBTRACT', dist_p, 22.0), 0.0), R - 22.0)), 0.0), 2.0)

    field = attr('lu_a', 'Red'); meadow = attr('lu_a', 'Green'); dev = attr('lu_a', 'Blue'); rock = attr('lu_a', 'Alpha')
    forest = attr('lu_b', 'Red'); bank = attr('lu_b', 'Green'); shoulder = attr('lu_b', 'Blue')
    moist = attr('eco_a', 'Red'); disturbed = attr('eco_a', 'Green'); canopy = attr('eco_a', 'Blue'); hedge = attr('eco_a', 'Alpha')
    pasture = attr('eco_b', 'Red'); hay = attr('eco_b', 'Green'); plowed = attr('eco_b', 'Blue'); fallow = attr('eco_b', 'Alpha')
    lawn = attr('eco_c', 'Red')
    road_m = math('MULTIPLY', attr('eco_d', 'Red'), 25.5)      # distance to the paved shoulder edge (m)
    rail_m = math('MULTIPLY', attr('eco_d', 'Green'), 25.5)    # distance to ballast / bridge deck edge (m)
    # nothing on pavement / ballast / decks; low grass from 0.5 m, tall plants from ~1.5 m
    offroad = math('MULTIPLY', smooth(road_m, 0.5, 1.6), smooth(rail_m, 0.8, 2.0))
    offroad_tall = math('MULTIPLY', smooth(road_m, 1.2, 2.5), smooth(rail_m, 1.5, 3.0))
    under = smooth(canopy, 0.35, 0.75)
    open_ = math('MULTIPLY', math('SUBTRACT', 1.0, under), offroad_tall)
    open_low = math('MULTIPLY', math('SUBTRACT', 1.0, under), offroad)
    verge = math('MULTIPLY', math('SUBTRACT', 1.0, smooth(road_m, 5.0, 9.0)), offroad_tall)   # roadside verge band (weeds, tall grass)

    def layer(key, density, weight, scale=(0.8, 1.25), seed=0, align=0.3):
        if key not in protos:
            return
        c, objs = protos[key]
        dist = N.new('GeometryNodeDistributePointsOnFaces'); dist.distribute_method = 'RANDOM'
        dist.inputs['Density'].default_value = density; dist.inputs['Seed'].default_value = seed
        # faces: near the camera AND where this layer can grow at all (keeps point counts small)
        L.new(gi.outputs[0], dist.inputs['Mesh']); L.new(math('MULTIPLY', sel_face, math('GREATER_THAN', weight, 0.02)), dist.inputs['Selection'])
        # keep point if random < weight * falloff * flatness (normal from the distribution)
        dn = N.new('ShaderNodeSeparateXYZ'); L.new(dist.outputs['Normal'], dn.inputs[0])
        flat = smooth(dn.outputs['Z'], 0.72, 0.9)
        rv = N.new('FunctionNodeRandomValue'); rv.data_type = 'FLOAT'; rv.inputs['Seed'].default_value = seed + 7
        keep = math('LESS_THAN', rv.outputs[1], math('MULTIPLY', math('MULTIPLY', weight, fall), flat))
        dl = N.new('GeometryNodeDeleteGeometry'); dl.domain = 'POINT'
        L.new(dist.outputs['Points'], dl.inputs['Geometry']); L.new(math('SUBTRACT', 1.0, keep), dl.inputs['Selection'])
        ci = N.new('GeometryNodeCollectionInfo'); ci.inputs['Collection'].default_value = c
        ci.inputs['Separate Children'].default_value = True; ci.inputs['Reset Children'].default_value = True
        iop = N.new('GeometryNodeInstanceOnPoints'); iop.inputs['Pick Instance'].default_value = True
        ri = N.new('FunctionNodeRandomValue'); ri.data_type = 'INT'; ri.inputs[4].default_value = 0; ri.inputs[5].default_value = len(objs) - 1; ri.inputs['Seed'].default_value = seed + 1
        rs = N.new('FunctionNodeRandomValue'); rs.data_type = 'FLOAT'; rs.inputs[2].default_value = scale[0]; rs.inputs[3].default_value = scale[1]; rs.inputs['Seed'].default_value = seed + 2
        rr = N.new('FunctionNodeRandomValue'); rr.data_type = 'FLOAT_VECTOR'; rr.inputs[0].default_value = (-0.06, -0.06, 0); rr.inputs[1].default_value = (0.06, 0.06, 6.283); rr.inputs['Seed'].default_value = seed + 3
        L.new(dl.outputs[0], iop.inputs['Points']); L.new(ci.outputs[0], iop.inputs['Instance'])
        L.new(ri.outputs[2], iop.inputs['Instance Index']); L.new(rs.outputs[1], iop.inputs['Scale']); L.new(rr.outputs[0], iop.inputs['Rotation'])
        L.new(iop.outputs[0], join.inputs[0])

    # fields
    tallgrass = math('MULTIPLY', math('MAXIMUM', math('MAXIMUM', pasture, math('MULTIPLY', fallow, 0.6)), math('MULTIPLY', meadow, 0.8)), open_)
    layer('pasture', 5.0, math('MAXIMUM', tallgrass, math('MULTIPLY', verge, 0.8)), (0.8, 1.3), 10)
    layer('pasture', 1.2, math('MULTIPLY', hedge, offroad_tall), (1.0, 1.5), 11)
    layer('broomsedge', 3.0, math('MULTIPLY', fallow, open_), (0.8, 1.3), 20)
    layer('stubble', 4.0, math('MULTIPLY', hay, open_), (0.9, 1.3), 30)
    layer('short', 3.0, math('MULTIPLY', math('MAXIMUM', lawn, math('MULTIPLY', dev, 0.7)), open_low), (0.8, 1.4), 40)
    layer('short', 1.2, math('MULTIPLY', math('MULTIPLY', plowed, 0.25), open_), (0.8, 1.2), 41)
    # wildflowers / weeds: fallow fields, verges, hedges, pasture edges
    fl = math('MAXIMUM', math('MAXIMUM', math('MULTIPLY', fallow, 0.8), math('MULTIPLY', pasture, 0.2)), math('MAXIMUM', verge, hedge))
    layer('flowers', 0.35, math('MULTIPLY', fl, open_), (0.8, 1.2), 50)
    layer('weeds', 0.3, math('MULTIPLY', math('MAXIMUM', math('MAXIMUM', verge, disturbed), math('MULTIPLY', pasture, 0.3)), open_), (0.8, 1.3), 60)
    # creek banks / wet ground: rushes + ferns
    wet = math('MULTIPLY', math('MAXIMUM', bank, smooth(moist, 0.7, 0.95)), offroad_tall)
    layer('rush', 1.6, wet, (0.8, 1.3), 70)
    # forest floor: ferns (moist), sparse grass, in canopy gaps more
    layer('fern', 0.35, math('MULTIPLY', math('MULTIPLY', under, math('ADD', 0.35, moist)), offroad_tall), (1.0, 1.8), 80)
    layer('forest_grass', 0.25, math('MULTIPLY', math('MULTIPLY', forest, math('SUBTRACT', 1.1, canopy)), offroad), (0.8, 1.2), 90)
    for t in terrain_objs:
        m = t.modifiers.get('GroundCover') or t.modifiers.new('GroundCover', 'NODES'); m.node_group = ng
    return ng


def load_protos(coll):
    return build_protos(coll)
