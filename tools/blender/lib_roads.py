"""Road, bridge and railway geometry for the Blender world (Agent 2: roads & infrastructure).

Built entirely from the canonical data (never from the map image):
    data/roads/roads.geojson        edges [x_px, y_px, z_m] junction-to-junction (graded profiles)
    data/roads/road_nodes.geojson   junctions
    data/roads/road_types.json      classes + engineered cross-sections ('section')
    data/roads/bridges.geojson      bridge / viaduct / overpass spans
    data/railways/railways.geojson  railway centrelines (graded)
    data/terrain/height_graded_f32.bin  graded terrain (the road beds / ditches are cut into it)

Geometry
  * Every edge is swept with its engineered cross-section: crowned carriageway (2-2.5 %),
    superelevation on curves, paved shoulders, gravel shoulders + side ditch + backslope
    (cut) or embankment slope (fill) that end in a skirt tucked under the terrain; curbs
    and sidewalks in town cores; bridge decks get a safety curb (the structure object adds
    parapets). Sections come from road_types.json so grading.py and this file agree.
  * Junctions are trimmed legs + one merged pavement polygon per node with curb returns
    (quadratic fillets between the adjacent legs' edge lines), plus a curb/sidewalk or
    gravel apron ring around the corners. No overlapping discs.
  * Markings are procedural in the asphalt shader, driven by per-vertex attributes
    (lateral offset, station, scheme, no-passing zones, stop bars, crosswalks), so they are
    crisp at driver eye height, never z-fight, and wear like 1974 paint.
  * Bridges by class: concrete girder (deck, girders, hammerhead piers to the river bed,
    abutments + wingwalls, parapet with steel rail), steel through truss, concrete slab,
    girder overpasses (median pier, slope paving), tall-pier viaducts; railway deck girders.
  * Railways: ballast prism, instanced ties, rail profiles.

Object names keep world ids so assets can be swapped: ROAD <id>, JUNCTION <node id>,
BRIDGE <bridge id>, RAIL <rail id>.

Entry point (build_world.py):  lib_roads.build(root_collection, ctx)
  ctx keys: T (height sampler with .at(px, py)), mesh_obj, collection, luw (optional
  land-use weights for the verge material), bbox (optional px box to limit the build).
"""
import bpy, json, math, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PH = os.path.join(ROOT, 'assets/external/polyhaven')
W, H, MPP, OX, OY = 2000, 667, 2.5, 1000.0, 333.5


def P(*p):
    return os.path.join(ROOT, *p)


def load(p):
    with open(P(p)) as f:
        return json.load(f)


def w2px(X, Y):
    return np.asarray(X) / MPP + OX, -np.asarray(Y) / MPP + OY


def px2w(x, y):
    return (np.asarray(x) - OX) * MPP, -(np.asarray(y) - OY) * MPP


# ------------------------------------------------------------------ mesh accumulator
class Mesh:
    """Accumulates verts / faces / per-face material / per-vertex float attributes."""

    def __init__(self, attrs=()):
        self.V, self.F, self.M = [], [], []
        self.n = 0
        self.attrs = {a: [] for a in attrs}

    def add(self, V, F, mats, **attrs):
        V = np.asarray(V, np.float64).reshape(-1, 3)
        if len(V) == 0 or len(F) == 0:
            return
        F = np.asarray(F, np.int64)
        self.V.append(V)
        self.F.append(F + self.n)
        self.M.append(np.broadcast_to(np.asarray(mats, np.int32), (len(F),)).copy())
        for a in self.attrs:
            v = attrs.get(a, 0.0)
            self.attrs[a].append(np.broadcast_to(np.asarray(v, np.float32), (len(V),)).copy())
        self.n += len(V)

    def grid(self, G, band_mats, **attrs):
        """G: (rows, cols, 3) grid -> quads between rows; band_mats: material per column band."""
        r, c, _ = G.shape
        if r < 2 or c < 2:
            return
        idx = np.arange(r * c).reshape(r, c)
        a = idx[:-1, :-1]; b = idx[:-1, 1:]; d = idx[1:, :-1]; e = idx[1:, 1:]
        F = np.stack([a, d, e, b], -1).reshape(-1, 4)
        # faces up (outward for caps): flip the winding if the summed normal points down
        Q = G.reshape(-1, 3)
        d1 = Q[F[:, 2]] - Q[F[:, 0]]; d2 = Q[F[:, 3]] - Q[F[:, 1]]
        if (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]).sum() < 0:
            F = F[:, ::-1]
        mats = np.broadcast_to(np.asarray(band_mats, np.int32)[None, :], (r - 1, c - 1)).reshape(-1)
        at = {k: (np.asarray(v).reshape(-1) if np.ndim(v) else v) for k, v in attrs.items()}
        self.add(G.reshape(-1, 3), F, mats, **at)

    def empty(self):
        return self.n == 0

    def to_object(self, name, coll, materials, smooth=True, uv_from_px=False, luw=None):
        V = np.vstack(self.V).astype(np.float32)
        tris = [f for f in self.F if f.shape[1] == 3]
        quads = [f for f in self.F if f.shape[1] == 4]
        mt = [m for f, m in zip(self.F, self.M) if f.shape[1] == 3]
        mq = [m for f, m in zip(self.F, self.M) if f.shape[1] == 4]
        faces = []
        mats = []
        if quads:
            faces.append(np.vstack(quads)); mats.append(np.concatenate(mq))
        if tris:
            T3 = np.vstack(tris)
            faces.append(T3); mats.append(np.concatenate(mt))
        me = bpy.data.meshes.new(name)
        me.vertices.add(len(V))
        me.vertices.foreach_set('co', V.ravel())
        loops = np.concatenate([f.ravel() for f in faces]).astype(np.int32)
        sizes = np.concatenate([np.full(len(f), f.shape[1], np.int32) for f in faces])
        starts = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int32)
        me.loops.add(len(loops))
        me.loops.foreach_set('vertex_index', loops)
        me.polygons.add(len(sizes))
        me.polygons.foreach_set('loop_start', starts)
        me.polygons.foreach_set('loop_total', sizes)
        me.update(calc_edges=True)
        me.polygons.foreach_set('material_index', np.concatenate(mats).astype(np.int32))
        if smooth:
            me.polygons.foreach_set('use_smooth', np.ones(len(sizes), bool))
        for a, vals in self.attrs.items():
            at = me.attributes.new(a, 'FLOAT', 'POINT')
            at.data.foreach_set('value', np.concatenate(vals).astype(np.float32))
        if uv_from_px:
            x, y = w2px(V[:, 0], V[:, 1])
            uv = me.uv_layers.new(name='UVMap')
            uvv = np.stack([x / W, 1 - y / H], 1)[loops]
            uv.data.foreach_set('uv', uvv.astype(np.float32).ravel())
        if luw is not None:
            x, y = w2px(V[:, 0], V[:, 1])
            xi = np.clip(x.astype(int), 0, W - 1); yi = np.clip(y.astype(int), 0, H - 1)
            A = luw[:, yi, xi].T.copy()  # field, meadow, developed, rock, forest, bank, shoulder
            A[:, 5] *= 0.15  # road fills / verges are grassed, not pale river bank
            for an, data in (('lu_a', np.c_[A[:, 0], A[:, 1], A[:, 2], A[:, 3]]), ('lu_b', np.c_[A[:, 4], A[:, 5], A[:, 6], np.ones(len(A))])):
                at = me.attributes.new(an, 'FLOAT_COLOR', 'POINT')
                at.data.foreach_set('color', data.astype(np.float32).ravel())
        for m in materials:
            me.materials.append(m)
        me.validate(clean_customdata=False)
        ob = bpy.data.objects.new(name, me)
        coll.objects.link(ob)
        return ob


# ------------------------------------------------------------------ shader helpers
class NB:
    """Tiny node-building DSL (links sockets, sets constants)."""

    def __init__(self, mat):
        mat.use_nodes = True
        self.nt = mat.node_tree
        for n in list(self.nt.nodes):
            self.nt.nodes.remove(n)
        self.out = self.nt.nodes.new('ShaderNodeOutputMaterial')
        self.bsdf = self.nt.nodes.new('ShaderNodeBsdfPrincipled')
        self.nt.links.new(self.bsdf.outputs[0], self.out.inputs['Surface'])
        self.tc = self.nt.nodes.new('ShaderNodeTexCoord')

    def _in(self, sock, v):
        if v is None:
            return
        if hasattr(v, 'is_output'):
            self.nt.links.new(v, sock)
        else:
            sock.default_value = v

    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.nt.nodes.new('ShaderNodeMath'); n.operation = op; n.use_clamp = clamp
        self._in(n.inputs[0], a)
        if b is not None:
            self._in(n.inputs[1], b)
        if c is not None:
            self._in(n.inputs[2], c)
        return n.outputs[0]

    def add(self, a, b): return self.math('ADD', a, b)
    def sub(self, a, b): return self.math('SUBTRACT', a, b)
    def mul(self, a, b): return self.math('MULTIPLY', a, b)
    def mx(self, a, b): return self.math('MAXIMUM', a, b)
    def mn(self, a, b): return self.math('MINIMUM', a, b)
    def absv(self, a): return self.math('ABSOLUTE', a)
    def fract(self, a): return self.math('FRACT', a)
    def lt(self, a, b): return self.math('LESS_THAN', a, b)
    def gt(self, a, b): return self.math('GREATER_THAN', a, b)

    def smooth(self, x, lo, hi):
        n = self.nt.nodes.new('ShaderNodeMapRange'); n.interpolation_type = 'SMOOTHSTEP'; n.clamp = True
        self._in(n.inputs['Value'], x); self._in(n.inputs['From Min'], lo); self._in(n.inputs['From Max'], hi)
        return n.outputs['Result']

    def attr(self, name):
        n = self.nt.nodes.new('ShaderNodeAttribute'); n.attribute_name = name; n.attribute_type = 'GEOMETRY'
        return n.outputs['Fac']

    def xyz(self, x, y, z=0.0):
        n = self.nt.nodes.new('ShaderNodeCombineXYZ')
        self._in(n.inputs[0], x); self._in(n.inputs[1], y); self._in(n.inputs[2], z)
        return n.outputs[0]

    def mix(self, fac, a, b, blend='MIX'):
        n = self.nt.nodes.new('ShaderNodeMix'); n.data_type = 'RGBA'; n.blend_type = blend; n.clamp_factor = True
        self._in(n.inputs['Factor'], fac)
        self._in(n.inputs['A'], a if not isinstance(a, tuple) else (a + (1.0,) if len(a) == 3 else a))
        self._in(n.inputs['B'], b if not isinstance(b, tuple) else (b + (1.0,) if len(b) == 3 else b))
        return n.outputs['Result']

    def mixf(self, fac, a, b):
        n = self.nt.nodes.new('ShaderNodeMix'); n.data_type = 'FLOAT'; n.clamp_factor = True
        self._in(n.inputs['Factor'], fac); self._in(n.inputs['A'], a); self._in(n.inputs['B'], b)
        return n.outputs['Result']

    def noise(self, vec, scale, detail=3.0, rough=0.55, out='Fac'):
        n = self.nt.nodes.new('ShaderNodeTexNoise'); n.inputs['Scale'].default_value = scale
        n.inputs['Detail'].default_value = detail; n.inputs['Roughness'].default_value = rough
        self._in(n.inputs['Vector'], vec)
        return n.outputs[out]

    def voronoi(self, vec, scale, feature='F1', out='Distance', rand=1.0):
        n = self.nt.nodes.new('ShaderNodeTexVoronoi'); n.feature = feature
        n.inputs['Scale'].default_value = scale
        if 'Randomness' in n.inputs:
            n.inputs['Randomness'].default_value = rand
        self._in(n.inputs['Vector'], vec)
        return n.outputs[out]

    def mapping(self, vec, scale, rot=0.0, loc=(0, 0, 0)):
        n = self.nt.nodes.new('ShaderNodeMapping')
        n.inputs['Scale'].default_value = (1 / scale,) * 3 if not isinstance(scale, tuple) else scale
        n.inputs['Rotation'].default_value = (0, 0, rot); n.inputs['Location'].default_value = loc
        self._in(n.inputs['Vector'], vec)
        return n.outputs['Vector']

    def tex(self, aid, kind, vec, color=True):
        d = os.path.join(PH, aid, 'textures')
        if not os.path.isdir(d):
            return None
        f = [x for x in os.listdir(d) if f'_{kind}_' in x]
        if not f:
            return None
        im = bpy.data.images.load(os.path.join(d, f[0]), check_existing=True)
        if not color:
            im.colorspace_settings.name = 'Non-Color'
        t = self.nt.nodes.new('ShaderNodeTexImage'); t.image = im
        self._in(t.inputs['Vector'], vec)
        return t

    def bw(self, col):
        n = self.nt.nodes.new('ShaderNodeRGBToBW'); self._in(n.inputs[0], col)
        return n.outputs[0]

    def bump(self, h, strength, dist=0.02, normal=None):
        n = self.nt.nodes.new('ShaderNodeBump'); n.inputs['Strength'].default_value = strength; n.inputs['Distance'].default_value = dist
        self._in(n.inputs['Height'], h)
        if normal is not None:
            self._in(n.inputs['Normal'], normal)
        return n.outputs['Normal']

    def inp(self, *names):
        for n in names:
            if n in self.bsdf.inputs:
                return self.bsdf.inputs[n]
        raise KeyError(names)


# ------------------------------------------------------------------ materials
def _asphalt_material():
    """One asphalt shader for every paved class. Attributes (per vertex):
    rl lateral offset (m, + = right of the edge direction), rs station (m), hw carriageway
    half width, mk marking scheme, esh paved-shoulder width, lw lane width, uin median
    half width (divided), np no-passing (0/1), age (0 fresh .. 1 old), surf (0 asphalt,
    1 chip seal), stop/xw distances for stop bars and crosswalks (sa, sb, xa, xb), len."""
    m = bpy.data.materials.new('MAT_Road_Asphalt')
    b = NB(m)
    obj = b.tc.outputs['Object']
    rl, rs, hw, mk = b.attr('rl'), b.attr('rs'), b.attr('hw'), b.attr('mk')
    esh, lw, uin, npz, age, surf = b.attr('esh'), b.attr('lw'), b.attr('uin'), b.attr('np'), b.attr('age'), b.attr('surf')
    au = b.absv(rl)
    road_uv = b.xyz(rl, rs)
    # --- base asphalt: texture detail x class tint
    t1 = b.tex('asphalt_02', 'diff', b.mapping(obj, 3.2))
    t2 = b.tex('asphalt_02', 'diff', b.mapping(obj, 11.0, 0.7))
    det = b.bw(t1.outputs['Color']) if t1 else 0.35
    det2 = b.bw(t2.outputs['Color']) if t2 else 0.35
    det = b.mixf(0.35, det, det2)
    det = b.math('POWER', b.mul(det, 2.6), 0.55)
    fresh = (0.030, 0.030, 0.032)
    old = (0.085, 0.083, 0.078)
    chip = (0.115, 0.100, 0.082)
    base = b.mix(age, fresh, old)
    base = b.mix(surf, base, chip)
    # large-scale weathering (sun-bleached patches, repairs of different ages)
    big = b.noise(obj, 0.035, 2.0)
    base = b.mix(b.mul(b.smooth(big, 0.45, 0.75), 0.35), base, b.mix(0.5, base, (0.13, 0.125, 0.115)))
    col = b.mix(1.0, base, b.xyz(det, det, det), 'MULTIPLY')
    # --- lanes: oil drip strip in each lane centre, polished wheel paths
    u_l = b.math('DIVIDE', b.mx(b.sub(au, uin), 0.0), b.mx(lw, 0.5))
    p = b.fract(u_l)
    oil = b.sub(1.0, b.smooth(b.absv(b.sub(p, 0.5)), 0.03, 0.16))
    oil = b.mul(oil, b.mul(b.smooth(b.noise(road_uv, 0.25, 2.0), 0.35, 0.7), b.lt(au, b.sub(hw, esh))))
    oil = b.mul(oil, b.sub(1.0, b.mul(surf, 0.6)))
    traffic = b.add(0.3, b.mul(b.gt(mk, 0.5), 0.7))   # busy roads show oil strips, quiet streets barely
    col = b.mix(b.mul(oil, b.mul(traffic, 0.35)), col, (0.016, 0.016, 0.017))
    wheel = b.sub(1.0, b.smooth(b.absv(b.sub(b.absv(b.sub(p, 0.5)), 0.27)), 0.05, 0.14))
    wheel = b.mul(wheel, b.lt(au, b.sub(hw, esh)))
    col = b.mix(b.mul(wheel, b.mul(age, 0.12)), col, b.mix(0.5, col, (0.2, 0.19, 0.17)))
    # --- rectangular patch repairs aligned with the road (u across, s along)
    pv = b.mapping(road_uv, (1 / 3.4, 1 / 8.5, 1.0))
    cell = b.voronoi(pv, 1.0, 'F1', 'Color')
    cr = b.nt.nodes.new('ShaderNodeSeparateColor'); b._in(cr.inputs[0], cell)
    patch = b.lt(cr.outputs['Red'], b.add(0.03, b.mul(age, 0.13)))
    patch = b.mul(patch, b.lt(au, b.sub(hw, 0.3)))
    col = b.mix(b.mul(patch, 0.6), col, b.mix(cr.outputs['Green'], (0.03, 0.03, 0.032), (0.05, 0.049, 0.046)))
    # --- cracks: alligator clusters, transverse cracks, sealed centre joint (tar snakes)
    # alligator cracking: small cells, in patches, mostly in the wheel paths of old roads
    ed = b.voronoi(b.mapping(road_uv, 0.32), 1.0, 'DISTANCE_TO_EDGE', 'Distance')
    cluster = b.smooth(b.noise(b.mapping(road_uv, 7.0), 1.0, 2.0), b.sub(0.78, b.mul(age, 0.16)), b.sub(0.86, b.mul(age, 0.14)))
    crack = b.mul(b.mul(b.lt(ed, 0.02), cluster), b.gt(age, 0.4))
    tr = b.fract(b.add(b.math('DIVIDE', rs, 13.7), b.mul(b.noise(road_uv, 0.6, 1.0), 0.15)))
    trans = b.mul(b.lt(tr, 0.006), b.gt(age, 0.35))
    joint = b.mul(b.lt(b.absv(b.add(rl, b.mul(b.sub(b.noise(road_uv, 0.8, 1.0), 0.5), 0.1))), 0.05), b.gt(age, 0.25))
    tar = b.mx(b.mx(crack, trans), b.mul(joint, b.lt(mk, 5.5)))
    tar = b.mul(tar, b.lt(au, b.sub(hw, 0.15)))
    col = b.mix(b.mul(tar, 0.7), col, (0.012, 0.012, 0.013))
    # --- pavement edge raveling (grey-brown aggregate)
    ravel = b.mul(b.smooth(au, b.sub(hw, 0.45), hw), b.smooth(b.noise(road_uv, 1.3, 3.0), 0.4, 0.62))
    col = b.mix(b.mul(ravel, 0.75), col, (0.12, 0.105, 0.085))
    # --- markings (1971 MUTCD: yellow centre lines, white edges / lane lines)
    aa = 0.012

    def band(u, c, w):  # 1 inside |u-c| < w/2 (anti-aliased)
        return b.sub(1.0, b.smooth(b.absv(b.sub(u, c)), b.sub(w * 0.5, aa), w * 0.5 + aa))

    def dash(on, period):
        return b.lt(b.fract(b.math('DIVIDE', rs, period)), on / period)

    def is_(code):
        return b.lt(b.absv(b.sub(mk, code)), 0.5)
    edge_u = b.sub(hw, esh)
    white = b.mul(band(au, edge_u, 0.12), b.mx(is_(1), b.mx(is_(6), is_(7))))
    # 2-lane rural / highway centre: double solid in no-passing zones, dashed otherwise
    dbl = b.mx(band(rl, -0.14, 0.1), band(rl, 0.14, 0.1))
    single_dash = b.mul(band(rl, 0.0, 0.1), dash(3.05, 12.2))
    c2 = b.mixf(npz, single_dash, dbl)
    yellow = b.mul(c2, b.mx(is_(1), is_(2)))
    # urban 2-lane: dashed yellow centre;  arterial / main street: double yellow
    yellow = b.mx(yellow, b.mul(b.mul(band(rl, 0.0, 0.1), dash(3.05, 9.1)), is_(3)))
    yellow = b.mx(yellow, b.mul(dbl, b.mx(is_(4), is_(5))))
    # multi-lane: dashed white lane lines
    white = b.mx(white, b.mul(b.mul(band(au, lw, 0.1), dash(3.05, 12.2)), is_(4)))
    # freeway: yellow at the median edge, dashed white between the two lanes
    white = b.mx(white, b.mul(b.mul(band(au, b.add(uin, lw), 0.1), dash(3.05, 12.2)), is_(6)))
    yellow = b.mx(yellow, b.mul(band(au, b.add(uin, 0.08), 0.1), is_(6)))
    # main street parking: T marks every 6.7 m in the parking lanes
    park = b.mul(b.mul(b.lt(b.fract(b.math('DIVIDE', rs, 6.7)), 0.02), b.gt(au, b.add(lw, 0.2))), is_(5))
    white = b.mx(white, b.mul(park, b.lt(au, b.sub(hw, 0.2))))
    # stop bars (right half of the approach) and crosswalks at junction mouths
    sa, sbb, xa, xb, ln = b.attr('sa'), b.attr('sb'), b.attr('xa'), b.attr('xb'), b.attr('len')
    eb = b.sub(ln, rs)
    stop_b = b.mul(b.mul(band(eb, b.add(sbb, 0.3), 0.45), b.gt(rl, 0.1)), b.gt(sbb, 0.01))
    stop_a = b.mul(b.mul(band(rs, b.add(sa, 0.3), 0.45), b.lt(rl, -0.1)), b.gt(sa, 0.01))
    xw_b = b.mul(b.mx(band(eb, b.add(xb, 0.2), 0.3), band(eb, b.add(xb, 2.8), 0.3)), b.gt(xb, 0.01))
    xw_a = b.mul(b.mx(band(rs, b.add(xa, 0.2), 0.3), band(rs, b.add(xa, 2.8), 0.3)), b.gt(xa, 0.01))
    white = b.mx(white, b.mul(b.mx(b.mx(stop_a, stop_b), b.mx(xw_a, xw_b)), b.lt(au, b.sub(hw, 0.3))))
    # paint wear: rural paint is thin and faded, highway paint fresher
    wear = b.add(b.mul(age, 0.55), b.mul(b.mx(is_(2), is_(3)), 0.25))
    wn = b.noise(b.mapping(road_uv, 0.6), 1.0, 4.0, 0.7)
    keep = b.smooth(b.add(wn, b.mul(b.noise(obj, 0.02, 1.0), 0.2)), b.mul(wear, 0.75), b.add(b.mul(wear, 0.75), 0.18))
    keep = b.mul(keep, b.sub(1.0, b.mul(tar, 0.8)))
    paint_w = b.mul(white, keep)
    paint_y = b.mul(yellow, keep)
    col = b.mix(paint_w, col, (0.62, 0.62, 0.58))
    col = b.mix(paint_y, col, (0.62, 0.38, 0.04))
    b.nt.links.new(col, b.inp('Base Color'))
    rough = b.mixf(b.mul(wheel, 0.5), 0.9, 0.72)
    rough = b.mixf(b.mx(paint_w, paint_y), rough, 0.55)
    rough = b.mixf(b.mul(oil, 0.6), rough, 0.6)
    b.nt.links.new(rough, b.inp('Roughness'))
    h = b.tex('asphalt_02', 'disp', b.mapping(obj, 3.2), False)
    hh = b.bw(h.outputs['Color']) if h else 0.5
    hh = b.sub(hh, b.mul(tar, 0.6))
    hh = b.add(hh, b.mul(b.mx(paint_w, paint_y), 0.25))
    b.nt.links.new(b.bump(hh, 0.35, 0.01), b.inp('Normal'))
    return m


TEX_AVG_L = {'gravel_road': 0.3 * 0.192 + 0.59 * 0.099 + 0.11 * 0.051, 'red_dirt_mud_01': 0.3 * 0.214 + 0.59 * 0.091 + 0.11 * 0.038}


def _gravel_material(name='MAT_Road_Gravel', col=(0.15, 0.135, 0.11), aid='gravel_road', size=2.6, col2=None, ruts=True):
    """Texture supplies detail only (luminance / average); colour is art-directed
    (north Georgia crushed granite: grey-tan, reddened by clay dust)."""
    m = bpy.data.materials.new(name)
    b = NB(m)
    obj = b.tc.outputs['Object']
    t = b.tex(aid, 'diff', b.mapping(obj, size))
    t2 = b.tex(aid, 'diff', b.mapping(obj, size * 3.3, 1.2))
    det = b.bw(t.outputs['Color']) if t else 0.1
    if t2:
        det = b.mixf(b.smooth(b.noise(obj, 0.08, 2.0), 0.4, 0.6), det, b.bw(t2.outputs['Color']))
    det = b.math('POWER', b.math('DIVIDE', det, TEX_AVG_L.get(aid, 0.1)), 0.85)
    base = b.mix(b.smooth(b.noise(obj, 0.05, 2.0), 0.3, 0.7), col, col2 or tuple(x * 0.85 for x in col))
    c = b.mix(1.0, base, b.xyz(det, det, det), 'MULTIPLY')
    if ruts:
        rl = b.attr('rl')
        rut = b.sub(1.0, b.smooth(b.absv(b.sub(b.absv(rl), 0.85)), 0.15, 0.45))
        rut = b.mul(rut, b.lt(b.absv(rl), 3.0))
        c = b.mix(b.mul(rut, 0.3), c, (0.07, 0.06, 0.05))
    b.nt.links.new(c, b.inp('Base Color'))
    b.inp('Roughness').default_value = 0.97
    h = b.tex(aid, 'disp', b.mapping(obj, size), False)
    if h:
        b.nt.links.new(b.bump(h.outputs['Color'], 0.6, 0.03), b.inp('Normal'))
    return m


def _concrete_material(name='MAT_Road_Concrete', col=(0.3, 0.295, 0.275), joints=1.5):
    m = bpy.data.materials.new(name)
    b = NB(m)
    obj = b.tc.outputs['Object']
    n1 = b.noise(obj, 0.9, 5.0)
    n2 = b.noise(obj, 0.05, 2.0)
    c = b.mix(b.smooth(n1, 0.3, 0.75), col, tuple(x * 0.78 for x in col))
    c = b.mix(b.mul(b.smooth(n2, 0.45, 0.75), 0.5), c, (col[0] * 0.7, col[1] * 0.72, col[2] * 0.66))
    # grime streaks (vertical) + scored joints along the station
    st = b.noise(b.mapping(obj, (1 / 0.35, 1 / 0.35, 1 / 4.0)), 1.0, 2.0)
    c = b.mix(b.mul(b.smooth(st, 0.55, 0.8), 0.35), c, (0.14, 0.13, 0.11))
    if joints:
        j = b.lt(b.fract(b.math('DIVIDE', b.attr('rs'), joints)), 0.012)
        c = b.mix(b.mul(j, 0.6), c, (0.08, 0.08, 0.075))
    b.nt.links.new(c, b.inp('Base Color'))
    b.inp('Roughness').default_value = 0.88
    b.nt.links.new(b.bump(n1, 0.15, 0.01), b.inp('Normal'))
    return m


def _verge_material():
    """Ditches / embankments / sidewalk backs: the terrain material itself, so the road
    corridor blends seamlessly (needs UVMap + lu_a/lu_b like the terrain chunks)."""
    m = bpy.data.materials.get('MAT_Terrain_PBR') or bpy.data.materials.get('MAT_Terrain')
    if m is not None:
        return m
    m = bpy.data.materials.new('MAT_Road_Verge')
    b = NB(m)
    b.inp('Base Color').default_value = (0.09, 0.1, 0.04, 1)
    b.inp('Roughness').default_value = 0.95
    return m


def _steel_material(name, col, rough=0.55, rust=0.25):
    m = bpy.data.materials.new(name)
    b = NB(m)
    obj = b.tc.outputs['Object']
    n = b.noise(obj, 0.7, 5.0)
    r = b.noise(obj, 2.5, 3.0)
    c = b.mix(b.smooth(n, 0.35, 0.7), col, tuple(x * 0.8 for x in col))
    c = b.mix(b.mul(b.smooth(r, 0.62, 0.78), rust), c, (0.16, 0.06, 0.025))
    b.nt.links.new(c, b.inp('Base Color'))
    b.inp('Metallic').default_value = 0.55
    b.inp('Roughness').default_value = rough
    return m


def _simple(name, col, rough=0.9, metal=0.0):
    m = bpy.data.materials.new(name)
    b = NB(m)
    obj = b.tc.outputs['Object']
    n = b.noise(obj, 3.0, 4.0)
    c = b.mix(b.smooth(n, 0.3, 0.7), col, tuple(x * 0.75 for x in col))
    b.nt.links.new(c, b.inp('Base Color'))
    b.inp('Roughness').default_value = rough
    b.inp('Metallic').default_value = metal
    return m


def _riprap_material():
    """Granite riprap: angular stones (voronoi cells) with dark voids, some moss."""
    m = bpy.data.materials.new('MAT_Bridge_Riprap')
    b = NB(m)
    obj = b.tc.outputs['Object']
    v = b.mapping(obj, 0.55)
    cell = b.voronoi(v, 1.0, 'F1', 'Color')
    ed = b.voronoi(v, 1.0, 'DISTANCE_TO_EDGE', 'Distance')
    sep = b.nt.nodes.new('ShaderNodeSeparateColor'); b._in(sep.inputs[0], cell)
    st = b.mix(sep.outputs['Red'], (0.11, 0.105, 0.1), (0.24, 0.23, 0.21))
    st = b.mix(b.mul(b.smooth(b.noise(obj, 0.3, 3.0), 0.55, 0.8), 0.6), st, (0.06, 0.08, 0.035))
    c = b.mix(b.lt(ed, 0.05), st, (0.015, 0.015, 0.012))
    b.nt.links.new(c, b.inp('Base Color'))
    b.inp('Roughness').default_value = 0.9
    b.nt.links.new(b.bump(b.smooth(ed, 0.0, 0.2), 1.0, 0.15), b.inp('Normal'))
    return m


_MATS = {}


def materials():
    if _MATS:
        return _MATS
    _MATS.update({
        'asphalt': _asphalt_material(),
        'gravel': _gravel_material('MAT_Road_Gravel', (0.15, 0.13, 0.105), col2=(0.17, 0.11, 0.07)),
        'shoulder': _gravel_material('MAT_Road_GravelShoulder', (0.13, 0.12, 0.1), 'gravel_road', 2.0, col2=(0.11, 0.1, 0.07), ruts=False),
        'clay': _gravel_material('MAT_Road_RedClay', (0.24, 0.105, 0.05), 'red_dirt_mud_01', 3.0, col2=(0.2, 0.1, 0.055)),
        'concrete': _concrete_material(),
        'sidewalk': _concrete_material('MAT_Road_Sidewalk', (0.27, 0.265, 0.245), 1.5),
        'bridge_concrete': _concrete_material('MAT_Bridge_Concrete', (0.44, 0.43, 0.40), 0.0),
        'verge': _verge_material(),
        'truss': _steel_material('MAT_Bridge_Steel_Truss', (0.2, 0.225, 0.215), 0.55, 0.35),
        'riprap': _riprap_material(),
        'girder_steel': _steel_material('MAT_Bridge_Steel_Girder', (0.22, 0.24, 0.23), 0.6, 0.4),
        'rail_steel': _steel_material('MAT_Rail_Steel', (0.32, 0.29, 0.26), 0.45, 0.6),
        'ballast': _gravel_material('MAT_Rail_Ballast', (0.12, 0.115, 0.11), 'gravel_road', 1.1, col2=(0.1, 0.085, 0.07), ruts=False),
        'tie': _simple('MAT_Rail_Tie', (0.075, 0.055, 0.04), 0.95),
        'railing': _steel_material('MAT_Bridge_Railing', (0.5, 0.52, 0.52), 0.4, 0.15),
    })
    return _MATS


# ------------------------------------------------------------------ network + cross-sections
CURB_ZONES = {'downtown', 'industrial', 'town_center', 'city', 'town'}
MARK = {'freeway': 6, 'ramp': 7, 'highway': 1, 'rural': 2, 'collector': 3, 'urban_street': 3, 'arterial': 4, 'main_street': 5}
AGE = {'freeway': 0.12, 'ramp': 0.15, 'highway': 0.3, 'arterial': 0.45, 'main_street': 0.55, 'collector': 0.55,
       'urban_street': 0.62, 'residential': 0.72, 'rural': 0.8, 'gravel': 0.5, 'dirt': 0.5, 'driveway': 0.6}
SURF = {'gravel': 'gravel', 'driveway': 'gravel', 'dirt': 'clay'}
RANK = {'driveway': 0, 'dirt': 1, 'gravel': 2, 'residential': 3, 'urban_street': 4, 'rural': 4, 'collector': 5,
        'main_street': 6, 'arterial': 6, 'highway': 8, 'ramp': 7, 'freeway': 9}
CURB_R = {'urban_street': 4.5, 'main_street': 4.5, 'residential': 6.0, 'collector': 7.5, 'arterial': 7.5, 'rural': 9.0,
          'highway': 12.0, 'ramp': 12.0, 'gravel': 5.0, 'dirt': 4.0, 'driveway': 3.0, 'freeway': 15.0}
DESIGN_KMH = {'freeway': 90, 'highway': 90, 'ramp': 55, 'arterial': 65, 'rural': 72, 'collector': 56}
# mesh material slots for ROAD / JUNCTION objects
SLOT = {'asphalt': 0, 'gravel': 1, 'clay': 2, 'shoulder': 3, 'concrete': 4, 'sidewalk': 5, 'verge': 6}
SLOT_ORDER = ['asphalt', 'gravel', 'clay', 'shoulder', 'concrete', 'sidewalk', 'verge']
ATTRS = ('rl', 'rs', 'hw', 'mk', 'esh', 'lw', 'uin', 'np', 'age', 'surf', 'sa', 'sb', 'xa', 'xb', 'len')


class Net:
    def __init__(self, bbox=None):
        self.types = load('data/roads/road_types.json')['types']
        zk = {z['id']: z['kind'] for z in load('data/manual/zones.json')['zones']}
        self.nodes = {f['properties']['id']: f for f in load('data/roads/road_nodes.geojson')['features']}
        self.E = []
        for f in load('data/roads/roads.geojson')['features']:
            p = f['properties']
            if p.get('virtual'):
                continue
            c = np.asarray(f['geometry']['coordinates'], float)
            if c.shape[1] < 3:
                c = np.c_[c, np.zeros(len(c))]
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                if c[:, 0].max() < x0 or c[:, 0].min() > x1 or c[:, 1].max() < y0 or c[:, 1].min() > y1:
                    continue
            X, Y = px2w(c[:, 0], c[:, 1])
            P3 = np.c_[X, Y, c[:, 2]]
            keep = np.r_[True, np.hypot(*np.diff(P3[:, :2], axis=0).T) > 1e-4]
            P3 = P3[keep]
            if len(P3) < 2:
                continue
            e = {'p': p, 'P': P3, 'zone': zk.get(p.get('zone')), 'trim': [0.0, 0.0], 'tan': [None, None], 'hwend': [None, None]}
            e['sec'] = self.section(e)
            self.E.append(e)
        self.legs = {}
        for i, e in enumerate(self.E):
            for end, n in ((0, e['p']['from']), (1, e['p']['to'])):
                self.legs.setdefault(n, []).append((i, end))

    def section(self, e):
        p = e['p']
        t = p['type']
        sec = dict(self.types[t].get('section', {}))
        hw = p.get('width_m', self.types[t]['width_m']) / 2
        curb = (bool(sec.get('curb')) or (bool(sec.get('curb_in_zones')) and e['zone'] in sec.get('curb_zones', CURB_ZONES))) and t not in ('freeway', 'ramp', 'highway')
        lanes = p.get('lanes') or self.types[t]['lanes']
        lw = sec.get('lane_w', 3.3)
        esh = sec.get('parking_m', sec.get('shoulder_paved_m', 0.0))
        uin = sec.get('median_m', 0.0) / 2 + sec.get('inner_shoulder_m', 0.0) if sec.get('median_m') else 0.0
        mk = MARK.get(t, 0)
        if t == 'collector' and lanes >= 4:
            mk = 4
        if t == 'highway' and lanes >= 4:
            mk = 4
        if t == 'residential' and e['zone'] in ('city', 'downtown'):
            mk = 0
        sw = sec.get('sidewalk_m', 1.5) if curb and ('sidewalk_zones' not in sec or e['zone'] in sec['sidewalk_zones']) else 0.0
        return {'t': t, 'hw': hw, 'curb': curb, 'sw': max(sw, 0.3 if curb else 0.0),
                'g': 0.0 if curb else sec.get('shoulder_gravel_m', 0.0), 'dw': 0.0 if curb else sec.get('ditch_w_m', 0.0),
                'dd': 0.0 if curb else sec.get('ditch_d_m', 0.0), 'crown': sec.get('crown', 0.02), 'lw': lw, 'esh': esh,
                'uin': uin, 'mk': mk, 'age': AGE.get(t, 0.5), 'surf': 1.0 if t == 'rural' else 0.0,
                'mat': SURF.get(t, 'asphalt'), 'lanes': lanes, 'median': sec.get('median_m', 0.0)}

    # direction of an edge leaving node end (0 = start, 1 = end), in world XY
    def out_dir(self, i, end, dist=6.0):
        P = self.E[i]['P'][:, :2]
        if end == 1:
            P = P[::-1]
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
        q = np.array([np.interp(min(dist, s[-1]), s, P[:, 0]), np.interp(min(dist, s[-1]), s, P[:, 1])])
        d = q - P[0]
        return d / max(np.hypot(*d), 1e-9), s[-1]


def _intersect(p, d, q, e):
    """Lines p + t d and q + u e -> (t, u) or None."""
    A = np.array([[d[0], -e[0]], [d[1], -e[1]]])
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None
    t, u = np.linalg.solve(A, q - p)
    return t, u


def _through_pair(net, info):
    """Two legs that form the continuing road through a junction (same road / class,
    near-straight, ranked above every other leg) or None."""
    best = None
    for x in range(len(info)):
        for y in range(x + 1, len(info)):
            a, b = info[x], info[y]
            dev = abs(math.degrees(abs((a['ang'] - b['ang'] + math.pi) % (2 * math.pi) - math.pi)) - 180.0)
            if dev > 38:
                continue
            ra, rb = RANK.get(a['t'], 3), RANK.get(b['t'], 3)
            pa, pb = net.E[a['i']]['p'], net.E[b['i']]['p']
            same = (pa.get('def_id') and pa.get('def_id') == pb.get('def_id')) or (pa.get('route') and pa.get('route') == pb.get('route'))
            others = [RANK.get(c['t'], 3) for k, c in enumerate(info) if k not in (x, y)]
            if not others:
                continue
            if min(ra, rb) < max(others):
                continue
            if not same and min(ra, rb) == max(others):
                continue
            score = (min(ra, rb), 1 if same else 0, -dev)
            if best is None or score > best[0]:
                best = (score, x, y)
    return None if best is None else (best[1], best[2])


def plan_junctions(net):
    """Per junction: either 'through' (the major road runs through untrimmed with its
    markings; minor legs are trimmed to its edge and joined by curb-return fillets) or
    'pad' (all legs trimmed, one merged intersection surface: city grids, equal roads)."""
    J = {}
    for n, legs in net.legs.items():
        if len(legs) < 3:
            continue
        types = {net.E[i]['sec']['t'] for i, _ in legs}
        if 'freeway' in types:
            continue  # freeway merges / wyes: carriageways run through
        node_xy = None
        info = []
        for i, end in legs:
            d, L = net.out_dir(i, end)
            P = net.E[i]['P']
            xy = P[0, :2] if end == 0 else P[-1, :2]
            node_xy = xy if node_xy is None else node_xy
            info.append({'i': i, 'end': end, 'd': d, 'L': L, 'hw': net.E[i]['sec']['hw'], 'ang': math.atan2(d[1], d[0]),
                         't': net.E[i]['sec']['t'], 'z': P[0, 2] if end == 0 else P[-1, 2]})
        info.sort(key=lambda a: a['ang'])
        m = len(info)
        zone = net.E[info[0]['i']]['zone']
        dense = zone in ('downtown', 'town_center') and all(RANK.get(a['t'], 3) >= 3 for a in info)
        tp = None if dense else _through_pair(net, info)
        mode = 'pad'
        if tp is not None:
            # every minor leg must sit between the two through legs
            ok = True
            for k in range(m):
                if k in tp:
                    continue
                if (k - 1) % m not in tp and (k + 1) % m not in tp:
                    ok = False
            if ok:
                mode = 'through'
        need = [[max(2.0, a['hw'] * 0.7)] for a in info]
        if mode == 'through':
            for k in tp:
                need[k] = [0.0]
        corner = {}
        for k in range(m):
            a, b = info[k], info[(k + 1) % m]
            if mode == 'through' and k in tp and (k + 1) % m in tp:
                continue
            gap = (b['ang'] - a['ang']) % (2 * math.pi)
            if gap < math.radians(8) or gap > math.radians(172):
                continue
            la = np.array([-a['d'][1], a['d'][0]])   # left of leg a (CCW side, towards b)
            lb = np.array([-b['d'][1], b['d'][0]])
            r = _intersect(node_xy + la * a['hw'], a['d'], node_xy - lb * b['hw'], b['d'])
            if r is None:
                continue
            ta, tb = r
            R = min(CURB_R.get(a['t'], 6), CURB_R.get(b['t'], 6))
            tl = R / math.tan(gap / 2)
            if not (mode == 'through' and k in tp):
                need[k].append(ta + tl)
            if not (mode == 'through' and (k + 1) % m in tp):
                need[(k + 1) % m].append(tb + tl)
            corner[k] = (ta, tb, tl)
        for k, a in enumerate(info):
            tr = max(need[k]) + (0.4 if max(need[k]) > 0 else 0.0)
            tr = min(tr, 38.0, 0.45 * a['L'])
            a['trim'] = max(tr, 0.0)
            net.E[a['i']]['trim'][a['end']] = a['trim']
        J[n] = {'xy': node_xy, 'legs': info, 'z': float(np.mean([a['z'] for a in info])), 'mode': mode, 'tp': tp, 'corner': corner}
        if mode == 'through':
            A, B = info[tp[0]], info[tp[1]]
            t = (-A['d'] + B['d'])
            t = t / max(np.hypot(*t), 1e-9)
            net.E[A['i']]['tan'][A['end']] = -t if A['end'] == 0 else t
            net.E[B['i']]['tan'][B['end']] = t if B['end'] == 0 else -t
            hw = 0.5 * (A['hw'] + B['hw'])
            net.E[A['i']]['hwend'][A['end']] = hw
            net.E[B['i']]['hwend'][B['end']] = hw
            # curb cuts on the through road where minor legs attach (filled by the junction)
            for k in range(m):
                if k in tp:
                    continue
                for nb, side_of_nb in (((k - 1) % m, 'left'), ((k + 1) % m, 'right')):
                    if nb not in tp:
                        continue
                    T_ = info[nb]
                    # extent along the through leg: corner intersection + fillet length
                    ck = (k - 1) % m if side_of_nb == 'left' else k
                    tq = 8.0
                    if ck in corner:
                        ta, tb, tl = corner[ck]
                        tq = (ta if side_of_nb == 'left' else tb) + max(tl, info[k]['trim'] - (tb if side_of_nb == 'left' else ta))
                    tq = float(np.clip(tq, 2.0, 24.0))
                    lsign = -1 if T_['end'] == 0 else 1          # 'left of outward' in the edge frame
                    sg = lsign if side_of_nb == 'left' else -lsign
                    net.E[T_['i']].setdefault('cut', []).append((T_['end'], sg, tq))
                    info[k].setdefault('q', {})[side_of_nb] = (nb, tq, sg)
    # shared tangents at degree-2 nodes (no cracks between consecutive edges)
    for n, legs in net.legs.items():
        if len(legs) != 2:
            continue
        (i, ei), (j, ej) = legs
        di, _ = net.out_dir(i, ei, 4.0)
        dj, _ = net.out_dir(j, ej, 4.0)
        t = (-di + dj)  # through direction from leg i into leg j
        if np.hypot(*t) < 1e-6:
            continue
        t = t / np.hypot(*t)
        net.E[i]['tan'][ei] = t if ei == 1 else -t   # tangent in the edge's own direction
        net.E[j]['tan'][ej] = t if ej == 0 else -t
        hw = 0.5 * (net.E[i]['sec']['hw'] + net.E[j]['sec']['hw'])
        net.E[i]['hwend'][ei] = hw
        net.E[j]['hwend'][ej] = hw
    return J


def _gauss(a, sig):
    if sig <= 0 or len(a) < 3:
        return a
    r = int(3 * sig) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); k /= k.sum()
    pad = np.r_[np.full(r, a[0]), a, np.full(r, a[-1])]
    return np.convolve(pad, k, mode='valid')


def sample_edge(e, step=2.0):
    """Resample the edge between its trims. Returns dict with P (n,3), s (station from
    the edge start, m), T (tangent), N (right normal), curvature k (1/m, + = left turn)."""
    P = e['P']
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(P[:, :2], axis=0).T))]
    L = s[-1]
    a, b = e['trim'][0], L - e['trim'][1]
    if b - a < 0.8:
        return None
    n = max(2, int(math.ceil((b - a) / step)) + 1)
    ss = np.linspace(a, b, n)
    X = np.interp(ss, s, P[:, 0]); Y = np.interp(ss, s, P[:, 1]); Z = np.interp(ss, s, P[:, 2])
    # light smoothing of the plan polyline (keeps ends) - removes 2.5 m facets on curves
    if n > 6:
        Xs, Ys = _gauss(X, 1.0), _gauss(Y, 1.0)
        w = np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / 3.0, 0, 1)
        X = X * (1 - w) + Xs * w; Y = Y * (1 - w) + Ys * w
    Z = _gauss(Z, 1.0) if n > 6 else Z
    Z[0], Z[-1] = np.interp(a, s, P[:, 2]), np.interp(b, s, P[:, 2])
    T = np.stack([np.gradient(X), np.gradient(Y)], 1)
    T /= np.maximum(np.hypot(T[:, 0], T[:, 1]), 1e-9)[:, None]
    for end, k in ((0, 0), (1, -1)):
        if e['tan'][end] is not None and e['trim'][end] < 0.01:
            T[k] = e['tan'][end]
    N = np.stack([T[:, 1], -T[:, 0]], 1)  # right-hand normal
    ang = np.unwrap(np.arctan2(T[:, 1], T[:, 0]))
    ds = np.maximum(np.gradient(ss), 1e-6)
    k = _gauss(np.gradient(ang) / ds, 3.0)
    return {'P': np.c_[X, Y, Z], 's': ss, 'T': T, 'N': N, 'k': k, 'L': L}


def _spans_mask(e, ss):
    """Deck samples: stations (m) inside the edge's bridge spans (px stations in data)."""
    on = np.zeros(len(ss), bool)
    for a, b in e['p'].get('bridge_spans') or []:
        on |= (ss >= a * MPP - 0.5) & (ss <= b * MPP + 0.5)
    return on


def _probe(T, P, N, sgn, offs):
    """Terrain heights at offsets (m) along +-normal: (n, len(offs))."""
    X = P[:, 0][:, None] + N[:, 0][:, None] * offs[None] * sgn
    Y = P[:, 1][:, None] + N[:, 1][:, None] * offs[None] * sgn
    x, y = w2px(X, Y)
    return T.at(x, y)


def _catch(z0, u0, slope, terr, offs, rising):
    """First crossing of the line z0 + slope*(u - u0) (u >= u0) with the terrain profile.
    rising: the line rises (cut backslope) -> stop where line >= terrain; else falls."""
    n = len(z0)
    line = z0[:, None] + slope * (offs[None] - u0[:, None])
    valid = offs[None] >= u0[:, None] - 1e-6
    hit = (line >= terr) if rising else (line <= terr)
    hit &= valid
    j = np.where(hit.any(1), hit.argmax(1), len(offs) - 1)
    uc = offs[j]
    zc = terr[np.arange(n), j]
    # refine between j-1 and j
    jm = np.maximum(j - 1, 0)
    d0 = (line[np.arange(n), jm] - terr[np.arange(n), jm])
    d1 = (line[np.arange(n), j] - terr[np.arange(n), j])
    f = np.clip(d0 / np.where(np.abs(d0 - d1) > 1e-6, d0 - d1, 1.0), 0, 1)
    ok = hit.any(1) & (j > 0) & valid[np.arange(n), jm]
    uc = np.where(ok, offs[jm] + f * (offs[j] - offs[jm]), uc)
    zc = np.where(ok, terr[np.arange(n), jm] + f * (terr[np.arange(n), j] - terr[np.arange(n), jm]), zc)
    uc = np.maximum(uc, u0 + 0.05)
    return uc, zc


JERSEY = np.array([(-0.30, 0.0), (-0.24, 0.08), (-0.09, 0.33), (-0.075, 0.81), (0.075, 0.81), (0.09, 0.33), (0.24, 0.08), (0.30, 0.0)])


def controls(net, J):
    """Stop bars / crosswalks per (edge, end): distance (m) from that end, 0 = none."""
    ctl = {}
    for n, j in J.items():
        ranks = []
        for a in j['legs']:
            t = a['t']
            ranks.append(0.5 if t == 'ramp' else RANK.get(t, 3))
        rmax = max(ranks)
        zone = net.E[j['legs'][0]['i']]['zone']
        dense = zone in ('downtown', 'town_center') or any(a['t'] == 'main_street' for a in j['legs'])
        for a, r in zip(j['legs'], ranks):
            e = net.E[a['i']]
            if e['sec']['mat'] != 'asphalt':
                continue
            xw = a['trim'] + 0.6 if (dense and e['sec']['curb']) else 0.0
            stop = 0.0
            if r < rmax or (dense and len(j['legs']) >= 4):
                stop = (xw + 3.6) if xw else a['trim'] + 1.2
            ctl[(a['i'], a['end'])] = (stop, xw)
    return ctl


def _side_columns(S, e, sec, zoff, hwv, deck, T, sgn):
    """Outer columns (6) of one side: arrays (n,6) of lateral offset u>=0 and z."""
    P, N = S['P'], S['N']
    n = len(P)
    z_e = P[:, 2] + zoff(sgn * hwv)
    U = np.zeros((n, 6)); Z = np.zeros((n, 6))
    offs = np.arange(0.0, 24.01, 0.5)
    if sec['curb']:
        sw = sec['sw']
        U[:, 0] = hwv; Z[:, 0] = z_e
        U[:, 1] = hwv + 0.001; Z[:, 1] = z_e + 0.15
        U[:, 2] = hwv + 0.15; Z[:, 2] = z_e + 0.15
        U[:, 3] = hwv + 0.15 + sw; Z[:, 3] = z_e + 0.15 + 0.02 * sw
        terr = _probe(T, P, N, sgn, offs + 0.0)
        u0 = U[:, 3]; z0 = Z[:, 3]
        tb = terr[np.arange(n), np.clip(np.searchsorted(offs, u0 + 0.4), 0, len(offs) - 1)]
        up = tb > z0
        uc1, zc1 = _catch(z0, u0, 1.0, terr, offs, True)
        uc2, zc2 = _catch(z0, u0, -0.67, terr, offs, False)
        U[:, 4] = np.where(up, uc1, uc2); Z[:, 4] = np.where(up, zc1, zc2)
    else:
        g, dw, dd = sec['g'], sec['dw'], sec['dd']
        bed = hwv + g
        z_bed = z_e - 0.04 * g - 0.02
        U[:, 0] = hwv; Z[:, 0] = z_e
        U[:, 1] = bed; Z[:, 1] = z_bed
        terr = _probe(T, P, N, sgn, offs)
        j_test = np.clip(np.searchsorted(offs, bed + dw + 1.0), 0, len(offs) - 1)
        cut = terr[np.arange(n), j_test] > z_bed - 0.25
        if dw > 0 and dd > 0:
            fs = min(3.0 * dd, 0.6 * dw)
            u3 = bed + fs; u4 = bed + dw
            # cut: ditch then backslope 1:1.5
            ucc, zcc = _catch(z_bed - dd, u4, 1 / 1.5, terr, offs, True)
        else:
            u3 = bed + 0.3; u4 = bed + 0.6
            ucc, zcc = _catch(z_bed - 0.05, u4, 1 / 1.5, terr, offs, True)
        # fill: embankment 1:2 from the bed edge
        ucf, zcf = _catch(z_bed, bed, -0.5, terr, offs, False)
        U[:, 2] = np.where(cut, u3, bed + (ucf - bed) / 3)
        Z[:, 2] = np.where(cut, z_bed - (dd if dw > 0 else 0.05), z_bed + (zcf - z_bed) / 3)
        U[:, 3] = np.where(cut, u4, bed + 2 * (ucf - bed) / 3)
        Z[:, 3] = np.where(cut, z_bed - (dd if dw > 0 else 0.05), z_bed + 2 * (zcf - z_bed) / 3)
        U[:, 4] = np.where(cut, ucc, ucf); Z[:, 4] = np.where(cut, zcc, zcf)
        U[:, 4] = np.maximum(U[:, 4], U[:, 3] + 0.05)
    # skirt: tucked under the terrain
    U[:, 5] = U[:, 4] + 0.7
    ts = _probe(T, P, N, sgn, np.array([0.0]))[:, 0] * 0  # placeholder shape
    x, y = w2px(P[:, 0] + N[:, 0] * U[:, 5] * sgn, P[:, 1] + N[:, 1] * U[:, 5] * sgn)
    Z[:, 5] = T.at(x, y) - 0.4
    # curb cuts: where a minor road joins, the junction surface replaces the side detail
    cut = np.zeros(n, bool)
    for end, sg, tq in e.get('cut', []):
        if sg != sgn:
            continue
        d = (S['s'] - S['s'][0]) if end == 0 else (S['s'][-1] - S['s'])
        cut |= d < tq
    if cut.any():
        for c in (1, 2, 3, 4):
            U[cut, c] = hwv[cut] + 0.001 * c; Z[cut, c] = z_e[cut] - 0.03
        U[cut, 5] = hwv[cut] + 0.3; Z[cut, 5] = z_e[cut] - 0.3
    # bridge decks: safety curb, the structure object adds parapets
    if deck.any():
        U[deck, 1] = hwv[deck] + 0.001; Z[deck, 1] = z_e[deck] + 0.22
        for c in (2, 3, 4, 5):
            U[deck, c] = hwv[deck] + 0.55; Z[deck, c] = z_e[deck] + 0.22
    return U, Z


def build_edge(net, idx, T, ctl, shared=None):
    e = net.E[idx]
    sec = e['sec']
    S = sample_edge(e)
    if S is None:
        return None
    P, N, ss = S['P'], S['N'], S['s']
    n = len(P)
    t = sec['t']
    # width: blend towards the shared width at degree-2 nodes (tapers between classes)
    hwv = np.full(n, sec['hw'])
    for end in (0, 1):
        he = e['hwend'][end]
        if he is not None and abs(he - sec['hw']) > 0.05:
            d = (ss - ss[0]) if end == 0 else (ss[-1] - ss)
            w = np.clip(1 - d / 25.0, 0, 1)
            hwv = hwv + (he - sec['hw']) * w * w * (3 - 2 * w)
    # superelevation on curves (rural design), crown elsewhere
    es = np.zeros(n)
    if t in DESIGN_KMH and not sec['curb']:
        V = DESIGN_KMH[t]
        R = 1.0 / np.maximum(np.abs(S['k']), 1e-6)
        es = np.clip(0.55 * V * V / (127.0 * R), 0, 0.06) * np.sign(S['k'])
        es = _gauss(es, 6.0)
        es[np.abs(es) < 0.004] = 0.0
    crown = sec['crown']
    wb = np.clip(np.abs(es) / (2 * crown), 0, 1)

    def zoff(u):
        return (1 - wb) * (-crown * np.abs(u)) + wb * es * u
    deck = _spans_mask(e, ss)
    e['_deck'] = {'s': ss, 'P': P, 'N': N, 'T': S['T'], 'zl': P[:, 2] + zoff(-hwv), 'zr': P[:, 2] + zoff(hwv), 'hw': hwv, 'L': S['L']}
    L_ = {}
    for sgn in (-1, 1):
        L_[sgn] = _side_columns(S, e, sec, zoff, hwv, deck, T, sgn)
    # assemble 13 columns: left outer .. left edge, centre, right edge .. right outer
    cols_u = np.concatenate([-L_[-1][0][:, ::-1], np.zeros((n, 1)), L_[1][0]], 1)
    cols_z = np.concatenate([L_[-1][1][:, ::-1], P[:, 2:3], L_[1][1]], 1)
    G = np.zeros((n, 13, 3))
    G[:, :, 0] = P[:, 0:1] + N[:, 0:1] * cols_u
    G[:, :, 1] = P[:, 1:2] + N[:, 1:2] * cols_u
    G[:, :, 2] = cols_z
    # watertight joins: consecutive edges share the exact end row at their common node
    if shared is not None:
        for end, r, nid in ((0, 0, e['p']['from']), (1, n - 1, e['p']['to'])):
            if e['trim'][end] > 0.01:
                continue
            key = nid
            tdir = S['T'][r]
            if key in shared:
                row, t0 = shared[key]
                if np.hypot(*(row[6, :2] - G[r, 6, :2])) < 0.5:
                    G[r] = row if np.dot(t0, tdir) > 0 else row[::-1]
            else:
                shared[key] = (G[r].copy(), tdir.copy())
    # per-row no-passing zones (2-lane): curves < 500 m radius, junction approaches, decks
    npz = np.zeros(n)
    if sec['mk'] in (1, 2):
        R = 1.0 / np.maximum(np.abs(S['k']), 1e-6)
        raw = (R < 520).astype(float)
        raw[(ss - ss[0] < 70) | (ss[-1] - ss < 70)] = 1.0
        vz = np.gradient(np.gradient(P[:, 2])) / np.maximum(np.gradient(ss), 1e-3) ** 2
        raw[vz < -0.0012] = 1.0  # crest vertical curves
        raw[deck] = 1.0
        npz = (_gauss(raw, 12.0) > 0.25).astype(float)
    stop_a, xw_a = ctl.get((idx, 0), (0.0, 0.0))
    stop_b, xw_b = ctl.get((idx, 1), (0.0, 0.0))
    mat_pav = SLOT[sec['mat']]
    if sec['curb']:
        side = [SLOT['concrete'], SLOT['concrete'], SLOT['sidewalk'], SLOT['verge'], SLOT['verge']]
    else:
        sh = SLOT['shoulder'] if sec['mat'] == 'asphalt' else mat_pav
        side = [sh, SLOT['verge'], SLOT['verge'], SLOT['verge'], SLOT['verge']]
    deck_side = [SLOT['concrete']] * 5
    M = Mesh(ATTRS)
    const = dict(hw=sec['hw'], mk=sec['mk'], esh=sec['esh'], lw=sec['lw'], uin=sec['uin'], age=sec['age'], surf=sec['surf'],
                 sa=stop_a, sb=stop_b, xa=xw_a, xb=xw_b, len=S['L'])
    # runs of ground / deck rows
    k = 0
    while k < n - 1:
        j = k
        while j + 1 < n and deck[j + 1] == deck[k]:
            j += 1
        j = min(j + 1, n - 1)
        sl = slice(k, j + 1)
        sm = deck_side if deck[k] else side
        bands = sm[::-1] + [mat_pav, mat_pav] + sm
        rl = np.broadcast_to(cols_u[sl], (j + 1 - k, 13))
        rs = np.broadcast_to(ss[sl][:, None], (j + 1 - k, 13))
        npa = np.broadcast_to(npz[sl][:, None], (j + 1 - k, 13))
        M.grid(G[sl], bands, rl=rl, rs=rs, np=npa, **const)
        k = j
    # freeway median barrier (concrete safety shape)
    if sec['median'] > 0:
        prof = JERSEY
        m = len(prof)
        GB = np.zeros((n, m, 3))
        GB[:, :, 0] = P[:, 0:1] + N[:, 0:1] * prof[None, :, 0]
        GB[:, :, 1] = P[:, 1:2] + N[:, 1:2] * prof[None, :, 0]
        GB[:, :, 2] = P[:, 2:3] + prof[None, :, 1] - 0.01
        M.grid(GB, [SLOT['concrete']] * (m - 1), rl=0.0, rs=np.broadcast_to(ss[:, None], (n, m)), **dict(const, mk=0))
    # rounded ends at intentional dead ends (cul-de-sac / end of pavement), not at the map border
    for end, r, nid in ((0, 0, e['p']['from']), (1, n - 1, e['p']['to'])):
        if len(net.legs.get(nid, [])) != 1 or t not in ('residential', 'urban_street', 'collector', 'rural', 'main_street'):
            continue
        px_, py_ = w2px(P[r, 0], P[r, 1])
        if min(px_, py_, W - px_, H - py_) < 3:
            continue
        out = -S['T'][r] if end == 0 else S['T'][r]
        R = hwv[r] * (1.35 if sec['curb'] else 1.0)
        ang0 = math.atan2(N[r, 1], N[r, 0])
        sgn_ = 1 if np.cross(np.r_[N[r], 0], np.r_[out, 0])[2] > 0 else -1
        m_ = 12
        arc = [P[r, :2] + R * np.array([math.cos(ang0 + sgn_ * math.pi * q / m_), math.sin(ang0 + sgn_ * math.pi * q / m_)]) for q in range(m_ + 1)]
        V = np.array([[P[r, 0], P[r, 1], P[r, 2]]] + [[a[0], a[1], P[r, 2] - sec['crown'] * R] for a in arc])
        F = np.array([(0, 1 + q, 2 + q) for q in range(m_)])
        M.add(V, F, mat_pav, rl=99.0, rs=0.0, **dict(const, mk=0))
    # mouth rows for the junction builder (left/right relative to the outward direction)
    mouths = {}
    for end, r in ((0, 0), (1, n - 1)):
        out = S['T'][r] if end == 0 else -S['T'][r]
        row = G[r]
        if end == 0:
            left, right = row[:7][::-1][1:], row[6:][1:]   # left of +T is -N
        else:
            left, right = row[6:][1:], row[:7][::-1][1:]
        mouths[end] = {'C': row[6], 'L': left, 'R': right, 'd': out}
    return M, mouths


def _bezier(a, c, b, n):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t * t * b


def _edge_line(net, leg, sg, t0, t1, step=1.0):
    """Pavement-edge points of a (through) leg between distances t0..t1 from the node."""
    e = net.E[leg['i']]
    D = e.get('_deck')
    if D is None:
        return None
    L = D['s'][-1]
    ts = np.linspace(t0, t1, max(2, int(abs(t1 - t0) / step) + 1))
    st = ts if leg['end'] == 0 else (D['s'][-1] - ts)
    st = np.clip(st, D['s'][0], D['s'][-1])
    f = lambda arr: np.interp(st, D['s'], arr)
    hw = f(D['hw'])
    x = f(D['P'][:, 0]) + f(D['N'][:, 0]) * hw * sg
    y = f(D['P'][:, 1]) + f(D['N'][:, 1]) * hw * sg
    z = f(D['zr'] if sg > 0 else D['zl'])
    return np.c_[x, y, z]


def triangulate(P2):
    """Ear clipping of a simple polygon (n,2) -> list of index triangles (CCW)."""
    P2 = np.asarray(P2, float)
    n = len(P2)
    if n < 3:
        return []
    area = 0.5 * sum(P2[i, 0] * P2[(i + 1) % n, 1] - P2[(i + 1) % n, 0] * P2[i, 1] for i in range(n))
    idx = list(range(n)) if area > 0 else list(range(n))[::-1]

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10 * n:
        guard += 1
        m = len(idx)
        found = False
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = P2[i0], P2[i1], P2[i2]
            if cross(a, b, c) <= 1e-9:
                continue
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = P2[j]
                if cross(a, b, p) >= -1e-9 and cross(b, c, p) >= -1e-9 and cross(c, a, p) >= -1e-9:
                    ok = False
                    break
            if ok:
                tris.append((i0, i1, i2))
                idx.pop(k)
                found = True
                break
        if not found:  # degenerate: clip the flattest vertex
            k = int(np.argmin([abs(cross(P2[idx[(q - 1) % m]], P2[idx[q]], P2[idx[(q + 1) % m]])) for q in range(m)]))
            idx.pop(k)
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def _dedupe(ring, tol=0.05):
    out = [ring[0]]
    for p in ring[1:]:
        if np.hypot(*(p[:2] - out[-1][:2])) > tol:
            out.append(p)
    if len(out) > 2 and np.hypot(*(out[0][:2] - out[-1][:2])) <= tol:
        out.pop()
    return np.asarray(out)


def _subdivide(V, F, max_len=3.0, levels=3):
    """Midpoint-subdivide triangles until edges are shorter than max_len."""
    V = [np.asarray(v, float) for v in V]
    F = [tuple(f) for f in F]
    nb = len(V)
    for _ in range(levels):
        if not F:
            break
        longest = max(max(np.hypot(*(V[a][:2] - V[b][:2])) for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))) for f in F)
        if longest <= max_len:
            break
        mid = {}

        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                mid[k] = len(V)
                V.append((V[a] + V[b]) / 2)
            return mid[k]
        G = []
        for a, b, c in F:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            G += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        F = G
    return np.asarray(V), np.asarray(F), nb


def _fan(M, ring, slot, top, T=None, center=None):
    ring = _dedupe(np.asarray(ring))
    tris = None
    if center is not None:
        # star-shaped around the node: fan from the node (keeps the road profile at the centre)
        n = len(ring)
        cr = [(ring[i, 0] - center[0]) * (ring[(i + 1) % n, 1] - center[1]) - (ring[i, 1] - center[1]) * (ring[(i + 1) % n, 0] - center[0]) for i in range(n)]
        if all(c > 1e-6 for c in cr) or all(c < -1e-6 for c in cr):
            V = np.vstack([ring, np.asarray(center)[None]])
            F = np.array([(i, (i + 1) % n, n) for i in range(n)])
            if cr[0] < 0:
                F = F[:, ::-1]
            tris = (V, F)
    if tris is None:
        t = triangulate(ring[:, :2])
        if not t:
            return
        tris = (ring, np.asarray(t))
    V, F = tris
    nb = len(ring)
    if T is not None:
        V, F, nb = _subdivide(V, F)
        # drape interior vertices over the graded bed (never below the road profile)
        x, y = w2px(V[nb:, 0], V[nb:, 1])
        if len(x):
            V[nb:, 2] = np.maximum(V[nb:, 2], T.at(x, y) + 0.22)
    M.add(V, F, slot, rl=99.0, rs=0.0, hw=top['hw'], mk=0, esh=0, lw=top['lw'], uin=0, np=0, age=top['age'], surf=top['surf'],
          sa=0, sb=0, xa=0, xb=0, len=1.0)


def _ring_sign(ring):
    r = np.asarray(ring)
    return 1.0 if (r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1]).sum() > 0 else -1.0


def _apron(M, curve, sec, C, T, first=None, last=None, out_sign=1.0):
    """Curb/sidewalk or shoulder/verge strip outside a junction boundary curve.
    out_sign: +1 = outward is to the right of the curve's direction of travel."""
    m = len(curve)
    if m < 2:
        return
    tang = np.gradient(curve[:, :2], axis=0)
    tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-9)[:, None]
    nb = np.stack([tang[:, 1], -tang[:, 0]], 1) * out_sign
    offs = np.arange(0.0, 16.01, 0.5)
    X = curve[:, 0][:, None] + nb[:, 0][:, None] * offs[None]
    Y = curve[:, 1][:, None] + nb[:, 1][:, None] * offs[None]
    terr = T.at(*w2px(X, Y))
    z0 = curve[:, 2]
    if sec['curb']:
        sw = sec['sw']
        u = np.array([0.0, 0.001, 0.15, 0.15 + sw])
        z = np.stack([z0, z0 + 0.15, z0 + 0.15, z0 + 0.15 + 0.02 * sw], 1)
        uc, zc = _catch(z[:, 3], np.full(m, u[3]), -0.67, terr, offs, False)
        up = terr[:, min(len(offs) - 1, int((u[3] + 0.5) / 0.5))] > z[:, 3]
        uc2, zc2 = _catch(z[:, 3], np.full(m, u[3]), 1.0, terr, offs, True)
        uc = np.where(up, uc2, uc); zc = np.where(up, zc2, zc)
        UU = np.c_[np.broadcast_to(u, (m, 4)), uc, uc + 0.7]
        ZZ = np.c_[z, zc, np.zeros(m)]
        bands = [SLOT['concrete'], SLOT['concrete'], SLOT['sidewalk'], SLOT['verge'], SLOT['verge']]
    else:
        g = max(sec['g'], 0.5)
        zb = z0 - 0.04 * g - 0.02
        ucf, zcf = _catch(zb, np.full(m, g), -0.5, terr, offs, False)
        up = terr[:, min(len(offs) - 1, int((g + 1.0) / 0.5))] > zb
        ucc, zcc = _catch(zb, np.full(m, g), 0.67, terr, offs, True)
        uc = np.where(up, ucc, ucf); zc = np.where(up, zcc, zcf)
        UU = np.c_[np.zeros(m), np.full(m, g), g + (uc - g) / 3, g + 2 * (uc - g) / 3, uc, uc + 0.7]
        ZZ = np.c_[z0, zb, zb + (zc - zb) / 3, zb + 2 * (zc - zb) / 3, zc, np.zeros(m)]
        sh = SLOT['shoulder'] if sec['mat'] == 'asphalt' else SLOT[sec['mat']]
        bands = [sh, SLOT['verge'], SLOT['verge'], SLOT['verge'], SLOT['verge']]
    xs = curve[:, 0][:, None] + nb[:, 0][:, None] * UU
    ys = curve[:, 1][:, None] + nb[:, 1][:, None] * UU
    ZZ[:, -1] = T.at(*w2px(xs[:, -1], ys[:, -1])) - 0.4
    G = np.stack([xs, ys, ZZ], -1)
    if first is not None and len(first) == 6:
        G[0] = first
    if last is not None and len(last) == 6:
        G[-1] = last
    M.grid(G, bands, rl=99.0, rs=0.0, hw=0.0, mk=0, esh=0, lw=3, uin=0, np=0, age=sec['age'], surf=0, sa=0, sb=0, xa=0, xb=0, len=1.0)


def _drape(curve, T, lift=0.22):
    """Raise interior points of a boundary curve that dip under the graded terrain."""
    c = np.asarray(curve, float).copy()
    if len(c) > 2:
        x, y = w2px(c[1:-1, 0], c[1:-1, 1])
        c[1:-1, 2] = np.maximum(c[1:-1, 2], T.at(x, y) + lift)
    return c


def _fillet(pm, dm, qpt, dq, step=1.0):
    """Quadratic curb-return from point pm (leaving along -dm) to qpt (arriving along +dq)."""
    r = _intersect(pm[:2], dm, qpt[:2], dq)
    if r is not None and r[0] < -0.1 and r[1] < -0.1:
        cc = pm[:2] + dm * r[0]
        ln = np.hypot(*(pm[:2] - cc)) + np.hypot(*(qpt[:2] - cc))
        nn = max(4, int(ln / step))
        return np.c_[_bezier(pm[:2], cc, qpt[:2], nn), np.linspace(pm[2], qpt[2], nn)]
    nn = max(2, int(np.hypot(*(pm[:2] - qpt[:2])) / 1.5))
    return np.c_[np.linspace(pm[:2], qpt[:2], nn), np.linspace(pm[2], qpt[2], nn)]


def build_junction_through(net, nid, j, mouths, T):
    """Minor legs joining a through road: for each minor leg one surface bounded by its
    mouth, two curb-return fillets and the through road's pavement edge (whose shoulder /
    curb is cut there), plus the fillets' aprons."""
    legs = j['legs']
    tp = j['tp']
    C = np.array([j['xy'][0], j['xy'][1], j['z']])
    M = Mesh(ATTRS)
    top = net.E[legs[tp[0]]['i']]['sec']
    for k, a in enumerate(legs):
        if k in tp or (a['i'], a['end']) not in mouths:
            continue
        q = a.get('q', {})
        if 'left' not in q or 'right' not in q:
            continue
        mo = mouths[(a['i'], a['end'])]
        sec = net.E[a['i']]['sec']
        nbL, tqL, sgL = q['right']   # CCW neighbour, faced by the minor's left mouth corner
        nbR, tqR, sgR = q['left']    # CW neighbour, faced by the right mouth corner
        eL = _edge_line(net, legs[nbL], sgL, 0.0, tqL)
        eR = _edge_line(net, legs[nbR], sgR, 0.0, tqR)
        if eL is None or eR is None:
            continue
        dL = eL[-1, :2] - eL[max(0, len(eL) - 3), :2]; dL /= max(np.hypot(*dL), 1e-9)
        dR = eR[-1, :2] - eR[max(0, len(eR) - 3), :2]; dR /= max(np.hypot(*dR), 1e-9)
        fL = _drape(_fillet(mo['L'][0], mo['d'], eL[-1], dL), T)
        fR = _drape(_fillet(mo['R'][0], mo['d'], eR[-1], dR), T)
        ring = np.asarray([mo['R'][0], mo['L'][0]] + list(fL[1:]) + list(eL[::-1][1:]) + list(eR[1:]) + list(fR[::-1][1:-1]))
        slot = SLOT['asphalt'] if (sec['mat'] == 'asphalt' or top['mat'] == 'asphalt') else SLOT[sec['mat']]
        _fan(M, ring, slot, top, T)
        rs_ = _ring_sign(ring)
        _apron(M, fL, sec, C, T, first=mo['L'], out_sign=rs_)
        _apron(M, fR, sec, C, T, first=mo['R'], out_sign=-rs_)
    return M


def build_junction(net, nid, j, mouths, T):
    """Merged pavement polygon with curb returns + corner aprons (curb/sidewalk or gravel)."""
    if j.get('mode') == 'through':
        return build_junction_through(net, nid, j, mouths, T)
    legs = [a for a in j['legs'] if (a['i'], a['end']) in mouths]
    if len(legs) < 3:
        return None
    ring = []      # boundary points (x, y, z)
    corners = []   # (points, outward-normal sign, leg a, leg b)
    for k, a in enumerate(legs):
        b = legs[(k + 1) % len(legs)]
        ma, mb = mouths[(a['i'], a['end'])], mouths[(b['i'], b['end'])]
        ring.append(ma['R'][0]); ring.append(ma['L'][0])
        pa, pb = ma['L'][0], mb['R'][0]
        gap = (b['ang'] - a['ang']) % (2 * math.pi)
        curve = None
        if math.radians(8) < gap < math.radians(172):
            r = _intersect(pa[:2], ma['d'], pb[:2], mb['d'])
            if r is not None and r[0] < -0.2 and r[1] < -0.2:
                c = pa[:2] + ma['d'] * r[0]
                ln = np.hypot(*(pa[:2] - c)) + np.hypot(*(pb[:2] - c))
                m = max(4, int(ln / 1.2))
                q = _bezier(pa[:2], c, pb[:2], m)
                zz = np.linspace(pa[2], pb[2], m)
                curve = np.c_[q, zz]
        if curve is None:
            m = max(2, int(np.hypot(*(pa[:2] - pb[:2])) / 2.0))
            curve = np.c_[np.linspace(pa[:2], pb[:2], m), np.linspace(pa[2], pb[2], m)]
        curve = _drape(curve, T)
        ring.extend(curve[1:-1])
        corners.append((curve, ma, mb, a, b))
    ring = np.asarray(ring)
    C = np.array([j['xy'][0], j['xy'][1], j['z']])
    secs = [net.E[a['i']]['sec'] for a in legs]
    top = max(secs, key=lambda s_: RANK.get(s_['t'], 0))
    allgravel = all(s_['mat'] != 'asphalt' for s_ in secs)
    M = Mesh(ATTRS)
    # the node itself is inside; insert it via a fan-free ear clipping of the boundary
    _fan(M, ring, SLOT[top['mat'] if allgravel else 'asphalt'], top, T, center=C)
    # corner aprons (outside the ring: the ring runs CCW/CW consistently with the corners)
    rs_ = _ring_sign(ring)
    for curve, ma, mb, a, b in corners:
        sa_, sb_ = net.E[a['i']]['sec'], net.E[b['i']]['sec']
        sec = sa_ if RANK.get(sa_['t'], 0) <= RANK.get(sb_['t'], 0) else sb_  # the minor road's verge detail
        _apron(M, curve, sec, C, T, first=ma['L'], last=mb['R'], out_sign=rs_)
    return M


# ------------------------------------------------------------------ structures
def _frame(p0, p1, up=(0, 0, 1)):
    d = np.asarray(p1, float) - np.asarray(p0, float)
    L = np.linalg.norm(d)
    if L < 1e-6:
        return None
    x = d / L
    u = np.asarray(up, float)
    if abs(np.dot(u, x)) > 0.98:
        u = np.array([1.0, 0, 0]) if abs(x[0]) < 0.9 else np.array([0, 1.0, 0])
    y = np.cross(u, x); y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return x, y, z, L


def box_between(M, p0, p1, w, h, mat, up=(0, 0, 1), **at):
    """Rectangular member from p0 to p1 (w wide, h tall, centred on the axis)."""
    fr = _frame(p0, p1, up)
    if fr is None:
        return
    x, y, z, L = fr
    p0 = np.asarray(p0, float)
    V = []
    for t in (0.0, L):
        for sy, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            V.append(p0 + x * t + y * sy * w / 2 + z * sz * h / 2)
    F = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    M.add(np.asarray(V), np.asarray(F), mat, **at)


def box_at(M, c, ax, ay, sx, sy, z0, z1, mat, **at):
    """Vertical prism centred at c (xy) with local axes ax, ay (unit xy), half sizes sx, sy."""
    c = np.asarray(c, float)[:2]
    ax = np.asarray(ax, float); ay = np.asarray(ay, float)
    V = []
    for z in (z0, z1):
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            q = c + ax * a * sx + ay * b * sy
            V.append((q[0], q[1], z))
    F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    M.add(np.asarray(V), np.asarray(F), mat, **at)


def cyl_at(M, c, r, z0, z1, mat, seg=12, rx=None, ax=None, **at):
    """Vertical cylinder / rounded-end wall (rx = half length along ax for a stadium shape)."""
    c = np.asarray(c, float)[:2]
    pts = []
    if rx and ax is not None:
        ax = np.asarray(ax, float); ay = np.array([-ax[1], ax[0]])
        for side in (1, -1):
            cc = c + ax * side * (rx - r)
            for a in np.linspace(-math.pi / 2, math.pi / 2, seg // 2 + 1):
                v = ax * math.cos(a) * side + ay * math.sin(a) * side
                pts.append(cc + v * r)
    else:
        for a in np.linspace(0, 2 * math.pi, seg, endpoint=False):
            pts.append(c + np.array([math.cos(a), math.sin(a)]) * r)
    pts = np.asarray(pts)
    m = len(pts)
    V = np.r_[np.c_[pts, np.full(m, z0)], np.c_[pts, np.full(m, z1)], [[c[0], c[1], z1]]]
    F = [(i, (i + 1) % m, m + (i + 1) % m, m + i) for i in range(m)]
    F += [(m + i, m + (i + 1) % m, 2 * m) for i in range(m)]
    M.add(V[:2 * m + 1], [f for f in F if len(f) == 4], mat, **at)
    M.add(V[:2 * m + 1], [f for f in F if len(f) == 3], mat, **at)


B_SLOT = {'concrete': 0, 'steel': 1, 'railing': 2, 'girder': 3, 'riprap': 4}


def _deck_frame(e, s0, s1, step=1.5):
    D = e.get('_deck')
    if D is None:
        return None
    ss = D['s']
    a, b = max(s0, ss[0]), min(s1, ss[-1])
    if b - a < 1.0:
        return None
    n = max(2, int(math.ceil((b - a) / step)) + 1)
    st = np.linspace(a, b, n)
    f = lambda arr: np.interp(st, ss, arr)
    P = np.c_[f(D['P'][:, 0]), f(D['P'][:, 1]), f(D['P'][:, 2])]
    T = np.c_[f(D['T'][:, 0]), f(D['T'][:, 1])]
    T /= np.maximum(np.hypot(T[:, 0], T[:, 1]), 1e-9)[:, None]
    N = np.stack([T[:, 1], -T[:, 0]], 1)
    return {'s': st, 'P': P, 'T': T, 'N': N, 'zl': f(D['zl']), 'zr': f(D['zr']), 'hw': f(D['hw'])}


def _sweep(M, Fr, prof_fn, mat, closed=False, **at):
    """Sweep a per-sample cross-section: prof_fn(Fr) -> (U (n,m), Z (n,m)) along the deck frame."""
    U, Z = prof_fn(Fr)
    G = np.stack([Fr['P'][:, 0:1] + Fr['N'][:, 0:1] * U, Fr['P'][:, 1:2] + Fr['N'][:, 1:2] * U, Z], -1)
    M.grid(G, [mat] * (U.shape[1] - 1), **at)


def riprap(M, c, t_, n_, dirn, half_w, T, reach=9.0, lift=0.06):
    """Draped stone apron on the slope under a bridge end (from the abutment towards the
    span, and a little past the deck edges) - protects the fill against scour."""
    us = np.linspace(-(half_w + 3.0), half_w + 3.0, max(3, int(2 * half_w / 1.5) + 5))
    ts = np.linspace(0.3, reach, max(3, int(reach / 1.2)))
    G = np.zeros((len(ts), len(us), 3))
    for i, tt in enumerate(ts):
        p = np.asarray(c)[None, :2] - t_[None] * dirn * tt + n_[None] * us[:, None]
        z = T.at(*w2px(p[:, 0], p[:, 1])) + lift
        G[i] = np.c_[p, z]
    M.grid(G, [B_SLOT['riprap']] * (len(us) - 1))


def girder_bridge(M, Fr, T, kind, structure, over=None, lower_pts=None):
    n = len(Fr['s'])
    hw = Fr['hw']
    Wd = hw + 0.55 + 0.32          # outer face of the parapet
    zc = Fr['P'][:, 2]
    L = Fr['s'][-1] - Fr['s'][0]
    slab = 'slab' in structure
    steel = 'steel' in structure
    D = 0.55 if slab else float(np.clip(L / 22.0, 1.0, 2.0)) if not steel else float(np.clip(L / 18, 1.6, 2.6))
    zb = zc - 0.25 - 0.1           # deck soffit (slab bottom)
    # parapets + fascia (both sides)
    for sgn, ze in ((-1, Fr['zl']), (1, Fr['zr'])):
        def prof(_):
            u0 = hw + 0.55
            U = np.stack([u0, u0, u0 + 0.06, u0 + 0.32, u0 + 0.36, u0 + 0.36, hw - 0.2], 1) * sgn
            Z = np.stack([ze + 0.22, ze + 0.95, ze + 1.02, ze + 1.02, ze + 0.95, zb - 0.1, zb], 1)
            return U, Z
        _sweep(M, Fr, prof, B_SLOT['concrete'], rs=Fr['s'][:, None] * np.ones((1, 7)))
        # steel rail on posts above the parapet (1970s highway bridge rail)
        if not slab:
            def rail(_):
                u = (hw + 0.55 + 0.16) * sgn
                U = np.stack([u - 0.05, u + 0.05, u + 0.05, u - 0.05, u - 0.05], 1)
                zr_ = ze + 1.02 + 0.32
                Z = np.stack([zr_, zr_, zr_ + 0.1, zr_ + 0.1, zr_], 1)
                return U, Z
            _sweep(M, Fr, rail, B_SLOT['railing'])
            for k in range(0, n, max(1, int(2.0 / max(L / max(n - 1, 1), 1e-3)))):
                c = Fr['P'][k, :2] + Fr['N'][k] * (hw[k] + 0.71) * sgn
                box_at(M, c, Fr['T'][k], Fr['N'][k], 0.05, 0.05, ze[k] + 1.0, ze[k] + 1.34, B_SLOT['railing'])
    # deck soffit
    def soffit(_):
        U = np.stack([-(hw - 0.2), hw - 0.2], 1)
        Z = np.stack([zb, zb], 1)
        return U, Z
    U, Z = soffit(Fr)
    G = np.stack([Fr['P'][:, 0:1] + Fr['N'][:, 0:1] * U, Fr['P'][:, 1:2] + Fr['N'][:, 1:2] * U, Z], -1)
    Mtmp = Mesh(); Mtmp.grid(G, [B_SLOT['concrete']])
    for V, F, m in zip(Mtmp.V, Mtmp.F, Mtmp.M):
        M.add(V, F[:, ::-1], m)  # faces down
    # girders
    if not slab:
        ng = max(3, int(round(2 * float(hw.mean()) / 2.7)) + 1)
        us = np.linspace(-(hw.mean() - 0.9), hw.mean() - 0.9, ng)
        for u in us:
            def gird(_, u=u):
                U = np.stack([np.full(n, u - 0.18), np.full(n, u - 0.18), np.full(n, u - 0.35), np.full(n, u + 0.35), np.full(n, u + 0.18), np.full(n, u + 0.18)], 1)
                Z = np.stack([zb, zb - D + 0.2, zb - D, zb - D, zb - D + 0.2, zb], 1)
                return U, Z
            _sweep(M, Fr, gird, B_SLOT['girder'] if steel else B_SLOT['concrete'])
    # piers
    target = {'viaduct': 34.0, 'overpass': 22.0}.get(kind, 30.0)
    if steel:
        target = 45.0
    stations = []
    if not slab and L > target * 1.2:
        ns = max(2, int(round(L / target)))
        stations = [Fr['s'][0] + L * i / ns for i in range(1, ns)]
    if kind == 'overpass' and lower_pts is not None and len(lower_pts):
        # a single pier in the median / between the lower road's carriageways, else spans over
        stations = []
        from_s = []
        for q in lower_pts:
            k = int(np.argmin(np.hypot(*(Fr['P'][:, :2] - q[:2]).T)))
            from_s.append(Fr['s'][k])
        if over == 'freeway':
            stations = from_s
        elif L > 36:
            stations = [Fr['s'][0] + L / 2]
    zbot = zb - D
    for st in stations:
        k = int(np.argmin(np.abs(Fr['s'] - st)))
        c = Fr['P'][k, :2]; t_ = Fr['T'][k]; n_ = Fr['N'][k]
        ground = float(T.at(*w2px(c[0], c[1]))) - 0.6
        top = float(zbot[k])
        if top - ground < 1.0:
            continue
        # hammerhead cap
        box_at(M, c, t_, n_, 0.75, hw[k] * 0.92, top - 1.1, top, B_SLOT['concrete'])
        if kind in ('bridge',) or top - ground > 14:
            cyl_at(M, c, 0.75, ground - 1.0, top - 1.1, B_SLOT['concrete'], seg=14, rx=min(hw[k] * 0.55, 5.5), ax=n_)
            box_at(M, c, t_, n_, 1.4, min(hw[k] * 0.55, 5.5) + 0.7, ground - 1.5, ground + 0.35, B_SLOT['concrete'])
        else:
            for u in ((-hw[k] * 0.45, hw[k] * 0.45) if hw[k] > 5.5 else (0.0,)):
                cc = c + n_ * u
                g2 = float(T.at(*w2px(cc[0], cc[1]))) - 0.6
                cyl_at(M, cc, 0.55, g2 - 1.0, top - 1.1, B_SLOT['concrete'], seg=12)
                box_at(M, cc, t_, n_, 1.0, 1.0, g2 - 1.2, g2 + 0.25, B_SLOT['concrete'])
    # abutments + wingwalls at both ends
    for k, dirn in ((0, -1), (n - 1, 1)):
        c = Fr['P'][k, :2]; t_ = Fr['T'][k]; n_ = Fr['N'][k]
        zt = float(zbot[k]) if not slab else float(zb[k])
        g = float(T.at(*w2px(c[0], c[1])))
        cc = c + t_ * dirn * 0.5
        box_at(M, cc, t_, n_, 0.6, Wd[k], min(g, zt) - 2.0, float(zc[k]) - 0.25, B_SLOT['concrete'])
        if kind in ('bridge', 'viaduct'):
            riprap(M, c, t_, n_, dirn, float(Wd[k]), T)
        for sgn in (-1, 1):
            p0 = c + n_ * sgn * (Wd[k] - 0.2)
            for j in range(4):
                a0 = p0 + t_ * dirn * (j * 1.8); a1 = p0 + t_ * dirn * ((j + 1) * 1.8)
                ztop = float(zc[k]) + 0.9 - j * 0.9
                gg = float(T.at(*w2px(a1[0], a1[1])))
                if ztop < gg - 0.2:
                    break
                box_at(M, (a0 + a1) / 2, t_, n_, 0.9, 0.2, gg - 1.5, ztop, B_SLOT['concrete'])


def truss_bridge(M, Fr, T):
    n = len(Fr['s'])
    L = Fr['s'][-1] - Fr['s'][0]
    hw = float(Fr['hw'].mean())
    ut = hw + 0.55 + 0.25
    nsp = max(1, int(math.ceil(L / 58.0)))
    zc = Fr['P'][:, 2]

    def at(s_):
        k = np.clip(np.searchsorted(Fr['s'], s_), 1, n - 1)
        f = (s_ - Fr['s'][k - 1]) / max(Fr['s'][k] - Fr['s'][k - 1], 1e-6)
        p = Fr['P'][k - 1] * (1 - f) + Fr['P'][k] * f
        nn = Fr['N'][k - 1] * (1 - f) + Fr['N'][k] * f
        tt = Fr['T'][k - 1] * (1 - f) + Fr['T'][k] * f
        return p, nn / np.hypot(*nn), tt / np.hypot(*tt)
    for si in range(nsp):
        s0 = Fr['s'][0] + L * si / nsp
        s1 = Fr['s'][0] + L * (si + 1) / nsp
        Ls = s1 - s0
        npan = max(4, int(round(Ls / 5.5)))
        Ht = float(np.clip(0.14 * Ls, 4.8, 7.0))
        nodes_b = {}; nodes_t = {}
        for sgn in (-1, 1):
            for i in range(npan + 1):
                p, nn, tt = at(s0 + Ls * i / npan)
                base = np.array([p[0] + nn[0] * ut * sgn, p[1] + nn[1] * ut * sgn, p[2] - 0.15])
                nodes_b[(sgn, i)] = base
                nodes_t[(sgn, i)] = base + np.array([0, 0, Ht])
            for i in range(npan):
                box_between(M, nodes_b[(sgn, i)], nodes_b[(sgn, i + 1)], 0.36, 0.5, B_SLOT['steel'])      # bottom chord
            for i in range(1, npan - 1):
                box_between(M, nodes_t[(sgn, i)], nodes_t[(sgn, i + 1)], 0.42, 0.42, B_SLOT['steel'])    # top chord
            box_between(M, nodes_b[(sgn, 0)], nodes_t[(sgn, 1)], 0.42, 0.42, B_SLOT['steel'])            # end posts
            box_between(M, nodes_b[(sgn, npan)], nodes_t[(sgn, npan - 1)], 0.42, 0.42, B_SLOT['steel'])
            for i in range(1, npan):
                box_between(M, nodes_b[(sgn, i)], nodes_t[(sgn, i)], 0.24, 0.3, B_SLOT['steel'])        # verticals
            for i in range(1, npan - 1):                                                                # Pratt diagonals
                if i < npan / 2:
                    box_between(M, nodes_t[(sgn, i)], nodes_b[(sgn, i + 1)], 0.2, 0.08, B_SLOT['steel'])
                else:
                    box_between(M, nodes_b[(sgn, i)], nodes_t[(sgn, i + 1)], 0.2, 0.08, B_SLOT['steel'])
            # low railing inside the truss plane
            for i in range(npan):
                a = nodes_b[(sgn, i)] + np.array([0, 0, 1.05]); b = nodes_b[(sgn, i + 1)] + np.array([0, 0, 1.05])
                box_between(M, a, b, 0.08, 0.12, B_SLOT['steel'])
        for i in range(1, npan):
            box_between(M, nodes_t[(-1, i)], nodes_t[(1, i)], 0.2, 0.3, B_SLOT['steel'])                 # top struts
            box_between(M, nodes_b[(-1, i)] - np.array([0, 0, 0.45]), nodes_b[(1, i)] - np.array([0, 0, 0.45]), 0.3, 0.6, B_SLOT['steel'])  # floor beams
        for i in range(1, npan - 1):
            box_between(M, nodes_t[(-1, i)], nodes_t[(1, i + 1)], 0.1, 0.1, B_SLOT['steel'])             # lateral bracing
            box_between(M, nodes_t[(1, i)], nodes_t[(-1, i + 1)], 0.1, 0.1, B_SLOT['steel'])
        # portal knee braces
        for i in (1, npan - 1):
            box_between(M, nodes_t[(-1, i)] - np.array([0, 0, 1.2]), nodes_t[(1, i)] - np.array([0, 0, 1.2]), 0.12, 0.35, B_SLOT['steel'])
        # deck slab under the road surface
    def soffit(_):
        return np.stack([np.full(n, -ut), np.full(n, ut)], 1), np.stack([zc - 0.35, zc - 0.35], 1)
    _sweep(M, Fr, soffit, B_SLOT['concrete'])
    # piers between spans, masonry-look abutments
    for si in range(1, nsp):
        p, nn, tt = at(Fr['s'][0] + L * si / nsp)
        g = float(T.at(*w2px(p[0], p[1]))) - 0.6
        cyl_at(M, p[:2], 0.9, g - 1.0, p[2] - 0.8, B_SLOT['concrete'], seg=14, rx=ut + 0.8, ax=nn)
    for k, dirn in ((0, -1), (n - 1, 1)):
        c = Fr['P'][k, :2]; t_ = Fr['T'][k]; n_ = Fr['N'][k]
        g = float(T.at(*w2px(c[0], c[1])))
        box_at(M, c + t_ * dirn * 0.7, t_, n_, 0.9, ut + 0.9, min(g, zc[k] - 1) - 2.0, float(zc[k]) - 0.3, B_SLOT['concrete'])
        riprap(M, c, t_, n_, dirn, ut + 0.9, T)
        for sgn in (-1, 1):   # short concrete wingwalls retaining the approach fill
            p0 = c + n_ * sgn * (ut + 0.8)
            for j in range(3):
                a0 = p0 + t_ * dirn * (0.9 + j * 1.8)
                gg = float(T.at(*w2px(a0[0], a0[1])))
                ztop = float(zc[k]) + 0.3 - j * 0.9
                if ztop < gg - 0.2:
                    break
                box_at(M, a0, t_, n_, 0.9, 0.2, gg - 1.2, ztop, B_SLOT['concrete'])


def _seg_x(a0, a1, b0, b1):
    d1 = a1 - a0; d2 = b1 - b0
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((b0[0] - a0[0]) * d2[1] - (b0[1] - a0[1]) * d2[0]) / den
    u = ((b0[0] - a0[0]) * d1[1] - (b0[1] - a0[1]) * d1[0]) / den
    if 0 <= t <= 1 and 0 <= u <= 1:
        return a0 + d1 * t
    return None


_ALL_LINES = {}


def _lines_by_id():
    if not _ALL_LINES:
        for f in load('data/roads/roads.geojson')['features'] + load('data/railways/railways.geojson')['features']:
            c = np.asarray(f['geometry']['coordinates'], float)[:, :2]
            X, Y = px2w(c[:, 0], c[:, 1])
            _ALL_LINES[f['properties']['id']] = np.c_[X, Y]
    return _ALL_LINES


def _lower_pts(e, pr, road_by_id):
    """World XY points where the lower road/railway crosses under this deck."""
    if pr['kind'] != 'overpass' or not pr.get('over'):
        return None
    lo = _lines_by_id().get(pr['over'])
    if lo is None:
        return None
    A = e['P'][:, :2]
    out = []
    for i in range(len(A) - 1):
        for j in range(len(lo) - 1):
            q = _seg_x(A[i], A[i + 1], lo[j], lo[j + 1])
            if q is not None:
                out.append(q)
    return np.asarray(out) if out else None


def build_bridges(coll, net, T, mats):
    idx = {e['p']['id']: e for e in net.E}
    road_by_id = idx
    out = 0
    bm = [mats['bridge_concrete'], mats['truss'], mats['railing'], mats['girder_steel'], mats['riprap']]
    for b in load('data/roads/bridges.geojson')['features']:
        pr = b['properties']
        if pr['kind'] not in ('bridge', 'viaduct', 'overpass'):
            continue
        e = idx.get(pr['road'])
        if e is None or '_deck' not in e:
            continue
        c = np.asarray(b['geometry']['coordinates'], float)[:, :2]
        X, Y = px2w(c[:, 0], c[:, 1])
        Pe = e['P'][:, :2]
        se = np.r_[0, np.cumsum(np.hypot(*np.diff(Pe, axis=0).T))]
        def station(q):
            best = None
            for i in range(len(Pe) - 1):
                a, bb = Pe[i], Pe[i + 1]
                v = bb - a
                t = np.clip(np.dot(q - a, v) / max(np.dot(v, v), 1e-9), 0, 1)
                d = np.hypot(*(a + v * t - q))
                if best is None or d < best[0]:
                    best = (d, se[i] + t * np.hypot(*v))
            return best[1]
        s0, s1 = sorted([station(np.array([X[0], Y[0]])), station(np.array([X[-1], Y[-1]]))])
        Fr = _deck_frame(e, s0, s1)
        if Fr is None:
            continue
        M = Mesh()
        st = pr.get('structure', 'concrete girder')
        if 'truss' in st or 'trestle' in st:
            truss_bridge(M, Fr, T)
        else:
            lower = None; over = None
            if pr['kind'] == 'overpass':
                lo = road_by_id.get(pr.get('over'))
                over = lo['sec']['t'] if lo is not None else ('rail' if str(pr.get('over', '')).startswith('RR') else None)
            girder_bridge(M, Fr, T, pr['kind'], st, over=over, lower_pts=_lower_pts(e, pr, road_by_id))
        if M.empty():
            continue
        ob = M.to_object(f"BRIDGE {pr['id']}", coll, bm, smooth=False)
        ob['kind'] = pr['kind']; ob['structure'] = st; ob['world_id'] = pr['id']; ob['road'] = pr['road']
        out += 1
    return out


# ------------------------------------------------------------------ instancing helper
def instance_points(name, coll, pts, rotz, proto, scale=None, extra=None, rotx=None, roty=None):
    """Geometry-Nodes instancer: one vertex per instance (attributes rotx/roty/rotz, scl)."""
    pts = np.asarray(pts, np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return None
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(pts))
    me.vertices.foreach_set('co', pts.ravel())
    a = me.attributes.new('rotz', 'FLOAT', 'POINT'); a.data.foreach_set('value', np.asarray(rotz, np.float32))
    for an, v in (('rotx', rotx), ('roty', roty)):
        a = me.attributes.new(an, 'FLOAT', 'POINT')
        a.data.foreach_set('value', np.zeros(len(pts), np.float32) if v is None else np.asarray(v, np.float32))
    sc = np.ones(len(pts), np.float32) if scale is None else np.asarray(scale, np.float32)
    a = me.attributes.new('scl', 'FLOAT', 'POINT'); a.data.foreach_set('value', sc)
    for k, v in (extra or {}).items():
        a = me.attributes.new(k, 'FLOAT', 'POINT'); a.data.foreach_set('value', np.asarray(v, np.float32))
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    ng = bpy.data.node_groups.get('GN_A2_Instancer_' + proto.name)
    if ng is None:
        ng = bpy.data.node_groups.new('GN_A2_Instancer_' + proto.name, 'GeometryNodeTree')
        ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        N_ = ng.nodes; L = ng.links
        gi = N_.new('NodeGroupInput'); go = N_.new('NodeGroupOutput')
        oi = N_.new('GeometryNodeObjectInfo'); oi.inputs['Object'].default_value = proto
        iop = N_.new('GeometryNodeInstanceOnPoints')
        ar = N_.new('GeometryNodeInputNamedAttribute'); ar.data_type = 'FLOAT'; ar.inputs['Name'].default_value = 'rotz'
        ax_ = N_.new('GeometryNodeInputNamedAttribute'); ax_.data_type = 'FLOAT'; ax_.inputs['Name'].default_value = 'rotx'
        ay_ = N_.new('GeometryNodeInputNamedAttribute'); ay_.data_type = 'FLOAT'; ay_.inputs['Name'].default_value = 'roty'
        asc = N_.new('GeometryNodeInputNamedAttribute'); asc.data_type = 'FLOAT'; asc.inputs['Name'].default_value = 'scl'
        cmb = N_.new('ShaderNodeCombineXYZ')
        L.new(ar.outputs['Attribute'], cmb.inputs['Z']); L.new(ax_.outputs['Attribute'], cmb.inputs['X']); L.new(ay_.outputs['Attribute'], cmb.inputs['Y'])
        L.new(gi.outputs[0], iop.inputs['Points']); L.new(oi.outputs['Geometry'], iop.inputs['Instance'])
        L.new(cmb.outputs[0], iop.inputs['Rotation']); L.new(asc.outputs['Attribute'], iop.inputs['Scale'])
        L.new(iop.outputs[0], go.inputs[0])
    mod = ob.modifiers.new('Instancer', 'NODES'); mod.node_group = ng
    return ob


def proto_collection(root_coll):
    c = bpy.data.collections.get('A2_PROTOTYPES')
    if c is None:
        c = bpy.data.collections.new('A2_PROTOTYPES')
        root_coll.children.link(c)
        c.hide_render = True
        c.hide_viewport = True
    return c


def proto_object(name, M, mats, root_coll):
    if name in bpy.data.objects:
        return bpy.data.objects[name]
    ob = M.to_object(name, proto_collection(root_coll), mats, smooth=False)
    return ob


# ------------------------------------------------------------------ railways
def build_rail(coll, T, mats, bbox=None, root_coll=None):
    tie = Mesh()
    box_at(tie, (0, 0), (1, 0), (0, 1), 0.11, 1.3, -0.1, 0.08, 0)
    tie_proto = proto_object('A2_PROTO_RailTie', tie, [mats['tie']], root_coll or coll)
    n_obj = 0
    for f in load('data/railways/railways.geojson')['features']:
        p = f['properties']
        c = np.asarray(f['geometry']['coordinates'], float)
        if c.shape[1] < 3:
            continue
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            if c[:, 0].max() < x0 or c[:, 0].min() > x1 or c[:, 1].max() < y0 or c[:, 1].min() > y1:
                continue
        X, Y = px2w(c[:, 0], c[:, 1])
        e = {'P': np.c_[X, Y, c[:, 2]], 'trim': [0, 0], 'tan': [None, None]}
        S = sample_edge(e, 2.0)
        if S is None:
            continue
        P, N, Tn, ss = S['P'], S['N'], S['T'], S['s']
        n = len(P)
        tracks = p.get('tracks', 1)
        spacing = 4.0
        centres = [(t - (tracks - 1) / 2) * spacing for t in range(tracks)]
        top_hw = (tracks - 1) * spacing / 2 + 1.9
        span = np.zeros(n, bool)
        for a, b in p.get('bridge_spans') or []:
            span |= (ss >= a * MPP) & (ss <= b * MPP)
        M = Mesh(('rs',))
        offs = np.arange(0.0, 20.01, 0.5)
        cols = {}
        for sgn in (-1, 1):
            terr = _probe(T, P, N, sgn, offs)
            ztop = P[:, 2] + 0.3
            u_t = np.full(n, top_hw)
            u_toe = top_hw + 0.45 * 1.5
            z_toe = P[:, 2] - 0.15
            cut = terr[:, np.searchsorted(offs, u_toe + 1.5)] > z_toe
            ucc, zcc = _catch(z_toe - 0.3, np.full(n, u_toe + 0.8), 1 / 1.5, terr, offs, True)
            ucf, zcf = _catch(z_toe, np.full(n, u_toe), -0.5, terr, offs, False)
            U = np.stack([u_t, np.full(n, u_toe), np.where(cut, u_toe + 0.8, ucf), np.where(cut, ucc, ucf + 0.4), np.where(cut, ucc + 0.7, ucf + 1.1)], 1)
            Z = np.stack([ztop, z_toe, np.where(cut, z_toe - 0.3, zcf), np.where(cut, zcc, zcf - 0.1), np.zeros(n)], 1)
            xs = P[:, 0:1] + N[:, 0:1] * U * sgn; ys = P[:, 1:2] + N[:, 1:2] * U * sgn
            Z[:, 4] = T.at(*w2px(xs[:, 4], ys[:, 4])) - 0.4
            cols[sgn] = (U * sgn, Z)
        Uall = np.concatenate([cols[-1][0][:, ::-1], cols[1][0]], 1)
        Zall = np.concatenate([cols[-1][1][:, ::-1], cols[1][1]], 1)
        G = np.stack([P[:, 0:1] + N[:, 0:1] * Uall, P[:, 1:2] + N[:, 1:2] * Uall, Zall], -1)
        # ballast (0), verge (1)
        bands = [1, 1, 1, 0, 0, 0, 1, 1, 1]
        k = 0
        while k < n - 1:
            j = k
            while j + 1 < n and span[j + 1] == span[k]:
                j += 1
            j = min(j + 1, n - 1)
            if not span[k]:
                M.grid(G[k:j + 1], bands, rs=np.broadcast_to(ss[k:j + 1, None], (j + 1 - k, 10)))
            k = j
        # rails: head/web/base profile swept per track
        rp = np.array([(-0.075, 0.0), (-0.035, 0.012), (-0.009, 0.03), (-0.01, 0.12), (-0.036, 0.13), (-0.036, 0.165), (0.036, 0.165), (0.036, 0.13), (0.01, 0.12), (0.009, 0.03), (0.035, 0.012), (0.075, 0.0)])
        zr0 = P[:, 2] + 0.38
        for cu in centres:
            for rail_u in (cu - 0.7525, cu + 0.7525):
                GU = rail_u + rp[None, :, 0]
                GR = np.stack([P[:, 0:1] + N[:, 0:1] * GU, P[:, 1:2] + N[:, 1:2] * GU, zr0[:, None] + rp[None, :, 1]], -1)
                M.grid(GR, [2] * (len(rp) - 1), rs=0.0)
        mats_r = [mats['ballast'], mats['verge'], mats['rail_steel']]
        ob = M.to_object(f"RAIL {p['id']}", coll, mats_r, smooth=True, uv_from_px=True, luw=_LUW[0])
        ob['world_id'] = p['id']; ob['tracks'] = tracks
        # ties (instanced)
        st = np.arange(ss[0] + 0.3, ss[-1] - 0.3, 0.6)
        tp, tr = [], []
        ang = np.arctan2(Tn[:, 1], Tn[:, 0])
        for cu in centres:
            xs_ = np.interp(st, ss, P[:, 0] + N[:, 0] * cu); ys_ = np.interp(st, ss, P[:, 1] + N[:, 1] * cu)
            zs_ = np.interp(st, ss, P[:, 2]) + 0.3
            an = np.interp(st, ss, np.unwrap(ang))
            tp.append(np.c_[xs_, ys_, zs_]); tr.append(an)
        tp = np.vstack(tp); tr = np.concatenate(tr)
        jit = np.random.default_rng(len(tp)).normal(0, 0.02, len(tr))
        instance_points(f"RAIL {p['id']} TIES", coll, tp, tr + jit, tie_proto)
        # railway bridges: deck plate girders on concrete piers
        for a, b in p.get('bridge_spans') or []:
            k0, k1 = np.searchsorted(ss, a * MPP), np.searchsorted(ss, b * MPP)
            k0, k1 = max(0, k0 - 1), min(n - 1, k1 + 1)
            if k1 - k0 < 2:
                continue
            BM = Mesh()
            sl = slice(k0, k1 + 1)
            Fr = {'s': ss[sl], 'P': P[sl], 'T': Tn[sl], 'N': N[sl]}
            m_ = k1 - k0 + 1
            for cu in centres:
                for gu in (cu - 1.0, cu + 1.0):
                    U = np.stack([np.full(m_, gu - 0.25), np.full(m_, gu + 0.25), np.full(m_, gu + 0.25), np.full(m_, gu - 0.25), np.full(m_, gu - 0.25)], 1)
                    Zg = np.stack([P[sl, 2] + 0.2, P[sl, 2] + 0.2, P[sl, 2] - 2.4, P[sl, 2] - 2.4, P[sl, 2] + 0.2], 1)
                    _sweep(BM, Fr, lambda _f, U=U, Zg=Zg: (U, Zg), 1)
                # deck (ballasted) + walkway edges
                _sweep(BM, Fr, lambda _f, cu=cu: (np.stack([np.full(m_, cu - 1.6), np.full(m_, cu + 1.6)], 1), np.stack([P[sl, 2] + 0.22, P[sl, 2] + 0.22], 1)), 0)
            L = ss[k1] - ss[k0]
            nsp = max(1, int(round(L / 24)))
            for i in range(1, nsp):
                sst = ss[k0] + L * i / nsp
                kk = int(np.argmin(np.abs(ss - sst)))
                g = float(T.at(*w2px(P[kk, 0], P[kk, 1]))) - 0.6
                cyl_at(BM, P[kk, :2], 0.8, g - 1.0, P[kk, 2] - 2.4, 0, seg=14, rx=top_hw, ax=N[kk])
            for kk, dirn in ((k0, -1), (k1, 1)):
                g = float(T.at(*w2px(P[kk, 0], P[kk, 1])))
                box_at(BM, P[kk, :2] + Tn[kk] * dirn * 0.6, Tn[kk], N[kk], 0.8, top_hw + 0.6, min(g, P[kk, 2] - 2) - 2, P[kk, 2] + 0.2, 0)
            if not BM.empty():
                BM.to_object(f"BRIDGE {p['id']}_RB{k0}", coll, [mats['bridge_concrete'], mats['girder_steel']], smooth=False)
        n_obj += 1
    return n_obj


# ------------------------------------------------------------------ entry points
_LUW = [None]


def build_roads(coll, net, T, mats, luw=None):
    J = plan_junctions(net)
    ctl = controls(net, J)
    net.J, net.ctl = J, ctl
    mouths = {}
    slots = [mats[k] for k in SLOT_ORDER]
    n_obj = 0
    shared = {}
    for i, e in enumerate(net.E):
        r = build_edge(net, i, T, ctl, shared)
        if r is None:
            continue
        M, mo = r
        for end in (0, 1):
            mouths[(i, end)] = mo[end]
        ob = M.to_object(f"ROAD {e['p']['id']}", coll, slots, smooth=True, uv_from_px=True, luw=luw)
        p = e['p']
        for k in ('type', 'name', 'route', 'material', 'surface', 'lanes', 'width_m', 'speed_mph'):
            if p.get(k) is not None:
                ob[k] = p[k]
        ob['world_id'] = p['id']
        n_obj += 1
    jc = bpy.data.collections.get('JUNCTIONS') or bpy.data.collections.new('JUNCTIONS')
    if jc.name not in coll.children:
        coll.children.link(jc)
    nj = 0
    for nid, j in J.items():
        M = build_junction(net, nid, j, mouths, T)
        if M is None or M.empty():
            continue
        ob = M.to_object(f'JUNCTION {nid}', jc, slots, smooth=True, uv_from_px=True, luw=luw)
        ob['world_id'] = nid
        ob['kind'] = net.nodes.get(nid, {}).get('properties', {}).get('kind', '')
        nj += 1
    return n_obj, nj


def build(root, ctx):
    """Roads + junctions + bridges + railways. Returns the road network (reused by
    lib_infrastructure through ctx['road_net'])."""
    T = ctx['T']
    bbox = ctx.get('bbox')
    luw = ctx.get('luw')
    _LUW[0] = luw
    mats = materials()
    mk = ctx.get('collection')

    def coll(name):
        if mk:
            return mk(name, root)
        c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
        if c.name not in root.children:
            root.children.link(c)
        return c
    net = Net(bbox)
    cr = coll('ROADS')
    ne, nj = build_roads(cr, net, T, mats, luw)
    nb = build_bridges(coll('BRIDGES'), net, T, mats)
    nr = build_rail(coll('RAIL'), T, mats, bbox, root)
    ctx['road_net'] = net
    # verges use the terrain material: give them the same ecology attributes as the terrain chunks
    try:
        import lib_materials as LM
        objs = [o for c in (cr, bpy.data.collections.get('JUNCTIONS'), bpy.data.collections.get('RAIL')) if c for o in c.objects if o.type == 'MESH' and not o.modifiers]
        LM.add_terrain_attributes(objs)
    except Exception as ex:
        print('  (terrain attributes on road verges skipped:', ex, ')')
    print(f'  lib_roads: {ne} road edges, {nj} junctions, {nb} bridges, {nr} railways')
    return net
