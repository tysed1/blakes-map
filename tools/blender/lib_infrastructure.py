"""Roadside infrastructure for the 1974 north-Georgia world (Agent 2: roads & infrastructure).

Everything is placed by rules from the road data (never scattered):
  * wooden utility poles (crossarm, glass insulators, some transformer cans) with sagging
    catenary power conductors + a telephone cable, along highways / rural / collector /
    main streets, one side per route, ~45 m spans, skipping decks and junction mouths
  * W-beam guardrail on timber posts (turned-down ends): high fills / steep drops, the
    outside of sharp curves over a drop, and 25 m approaches at every bridge end
  * MUTCD-1971 signs: STOP on stop-controlled approaches, curve warnings before sharp
    curves, SPEED LIMIT 55 (national limit, Jan 1974) after highway junctions, US / GA route
    markers, railroad crossbucks + advance-warning discs at every at-grade crossing
  * white delineator posts along highways and rural roads
  * farm fences (timber posts, barbed wire) where fields / pastures border rural roads,
    with tube gates at driveways
  * culvert headwalls where roads cross mapped creeks without a bridge
  * stone retaining walls from data/roads/walls.geojson (dense town cores)
  * the railway tunnel portal at the Georgia Northern's west end
Repeated props are Geometry-Nodes instances of prototypes (collection A2_PROTOTYPES).

Entry point: lib_infrastructure.build(root_collection, ctx)  (after lib_roads.build: it
reuses ctx['road_net'], the per-edge sampled frames built by lib_roads).
"""
import bpy, bmesh, math, os
import numpy as np
import lib_roads as LR
from lib_roads import Mesh, NB, box_at, box_between, cyl_at, w2px, px2w, load, instance_points, proto_object

RNG = np.random.default_rng(1974)


# ------------------------------------------------------------------ materials
def _mat_wood(name, col=(0.13, 0.1, 0.075)):
    m = bpy.data.materials.new(name)
    b = NB(m)
    obj = b.tc.outputs['Object']
    grain = b.noise(b.mapping(obj, (1 / 0.08, 1 / 0.08, 1 / 1.5)), 1.0, 6.0, 0.7)
    c = b.mix(b.smooth(grain, 0.3, 0.75), col, tuple(x * 0.62 for x in col))
    c = b.mix(b.mul(b.smooth(b.noise(obj, 0.9, 3.0), 0.55, 0.8), 0.5), c, (0.16, 0.155, 0.15))  # silvered weathering
    b.nt.links.new(c, b.inp('Base Color'))
    b.inp('Roughness').default_value = 0.93
    b.nt.links.new(b.bump(grain, 0.4, 0.01), b.inp('Normal'))
    return m


def _mat_flat(name, col, rough=0.6, metal=0.0, emit=0.0):
    m = bpy.data.materials.new(name)
    b = NB(m)
    obj = b.tc.outputs['Object']
    n = b.noise(obj, 6.0, 3.0)
    c = b.mix(b.mul(b.smooth(n, 0.4, 0.8), 0.25), col, tuple(x * 0.7 for x in col))
    b.nt.links.new(c, b.inp('Base Color'))
    b.inp('Roughness').default_value = rough
    b.inp('Metallic').default_value = metal
    return m


def _mat_glass(name, col=(0.35, 0.55, 0.45)):
    m = bpy.data.materials.new(name)
    b = NB(m)
    b.inp('Base Color').default_value = (*col, 1)
    b.inp('Roughness').default_value = 0.15
    b.inp('Transmission Weight' if 'Transmission Weight' in b.bsdf.inputs else 'Transmission').default_value = 0.6
    return m


def _mat_stone(name='MAT_Infra_StoneWall'):
    """Rubble masonry (local granite / fieldstone) with recessed mortar."""
    m = bpy.data.materials.new(name)
    b = NB(m)
    obj = b.tc.outputs['Object']
    v = b.mapping(obj, (1 / 0.55, 1 / 0.55, 1 / 0.32))
    cell = b.voronoi(v, 1.0, 'F1', 'Color')
    ed = b.voronoi(v, 1.0, 'DISTANCE_TO_EDGE', 'Distance')
    sep = b.nt.nodes.new('ShaderNodeSeparateColor'); b._in(sep.inputs[0], cell)
    stone = b.mix(sep.outputs['Red'], (0.2, 0.19, 0.17), (0.34, 0.31, 0.27))
    stone = b.mix(b.mul(sep.outputs['Green'], 0.5), stone, (0.28, 0.2, 0.14))
    mortar = b.lt(ed, 0.06)
    c = b.mix(mortar, stone, (0.42, 0.41, 0.38))
    c = b.mix(b.mul(b.smooth(b.noise(obj, 0.6, 3.0), 0.55, 0.8), 0.5), c, (0.08, 0.1, 0.05))  # moss/grime
    b.nt.links.new(c, b.inp('Base Color'))
    b.inp('Roughness').default_value = 0.92
    b.nt.links.new(b.bump(b.smooth(ed, 0.0, 0.12), 0.8, 0.05), b.inp('Normal'))
    return m


_M = {}


def mats():
    if _M:
        return _M
    _M.update({
        'pole': _mat_wood('MAT_Infra_PoleWood', (0.11, 0.085, 0.065)),
        'wood': _mat_wood('MAT_Infra_Timber', (0.14, 0.11, 0.08)),
        'glass': _mat_glass('MAT_Infra_InsulatorGlass'),
        'galv': _mat_flat('MAT_Infra_Galvanized', (0.42, 0.43, 0.43), 0.45, 0.8),
        'steel_dark': _mat_flat('MAT_Infra_SteelDark', (0.12, 0.12, 0.12), 0.5, 0.6),
        'wire': _mat_flat('MAT_Infra_Wire', (0.05, 0.05, 0.05), 0.4, 0.7),
        'white': _mat_flat('MAT_Sign_White', (0.72, 0.72, 0.7), 0.5),
        'black': _mat_flat('MAT_Sign_Black', (0.012, 0.012, 0.012), 0.6),
        'red': _mat_flat('MAT_Sign_Red', (0.45, 0.018, 0.015), 0.5),
        'yellow': _mat_flat('MAT_Sign_Yellow', (0.78, 0.55, 0.03), 0.5),
        'signback': _mat_flat('MAT_Sign_Back', (0.3, 0.3, 0.3), 0.5, 0.7),
        'stone': _mat_stone(),
        'concrete': LR.materials()['bridge_concrete'],
        'dark': _mat_flat('MAT_Infra_Void', (0.005, 0.005, 0.005), 1.0),
        'reflector': _mat_flat('MAT_Infra_Reflector', (0.8, 0.45, 0.05), 0.2),
    })
    return _M


# ------------------------------------------------------------------ prototypes
def _text_mesh(M, text, size, center, mat, depth=0.004, face=-1):
    """Add text (font -> mesh) centred at center (x, z) on the plane y = face*depth."""
    cu = bpy.data.curves.new('A2_txt', 'FONT')
    cu.body = text
    cu.size = size
    cu.align_x = 'CENTER'
    cu.align_y = 'CENTER'
    cu.extrude = 0.0
    ob = bpy.data.objects.new('A2_txt', cu)
    bpy.context.scene.collection.objects.link(ob)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    V = np.array([v.co[:] for v in bm.verts])
    F = np.array([[v.index for v in f.verts] for f in bm.faces])
    bm.free()
    bpy.data.objects.remove(ob); bpy.data.curves.remove(cu); bpy.data.meshes.remove(me)
    if len(F) == 0:
        return
    # text lies in XY facing +Z -> stand it up in the XZ plane facing -Y (towards traffic)
    P = np.c_[V[:, 0] + center[0], np.full(len(V), face * depth), V[:, 1] + center[1]]
    F = F[:, ::-1] if face < 0 else F
    M.add(P, F, mat)


def _plate(M, poly_xz, mat, y=0.0, thick=0.004, back_mat=None):
    """Flat sign plate from an XZ polygon, front face at y facing -Y, back face behind."""
    P = np.asarray(poly_xz, float)
    n = len(P)
    tris = LR.triangulate(P)
    Vf = np.c_[P[:, 0], np.full(n, y), P[:, 1]]
    Vb = np.c_[P[:, 0], np.full(n, y + thick), P[:, 1]]
    F = np.asarray(tris)
    # front faces must face -Y: triangles CCW in XZ seen from -Y
    M.add(Vf, F[:, ::-1], mat)
    M.add(Vb, F, back_mat if back_mat is not None else mat)


def _circle(r, n=24, cx=0.0, cz=0.0):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.c_[cx + r * np.cos(a), cz + r * np.sin(a)]


def _post(M, height, mat, w=0.06):
    box_at(M, (0, 0.03), (1, 0), (0, 1), w / 2, w / 2, -0.4, height, mat)


SIGN_SLOTS = ['galv', 'white', 'black', 'red', 'yellow', 'signback', 'wood', 'glass', 'reflector', 'steel_dark']
S_ = {k: i for i, k in enumerate(SIGN_SLOTS)}


def _slots():
    m = mats()
    return [m[k] for k in SIGN_SLOTS]


def build_prototypes(root):
    """Prototype objects (hidden collection) -> dict name -> object."""
    P = {}
    sl = _slots()
    # --- utility pole (35 ft class, 9.5 m above ground), local X = crossarm axis, pole at origin
    M = Mesh()
    cyl_at(M, (0, 0), 0.14, -1.5, 9.6, S_['wood'], seg=10)
    box_at(M, (0, -0.16), (1, 0), (0, 1), 1.22, 0.05, 9.05, 9.17, S_['wood'])          # crossarm (8 ft)
    for x in (-1.1, -0.4, 1.1):
        cyl_at(M, (x, -0.16), 0.012, 9.17, 9.27, S_['galv'], seg=6)
        cyl_at(M, (x, -0.16), 0.05, 9.27, 9.38, S_['glass'], seg=8)
    box_between(M, (-0.55, -0.16, 9.05), (0.0, -0.16, 8.55), 0.03, 0.05, S_['galv'])  # braces
    box_between(M, (0.55, -0.16, 9.05), (0.0, -0.16, 8.55), 0.03, 0.05, S_['galv'])
    cyl_at(M, (0, 0.14), 0.035, 7.85, 7.95, S_['glass'], seg=8)                        # neutral spool
    box_at(M, (0, 0.2), (1, 0), (0, 1), 0.05, 0.06, 5.45, 5.6, S_['steel_dark'])       # telephone cable clamp
    P['pole'] = proto_object('A2_PROTO_Pole', M, sl, root)
    M2 = Mesh()
    _copy(M, M2)
    cyl_at(M2, (0, 0.42), 0.26, 7.3, 8.3, S_['steel_dark'], seg=14)
    box_at(M2, (0, 0.2), (1, 0), (0, 1), 0.06, 0.1, 8.0, 8.1, S_['galv'])
    P['pole_tx'] = proto_object('A2_PROTO_PoleTransformer', M2, sl, root)
    # --- guardrail post (timber 6x8 with block-out), rail attaches at y = -0.35
    M = Mesh()
    box_at(M, (0, 0.0), (1, 0), (0, 1), 0.075, 0.1, -0.9, 0.62, S_['wood'])
    box_at(M, (0, -0.2), (1, 0), (0, 1), 0.075, 0.1, 0.36, 0.62, S_['wood'])
    P['gr_post'] = proto_object('A2_PROTO_GuardrailPost', M, sl, root)
    # --- delineator (white steel post + amber reflector)
    M = Mesh()
    box_at(M, (0, 0), (1, 0), (0, 1), 0.04, 0.02, -0.4, 1.2, S_['white'])
    box_at(M, (0, -0.025), (1, 0), (0, 1), 0.035, 0.006, 1.0, 1.12, S_['reflector'])
    P['delineator'] = proto_object('A2_PROTO_Delineator', M, sl, root)
    # --- STOP (30 in octagon)
    M = Mesh(); _post(M, 1.95, S_['galv'])
    r = 0.4
    octo = [(r * math.cos(math.radians(22.5 + 45 * k)), 2.15 + r * math.sin(math.radians(22.5 + 45 * k))) for k in range(8)]
    _plate(M, octo, S_['red'], back_mat=S_['signback'])
    octi = [(0.92 * x, 2.15 + 0.92 * (z - 2.15)) for x, z in octo]
    _plate(M, octi, S_['red'], y=-0.001)
    _text_mesh(M, 'STOP', 0.2, (0.0, 2.15), S_['white'], depth=0.003)
    P['stop'] = proto_object('A2_PROTO_Sign_Stop', M, sl, root)
    # --- curve warnings (30 in diamond, black arrow)
    for nm, sgn in (('curve_r', 1), ('curve_l', -1)):
        M = Mesh(); _post(M, 1.9, S_['galv'])
        h = 0.42
        _plate(M, [(0, 2.1 - h), (h, 2.1), (0, 2.1 + h), (-h, 2.1)], S_['yellow'], back_mat=S_['signback'])
        arrow = [(-0.04 * sgn, 1.86), (0.04 * sgn, 1.86), (0.04 * sgn, 2.12), (0.12 * sgn, 2.24), (0.16 * sgn, 2.2), (0.2 * sgn, 2.34),
                 (0.06 * sgn, 2.33), (0.1 * sgn, 2.29), (-0.02 * sgn, 2.16), (-0.04 * sgn, 2.12)]
        if sgn < 0:
            arrow = arrow[::-1]
        _plate(M, arrow, S_['black'], y=-0.003, thick=0.001)
        P[nm] = proto_object(f'A2_PROTO_Sign_{nm}', M, sl, root)
    # --- SPEED LIMIT 55
    M = Mesh(); _post(M, 1.8, S_['galv'])
    _plate(M, [(-0.3, 1.8), (0.3, 1.8), (0.3, 2.55), (-0.3, 2.55)], S_['white'], back_mat=S_['signback'])
    _text_mesh(M, 'SPEED', 0.1, (0.0, 2.44), S_['black'], depth=0.003)
    _text_mesh(M, 'LIMIT', 0.1, (0.0, 2.32), S_['black'], depth=0.003)
    _text_mesh(M, '55', 0.28, (0.0, 2.04), S_['black'], depth=0.003)
    P['speed55'] = proto_object('A2_PROTO_Sign_Speed55', M, sl, root)
    # --- railroad crossbuck (+ number-of-tracks plate) and advance-warning disc
    M = Mesh()
    box_at(M, (0, 0.03), (1, 0), (0, 1), 0.05, 0.05, -0.4, 3.6, S_['white'])
    for ang in (35, -35):
        a = math.radians(ang)
        c, s_ = math.cos(a), math.sin(a)
        L_, Wb = 1.22, 0.11
        pts = [(-L_ * c - Wb * -s_, 3.15 - L_ * s_ - Wb * c), (L_ * c - Wb * -s_, 3.15 + L_ * s_ - Wb * c),
               (L_ * c + Wb * -s_, 3.15 + L_ * s_ + Wb * c), (-L_ * c + Wb * -s_, 3.15 - L_ * s_ + Wb * c)]
        _plate(M, pts, S_['white'], y=-0.01 if ang > 0 else -0.02, back_mat=S_['signback'])
    _text_mesh(M, 'RAILROAD', 0.085, (-0.02, 3.3), S_['black'], depth=0.025)
    _text_mesh(M, 'CROSSING', 0.085, (0.02, 3.0), S_['black'], depth=0.025)
    P['crossbuck'] = proto_object('A2_PROTO_Sign_Crossbuck', M, sl, root)
    M = Mesh(); _post(M, 1.8, S_['galv'])
    _plate(M, _circle(0.46, 28, 0, 2.25), S_['yellow'], back_mat=S_['signback'])
    for ang in (45, -45):
        a = math.radians(ang); c, s_ = math.cos(a), math.sin(a)
        _plate(M, [(-0.4 * c + 0.03 * s_, 2.25 - 0.4 * s_ - 0.03 * c), (0.4 * c + 0.03 * s_, 2.25 + 0.4 * s_ - 0.03 * c),
                   (0.4 * c - 0.03 * s_, 2.25 + 0.4 * s_ + 0.03 * c), (-0.4 * c - 0.03 * s_, 2.25 - 0.4 * s_ + 0.03 * c)], S_['black'], y=-0.003, thick=0.001)
    _text_mesh(M, 'R', 0.2, (-0.2, 2.25), S_['black'], depth=0.004)
    _text_mesh(M, 'R', 0.2, (0.2, 2.25), S_['black'], depth=0.004)
    P['rr_advance'] = proto_object('A2_PROTO_Sign_RRAdvance', M, sl, root)
    # --- route markers (1971 MUTCD US shield: white shield on black square; GA: white disc)
    for route in ('19', '76', '129', '400', '60', '9'):
        M = Mesh(); _post(M, 2.0, S_['galv'])
        s = 0.3
        _plate(M, [(-s, 2.0), (s, 2.0), (s, 2.0 + 2 * s), (-s, 2.0 + 2 * s)], S_['black'], back_mat=S_['signback'])
        if route in ('19', '76', '129'):
            sh = [(-0.25, 2.5), (-0.2, 2.56), (0.0, 2.52), (0.2, 2.56), (0.25, 2.5), (0.25, 2.26), (0.15, 2.12), (0.0, 2.05), (-0.15, 2.12), (-0.25, 2.26)]
            _plate(M, sh, S_['white'], y=-0.002, thick=0.001)
            _text_mesh(M, 'U.S.', 0.06, (0.0, 2.46), S_['black'], depth=0.004)
        else:
            _plate(M, _circle(0.25, 28, 0, 2.3), S_['white'], y=-0.002, thick=0.001)
        _text_mesh(M, route, 0.2 if len(route) < 3 else 0.15, (0.0, 2.28), S_['black'], depth=0.004)
        P['route_' + route] = proto_object(f'A2_PROTO_Sign_Route{route}', M, sl, root)
    # --- farm fence post, gate
    M = Mesh()
    cyl_at(M, (0, 0), 0.06, -0.6, 1.3, S_['wood'], seg=6)
    P['fence_post'] = proto_object('A2_PROTO_FencePost', M, sl, root)
    M = Mesh()
    for x in (0.0, 4.2):
        cyl_at(M, (x, 0), 0.1, -0.8, 1.5, S_['wood'], seg=8)
    for z in (0.25, 0.55, 0.85, 1.15):
        box_between(M, (0.1, 0, z), (4.1, 0, z), 0.04, 0.04, S_['galv'])
    box_between(M, (0.1, 0, 0.25), (4.1, 0, 1.15), 0.03, 0.03, S_['galv'])
    P['gate'] = proto_object('A2_PROTO_FarmGate', M, sl, root)
    return P


def _copy(src, dst):
    off = 0
    for V, F, m_ in zip(src.V, src.F, src.M):
        dst.add(V, F - off, m_)
        off += len(V)
