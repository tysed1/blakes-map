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
        'lens': _mat_flat('MAT_Infra_SignalLens', (0.32, 0.012, 0.008), 0.15),
        'refractor': _mat_flat('MAT_Infra_Refractor', (0.62, 0.6, 0.52), 0.2),
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


SIGN_SLOTS = ['galv', 'white', 'black', 'red', 'yellow', 'signback', 'wood', 'glass', 'reflector', 'steel_dark', 'lens', 'refractor']
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
    # --- YIELD (1971 MUTCD R1-2: red-bordered white down-pointing triangle, 36 in)
    M = Mesh(); _post(M, 1.75, S_['galv'])
    tri = [(-0.46, 2.36), (0.46, 2.36), (0.0, 1.56)]
    _plate(M, tri, S_['red'], back_mat=S_['signback'])
    cz = (2.36 * 2 + 1.56) / 3
    _plate(M, [(x * 0.62, cz + (z - cz) * 0.62) for x, z in tri], S_['white'], y=-0.001, thick=0.001)
    _text_mesh(M, 'YIELD', 0.085, (0.0, 2.2), S_['red'], depth=0.003)
    P['yield'] = proto_object('A2_PROTO_Sign_Yield', M, sl, root)
    # --- STOP + '4-WAY' plate (all-way stop, R1-3)
    M = Mesh(); _copy_proto(P['stop'], M)
    _plate(M, [(-0.26, 1.52), (0.26, 1.52), (0.26, 1.7), (-0.26, 1.7)], S_['red'], back_mat=S_['signback'])
    _text_mesh(M, '4-WAY', 0.1, (0.0, 1.61), S_['white'], depth=0.003)
    P['stop4'] = proto_object('A2_PROTO_Sign_Stop4Way', M, sl, root)
    # --- intersection warnings (W2-1 cross road, W2-2 side road, W2-4 T), 30 in diamonds
    for nm in ('xroad', 'side_l', 'side_r', 'tee'):
        M = Mesh(); _post(M, 1.9, S_['galv'])
        h = 0.42
        _plate(M, [(0, 2.1 - h), (h, 2.1), (0, 2.1 + h), (-h, 2.1)], S_['yellow'], back_mat=S_['signback'])
        bars = []
        if nm in ('xroad', 'side_l', 'side_r'):
            bars.append((-0.035, 1.86, 0.035, 2.34))                       # through road (vertical)
        if nm == 'xroad':
            bars.append((-0.24, 2.065, 0.24, 2.135))
        elif nm == 'side_r':
            bars.append((0.0, 2.065, 0.24, 2.135))
        elif nm == 'side_l':
            bars.append((-0.24, 2.065, 0.0, 2.135))
        else:  # T ahead
            bars += [(-0.035, 1.86, 0.035, 2.2), (-0.24, 2.2, 0.24, 2.27)]
        for x0, z0, x1, z1 in bars:
            _plate(M, [(x0, z0), (x1, z0), (x1, z1), (x0, z1)], S_['black'], y=-0.003, thick=0.001)
        P[nm] = proto_object(f'A2_PROTO_Sign_{nm}', M, sl, root)
    # --- JCT assemblies: 'JCT' plate over the route marker
    for route in ('19', '76', '129', '400', '60', '9'):
        M = Mesh(); _copy_proto(P['route_' + route], M)
        box_at(M, (0, 0.07), (1, 0), (0, 1), 0.03, 0.03, 1.9, 2.85, S_['galv'])   # post extension behind the plates
        _plate(M, [(-0.3, 2.64), (0.3, 2.64), (0.3, 2.85), (-0.3, 2.85)], S_['white'], back_mat=S_['signback'])
        _text_mesh(M, 'JCT', 0.13, (0.0, 2.745), S_['black'], depth=0.003)
        P['jct_' + route] = proto_object(f'A2_PROTO_Sign_Jct{route}', M, sl, root)
    # --- 1970s cobra-head street light: galvanized pole + davit arm, luminaire over the lane
    # (local -Y points from the pole towards the road)
    M = Mesh()
    cyl_at(M, (0, 0), 0.13, -0.3, 0.5, S_['galv'], seg=10)                             # base collar
    cyl_at(M, (0, 0), 0.105, 0.5, 5.0, S_['galv'], seg=10)
    cyl_at(M, (0, 0), 0.08, 5.0, 8.9, S_['galv'], seg=10)
    box_between(M, (0, 0.0, 8.75), (0, -1.2, 9.25), 0.045, 0.045, S_['galv'])            # davit arm, rising
    box_between(M, (0, -1.2, 9.25), (0, -2.35, 9.32), 0.04, 0.04, S_['galv'])
    box_between(M, (0, -0.1, 8.1), (0, -1.1, 9.2), 0.018, 0.018, S_['galv'])            # brace
    box_at(M, (0, -2.62), (1, 0), (0, 1), 0.2, 0.36, 9.12, 9.36, S_['galv'])            # cobra head housing
    box_at(M, (0, -2.66), (1, 0), (0, 1), 0.15, 0.27, 9.0, 9.12, S_['refractor'])       # prismatic refractor bowl
    P['streetlight'] = proto_object('A2_PROTO_StreetLight', M, sl, root)
    # --- railroad flashing-light signal (crossbuck + twin red flashers + bell) and gate
    def _rr_signal(M):
        cyl_at(M, (0, 0.03), 0.065, -0.4, 4.75, S_['galv'], seg=10)
        cyl_at(M, (0, 0.03), 0.09, 4.75, 4.9, S_['black'], seg=12)                    # bell
        for ang in (35, -35):
            a = math.radians(ang); c, s_ = math.cos(a), math.sin(a)
            L_, Wb = 1.22, 0.11
            pts = [(-L_ * c - Wb * -s_, 4.15 - L_ * s_ - Wb * c), (L_ * c - Wb * -s_, 4.15 + L_ * s_ - Wb * c),
                   (L_ * c + Wb * -s_, 4.15 + L_ * s_ + Wb * c), (-L_ * c + Wb * -s_, 4.15 - L_ * s_ + Wb * c)]
            _plate(M, pts, S_['white'], y=-0.02 if ang > 0 else -0.03, back_mat=S_['signback'])
        _text_mesh(M, 'RAILROAD', 0.085, (-0.02, 4.3), S_['black'], depth=0.035)
        _text_mesh(M, 'CROSSING', 0.085, (0.02, 4.0), S_['black'], depth=0.035)
        box_at(M, (0, -0.05), (1, 0), (0, 1), 0.72, 0.04, 2.98, 3.06, S_['galv'])       # flasher crossarm
        for x in (-0.58, 0.58):
            _plate(M, _circle(0.3, 20, x, 3.02), S_['black'], y=-0.1, thick=0.01)         # backgrounds
            cyl_at(M, (x, -0.16), 0.15, 2.9, 3.14, S_['black'], seg=12)                   # lamp housing
            _plate(M, _circle(0.13, 16, x, 3.02), S_['lens'], y=-0.285, thick=0.004)     # red roundel
            box_at(M, (x, -0.26), (1, 0), (0, 1), 0.16, 0.1, 3.15, 3.18, S_['black'])     # visor
        _plate(M, [(-0.2, 2.45), (0.2, 2.45), (0.2, 2.72), (-0.2, 2.72)], S_['white'], y=-0.02, back_mat=S_['signback'])
        _text_mesh(M, '1', 0.16, (0.0, 2.585), S_['black'], depth=0.025)
        _text_mesh(M, 'TRACK', 0.045, (0.0, 2.5), S_['black'], depth=0.025)
    M = Mesh(); _rr_signal(M)
    P['rr_flasher'] = proto_object('A2_PROTO_Sign_RRFlasher', M, sl, root)
    M = Mesh(); _rr_signal(M)
    box_at(M, (0.32, 0.1), (1, 0), (0, 1), 0.16, 0.12, 0.2, 1.35, S_['black'])          # gate mechanism
    for k in range(10):                                                                   # arm raised (clear)
        z0 = 1.2 + k * 0.42
        box_at(M, (0.52, 0.1), (1, 0), (0, 1), 0.045, 0.03, z0, z0 + 0.42, S_['red'] if k % 2 else S_['white'])
    box_at(M, (0.52, 0.1), (1, 0), (0, 1), 0.07, 0.05, 1.0, 1.25, S_['steel_dark'])     # counterweight
    P['rr_gate'] = proto_object('A2_PROTO_Sign_RRGate', M, sl, root)
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


def _copy_proto(ob, dst):
    """Copy a built prototype object's mesh (per-face slot) into a Mesh builder."""
    me = ob.data
    me.calc_loop_triangles()
    V = np.array([v.co[:] for v in me.vertices])
    tl = np.array([t.vertices[:] for t in me.loop_triangles], np.int64)
    tm = np.array([t.material_index for t in me.loop_triangles], np.int32)
    dst.add(V[tl.reshape(-1)], np.arange(len(tl) * 3).reshape(-1, 3), tm)


def _copy(src, dst):
    off = 0
    for V, F, m_ in zip(src.V, src.F, src.M):
        dst.add(V, F - off, m_)
        off += len(V)


# ------------------------------------------------------------------ road chains (continuous routes)
def _node_deg(net):
    deg = {}
    for n, legs in net.legs.items():
        deg[n] = len(legs)
    return deg


def chains(net, types):
    """Continuous sampled routes (edges of one road chained through their nodes).
    Returns dicts with s, P, N, T, hw, zl, zr, deck, near (junction mouth zone), sec, key."""
    deg = _node_deg(net)
    groups = {}
    for i, e in enumerate(net.E):
        if '_deck' not in e or e['sec']['t'] not in types:
            continue
        key = e['p'].get('def_id') or ('_' + e['p']['id'])
        groups.setdefault(key, []).append(i)
    out = []
    for key, idxs in groups.items():
        adj = {}
        for i in idxs:
            for end, nd in ((0, net.E[i]['p']['from']), (1, net.E[i]['p']['to'])):
                adj.setdefault(nd, []).append((i, end))
        used = set()
        starts = [nd for nd, l in adj.items() if len(l) == 1] + list(adj)
        for st in starts:
            cur = st
            parts = []
            while True:
                nxt = [(i, end) for i, end in adj.get(cur, []) if i not in used]
                if not nxt:
                    break
                i, end = nxt[0]
                used.add(i)
                parts.append((i, end == 0))
                e = net.E[i]
                cur = e['p']['to'] if end == 0 else e['p']['from']
            if not parts:
                continue
            S_, P_, N_, T_, hw_, zl_, zr_, dk_, nr_ = [], [], [], [], [], [], [], [], []
            off = 0.0
            for i, fwd in parts:
                e = net.E[i]
                D = e['_deck']
                s = D['s'] - D['s'][0]
                deck = LR._spans_mask(e, D['s'])
                near = np.zeros(len(s), bool)
                L = s[-1]
                if deg.get(e['p']['from'], 0) >= 3:
                    near |= D['s'] < e['trim'][0] + 9.0
                if deg.get(e['p']['to'], 0) >= 3:
                    near |= (D['L'] - D['s']) < e['trim'][1] + 9.0
                if fwd:
                    S_.append(off + s); P_.append(D['P']); N_.append(D['N']); T_.append(D['T']); hw_.append(D['hw'])
                    zl_.append(D['zl']); zr_.append(D['zr']); dk_.append(deck); nr_.append(near)
                else:
                    S_.append(off + (L - s[::-1])); P_.append(D['P'][::-1]); N_.append(-D['N'][::-1]); T_.append(-D['T'][::-1])
                    hw_.append(D['hw'][::-1]); zl_.append(D['zr'][::-1]); zr_.append(D['zl'][::-1]); dk_.append(deck[::-1]); nr_.append(near[::-1])
                off += L + 0.5 + e['trim'][0] + e['trim'][1]
            out.append({'key': key, 's': np.concatenate(S_), 'P': np.vstack(P_), 'N': np.vstack(N_), 'T': np.vstack(T_),
                        'hw': np.concatenate(hw_), 'zl': np.concatenate(zl_), 'zr': np.concatenate(zr_), 'deck': np.concatenate(dk_),
                        'near': np.concatenate(nr_), 'sec': net.E[parts[0][0]]['sec'], 'edges': [i for i, _ in parts],
                        'zone': net.E[parts[0][0]]['zone'], 'p': net.E[parts[0][0]]['p']})
    return out


def _at(ch, st):
    """Interpolated frame of a chain at stations st -> P (n,3), N, T, hw, zl, zr, deck, near."""
    s = ch['s']
    st = np.atleast_1d(st)
    f = lambda a: np.interp(st, s, a)
    P = np.c_[f(ch['P'][:, 0]), f(ch['P'][:, 1]), f(ch['P'][:, 2])]
    T = np.c_[f(ch['T'][:, 0]), f(ch['T'][:, 1])]
    T /= np.maximum(np.hypot(T[:, 0], T[:, 1]), 1e-9)[:, None]
    N = np.stack([T[:, 1], -T[:, 0]], 1)
    idx = np.clip(np.searchsorted(s, st), 0, len(s) - 1)
    return P, N, T, f(ch['hw']), f(ch['zl']), f(ch['zr']), ch['deck'][idx], ch['near'][idx]


def _side_offset(sec):
    """Distance from the pavement edge to the back of the road's side detail (m)."""
    if sec['curb']:
        return 0.15 + sec['sw']
    return sec['g'] + sec['dw']


def _rotz_facing(n):
    """Instance yaw so that the prototype's front (local -Y) faces direction n (xy)."""
    return math.atan2(-n[1], -n[0]) - math.pi / 2


# ------------------------------------------------------------------ utility lines
POLE_TYPES = ('highway', 'rural', 'collector', 'main_street', 'arterial', 'gravel', 'urban_street')


def build_poles(coll, net, T, P_, luw=None):
    pts, rz, rx, ry, pts_tx, rz_tx, rx_tx, ry_tx = [], [], [], [], [], [], [], []
    wires, tele = [], []
    for ch in chains(net, POLE_TYPES):
        t = ch['sec']['t']
        L = ch['s'][-1]
        if L < 90 or ch['zone'] == 'downtown':
            continue
        if t in ('gravel', 'urban_street') and L < 250:
            continue
        h = sum(ord(c) for c in ch['key'])
        sgn = 1 if h % 2 else -1
        spacing = 42.0 if t != 'main_street' else 36.0
        st = np.arange(12.0 + (h % 17), L - 5, spacing)
        st = st + RNG.uniform(-3, 3, len(st))
        st = st[(st > 0) & (st < L)]
        P, N, Tn, hw, zl, zr, deck, near = _at(ch, st)
        ok = ~deck & ~near
        u = hw + _side_offset(ch['sec']) + (0.6 if ch['sec']['curb'] else 1.6)
        prev = None
        for k in range(len(st)):
            if not ok[k]:
                prev = None
                continue
            c = P[k, :2] + N[k] * u[k] * sgn
            g = float(T.at(*w2px(c[0], c[1])))
            if abs(g - P[k, 2]) > 6:  # deep cut / high fill: the line would not be there
                prev = None
                continue
            yaw = math.atan2(N[k, 1], N[k, 0])      # crossarm (local X) across the road
            lx, ly = RNG.normal(0, 0.012), RNG.normal(0, 0.012)
            tx = (RNG.random() < 0.14) or (t == 'main_street' and RNG.random() < 0.3)
            if tx:
                pts_tx.append((c[0], c[1], g)); rz_tx.append(yaw); rx_tx.append(lx); ry_tx.append(ly)
            else:
                pts.append((c[0], c[1], g)); rz.append(yaw); rx.append(lx); ry.append(ly)
            X = np.array([math.cos(yaw), math.sin(yaw), 0.0]); Y = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
            base = np.array([c[0], c[1], g])
            att = [base + X * x + Y * -0.16 + np.array([0, 0, 9.38]) for x in (-1.1, -0.4, 1.1)]
            att.append(base + Y * 0.14 + np.array([0, 0, 7.95]))
            tel = base + Y * 0.2 + np.array([0, 0, 5.52])
            if prev is not None and np.hypot(*(c - prev[0][:2])) < 75:
                for a, b in zip(prev[1], att):
                    wires.append(_catenary(a, b, 0.018))
                tele.append(_catenary(prev[2], tel, 0.028))
            prev = (base, att, tel)
    root = coll
    if pts:
        instance_points('INFRA Utility Poles', coll, pts, rz, P_['pole'], rotx=rx, roty=ry)
    if pts_tx:
        instance_points('INFRA Utility Poles (transformer)', coll, pts_tx, rz_tx, P_['pole_tx'], rotx=rx_tx, roty=ry_tx)
    _curves('INFRA Power Lines', coll, wires, 0.009)
    _curves('INFRA Telephone Cable', coll, tele, 0.017)
    return len(pts) + len(pts_tx)


def _catenary(a, b, sag_ratio, n=10):
    a = np.asarray(a, float); b = np.asarray(b, float)
    span = np.hypot(*(b[:2] - a[:2]))
    t = np.linspace(0, 1, n)[:, None]
    p = a + (b - a) * t
    p[:, 2] -= sag_ratio * span * 4 * t[:, 0] * (1 - t[:, 0])
    return p


def _curves(name, coll, polylines, radius):
    if not polylines:
        return None
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = radius
    cu.bevel_resolution = 0
    cu.fill_mode = 'FULL'
    for pl in polylines:
        sp = cu.splines.new('POLY')
        sp.points.add(len(pl) - 1)
        co = np.c_[pl, np.ones(len(pl))].astype(np.float32).ravel()
        sp.points.foreach_set('co', co)
    cu.materials.append(mats()['wire'])
    ob = bpy.data.objects.new(name, cu)
    coll.objects.link(ob)
    return ob


# ------------------------------------------------------------------ guardrails + delineators
GR_TYPES = ('highway', 'rural', 'collector', 'ramp', 'arterial', 'freeway')
W_PROFILE = np.array([(0.0, -0.16), (0.035, -0.135), (0.075, -0.085), (0.075, -0.045), (0.03, -0.012), (0.03, 0.012),
                      (0.075, 0.045), (0.075, 0.085), (0.035, 0.135), (0.0, 0.16)])


def _runs(mask, min_len, s):
    out, k = [], 0
    while k < len(mask):
        if mask[k]:
            j = k
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            if s[j] - s[k] >= min_len:
                out.append((k, j))
            k = j + 1
        else:
            k += 1
    return out


def build_guardrails(coll, net, T, P_):
    M = Mesh()
    posts, prz = [], []
    dl_pts, dl_rz = [], []
    n_runs = 0
    for ch in chains(net, GR_TYPES):
        sec = ch['sec']
        if sec['curb']:
            continue
        s = ch['s']
        n = len(s)
        if n < 4:
            continue
        P, N, Tn = ch['P'], ch['N'], ch['T']
        hw = ch['hw']
        deck = ch['deck']
        # curvature along the chain
        ang = np.unwrap(np.arctan2(Tn[:, 1], Tn[:, 0]))
        k_ = np.gradient(ang) / np.maximum(np.gradient(s), 1e-3)
        k_ = LR._gauss(k_, 3.0)
        for sgn in (-1, 1):
            ze = ch['zr'] if sgn > 0 else ch['zl']
            u_probe = hw + sec['g'] + 3.5
            X = P[:, 0] + N[:, 0] * u_probe * sgn; Y = P[:, 1] + N[:, 1] * u_probe * sgn
            drop = ze - T.at(*w2px(X, Y))
            need = drop > 2.3
            outside = (k_ * -sgn) > 0          # left turn (k>0): outside is the right side
            need |= (np.abs(k_) > 1 / 130.0) & outside & (drop > 1.0)
            # bridge approaches (both sides, 25 m each end)
            if deck.any():
                for a, b in _runs(deck, 0.0, s):
                    need |= ((s > s[a] - 25) & (s < s[a])) | ((s > s[b]) & (s < s[b] + 25))
            need &= ~deck & ~ch['near']
            # close small gaps
            for a, b in _runs(~need, 0.0, s):
                if 0 < a and b < n - 1 and s[b] - s[a] < 10:
                    need[a:b + 1] = True
            for a, b in _runs(need, 12.0, s):
                n_runs += 1
                touches_deck = [a > 0 and deck[a - 1], b < n - 1 and deck[b + 1]]
                st = np.arange(s[a], s[b], 1.0)
                if len(st) < 3:
                    continue
                Pp, Nn, Tt, hww, zl, zr, dk, nr = _at(ch, st)
                zz = zr if sgn > 0 else zl
                ur = hww + max(sec['g'], 0.3) - 0.05
                # turned-down ends (1970s terminal) unless the run ends at a bridge parapet
                f = np.ones(len(st))
                if not touches_deck[0]:
                    f = np.minimum(f, np.clip((st - st[0]) / 4.0, 0, 1))
                if not touches_deck[1]:
                    f = np.minimum(f, np.clip((st[-1] - st) / 4.0, 0, 1))
                zc = zz - 0.04 * sec['g'] + (0.55 * f - 0.25 * (1 - f))
                U = (ur[:, None] - W_PROFILE[None, :, 0]) * sgn
                G = np.stack([Pp[:, 0:1] + Nn[:, 0:1] * U, Pp[:, 1:2] + Nn[:, 1:2] * U, zc[:, None] + W_PROFILE[None, :, 1]], -1)
                M.grid(G, [0] * (len(W_PROFILE) - 1))
                for q in np.arange(1.5, len(st) - 1.5, 1.9):
                    qi = int(q)
                    if f[qi] < 0.95:
                        continue
                    c = Pp[qi, :2] + Nn[qi] * (ur[qi] + 0.28) * sgn
                    posts.append((c[0], c[1], zz[qi] - 0.04 * sec['g']))
                    prz.append(_rotz_facing(-Nn[qi] * sgn))
        # delineators: highways + rural roads, both sides, tighter on curves
        if sec['t'] in ('highway', 'rural', 'ramp'):
            st = [0.0]
            while st[-1] < s[-1]:
                kk = abs(np.interp(st[-1], s, k_))
                st.append(st[-1] + (24.0 if kk > 1 / 250 else 48.0))
            st = np.asarray(st[1:-1])
            if len(st):
                Pp, Nn, Tt, hww, zl, zr, dk, nr = _at(ch, st)
                for sgn in (-1, 1):
                    zz = zr if sgn > 0 else zl
                    for q in range(len(st)):
                        if dk[q] or nr[q]:
                            continue
                        c = Pp[q, :2] + Nn[q] * (hww[q] + sec['g'] + 0.35) * sgn
                        dl_pts.append((c[0], c[1], zz[q] - 0.06))
                        dl_rz.append(_rotz_facing(-Tt[q] * sgn))
    if not M.empty():
        M.to_object('INFRA Guardrail W-beam', coll, [mats()['galv']], smooth=True)
    instance_points('INFRA Guardrail Posts', coll, posts, prz, P_['gr_post'])
    instance_points('INFRA Delineators', coll, dl_pts, dl_rz, P_['delineator'])
    return n_runs, len(dl_pts)


# ------------------------------------------------------------------ signs
# side-road importance for intersection warnings (driveways / farm tracks never get one)
RANK_SIDE = {'driveway': 0, 'dirt': 1, 'gravel': 2, 'residential': 3, 'urban_street': 3, 'rural': 4, 'collector': 5,
             'main_street': 6, 'arterial': 6, 'highway': 8, 'ramp': 7, 'freeway': 9}
IC_ROUTE = {'IC_LC_SR400_US19': '400', 'IC_TV_SR60_US76': '60', 'IC_TV_SR60_RIVERFRONT': '60'}
ROUTE_OF = {'HWY_US19': '19', 'HWY_US19_E': '19', 'HR_MAIN_ST': '19', 'HWY_US76': '76', 'HWY_US129': '129',
            'HWY_SR400': '400', 'HWY_SR60': '60', 'HWY_SR9_W': '9', 'HWY_SR9_N': '9', 'HWY_SR9_S': '9', 'LC_SR9_CONNECTOR': '9'}


def build_signs(coll, net, T, P_):
    place = {}

    def add(kind, c, z, face_dir):
        place.setdefault(kind, ([], []))
        place[kind][0].append((c[0], c[1], z))
        place[kind][1].append(_rotz_facing(face_dir) + RNG.normal(0, 0.03))
    edge_by_id = {e['p']['id']: e for e in net.E if '_deck' in e}
    route_of_edge = lambda e: ROUTE_OF.get(e['p'].get('def_id') or '')

    def at_edge(e, st, side, off, face_sign):
        """Point beside edge e at station st: side +1 = right of the edge direction."""
        D = e['_deck']
        st = float(np.clip(st, D['s'][0], D['s'][-1]))
        f = lambda a: np.interp(st, D['s'], a)
        p = np.array([f(D['P'][:, 0]), f(D['P'][:, 1])]); nrm = np.array([f(D['N'][:, 0]), f(D['N'][:, 1])])
        tg = np.array([f(D['T'][:, 0]), f(D['T'][:, 1])])
        c = p + nrm * (f(D['hw']) + off) * side
        return c, tg * face_sign

    def approach(e, end, dist, off, kind, clip=False):
        """Sign on the right of traffic approaching node `end` of edge e, dist m before it."""
        D = e['_deck']
        L = D['s'][-1] + e['trim'][1]
        st = (e['trim'][0] + dist) if end == 0 else (L - e['trim'][1] - dist)
        if st < D['s'][0] - 0.5 or st > D['s'][-1] + 0.5:
            if not clip or D['s'][-1] - D['s'][0] < 2.0:
                return False
            st = float(np.clip(st, D['s'][0] + 0.5, D['s'][-1] - 0.5))
        if LR._spans_mask(e, np.array([float(np.clip(st, D['s'][0], D['s'][-1]))]))[0]:
            return False  # never on a bridge deck
        sgn = -1 if end == 0 else 1
        c, face = at_edge(e, st, sgn, off, 1 if end == 0 else -1)
        add(kind, c, float(T.at(*w2px(c[0], c[1]))), face)
        return True
    # regulatory control at junction approaches (lib_roads.controls: STOP / YIELD / all-way STOP)
    kinds = getattr(net, 'ctl_kind', {})
    for (i, end), kind in kinds.items():
        if not kind:
            continue
        e = net.E[i]
        if e.get('_deck') is None:
            continue
        stop, xw = getattr(net, 'ctl', {}).get((i, end), (0.0, 0.0))
        dist = (stop + 1.2) if stop > 0 else 2.2
        approach(e, end, dist, _side_offset(e['sec']) * 0.5 + 0.9, {'stop': 'stop', 'yield': 'yield', 'all_stop': 'stop4'}[kind], clip=True)
    # numbered-route junction assemblies (JCT + shield) on every approach of one route to another
    for n, j in getattr(net, 'J', {}).items():
        legs = [(a['i'], a['end']) for a in j['legs']]
        routes = {route_of_edge(net.E[i]) for i, _ in legs} - {None}
        # interchange ramps carry the freeway's route to the crossroad terminal
        for i, _ in legs:
            ic = net.E[i]['p'].get('interchange')
            if ic:
                routes.add(IC_ROUTE.get(ic))
        routes.discard(None)
        if len(routes) < 2:
            continue
        for i, end in legs:
            e = net.E[i]
            r = route_of_edge(e)
            if r is None or e['sec']['t'] in ('ramp', 'freeway') or e.get('_deck') is None:
                continue
            if e['p'].get('oneway') and end == 0:
                continue
            for other in sorted(routes - {r}):
                if 'jct_' + other not in P_:
                    continue
                for d in (150.0, 115.0, 80.0, 55.0, 35.0):
                    if approach(e, end, d, _side_offset(e['sec']) + 1.0, 'jct_' + other):
                        break
    # per-route signs
    CURVE_R = {'highway': 230.0, 'rural': 150.0, 'collector': 100.0}
    jnodes = {n: j for n, j in getattr(net, 'J', {}).items()}
    for ch in chains(net, ('highway', 'rural', 'collector', 'main_street', 'arterial', 'freeway', 'ramp')):
        sec = ch['sec']
        s = ch['s']
        if s[-1] < 60:
            continue
        Tn = ch['T']
        ang = np.unwrap(np.arctan2(Tn[:, 1], Tn[:, 0]))
        k_ = LR._gauss(np.gradient(ang) / np.maximum(np.gradient(s), 1e-3), 4.0)

        def put(kind, st, direction, off=0.8, slide=0.0):
            """slide > 0: if the spot is on a deck / in a junction mouth, try further along travel."""
            for tries in range(6 if slide else 1):
                q = st + direction * slide * tries
                if q < 5 or q > s[-1] - 5:
                    return
                P, N, Tt, hw, zl, zr, dk, nr = _at(ch, q)
                if not (dk[0] or nr[0]):
                    break
            else:
                return
            if dk[0] or nr[0]:
                return
            sgn = 1 if direction > 0 else -1          # right-hand side of travel
            u = hw[0] + _side_offset(sec) + off
            c = P[0, :2] + N[0] * u * sgn
            add(kind, c, float(T.at(*w2px(c[0], c[1]))), -Tt[0] * direction)
        if sec['t'] in CURVE_R and not sec['curb'] and ch['zone'] is None:
            # warn where the curve needs a speed below the road's, one sign per curve group
            R = 1.0 / np.maximum(np.abs(k_), 1e-6)
            last = -1e9
            for a, b in _runs(R < CURVE_R[sec['t']], 18.0, s):
                if s[a] - last < 180:
                    last = s[b]
                    continue
                left = np.mean(k_[a:b + 1]) > 0
                put('curve_l' if left else 'curve_r', s[a] - 55, 1)
                put('curve_r' if left else 'curve_l', s[b] + 55, -1)
                last = s[b]
        if sec['t'] in ('highway', 'rural') and ch['zone'] is None:
            # intersection warnings ahead of public side roads (not driveways / farm tracks)
            chain_edges = set(ch['edges'])
            xy = ch['P'][:, :2]
            for n, j in jnodes.items():
                if not any(a['i'] in chain_edges for a in j['legs']):
                    continue
                side = [a for a in j['legs'] if a['i'] not in chain_edges and
                        RANK_SIDE.get(a['t'], 0) >= (2 if sec['t'] == 'highway' else 3)]
                if not side:
                    continue
                k = int(np.argmin(np.hypot(xy[:, 0] - j['xy'][0], xy[:, 1] - j['xy'][1])))
                if np.hypot(*(xy[k] - j['xy'])) > 8:
                    continue
                st0 = s[k]
                Nk = np.array([Tn[k, 1], -Tn[k, 0]])
                sides = {int(np.sign(np.dot(a['d'], Nk)) or 1) for a in side}
                for direction in (1, -1):
                    if len(sides) == 2:
                        kind = 'xroad'
                    else:
                        right = (next(iter(sides)) * direction) > 0
                        kind = 'side_r' if right else 'side_l'
                    put(kind, st0 - direction * 120.0, direction)
        if sec['t'] in ('highway', 'freeway'):
            step = 1600.0 if sec['t'] == 'highway' else 2400.0
            for st in np.arange(140.0 if sec['t'] == 'highway' else 400.0, s[-1] - 100, step):
                put('speed55', st, 1, 1.2)
                put('speed55', s[-1] - st, -1, 1.2)
        r = ROUTE_OF.get(ch['key'])
        if r:
            # reassurance markers after junctions and every ~1 km (freeway: every ~1.5 km);
            # a short route piece still gets one marker per direction
            step = 950.0 if sec['t'] != 'freeway' else 1500.0
            first = min(200.0 if sec['t'] != 'freeway' else 350.0, 0.35 * s[-1])
            for st in np.arange(first, max(s[-1] - 100, first + 1), step):
                put('route_' + r, st, 1, 1.0 if sec['t'] != 'freeway' else 1.6, slide=25.0)
                put('route_' + r, s[-1] - st, -1, 1.0 if sec['t'] != 'freeway' else 1.6, slide=25.0)
    # town-centre street lighting: cobra heads on davit poles (Hollow Ridge Main St, the downtowns)
    for ch in chains(net, ('main_street', 'arterial', 'urban_street', 'collector')):
        if ch['zone'] not in ('downtown', 'town_center') or not ch['sec']['curb']:
            continue
        L = ch['s'][-1]
        if L < 25:
            continue
        both = ch['sec']['t'] in ('main_street', 'arterial')
        h = sum(ord(c) for c in ch['key'])
        st = np.arange(10.0 + (h % 11), L - 8, 38.0 if both else 42.0)
        if len(st) == 0:
            continue
        P, N, Tt, hw, zl, zr, dk, nr = _at(ch, st)
        for k in range(len(st)):
            if dk[k] or nr[k]:
                continue
            sgn = (1 if k % 2 else -1) if both else (1 if h % 2 else -1)
            c = P[k, :2] + N[k] * (hw[k] + 0.55) * sgn
            add('streetlight', c, float(T.at(*w2px(c[0], c[1]))), -N[k] * sgn)
    # railroad crossings: crossbucks (1971: every public crossing), flashing-light signals on the
    # mainline and busy yard streets, automatic gates where a collector or bigger crosses the main
    rails_ = {f['properties']['id']: f['properties'] for f in load('data/railways/railways.geojson')['features']}
    for f in load('data/roads/rail_crossings.geojson')['features']:
        pr = f['properties']
        e = edge_by_id.get(pr['road'])
        if e is None:
            continue
        rt = pr.get('road_type') or e['sec']['t']
        tracks = rails_.get(pr['rail'], {}).get('tracks', pr.get('tracks', 1))
        main = tracks == 1
        if main and rt in ('collector', 'arterial', 'highway', 'main_street'):
            xk = 'rr_gate'
        elif (main and rt in ('rural', 'urban_street', 'residential')) or (not main and rt in ('urban_street', 'collector', 'arterial')):
            xk = 'rr_flasher'
        else:
            xk = 'crossbuck'
        D = e['_deck']
        x, y = px2w(*f['geometry']['coordinates'])
        k = int(np.argmin(np.hypot(D['P'][:, 0] - x, D['P'][:, 1] - y)))
        st0 = D['s'][k]
        for direction in (1, -1):
            for kind, dist, extra in ((xk, 4.6, 1.0), ('rr_advance', 100.0, 0.8)):
                st = st0 - direction * dist
                if st < D['s'][0] or st > D['s'][-1]:
                    if kind != 'rr_advance':
                        continue
                    st = float(np.clip(st, D['s'][0] + 3, D['s'][-1] - 3))
                    if abs(st - st0) < 35:
                        continue
                f_ = lambda a: np.interp(st, D['s'], a)
                p = np.array([f_(D['P'][:, 0]), f_(D['P'][:, 1])]); nrm = np.array([f_(D['N'][:, 0]), f_(D['N'][:, 1])])
                tg = np.array([f_(D['T'][:, 0]), f_(D['T'][:, 1])])
                c = p + nrm * (f_(D['hw']) + _side_offset(e['sec']) * 0.5 + 1.2 + extra) * direction
                add(kind, c, float(T.at(*w2px(c[0], c[1]))), -tg * direction)
    print('   signs:', {k: len(v[0]) for k, v in place.items()})
    # timber crossing panels on the road at every at-grade crossing (between and beside the rails)
    PM = Mesh()
    rails = {f['properties']['id']: f for f in load('data/railways/railways.geojson')['features']}
    for f in load('data/roads/rail_crossings.geojson')['features']:
        pr = f['properties']
        e = edge_by_id.get(pr['road'])
        rf = rails.get(pr['rail'])
        if e is None or rf is None:
            continue
        rc = np.asarray(rf['geometry']['coordinates'], float)
        X, Y = px2w(rc[:, 0], rc[:, 1])
        x, y = px2w(*f['geometry']['coordinates'])
        k = int(np.argmin(np.hypot(X - x, Y - y)))
        k2 = min(k + 1, len(X) - 1) if k + 1 < len(X) else k - 1
        td = np.array([X[k2] - X[k], Y[k2] - Y[k]]); td /= max(np.hypot(*td), 1e-9)
        tn = np.array([-td[1], td[0]])
        zr_ = float(np.interp(0, [0], [rc[k, 2]]))
        D = e['_deck']
        kk = int(np.argmin(np.hypot(D['P'][:, 0] - x, D['P'][:, 1] - y)))
        half = D['hw'][kk] + 0.8
        ang = abs(np.dot(td, D['T'][kk]))
        across = half / max(math.sqrt(max(1 - ang * ang, 0.05)), 0.3)
        for off, w in ((-1.1, 0.6), (0.0, 1.35), (1.1, 0.6)):
            c = np.array([x, y]) + tn * off
            box_at(PM, c, td, tn, across, w / 2, zr_ + 0.39, zr_ + 0.515, 0)
    if not PM.empty():
        PM.to_object('INFRA Railroad Crossing Panels', coll, [mats()['wood']], smooth=False)
    n = 0
    for kind, (pts, rz) in place.items():
        if kind in P_:
            instance_points(f'INFRA Signs {kind}', coll, pts, rz, P_[kind])
            n += len(pts)
    return n


# ------------------------------------------------------------------ fences
def build_fences(coll, net, T, P_, lu):
    posts, prz, gates, grz = [], [], [], []
    wires = []
    if lu is None:
        return 0
    deg = _node_deg(net)
    for ch in chains(net, ('rural', 'gravel', 'dirt', 'collector')):
        if ch['zone'] is not None or ch['sec']['curb']:
            continue
        s = ch['s']
        if s[-1] < 40:
            continue
        st = np.arange(2.0, s[-1] - 2, 3.2)
        P, N, Tt, hw, zl, zr, dk, nr = _at(ch, st)
        for sgn in (-1, 1):
            u = hw + _side_offset(ch['sec']) + 4.0
            X = P[:, 0] + N[:, 0] * u * sgn; Y = P[:, 1] + N[:, 1] * u * sgn
            x, y = w2px(X + N[:, 0] * 6 * sgn, Y + N[:, 1] * 6 * sgn)
            cls = lu[np.clip(y.astype(int), 0, lu.shape[0] - 1), np.clip(x.astype(int), 0, lu.shape[1] - 1)]
            ok = np.isin(cls, (4, 5)) & ~dk & ~nr
            for a, b in _runs(ok, 25.0, st):
                g = T.at(*w2px(X[a:b + 1], Y[a:b + 1]))
                for q in range(a, b + 1):
                    posts.append((X[q], Y[q], g[q - a] + RNG.normal(0, 0.03)))
                    prz.append(RNG.uniform(0, 6.28))
                for hgt in (0.55, 0.85, 1.12):
                    pl = np.c_[X[a:b + 1], Y[a:b + 1], g + hgt]
                    wires.append(pl)
                # a tube gate at the start of each fenced field (farm entrance)
                if b - a > 20:
                    q = a + 2
                    gates.append((X[q], Y[q], float(g[2])))
                    grz.append(math.atan2(Tt[q, 1], Tt[q, 0]))
    instance_points('INFRA Fence Posts', coll, posts, prz, P_['fence_post'])
    instance_points('INFRA Farm Gates', coll, gates, grz, P_['gate'])
    _curves('INFRA Fence Wire', coll, wires, 0.004)
    return len(posts)


# ------------------------------------------------------------------ culverts, walls, tunnel portal
def build_culverts(coll, net, T):
    M = Mesh()
    ww = []
    for f in load('data/water/waterways.geojson')['features']:
        c = np.asarray(f['geometry']['coordinates'], float)[:, :2]
        X, Y = px2w(c[:, 0], c[:, 1])
        ww.append(np.c_[X, Y])
    n = 0
    for e in net.E:
        D = e.get('_deck')
        if D is None:
            continue
        A = D['P'][:, :2]
        bb0, bb1 = A.min(0) - 5, A.max(0) + 5
        deck = LR._spans_mask(e, D['s'])
        for W_ in ww:
            if (W_[:, 0].max() < bb0[0]) or (W_[:, 0].min() > bb1[0]) or (W_[:, 1].max() < bb0[1]) or (W_[:, 1].min() > bb1[1]):
                continue
            for i in range(len(A) - 1):
                if deck[i] or deck[i + 1]:
                    continue
                for j in range(len(W_) - 1):
                    q = LR._seg_x(A[i], A[i + 1], W_[j], W_[j + 1])
                    if q is None:
                        continue
                    k = i
                    t_ = D['T'][k]; nrm = D['N'][k]
                    u = D['hw'][k] + _side_offset(e['sec']) + 0.8
                    for sgn in (-1, 1):
                        c = q + nrm * u * sgn
                        g = float(T.at(*w2px(c[0], c[1])))
                        top = min(float(D['P'][k, 2]) - 0.1, g + 1.3)
                        box_at(M, c, t_, nrm, 1.5, 0.18, g - 0.6, max(top, g + 0.9), 0)          # headwall
                        for w in (-1, 1):                                                          # wingwalls
                            box_at(M, c + t_ * w * 1.6 + nrm * sgn * 0.6, nrm, t_, 0.7, 0.14, g - 0.6, g + 0.6, 0)
                        box_at(M, c + nrm * sgn * 0.2, t_, nrm, 0.45, 0.05, g - 0.1, g + 0.55, 1)  # pipe mouth (dark)
                    n += 1
    # cross culverts at every sag of the road profile (where runoff from both sides collects)
    for ch in chains(net, ('highway', 'rural', 'collector', 'gravel', 'arterial', 'ramp')):
        if ch['sec']['curb']:
            continue
        s, z = ch['s'], ch['P'][:, 2]
        if len(s) < 20:
            continue
        for i in range(8, len(s) - 8):
            w = (s > s[i] - 45) & (s < s[i] + 45)
            if z[i] > z[w].min() + 1e-6 or ch['deck'][i] or ch['near'][i]:
                continue
            if min(z[w][0], z[w][-1]) - z[i] < 0.6:
                continue
            t_ = ch['T'][i]; nrm = ch['N'][i]
            u = ch['hw'][i] + _side_offset(ch['sec']) + 0.6
            for sgn in (-1, 1):
                c = ch['P'][i, :2] + nrm * u * sgn
                g = float(T.at(*w2px(c[0], c[1])))
                top = min(float(z[i]) - 0.15, g + 1.2)
                box_at(M, c, t_, nrm, 1.3, 0.16, g - 0.6, max(top, g + 0.8), 0)
                box_at(M, c + nrm * sgn * 0.17, t_, nrm, 0.4, 0.02, g - 0.05, g + 0.5, 1)
            n += 1
    if not M.empty():
        M.to_object('INFRA Culvert Headwalls', coll, [mats()['concrete'], mats()['dark']], smooth=False)
    return n


def build_walls(coll, T, bbox=None):
    M = Mesh(('rs',))
    n = 0
    for f in load('data/roads/walls.geojson')['features']:
        pr = f['properties']
        if pr['kind'] != 'retaining_wall':
            continue
        c = np.asarray(f['geometry']['coordinates'], float)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            if c[:, 0].max() < x0 or c[:, 0].min() > x1 or c[:, 1].max() < y0 or c[:, 1].min() > y1:
                continue
        X, Y = px2w(c[:, 0], c[:, 1])
        e = {'P': np.c_[X, Y, np.zeros(len(X))], 'trim': [0, 0], 'tan': [None, None]}
        S = LR.sample_edge(e, 1.5)
        if S is None:
            continue
        P, N = S['P'], S['N']
        tops = np.asarray(pr['top_z_m'], float)
        ztop = np.interp(np.linspace(0, 1, len(P)), np.linspace(0, 1, len(tops)), tops)
        g = T.at(*w2px(P[:, 0], P[:, 1]))
        base = np.minimum(g, pr['base_z_m']) - 0.6
        ztop = np.maximum(ztop, base + 1.0) + 0.3
        sgn = 1 if pr['side'] == 'right' else -1
        U = np.stack([np.zeros(len(P)), np.zeros(len(P)), np.full(len(P), 0.45), np.full(len(P), 0.45)], 1) * sgn
        Z = np.stack([base, ztop, ztop, base], 1)
        G = np.stack([P[:, 0:1] + N[:, 0:1] * U, P[:, 1:2] + N[:, 1:2] * U, Z], -1)
        M.grid(G, [0, 0, 0], rs=np.broadcast_to(S['s'][:, None], (len(P), 4)))
        n += 1
    if not M.empty():
        M.to_object('INFRA Retaining Walls', coll, [mats()['stone']], smooth=False)
    return n


def build_tunnel_portals(coll, T):
    M = Mesh()
    n = 0
    for f in load('data/railways/railways.geojson')['features']:
        c = np.asarray(f['geometry']['coordinates'], float)
        if c.shape[1] < 3:
            continue
        man = [r for r in LR.load('data/manual/roads/40_railways.json')['roads'] if r['id'] == f['properties']['id']]
        if not man or not man[0].get('tunnel_until_px'):
            continue
        tx, ty = man[0]['tunnel_until_px']
        k = 0 if np.hypot(c[0, 0] - tx, c[0, 1] - ty) < np.hypot(c[-1, 0] - tx, c[-1, 1] - ty) else -1
        k2 = 1 if k == 0 else -2
        X, Y = px2w(c[[k, k2], 0], c[[k, k2], 1])
        d = np.array([X[0] - X[1], Y[0] - Y[1]]); d /= np.hypot(*d)   # into the tunnel
        nrm = np.array([d[1], -d[0]])
        base = np.array([X[0], Y[0]])
        z0 = float(c[k, 2]) - 0.2
        wi, hi, Wd, Hd = 2.6, 6.2, 5.5, 10.0
        inner, outer = [], []
        m = 16
        for q in range(m + 1):
            a = math.pi * q / m
            ix = -math.cos(a) * wi; iz = hi - wi + math.sin(a) * wi
            inner.append((ix, iz))
        inner = [(-wi, 0.0)] + inner + [(wi, 0.0)]
        for q in range(len(inner)):
            t_ = q / (len(inner) - 1)
            if t_ < 0.33:
                ox, oz = -Wd, Hd * (t_ / 0.33)
            elif t_ < 0.67:
                ox, oz = -Wd + 2 * Wd * (t_ - 0.33) / 0.34, Hd
            else:
                ox, oz = Wd, Hd * (1 - (t_ - 0.67) / 0.33)
            outer.append((ox, oz))
        inner = np.asarray(inner); outer = np.asarray(outer)
        for depth, mat in ((0.0, 0), (1.2, 0)):
            G = np.zeros((len(inner), 2, 3))
            for q in range(len(inner)):
                for j, (xx, zz) in enumerate((inner[q], outer[q])):
                    p = base + nrm * xx + d * depth
                    G[q, j] = (p[0], p[1], z0 + zz)
            M.grid(G, [mat])
        # dark bore into the hill
        G = np.zeros((len(inner), 2, 3))
        for q in range(len(inner)):
            for j, dep in enumerate((0.3, 30.0)):
                p = base + nrm * inner[q, 0] + d * dep
                G[q, j] = (p[0], p[1], z0 + inner[q, 1])
        M.grid(G, [1])
        # wingwalls into the slope
        for sgn in (-1, 1):
            p0 = base + nrm * sgn * Wd
            for j in range(3):
                a0 = p0 - d * (j * 2.5) + nrm * sgn * j * 1.2
                box_at(M, a0, d, nrm, 1.3, 0.35, z0 - 1, z0 + Hd - j * 2.8, 0)
        n += 1
    if not M.empty():
        M.to_object('INFRA Tunnel Portals', coll, [mats()['concrete'], mats()['dark']], smooth=False)
    return n


# ------------------------------------------------------------------ entry point
def build(root, ctx):
    net = ctx.get('road_net')
    if net is None:
        net = LR.Net(ctx.get('bbox'))
    T = ctx['T']
    mk = ctx.get('collection')
    coll = mk('INFRASTRUCTURE', root) if mk else root
    P_ = build_prototypes(root)
    lu = None
    try:
        lu = np.fromfile(os.path.join(LR.ROOT, 'public/world/landuse_u8.bin'), np.uint8).reshape(LR.H, LR.W)
    except Exception:
        pass
    npole = build_poles(coll, net, T, P_)
    ngr, ndl = build_guardrails(coll, net, T, P_)
    nsg = build_signs(coll, net, T, P_)
    nfe = build_fences(coll, net, T, P_, lu)
    ncu = build_culverts(coll, net, T)
    nwa = build_walls(coll, T, ctx.get('bbox'))
    ntu = build_tunnel_portals(coll, T)
    print(f'  lib_infrastructure: {npole} poles, {ngr} guardrail runs, {ndl} delineators, {nsg} signs, {nfe} fence posts, '
          f'{ncu} culverts, {nwa} walls, {ntu} tunnel portals')
