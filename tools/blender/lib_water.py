"""Rivers and creeks for the Blender world (A1 Terrain & Water).

    import lib_water as LW
    LW.build_water(coll, mat=None)   # -> WATER_Cxx_yy surfaces, WATER_Bed_*, WATER_Shore_*, WATER_Boulders

Data (written by tools/pipeline/terrain.py; numpy only, Blender's Python has no scipy):
  data/terrain/water_level_f32.bin   water surface (NaN where dry), stepped at riffles/rapids
  data/terrain/water_fx_u8.bin       [4][H][W] depth, surface slope, flow direction, bank/bed type
  data/terrain/height_graded_f32.bin terrain (bed + banks), sampled exactly like build_world's terrain

Look (graphics ref.png): clear shallows showing a rocky bed, dark tea-green pools, sky reflections,
flow-aligned ripples, whitewater only on steep reaches, wet dark stone at the waterline, boulders
in rapids and along rocky banks.

Objects
  WATER_Cxx_yy        water surface per 100 px chunk (attrs: depth, foam, flow, shore)
  WATER_Bed_Cxx_yy    riverbed + gravel bars + rocky banks, 3 cm above the terrain (attrs: depth, bank, edge)
  WATER_Shore_Cxx_yy  wet film on the banks just above the waterline (attr: wet)
  WATER_Boulders      boulders in rapids, pools and on rocky banks (one mesh)
"""
import bpy, math, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PH = os.path.join(ROOT, 'assets/external/polyhaven')
W, H, MPP, OX, OY, CHUNK = 2000, 667, 2.5, 1000.0, 333.5, 100


def P(*p):
    return os.path.join(ROOT, *p)


# ------------------------------------------------------------------ data
def _fill_nearest(a, iters=4):
    """Fill NaNs from finite 8-neighbours (a few rings; enough for mesh skirts)."""
    a = a.copy()
    for _ in range(iters):
        nan = ~np.isfinite(a)
        if not nan.any():
            break
        acc = np.zeros_like(a); cnt = np.zeros_like(a)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                s = np.roll(np.roll(a, dy, 0), dx, 1)
                ok = np.isfinite(s)
                acc[ok] += s[ok]; cnt[ok] += 1
        fill = nan & (cnt > 0)
        a[fill] = acc[fill] / cnt[fill]
    return a


def _dilate(m, r=1):
    o = m.copy()
    for _ in range(r):
        o = o | np.roll(o, 1, 0) | np.roll(o, -1, 0) | np.roll(o, 1, 1) | np.roll(o, -1, 1)
    return o


def _erode_steps(m, n):
    """Number of 4-neighbour erosion steps each pixel survives (distance to dry, capped at n)."""
    d = np.zeros(m.shape, np.float32)
    cur = m.copy()
    for k in range(n):
        d += cur
        cur = cur & np.roll(cur, 1, 0) & np.roll(cur, -1, 0) & np.roll(cur, 1, 1) & np.roll(cur, -1, 1)
    return d


def _corner_avg(a):
    """Pixel-centre field -> pixel-corner field (H+1, W+1), averaging finite neighbours."""
    p = np.pad(a, 1, mode='edge')
    s = np.stack([p[:-1, :-1], p[1:, :-1], p[:-1, 1:], p[1:, 1:]])
    ok = np.isfinite(s)
    return np.where(ok.any(0), np.nansum(np.where(ok, s, 0), 0) / np.maximum(ok.sum(0), 1), np.nan)


class HF:
    """Same bilinear sampler as build_world.HF (terrain vertices sit at integer px corners)."""
    def __init__(self, a):
        self.a = a

    def at(self, x, y):
        a = self.a
        x = np.clip(np.asarray(x, float) - 0.5, 0, a.shape[1] - 1.001)
        y = np.clip(np.asarray(y, float) - 0.5, 0, a.shape[0] - 1.001)
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
        fx = x - x0; fy = y - y0
        return a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy) + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy


def load():
    WL = np.fromfile(P('data/terrain/water_level_f32.bin'), np.float32).reshape(H, W)
    T = np.fromfile(P('data/terrain/height_graded_f32.bin'), np.float32).reshape(H, W)
    fp = P('data/terrain/water_fx_u8.bin')
    if os.path.exists(fp):
        raw = np.fromfile(fp, np.uint8)
        nch = raw.size // (H * W)
        fx = raw.reshape(nch, H, W).astype(np.float32) / 255
        depth, ang, bank = fx[0] * 6.0, fx[2] * 2 * np.pi, fx[3] * 2 - 1
        slope = fx[1] * (0.25 if nch >= 5 else 0.1)
        foam = fx[4] if nch >= 5 else np.clip((slope - 0.01) / 0.035, 0, 1)
    else:  # older data: derive what we can
        depth = np.nan_to_num(WL - T, nan=0).clip(0); slope = np.zeros_like(T); ang = np.zeros_like(T); bank = np.zeros_like(T)
        foam = np.zeros_like(T)
    wet = np.isfinite(WL)
    return dict(WL=WL, T=T, wet=wet, depth=np.where(wet, np.maximum(depth, np.nan_to_num(WL - T, nan=0)), 0), slope=slope,
                fdx=np.cos(ang), fdy=np.sin(ang), bank=bank, foam=foam)


def px2b(x, y, z):
    return (np.asarray(x) - OX) * MPP, -(np.asarray(y) - OY) * MPP, np.asarray(z)


def _mesh(name, verts, faces, coll, mat, attrs=None, smooth=True):
    me = bpy.data.meshes.new(name)
    verts = np.asarray(verts, np.float32).reshape(-1, 3)
    faces = np.asarray(faces, np.int32)
    me.vertices.add(len(verts)); me.vertices.foreach_set('co', verts.ravel())
    nf, k = faces.shape
    me.loops.add(nf * k); me.loops.foreach_set('vertex_index', faces.ravel())
    me.polygons.add(nf)
    me.polygons.foreach_set('loop_start', np.arange(0, nf * k, k, dtype=np.int32))
    me.polygons.foreach_set('loop_total', np.full(nf, k, np.int32))
    me.update(calc_edges=True); me.validate(clean_customdata=False)
    if smooth:
        me.polygons.foreach_set('use_smooth', np.ones(nf, bool))
    for an, (typ, data) in (attrs or {}).items():
        a = me.attributes.new(an, typ, 'POINT')
        a.data.foreach_set('vector' if typ == 'FLOAT_VECTOR' else 'value', np.asarray(data, np.float32).ravel())
    if mat:
        me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def _grid_mesh(cells, zfun, x0, y0):
    """Quads for the given (ys, xs) cells (pixel squares), shared corner vertices."""
    ys, xs = cells
    key = (ys[:, None] + np.array([0, 1, 1, 0])[None]) * (W + 1) + (xs[:, None] + np.array([0, 0, 1, 1])[None])
    uk, inv = np.unique(key.ravel(), return_inverse=True)
    vy, vx = uk // (W + 1), uk % (W + 1)
    faces = inv.reshape(-1, 4)
    return vx.astype(np.float64), vy.astype(np.float64), faces


# ------------------------------------------------------------------ materials
def _inp(b, *names):
    for n in names:
        if n in b.inputs:
            return b.inputs[n]
    raise KeyError(names)


def _new_mat(name):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    return m, nt


def _attr(nt, name, kind='FLOAT'):
    a = nt.nodes.new('ShaderNodeAttribute'); a.attribute_name = name; a.attribute_type = 'GEOMETRY'
    return a.outputs['Fac'] if kind == 'FLOAT' else a.outputs['Vector']


def _math(nt, op, a, b=None, clamp=False):
    m = nt.nodes.new('ShaderNodeMath'); m.operation = op; m.use_clamp = clamp
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            m.inputs[i].default_value = v
        else:
            nt.links.new(v, m.inputs[i])
    return m.outputs[0]


def _maprange(nt, x, a, b, c=0.0, d=1.0, smooth=True):
    m = nt.nodes.new('ShaderNodeMapRange')
    if smooth:
        m.interpolation_type = 'SMOOTHSTEP'
    m.inputs['From Min'].default_value = a; m.inputs['From Max'].default_value = b
    m.inputs['To Min'].default_value = c; m.inputs['To Max'].default_value = d
    nt.links.new(x, m.inputs['Value'])
    return m.outputs['Result']


def _mixrgb(nt, fac, a, b, blend='MIX'):
    m = nt.nodes.new('ShaderNodeMix'); m.data_type = 'RGBA'; m.blend_type = blend
    for sock, v in ((m.inputs['Factor'], fac), (m.inputs[6], a), (m.inputs[7], b)):
        if isinstance(v, (int, float)):
            sock.default_value = v
        elif isinstance(v, tuple):
            sock.default_value = v
        else:
            nt.links.new(v, sock)
    return m.outputs[2]


def _mixsh(nt, fac, a, b):
    m = nt.nodes.new('ShaderNodeMixShader')
    if isinstance(fac, (int, float)):
        m.inputs[0].default_value = fac
    else:
        nt.links.new(fac, m.inputs[0])
    nt.links.new(a, m.inputs[1]); nt.links.new(b, m.inputs[2])
    return m.outputs[0]


def _flow_coords(nt, stretch=(0.18, 0.9)):
    """Object-space coordinates rotated into the local flow frame (along, across), stretched so noise
    becomes streaks along the current."""
    tc = nt.nodes.new('ShaderNodeTexCoord')
    flow = _attr(nt, 'flow', 'VECTOR')
    sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(flow, sep.inputs[0])
    pos = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(tc.outputs['Object'], pos.inputs[0])
    along = _math(nt, 'ADD', _math(nt, 'MULTIPLY', pos.outputs['X'], sep.outputs['X']), _math(nt, 'MULTIPLY', pos.outputs['Y'], sep.outputs['Y']))
    across = _math(nt, 'SUBTRACT', _math(nt, 'MULTIPLY', pos.outputs['Y'], sep.outputs['X']), _math(nt, 'MULTIPLY', pos.outputs['X'], sep.outputs['Y']))
    cmb = nt.nodes.new('ShaderNodeCombineXYZ')
    nt.links.new(_math(nt, 'MULTIPLY', along, stretch[0]), cmb.inputs['X'])
    nt.links.new(_math(nt, 'MULTIPLY', across, stretch[1]), cmb.inputs['Y'])
    nt.links.new(pos.outputs['Z'], cmb.inputs['Z'])
    return cmb.outputs[0]


def water_material():
    """Clear shallows (transmissive, bed visible) -> dark tea-green pools (absorbing, reflective);
    flow-streaked ripple normals; whitewater where the reach is steep."""
    m, nt = _new_mat('MAT_Water')
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    depth = _attr(nt, 'depth'); foam = _attr(nt, 'foam'); shore = _attr(nt, 'shore')
    fc = _flow_coords(nt)
    # ripples: streaky along the flow, finer + stronger in fast water
    n1 = nt.nodes.new('ShaderNodeTexNoise'); n1.inputs['Scale'].default_value = 0.9; n1.inputs['Detail'].default_value = 6
    n1.inputs['Roughness'].default_value = 0.55
    nt.links.new(fc, n1.inputs['Vector'])
    n2 = nt.nodes.new('ShaderNodeTexNoise'); n2.inputs['Scale'].default_value = 4.0; n2.inputs['Detail'].default_value = 4
    nt.links.new(fc, n2.inputs['Vector'])
    ht = _math(nt, 'ADD', n1.outputs['Fac'], _math(nt, 'MULTIPLY', n2.outputs['Fac'], _math(nt, 'ADD', 0.25, _math(nt, 'MULTIPLY', foam, 1.2))))
    bump = nt.nodes.new('ShaderNodeBump'); bump.inputs['Distance'].default_value = 0.15
    nt.links.new(_math(nt, 'ADD', 0.06, _math(nt, 'MULTIPLY', foam, 0.35)), bump.inputs['Strength'])
    nt.links.new(ht, bump.inputs['Height'])
    # clear shallow water: transmissive, faintly tinted
    clear = nt.nodes.new('ShaderNodeBsdfPrincipled')
    _inp(clear, 'Base Color').default_value = (0.80, 0.90, 0.84, 1)
    _inp(clear, 'Transmission Weight').default_value = 1.0
    _inp(clear, 'IOR').default_value = 1.333
    _inp(clear, 'Roughness').default_value = 0.03
    nt.links.new(bump.outputs['Normal'], _inp(clear, 'Normal'))
    # deep water: absorbing tea/green-blue body with the same surface reflection
    deep = nt.nodes.new('ShaderNodeBsdfPrincipled')
    _inp(deep, 'Base Color').default_value = (0.010, 0.030, 0.028, 1)
    _inp(deep, 'IOR').default_value = 1.333
    _inp(deep, 'Roughness').default_value = 0.04
    nt.links.new(bump.outputs['Normal'], _inp(deep, 'Normal'))
    # depth mix: e^(-d/1.1) -> clear within ~0.5 m, dark by ~2.5 m; always a little clear at the very edge
    dfac = _math(nt, 'SUBTRACT', 1.0, _math(nt, 'EXPONENT', _math(nt, 'DIVIDE', depth, -1.1)))
    dfac = _math(nt, 'MULTIPLY', dfac, _maprange(nt, shore, 0.0, 2.5, 0.35, 0.92))
    body = _mixsh(nt, dfac, clear.outputs[0], deep.outputs[0])
    # whitewater: steep reaches only, broken into flow-aligned streaks and clumps
    fn = nt.nodes.new('ShaderNodeTexNoise'); fn.inputs['Scale'].default_value = 1.2; fn.inputs['Detail'].default_value = 10
    fn.inputs['Roughness'].default_value = 0.7
    nt.links.new(fc, fn.inputs['Vector'])
    fv = nt.nodes.new('ShaderNodeTexVoronoi'); fv.inputs['Scale'].default_value = 0.9
    nt.links.new(fc, fv.inputs['Vector'])
    brk = _math(nt, 'ADD', _math(nt, 'MULTIPLY', fn.outputs['Fac'], 0.8), _math(nt, 'MULTIPLY', fv.outputs['Distance'], 0.5))
    fmask = _maprange(nt, _math(nt, 'ADD', _math(nt, 'MULTIPLY', foam, 0.75), brk), 1.05, 1.35)
    fmask = _math(nt, 'MULTIPLY', fmask, _maprange(nt, foam, 0.12, 0.5))
    white = nt.nodes.new('ShaderNodeBsdfPrincipled')
    _inp(white, 'Base Color').default_value = (0.80, 0.83, 0.82, 1)
    _inp(white, 'Roughness').default_value = 0.5
    _inp(white, 'Subsurface Weight').default_value = 0.3
    nt.links.new(bump.outputs['Normal'], _inp(white, 'Normal'))
    surf = _mixsh(nt, fmask, body, white.outputs[0])
    nt.links.new(surf, out.inputs['Surface'])
    return m


def _tex_or_none(nt, aid, kind, vec, color=True):
    d = os.path.join(PH, aid, 'textures')
    if not os.path.isdir(d):
        return None
    f = sorted(x for x in os.listdir(d) if f'_{kind}_' in x)
    if not f:
        return None
    t = nt.nodes.new('ShaderNodeTexImage')
    t.image = bpy.data.images.load(os.path.join(d, f[0]), check_existing=True)
    if not color:
        t.image.colorspace_settings.name = 'Non-Color'
    nt.links.new(vec, t.inputs['Vector'])
    return t


def bed_material():
    """Riverbed: rounded river stones and gravel (Poly Haven river_small_rocks / gravel_ground_01 when
    present, procedural pebbles otherwise), silt-darkened in pools, dry gravel bars pale, rock banks grey."""
    m, nt = _new_mat('MAT_Riverbed')
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    b = nt.nodes.new('ShaderNodeBsdfPrincipled')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    depth = _attr(nt, 'depth'); bank = _attr(nt, 'bank'); edge = _attr(nt, 'edge')
    mp = nt.nodes.new('ShaderNodeMapping'); mp.inputs['Scale'].default_value = (1 / 2.2,) * 3
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    mp2 = nt.nodes.new('ShaderNodeMapping'); mp2.inputs['Scale'].default_value = (1 / 3.1,) * 3
    mp2.inputs['Rotation'].default_value = (0, 0, 0.7)
    nt.links.new(tc.outputs['Object'], mp2.inputs['Vector'])
    rocks = _tex_or_none(nt, 'river_small_rocks', 'diff', mp.outputs[0])
    grav = _tex_or_none(nt, 'gravel_ground_01', 'diff', mp2.outputs[0])
    vor = nt.nodes.new('ShaderNodeTexVoronoi'); vor.inputs['Scale'].default_value = 3.5
    nt.links.new(tc.outputs['Object'], vor.inputs['Vector'])
    if rocks is None:
        col_r = _mixrgb(nt, vor.outputs['Distance'], (0.20, 0.18, 0.15, 1), (0.42, 0.39, 0.34, 1))
    else:
        col_r = rocks.outputs['Color']
    col_g = grav.outputs['Color'] if grav is not None else _mixrgb(nt, vor.outputs['Distance'], (0.35, 0.32, 0.27, 1), (0.55, 0.51, 0.44, 1))
    # bank < 0 -> gravel bar, > 0 -> rock; stone mix everywhere
    gfac = _maprange(nt, bank, -0.1, -0.6)
    col = _mixrgb(nt, gfac, col_r, col_g)
    # large-scale colour variation (tannin browns, algae greens) so the bed is not uniform
    ln = nt.nodes.new('ShaderNodeTexNoise'); ln.inputs['Scale'].default_value = 0.05; ln.inputs['Detail'].default_value = 3
    nt.links.new(tc.outputs['Object'], ln.inputs['Vector'])
    col = _mixrgb(nt, _math(nt, 'MULTIPLY', ln.outputs['Fac'], 0.45), col, (0.23, 0.26, 0.12, 1), 'MULTIPLY')
    # silt + wet darkening with depth; dry bar tops stay pale
    wetdark = _maprange(nt, depth, 0.0, 2.5, 0.0, 0.75)
    col = _mixrgb(nt, wetdark, col, (0.05, 0.055, 0.045, 1), 'MIX')
    col = _mixrgb(nt, _maprange(nt, depth, -0.05, 0.05, 0.0, 0.35), col, (0.3, 0.3, 0.3, 1), 'MULTIPLY')
    nt.links.new(col, _inp(b, 'Base Color'))
    nt.links.new(_maprange(nt, depth, -0.05, 0.05, 0.85, 0.35), _inp(b, 'Roughness'))
    disp = _tex_or_none(nt, 'river_small_rocks', 'disp', mp.outputs[0], False)
    bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.6; bp.inputs['Distance'].default_value = 0.06
    nt.links.new(disp.outputs['Color'] if disp else vor.outputs['Distance'], bp.inputs['Height'])
    nt.links.new(bp.outputs['Normal'], _inp(b, 'Normal'))
    # soft edge into the surrounding terrain
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    alpha = _maprange(nt, _math(nt, 'ADD', edge, _math(nt, 'MULTIPLY', ln.outputs['Fac'], 0.3)), 0.25, 0.75)
    nt.links.new(_mixsh(nt, alpha, tr.outputs[0], b.outputs[0]), out.inputs['Surface'])
    return m


def shore_material():
    """Wet film just above the waterline: darker, glossier ground/stone fading out upslope."""
    m, nt = _new_mat('MAT_WetShore')
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    wet = _attr(nt, 'wet')
    b = nt.nodes.new('ShaderNodeBsdfPrincipled')
    _inp(b, 'Base Color').default_value = (0.035, 0.035, 0.028, 1)
    _inp(b, 'Roughness').default_value = 0.16
    tc = nt.nodes.new('ShaderNodeTexCoord')
    n = nt.nodes.new('ShaderNodeTexNoise'); n.inputs['Scale'].default_value = 0.8; n.inputs['Detail'].default_value = 5
    nt.links.new(tc.outputs['Object'], n.inputs['Vector'])
    a = _maprange(nt, _math(nt, 'SUBTRACT', wet, _math(nt, 'MULTIPLY', n.outputs['Fac'], 0.35)), 0.05, 0.55, 0.0, 0.8)
    tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    nt.links.new(_mixsh(nt, a, tr.outputs[0], b.outputs[0]), out.inputs['Surface'])
    return m


def rock_material():
    m, nt = _new_mat('MAT_RiverRock')
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    b = nt.nodes.new('ShaderNodeBsdfPrincipled')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    mp = nt.nodes.new('ShaderNodeMapping'); mp.inputs['Scale'].default_value = (1 / 1.5,) * 3
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    t = _tex_or_none(nt, 'mossy_rock', 'diff', mp.outputs[0])
    wetz = _attr(nt, 'wet')
    base = t.outputs['Color'] if t else (0.32, 0.30, 0.27, 1)
    n = nt.nodes.new('ShaderNodeTexNoise'); n.inputs['Scale'].default_value = 0.7
    nt.links.new(tc.outputs['Object'], n.inputs['Vector'])
    col = _mixrgb(nt, 0.35, base, (0.30, 0.29, 0.26, 1)) if t else _mixrgb(nt, n.outputs['Fac'], (0.16, 0.155, 0.14, 1), (0.34, 0.32, 0.29, 1))
    col = _mixrgb(nt, 1.0, col, (0.72, 0.72, 0.70, 1), 'MULTIPLY')
    col = _mixrgb(nt, _math(nt, 'MULTIPLY', wetz, 0.7), col, (0.05, 0.05, 0.045, 1))
    nt.links.new(col, _inp(b, 'Base Color'))
    nt.links.new(_maprange(nt, wetz, 0, 1, 0.85, 0.25), _inp(b, 'Roughness'))
    d = _tex_or_none(nt, 'mossy_rock', 'disp', mp.outputs[0], False)
    bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.8; bp.inputs['Distance'].default_value = 0.08
    nt.links.new(d.outputs['Color'] if d else n.outputs['Fac'], bp.inputs['Height'])
    nt.links.new(bp.outputs['Normal'], _inp(b, 'Normal'))
    nt.links.new(b.outputs[0], out.inputs['Surface'])
    return m


# ------------------------------------------------------------------ geometry
def _boulder_proto(rng, subdiv=2):
    """Icosphere-ish unit rock (numpy): verts, faces."""
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    v /= np.linalg.norm(v, axis=1)[:, None]
    for _ in range(subdiv):
        cache = {}
        vl = list(v); nf = []
        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                p = (vl[a] + vl[b]) / 2
                vl.append(p / np.linalg.norm(p)); cache[k] = len(vl) - 1
            return cache[k]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v = np.array(vl); f = np.array(nf)
    return v, f


def build_boulders(D, coll, mat, seed=7, max_n=4200):
    rng = np.random.default_rng(seed)
    wet, depth, slope, bank = D['wet'], D['depth'], D['slope'], D['bank']
    near = _dilate(wet, 3) & ~wet
    rapid = np.clip((slope - 0.012) / 0.03, 0, 1)
    # probability per pixel: rapids and riffles (many), rocky banks (some), pools (few), creeks' margins
    p = np.where(wet, 0.004 + 0.16 * np.maximum(rapid, D['foam']) + 0.02 * (depth < 0.8), 0.0)
    p += np.where(near, 0.05 * np.clip(bank, 0, 1) + 0.02 * rapid, 0.0)
    ys, xs = np.nonzero(p > 0)
    pick = rng.random(len(ys)) < p[ys, xs]
    ys, xs = ys[pick], xs[pick]
    if len(ys) > max_n:
        sel = rng.choice(len(ys), max_n, replace=False); ys, xs = ys[sel], xs[sel]
    if not len(ys):
        return None
    T = HF(D['T'])
    pv, pf = _boulder_proto(rng, 2)
    V, F, WZ = [], [], []
    off = 0
    for y, x in zip(ys, xs):
        fxp, fyp = x + rng.random(), y + rng.random()
        r = float(np.clip(rng.lognormal(-0.2, 0.55), 0.25, 2.6)) * (1.25 if rapid[y, x] > 0.5 else 1.0)
        sq = rng.uniform(0.45, 0.8)
        # per-rock lumpy deformation: low-order harmonics of the unit sphere
        k = rng.normal(0, 1, (3, 3))
        lump = 1 + 0.16 * np.tanh(pv @ k[0]) * np.sin(pv @ k[1] * 1.7) + 0.07 * np.sin(pv @ k[2] * 3.1)
        vv = pv * lump[:, None]
        # fracture planes -> angular, blocky river stone (weathered granite/gneiss)
        for _ in range(int(rng.integers(3, 7))):
            nrm = rng.normal(0, 1, 3); nrm /= np.linalg.norm(nrm)
            dcut = rng.uniform(0.55, 0.9)
            h = vv @ nrm
            over = h > dcut
            vv[over] -= np.outer(h[over] - dcut, nrm)
        vv = vv * np.array([r * rng.uniform(0.8, 1.3), r, r * sq])
        a = rng.uniform(0, 2 * np.pi)
        R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        vv = vv @ R.T
        bx, by, _ = px2b(fxp, fyp, 0)
        zb = float(T.at(fxp, fyp))
        vv += np.array([bx, by, zb + r * sq * rng.uniform(0.05, 0.45)])
        V.append(vv); F.append(pf + off); off += len(vv)
        wl = D['WL'][y, x] if np.isfinite(D['WL'][y, x]) else zb
        WZ.append(np.clip((wl + 0.4 - vv[:, 2]) / 0.8, 0, 1))
    ob = _mesh('WATER_Boulders', np.vstack(V), np.vstack(F), coll, mat, attrs={'wet': ('FLOAT', np.concatenate(WZ))})
    return ob


def build_water(coll, mat=None):
    """Water surfaces + riverbed + wet shore + boulders. Signature kept from build_world.build_water."""
    D = load()
    mat = mat if (mat is not None and mat.name == 'MAT_Water') else water_material()
    mbed, mshore, mrock = bed_material(), shore_material(), rock_material()
    wet = D['wet']
    T = HF(D['T'])
    WLf = _fill_nearest(D['WL'], 6)
    WLc = _corner_avg(WLf)
    shore_steps = _erode_steps(wet, 6)
    skirt = _dilate(wet, 1)
    near3 = _dilate(wet, 4) & ~wet
    bedcells = _dilate(wet, 1) | (near3 & (np.abs(D['bank']) > 0.15))
    objs = []
    for cy in range(math.ceil(H / CHUNK)):
        for cx in range(math.ceil(W / CHUNK)):
            sl = (slice(cy * CHUNK, (cy + 1) * CHUNK), slice(cx * CHUNK, (cx + 1) * CHUNK))
            # ---- water surface (skirted 2 px under the banks; the terrain clips the shoreline)
            ys, xs = np.nonzero(skirt[sl])
            if len(ys):
                ys = ys + cy * CHUNK; xs = xs + cx * CHUNK
                vx, vy, faces = _grid_mesh((ys, xs), None, cx * CHUNK, cy * CHUNK)
                z = WLc[vy.astype(int), vx.astype(int)]
                z = np.where(np.isfinite(z), z, T.at(vx, vy)) + 0.02
                bx, by, bz = px2b(vx, vy, z)
                pxi = np.clip(vx.astype(int), 0, W - 1); pyi = np.clip(vy.astype(int), 0, H - 1)
                dep = np.maximum(z - T.at(vx, vy), 0)
                foam = D['foam'][pyi, pxi]
                flow = np.stack([D['fdx'][pyi, pxi], -D['fdy'][pyi, pxi], np.zeros_like(bx)], 1)
                objs.append(_mesh(f'WATER_C{cx:02d}_{cy:02d}', np.stack([bx, by, bz], 1), faces, coll, mat,
                                  attrs={'depth': ('FLOAT', dep), 'foam': ('FLOAT', foam), 'shore': ('FLOAT', shore_steps[pyi, pxi]),
                                         'flow': ('FLOAT_VECTOR', flow)}))
            # ---- riverbed / bars / rocky banks
            ys, xs = np.nonzero(bedcells[sl])
            if len(ys):
                ys = ys + cy * CHUNK; xs = xs + cx * CHUNK
                vx, vy, faces = _grid_mesh((ys, xs), None, 0, 0)
                zt = T.at(vx, vy)
                bx, by, bz = px2b(vx, vy, zt + 0.03)
                pxi = np.clip(vx.astype(int), 0, W - 1); pyi = np.clip(vy.astype(int), 0, H - 1)
                wlv = WLc[vy.astype(int), vx.astype(int)]
                dep = np.where(np.isfinite(wlv), wlv - zt, -1.0)
                inside = bedcells.astype(np.float32)
                edge = (inside[pyi, pxi] + inside[np.clip(pyi - 1, 0, H - 1), pxi] + inside[pyi, np.clip(pxi - 1, 0, W - 1)] +
                        inside[np.clip(pyi - 1, 0, H - 1), np.clip(pxi - 1, 0, W - 1)]) / 4
                edge = np.maximum(edge * 0.6 + wet[pyi, pxi] * 0.4, 0) * (1 - 0.0)
                objs.append(_mesh(f'WATER_Bed_C{cx:02d}_{cy:02d}', np.stack([bx, by, bz], 1), faces, coll, mbed,
                                  attrs={'depth': ('FLOAT', dep), 'bank': ('FLOAT', D['bank'][pyi, pxi]), 'edge': ('FLOAT', edge)}))
            # ---- wet shore film
            ys, xs = np.nonzero(near3[sl])
            if len(ys):
                ys = ys + cy * CHUNK; xs = xs + cx * CHUNK
                vx, vy, faces = _grid_mesh((ys, xs), None, 0, 0)
                zt = T.at(vx, vy)
                bx, by, bz = px2b(vx, vy, zt + 0.05)
                wlv = WLc[vy.astype(int), vx.astype(int)]
                hab = zt - np.where(np.isfinite(wlv), wlv, zt - 5)
                wetf = np.clip(1 - hab / 0.9, 0, 1) * np.isfinite(wlv)
                objs.append(_mesh(f'WATER_Shore_C{cx:02d}_{cy:02d}', np.stack([bx, by, bz], 1), faces, coll, mshore,
                                  attrs={'wet': ('FLOAT', wetf)}))
    b = build_boulders(D, coll, mrock)
    if b is not None:
        objs.append(b)
    return objs
