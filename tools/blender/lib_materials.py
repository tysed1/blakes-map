"""Production materials for the Blender world (CC0 Poly Haven PBR sets + procedural glue).
Terrain = height-blended PBR layers driven by per-vertex land-use weights (lu_a / lu_b)
and slope, modulated by the macro albedo (large-scale variation consistent with the map)."""
import bpy, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PH = os.path.join(ROOT, 'assets/external/polyhaven')


def _img(path, color=True):
    im = bpy.data.images.load(path, check_existing=True)
    if not color:
        im.colorspace_settings.name = 'Non-Color'
    return im


def _tex(nt, aid, kind, vec, color=True):
    d = os.path.join(PH, aid, 'textures')
    f = [x for x in os.listdir(d) if f'_{kind}_' in x]
    if not f:
        return None
    t = nt.nodes.new('ShaderNodeTexImage')
    t.image = _img(os.path.join(d, f[0]), color)
    t.projection = 'BOX'; t.projection_blend = 0.3
    nt.links.new(vec, t.inputs['Vector'])
    return t


def _mapping(nt, tc, scale_m, rot=0.0):
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (1 / scale_m,) * 3
    mp.inputs['Rotation'].default_value = (0, 0, rot)
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    return mp.outputs['Vector']


def _mix(nt, fac, a, b, blend='MIX'):
    m = nt.nodes.new('ShaderNodeMix'); m.data_type = 'RGBA'; m.blend_type = blend
    nt.links.new(fac, m.inputs['Factor']) if not isinstance(fac, float) else None
    if isinstance(fac, float):
        m.inputs['Factor'].default_value = fac
    nt.links.new(a, m.inputs['A']) if not isinstance(a, tuple) else None
    if isinstance(a, tuple):
        m.inputs['A'].default_value = a
    nt.links.new(b, m.inputs['B']) if not isinstance(b, tuple) else None
    if isinstance(b, tuple):
        m.inputs['B'].default_value = b
    return m.outputs['Result']


def _math(nt, op, a, b=None, clamp=True):
    m = nt.nodes.new('ShaderNodeMath'); m.operation = op; m.use_clamp = clamp
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            m.inputs[i].default_value = v
        else:
            nt.links.new(v, m.inputs[i])
    return m.outputs[0]


def _layer_weight(nt, w, height, contrast=6.0):
    """Height-blend: weight sharpened by the layer's height map for natural transitions."""
    x = _math(nt, 'ADD', w, _math(nt, 'MULTIPLY', _math(nt, 'SUBTRACT', height, 0.5, False), 0.6, False), False)
    return _math(nt, 'MULTIPLY', _math(nt, 'SUBTRACT', x, 0.5, False), contrast, False) if False else _smooth(nt, x, 0.35, 0.65)


def _smooth(nt, x, lo, hi):
    mr = nt.nodes.new('ShaderNodeMapRange'); mr.interpolation_type = 'SMOOTHSTEP'
    mr.inputs['From Min'].default_value = lo; mr.inputs['From Max'].default_value = hi
    nt.links.new(x, mr.inputs['Value'])
    return mr.outputs['Result']


# approximate average albedo per texture (sRGB-linear) - used to tame tile contrast at distance
AVG = {'forest_floor': (0.09, 0.07, 0.045, 1), 'leafy_grass': (0.07, 0.11, 0.03, 1), 'sparse_grass': (0.12, 0.13, 0.06, 1),
       'aerial_grass_rock': (0.12, 0.12, 0.08, 1), 'river_small_rocks': (0.2, 0.18, 0.15, 1), 'rock_face_03': (0.22, 0.2, 0.17, 1),
       'red_dirt_mud_01': (0.25, 0.12, 0.07, 1)}
DETAIL = {'field': 0.45, 'meadow': 0.5, 'lawn': 0.5, 'forest': 0.7, 'rock': 0.85, 'bank': 0.85, 'dirt': 0.7}


def terrain_material(albedo_path):
    m = bpy.data.materials.new('MAT_Terrain_PBR'); m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(bsdf.outputs[0], out.inputs['Surface'])
    tc = nt.nodes.new('ShaderNodeTexCoord')
    la = nt.nodes.new('ShaderNodeVertexColor'); la.layer_name = 'lu_a'   # field, meadow, developed, rock
    lb = nt.nodes.new('ShaderNodeVertexColor'); lb.layer_name = 'lu_b'   # forest, bank, shoulder
    sa = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(la.outputs['Color'], sa.inputs[0])
    sb = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(lb.outputs['Color'], sb.inputs[0])
    layers = {}
    # art-directed palettes (linear): colour comes from here, texture only supplies detail (tex/avg)
    PALS = {
        'forest': [(0.075, 0.055, 0.03), (0.05, 0.05, 0.025), (0.09, 0.06, 0.03)],
        'field':  [(0.19, 0.17, 0.055), (0.12, 0.15, 0.04), (0.23, 0.18, 0.07)],   # golden pasture / green / hay
        'meadow': [(0.14, 0.14, 0.05), (0.18, 0.15, 0.06), (0.1, 0.12, 0.04)],
        'lawn':   [(0.1, 0.14, 0.04), (0.13, 0.15, 0.05), (0.09, 0.12, 0.035)],
        'bank':   [(0.2, 0.18, 0.15), (0.15, 0.14, 0.12), (0.24, 0.22, 0.18)],
        'rock':   [(0.1, 0.095, 0.085), (0.075, 0.075, 0.07), (0.12, 0.105, 0.085)],
        'dirt':   [(0.2, 0.1, 0.05), (0.16, 0.1, 0.06), (0.24, 0.13, 0.07)],
    }
    MEAS = {'forest_floor': (0.371, 0.227, 0.104), 'leafy_grass': (0.315, 0.232, 0.103), 'sparse_grass': (0.08, 0.048, 0.008),
            'aerial_grass_rock': (0.169, 0.122, 0.026), 'river_small_rocks': (0.193, 0.153, 0.107), 'rock_face_03': (0.231, 0.144, 0.083),
            'red_dirt_mud_01': (0.214, 0.091, 0.038)}
    for key, aid, size in (('forest', 'forest_floor', 3.0), ('field', 'leafy_grass', 2.5), ('meadow', 'aerial_grass_rock', 5.0),
                           ('lawn', 'aerial_grass_rock', 6.0), ('bank', 'river_small_rocks', 2.0), ('rock', 'rock_face_03', 7.0),
                           ('dirt', 'red_dirt_mud_01', 3.0)):
        vec = _mapping(nt, tc, size, 0.37 * len(layers))
        vec2 = _mapping(nt, tc, size * 3.7, 1.1 + 0.37 * len(layers))
        d = _tex(nt, aid, 'diff', vec); d2 = _tex(nt, aid, 'diff', vec2)
        h = _tex(nt, aid, 'disp', vec, False)
        nzm = nt.nodes.new('ShaderNodeTexNoise'); nzm.inputs['Scale'].default_value = 0.06; nzm.inputs['Detail'].default_value = 2
        nt.links.new(tc.outputs['Object'], nzm.inputs['Vector'])
        dm = _mix(nt, _smooth(nt, nzm.outputs['Fac'], 0.4, 0.6), d.outputs['Color'], d2.outputs['Color'])
        # detail = luminance(tex) / luminance(avg), softened
        bw = nt.nodes.new('ShaderNodeRGBToBW'); nt.links.new(dm, bw.inputs[0])
        mv = MEAS[aid]; avg_l = 0.3 * mv[0] + 0.59 * mv[1] + 0.11 * mv[2]
        det = _math(nt, 'DIVIDE', bw.outputs[0], avg_l, False)
        det = _math(nt, 'POWER', det, DETAIL.get(key, 0.6), False)
        # palette by two large noises (patchwork fields / variation)
        n1 = nt.nodes.new('ShaderNodeTexNoise'); n1.inputs['Scale'].default_value = 0.008 + 0.002 * len(layers); n1.inputs['Detail'].default_value = 1
        n2 = nt.nodes.new('ShaderNodeTexVoronoi'); n2.inputs['Scale'].default_value = 0.012
        nt.links.new(tc.outputs['Object'], n1.inputs['Vector']); nt.links.new(tc.outputs['Object'], n2.inputs['Vector'])
        p0, p1, p2 = PALS[key]
        c = _mix(nt, _smooth(nt, n1.outputs['Fac'], 0.35, 0.65), p0 + (1,), p1 + (1,))
        c = _mix(nt, _math(nt, 'MULTIPLY', _smooth(nt, n2.outputs['Color'] if False else n2.outputs['Distance'], 0.1, 0.5), 0.6), c, p2 + (1,))
        col = _mix(nt, 1.0, c, det, 'MULTIPLY')
        layers[key] = (col, h.outputs['Color'] if h else None)
    # slope (0 flat .. 1 cliff) from the geometric normal
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(geo.outputs['Normal'], sep.inputs[0])
    steep = _smooth(nt, _math(nt, 'SUBTRACT', 1.0, sep.outputs['Z']), 0.22, 0.42)
    bare = _smooth(nt, _math(nt, 'SUBTRACT', 1.0, sep.outputs['Z']), 0.12, 0.26)
    col, hgt = layers['forest']
    def blend(col, key, w):
        c2, h2 = layers[key]
        ww = _layer_weight(nt, w, h2) if h2 is not None else w
        return _mix(nt, ww, col, c2)
    col = blend(col, 'field', sa.outputs['Red'])
    col = blend(col, 'meadow', sa.outputs['Green'])
    col = blend(col, 'lawn', sa.outputs['Blue'])
    col = blend(col, 'bank', sb.outputs['Green'])
    col = blend(col, 'dirt', _math(nt, 'MULTIPLY', _math(nt, 'MULTIPLY', bare, 0.35), _math(nt, 'SUBTRACT', 1.0, sb.outputs['Red'])))
    col = blend(col, 'dirt', _math(nt, 'MULTIPLY', sb.outputs['Blue'], 0.7))
    col = blend(col, 'rock', _math(nt, 'MAXIMUM', steep, _math(nt, 'MULTIPLY', sa.outputs['Alpha'] if 'Alpha' in sa.outputs else sa.outputs['Blue'], 0.0)))
    # macro colour from the stylized albedo (keeps the map's large-scale patterns)
    alb = nt.nodes.new('ShaderNodeTexImage'); alb.image = _img(albedo_path); alb.interpolation = 'Cubic'
    uv = nt.nodes.new('ShaderNodeUVMap'); nt.links.new(uv.outputs[0], alb.inputs['Vector'])
    macro = _mix(nt, 1.0, alb.outputs['Color'], (2.4, 2.4, 2.4, 1), 'MULTIPLY')
    col = _mix(nt, 0.3, col, _mix(nt, 1.0, col, macro, 'MULTIPLY'))
    # far-distance softening: large noise breaks tiling
    nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 0.012; nz.inputs['Detail'].default_value = 3
    nt.links.new(tc.outputs['Object'], nz.inputs['Vector'])
    col = _mix(nt, 0.22, col, nz.outputs['Color'], 'OVERLAY')
    nt.links.new(col, bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.92
    # detail bump from the dominant layer heights
    bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.4; bp.inputs['Distance'].default_value = 0.08
    hsum = _math(nt, 'ADD', layers['forest'][1], layers['rock'][1], False)
    nt.links.new(hsum, bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], bsdf.inputs['Normal'])
    return m


def pbr(name, aid, size=4.0, tint=(1, 1, 1, 1), rough=None):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    vec = _mapping(nt, tc, size)
    d = _tex(nt, aid, 'diff', vec)
    nt.links.new(_mix(nt, 1.0, d.outputs['Color'], tint, 'MULTIPLY'), bsdf.inputs['Base Color'])
    h = _tex(nt, aid, 'disp', vec, False)
    if h:
        bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.5; bp.inputs['Distance'].default_value = 0.03
        nt.links.new(h.outputs['Color'], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], bsdf.inputs['Normal'])
    bsdf.inputs['Roughness'].default_value = rough if rough is not None else 0.85
    return m


def hdri_world(hdr_path, strength=1.0, rot_deg=0.0, sun_rot=None):
    sc = bpy.context.scene
    w = sc.world
    nt = w.node_tree
    for n in list(nt.nodes):
        if n.type in ('TEX_SKY', 'BACKGROUND', 'TEX_ENVIRONMENT', 'MAPPING', 'TEX_COORD'):
            nt.nodes.remove(n)
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_WORLD')
    tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Rotation'].default_value = (0, 0, __import__('math').radians(rot_deg))
    env = nt.nodes.new('ShaderNodeTexEnvironment'); env.image = _img(hdr_path)
    bg = nt.nodes.new('ShaderNodeBackground'); bg.inputs['Strength'].default_value = strength
    nt.links.new(tc.outputs['Generated'], mp.inputs['Vector']); nt.links.new(mp.outputs[0], env.inputs['Vector'])
    nt.links.new(env.outputs[0], bg.inputs[0]); nt.links.new(bg.outputs[0], out.inputs['Surface'])
