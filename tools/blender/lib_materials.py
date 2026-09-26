"""Production materials for the Blender world (A3: materials). CC0 Poly Haven PBR sets + procedural glue.

TERRAIN (terrain_material): one shader, many height-blended layers. Colour comes from art-directed
palettes (graphics ref.png: golden pastures, deep olive forest, grey-brown rock, red Georgia clay);
the photo textures only supply detail (texture / its measured mean), so layers never fight in hue
and distance reads as clean painterly colour. Anti-tiling: every texture is sampled at two scales
and rotations blended by noise, plus macro (250 m / 50 m) value + hue variation.
Drivers (per-vertex attributes):
    lu_a  (field, meadow, developed, rock)         lu_b (forest, bank, road shoulder)   - build_world
    eco_a (moisture, disturbed, canopy, hedge)     eco_b (pasture, hay, plowed, fallow)
    eco_c (lawn, field angle, tpi, talus)                                               - add_terrain_attributes
    + slope (geometric normal) + altitude.
Layers: forest-floor leaf litter, moss / dark humus in moist hollows, golden pasture, mown hay with
stripes, plowed red-clay furrows, broomsedge fallow, mown lawn, dry roadside grass, gravel shoulder,
red clay (road cuts / disturbed ground), wet mud (creek margins), river-bank gravel, talus / scree,
Appalachian rock (grey-brown sandstone / gneiss with lichen).

SHARED UTILITY MATERIALS (for Agent 2 / others, no UVs needed - world-space box projection):
    get('concrete' | 'bridge_concrete' | 'rusted_metal' | 'weathered_wood' | 'gravel' | 'dirt' |
        'red_clay' | 'rock' | 'mud' | 'dry_grass' | 'forest_floor' | 'grass')
"""
import bpy, os, math
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PH = os.path.join(ROOT, 'assets/external/polyhaven')

# measured mean linear albedo of each Poly Haven diffuse texture (tools: see A3 notes)
MEAN = {
    'aerial_grass_rock': (0.169, 0.122, 0.026), 'asphalt_02': (0.105, 0.103, 0.091), 'brown_mud_02': (0.077, 0.061, 0.040),
    'brown_mud_leaves_01': (0.133, 0.094, 0.034), 'cliff_side': (0.211, 0.091, 0.033), 'concrete_wall_006': (0.108, 0.083, 0.061),
    'dirt_floor': (0.471, 0.262, 0.128), 'dry_decay_leaves': (0.173, 0.076, 0.035), 'dry_ground_01': (0.234, 0.196, 0.145),
    'farm_soil': (0.098, 0.051, 0.021), 'forest_floor': (0.371, 0.227, 0.104), 'forest_leaves_02': (0.285, 0.171, 0.051),
    'grass_path_2': (0.279, 0.228, 0.133), 'gravel_ground_01': (0.291, 0.229, 0.142), 'gravel_road': (0.192, 0.099, 0.051),
    'leafy_grass': (0.315, 0.232, 0.103), 'leaves_forest_ground': (0.192, 0.114, 0.043), 'lichen_rock': (0.064, 0.049, 0.034),
    'mossy_rock': (0.161, 0.154, 0.100), 'mud_forest': (0.040, 0.020, 0.007), 'red_dirt_mud_01': (0.214, 0.091, 0.038),
    'red_mud_stones': (0.158, 0.057, 0.022), 'river_small_rocks': (0.193, 0.153, 0.107), 'rock_boulder_dry': (0.398, 0.337, 0.271),
    'rock_face': (0.121, 0.061, 0.032), 'rock_face_03': (0.231, 0.144, 0.083), 'rock_pitted_mossy': (0.556, 0.339, 0.188),
    'rocky_trail': (0.295, 0.220, 0.148), 'rough_concrete': (0.484, 0.465, 0.336), 'rusty_metal_02': (0.412, 0.288, 0.136),
    'sparse_grass': (0.080, 0.048, 0.008), 'weathered_planks': (0.081, 0.058, 0.044), 'withered_grass': (0.419, 0.297, 0.196),
    'worn_asphalt': (0.063, 0.041, 0.028),
}

# art-directed terrain palettes (linear albedo): 3 tones per layer (patch A / patch B / accent)
PAL = {
    'forest':  [(0.070, 0.045, 0.022), (0.090, 0.058, 0.026), (0.055, 0.042, 0.022)],   # autumn leaf litter
    'humus':   [(0.035, 0.036, 0.018), (0.045, 0.050, 0.022), (0.030, 0.028, 0.016)],   # moist hollows / moss
    'pasture': [(0.215, 0.180, 0.065), (0.150, 0.155, 0.050), (0.250, 0.195, 0.075)],   # golden-green pasture
    'hay':     [(0.270, 0.215, 0.095), (0.230, 0.195, 0.080), (0.300, 0.240, 0.110)],   # mown hay / straw
    'plowed':  [(0.160, 0.075, 0.038), (0.130, 0.068, 0.036), (0.185, 0.090, 0.045)],   # red-clay plow land
    'fallow':  [(0.270, 0.150, 0.060), (0.220, 0.160, 0.060), (0.300, 0.170, 0.070)],   # broomsedge
    'lawn':    [(0.120, 0.130, 0.042), (0.140, 0.135, 0.046), (0.105, 0.118, 0.036)],
    'meadow':  [(0.180, 0.165, 0.060), (0.130, 0.145, 0.050), (0.210, 0.170, 0.065)],
    'verge':   [(0.210, 0.170, 0.080), (0.170, 0.150, 0.065), (0.240, 0.190, 0.090)],   # dry roadside grass
    'gravel':  [(0.200, 0.180, 0.150), (0.170, 0.150, 0.125), (0.230, 0.200, 0.160)],
    'clay':    [(0.270, 0.105, 0.050), (0.230, 0.095, 0.045), (0.300, 0.130, 0.060)],   # Georgia red clay
    'mud':     [(0.060, 0.045, 0.030), (0.050, 0.040, 0.028), (0.075, 0.055, 0.035)],
    'bank':    [(0.170, 0.150, 0.120), (0.130, 0.120, 0.100), (0.200, 0.175, 0.140)],
    'talus':   [(0.100, 0.093, 0.080), (0.082, 0.078, 0.068), (0.120, 0.110, 0.092)],
    'rock':    [(0.095, 0.088, 0.075), (0.075, 0.072, 0.064), (0.115, 0.104, 0.086)],   # grey-brown sandstone / gneiss
}
# layer -> (texture id, tile size m, detail power, chroma keep, roughness)
LAYER = {
    'forest': ('forest_leaves_02', 2.6, 0.8, 0.35, 0.9), 'humus': ('brown_mud_leaves_01', 3.0, 0.7, 0.2, 0.8),
    'pasture': ('leafy_grass', 2.8, 0.45, 0.15, 0.95), 'hay': ('withered_grass', 2.4, 0.5, 0.1, 0.95),
    'plowed': ('farm_soil', 2.5, 0.8, 0.1, 0.95), 'fallow': ('withered_grass', 3.2, 0.55, 0.15, 0.95),
    'lawn': ('leafy_grass', 2.2, 0.35, 0.1, 0.95), 'meadow': ('aerial_grass_rock', 5.0, 0.45, 0.15, 0.95),
    'verge': ('withered_grass', 2.0, 0.5, 0.15, 0.95), 'gravel': ('gravel_ground_01', 1.8, 0.8, 0.2, 0.9),
    'clay': ('red_mud_stones', 2.5, 0.7, 0.25, 0.9), 'mud': ('brown_mud_02', 2.2, 0.8, 0.2, 0.55),
    'bank': ('river_small_rocks', 2.0, 0.9, 0.2, 0.85), 'talus': ('rocky_trail', 3.0, 0.9, 0.15, 0.9),
    'rock': ('rock_face_03', 6.0, 0.95, 0.25, 0.85),
}


# ------------------------------------------------------------------ node helpers
def _img(path, color=True):
    im = bpy.data.images.load(path, check_existing=True)
    if not color:
        im.colorspace_settings.name = 'Non-Color'
    return im


def _texfile(aid, kind):
    d = os.path.join(PH, aid, 'textures')
    if not os.path.isdir(d):
        return None
    keys = {'diff': ('_diff_', '_diffuse_'), 'disp': ('_disp_',), 'rough': ('_rough_',)}[kind]
    f = sorted(x for x in os.listdir(d) if any(k in x for k in keys))
    return os.path.join(d, f[0]) if f else None


def _tex(nt, aid, kind, vec, color=True, box=False):
    p = _texfile(aid, kind)
    if not p:
        return None
    t = nt.nodes.new('ShaderNodeTexImage')
    t.image = _img(p, color)
    if box:
        t.projection = 'BOX'; t.projection_blend = 0.3
    nt.links.new(vec, t.inputs['Vector'])
    return t


def _mapping(nt, tc, scale_m, rot=0.0, off=(0, 0, 0)):
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (1 / scale_m,) * 3
    mp.inputs['Rotation'].default_value = (0, 0, rot)
    mp.inputs['Location'].default_value = off
    nt.links.new(tc.outputs['Object'] if hasattr(tc, 'outputs') else tc, mp.inputs['Vector'])
    return mp.outputs['Vector']


def _mix(nt, fac, a, b, blend='MIX'):
    m = nt.nodes.new('ShaderNodeMix'); m.data_type = 'RGBA'; m.blend_type = blend
    for sock, v in ((m.inputs['Factor'], fac), (m.inputs['A'], a), (m.inputs['B'], b)):
        if isinstance(v, (int, float)):
            sock.default_value = v
        elif isinstance(v, tuple):
            sock.default_value = v if len(v) == 4 else v + (1,)
        else:
            nt.links.new(v, sock)
    return m.outputs['Result']


def _mixf(nt, fac, a, b):
    m = nt.nodes.new('ShaderNodeMix'); m.data_type = 'FLOAT'
    for sock, v in ((m.inputs['Factor'], fac), (m.inputs['A'], a), (m.inputs['B'], b)):
        if isinstance(v, (int, float)):
            sock.default_value = v
        else:
            nt.links.new(v, sock)
    return m.outputs['Result']


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


def _smooth(nt, x, lo, hi):
    mr = nt.nodes.new('ShaderNodeMapRange'); mr.interpolation_type = 'SMOOTHSTEP'
    mr.inputs['From Min'].default_value = lo; mr.inputs['From Max'].default_value = hi
    nt.links.new(x, mr.inputs['Value'])
    return mr.outputs['Result']


def _noise(nt, vec, scale, detail=2.0, rough=0.5):
    n = nt.nodes.new('ShaderNodeTexNoise'); n.inputs['Scale'].default_value = scale; n.inputs['Detail'].default_value = detail
    n.inputs['Roughness'].default_value = rough
    nt.links.new(vec, n.inputs['Vector'])
    return n


def _attr(nt, name):
    a = nt.nodes.new('ShaderNodeAttribute'); a.attribute_type = 'GEOMETRY'; a.attribute_name = name
    s = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(a.outputs['Color'], s.inputs[0])
    al = a.outputs['Alpha']
    return s.outputs['Red'], s.outputs['Green'], s.outputs['Blue'], al


def _detail_layer(nt, pos, key, n_macro, seed):
    """-> (colour socket, height socket, roughness value). Palette colour x texture detail, anti-tiled."""
    aid, size, dpow, chroma, rough = LAYER[key]
    rot1 = 0.37 * seed; rot2 = 1.9 + 0.61 * seed
    va = _mapping(nt, pos, size, rot1, (seed * 3.1, seed * 1.7, 0))
    vb = _mapping(nt, pos, size * 2.37, rot2, (seed * 5.3, -seed * 2.9, 0))
    ta = _tex(nt, aid, 'diff', va); tb = _tex(nt, aid, 'diff', vb)
    ha = _tex(nt, aid, 'disp', va, False)
    if ta is None:
        c = PAL[key][0]
        return _mix(nt, 0.0, c, c), None, rough
    # blend the two scales with a smooth noise (breaks the tile grid)
    bn = _noise(nt, pos, 0.045 + 0.01 * seed, 2)
    dm = _mix(nt, _smooth(nt, bn.outputs['Fac'], 0.42, 0.58), ta.outputs['Color'], tb.outputs['Color'])
    mean = MEAN.get(aid, (0.2, 0.2, 0.2))
    ml = 0.3 * mean[0] + 0.59 * mean[1] + 0.11 * mean[2]
    bw = nt.nodes.new('ShaderNodeRGBToBW'); nt.links.new(dm, bw.inputs[0])
    lum = _math(nt, 'DIVIDE', bw.outputs[0], ml)
    lumc = nt.nodes.new('ShaderNodeCombineColor')
    for i in range(3):
        nt.links.new(lum, lumc.inputs[i])
    rgb = _mix(nt, 1.0, dm, (1 / max(mean[0], 1e-3), 1 / max(mean[1], 1e-3), 1 / max(mean[2], 1e-3)), 'MULTIPLY')
    det = _mix(nt, chroma, lumc.outputs[0], rgb)
    g = nt.nodes.new('ShaderNodeGamma'); g.inputs['Gamma'].default_value = dpow
    nt.links.new(det, g.inputs['Color'])
    # palette: two tones by a 60 m noise, accent by a 25 m voronoi-ish noise
    p0, p1, p2 = PAL[key]
    n1 = _noise(nt, pos, 0.012 + 0.002 * seed, 1.5)
    n2 = _noise(nt, pos, 0.04 + 0.004 * seed, 2)
    c = _mix(nt, _smooth(nt, n1.outputs['Fac'], 0.38, 0.62), p0, p1)
    c = _mix(nt, _math(nt, 'MULTIPLY', _smooth(nt, n2.outputs['Fac'], 0.55, 0.72), 0.7), c, p2)
    col = _mix(nt, 1.0, c, g.outputs['Color'], 'MULTIPLY')
    return col, (ha.outputs['Color'] if ha else None), rough


class _Stack:
    """Height-blended layer stack carrying colour, height and roughness."""
    def __init__(self, nt, col, h, rough):
        self.nt, self.col, self.h, self.r = nt, col, h, rough

    def over(self, layer, w, sharp=0.6):
        nt = self.nt
        col, h, rough = layer
        if h is not None:
            # height-aware transition: w shifted by the incoming layer's height map
            x = _math(nt, 'ADD', w, _math(nt, 'MULTIPLY', _math(nt, 'SUBTRACT', h, 0.5), sharp))
            ww = _smooth(nt, x, 0.3, 0.7)
            ww = _math(nt, 'MINIMUM', ww, _math(nt, 'MULTIPLY', w, 3.0))  # never where w == 0
        else:
            ww = w
        self.col = _mix(nt, ww, self.col, col)
        if h is not None and self.h is not None:
            self.h = _mixf(nt, ww, self.h, h)
        self.r = _mixf(nt, ww, self.r, rough) if not isinstance(self.r, float) else _mixf(nt, ww, self.r, rough)
        return ww


def terrain_material(albedo_path=None):
    m = bpy.data.materials.new('MAT_Terrain_PBR'); m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(bsdf.outputs[0], out.inputs['Surface'])
    tc = nt.nodes.new('ShaderNodeTexCoord')
    pos = tc.outputs['Object']
    field, meadow, dev, _ = _attr(nt, 'lu_a'); rock_lu = _attr(nt, 'lu_a')[3]
    forest, bank, shoulder, _ = _attr(nt, 'lu_b')
    moist, disturbed, canopy, hedge = _attr(nt, 'eco_a')
    pasture, hay, plowed, fallow = _attr(nt, 'eco_b')
    lawn, fang, tpi, talus = _attr(nt, 'eco_c')
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    nz = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(geo.outputs['Normal'], nz.inputs[0])
    slope = _math(nt, 'SUBTRACT', 1.0, nz.outputs['Z'])      # 0 flat .. 1 vertical
    L = {k: _detail_layer(nt, pos, k, None, i + 1) for i, k in enumerate(LAYER)}

    # --- field stripes / furrows (rotated by the per-field angle)
    ang = _math(nt, 'MULTIPLY', fang, math.pi)
    sp = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(pos, sp.inputs[0])
    u = _math(nt, 'ADD', _math(nt, 'MULTIPLY', sp.outputs['X'], _math(nt, 'COSINE', ang)), _math(nt, 'MULTIPLY', sp.outputs['Y'], _math(nt, 'SINE', ang)))
    wob = _noise(nt, pos, 0.15, 1)
    u = _math(nt, 'ADD', u, _math(nt, 'MULTIPLY', wob.outputs['Fac'], 0.8))
    mow = _math(nt, 'SINE', _math(nt, 'MULTIPLY', u, 2 * math.pi / 4.5))            # 4.5 m mower passes
    furrow = _math(nt, 'SINE', _math(nt, 'MULTIPLY', u, 2 * math.pi / 0.9))         # 0.9 m plow furrows
    hayc, hayh, hayr = L['hay']
    hayc = _mix(nt, 1.0, hayc, _mix(nt, _smooth(nt, mow, -0.2, 0.2), (0.9, 0.9, 0.88), (1.1, 1.08, 1.02)), 'MULTIPLY')
    L['hay'] = (hayc, hayh, hayr)
    plc, plh, plr = L['plowed']
    fz = _smooth(nt, furrow, -0.6, 0.8)
    plc = _mix(nt, 1.0, plc, _mix(nt, fz, (0.72, 0.72, 0.74), (1.08, 1.06, 1.04)), 'MULTIPLY')
    plh = _math(nt, 'ADD', _math(nt, 'MULTIPLY', plh, 0.5), _math(nt, 'MULTIPLY', fz, 0.5)) if plh is not None else fz
    L['plowed'] = (plc, plh, plr)

    # --- blend stack (base: forest floor)
    S = _Stack(nt, *L['forest'])
    humus_w = _math(nt, 'MULTIPLY', _smooth(nt, moist, 0.55, 0.85), 0.8)
    S.over(L['humus'], humus_w)
    opn = _math(nt, 'SUBTRACT', 1.0, _smooth(nt, _math(nt, 'MAXIMUM', canopy, _math(nt, 'MULTIPLY', forest, 0.9)), 0.35, 0.8))
    S.over(L['meadow'], _math(nt, 'MULTIPLY', _math(nt, 'MAXIMUM', meadow, field), opn))
    S.over(L['pasture'], _math(nt, 'MULTIPLY', _math(nt, 'MAXIMUM', pasture, _math(nt, 'MULTIPLY', meadow, 0.5)), opn))
    S.over(L['fallow'], _math(nt, 'MULTIPLY', fallow, opn))
    S.over(L['hay'], _math(nt, 'MULTIPLY', hay, opn), 0.2)
    S.over(L['plowed'], _math(nt, 'MULTIPLY', plowed, opn), 0.3)
    S.over(L['lawn'], _math(nt, 'MULTIPLY', _math(nt, 'MAXIMUM', lawn, _math(nt, 'MULTIPLY', dev, 0.8)), opn))
    # roadside: dry grass verge, then gravel shoulder right at the pavement
    verge = _math(nt, 'MULTIPLY', _smooth(nt, shoulder, 0.1, 0.45), 0.9)
    S.over(L['verge'], verge)
    S.over(L['gravel'], _smooth(nt, shoulder, 0.55, 0.9))
    # red clay: disturbed ground, road cuts / fills (steep + near roads), bare steep open slopes
    cut = _math(nt, 'MULTIPLY', _smooth(nt, slope, 0.12, 0.3), _math(nt, 'MAXIMUM', _smooth(nt, shoulder, 0.05, 0.4), _math(nt, 'MULTIPLY', disturbed, 0.7)))
    clay_w = _math(nt, 'MAXIMUM', cut, _math(nt, 'MULTIPLY', _smooth(nt, disturbed, 0.6, 1.0), 0.5))
    clay_w = _math(nt, 'MULTIPLY', clay_w, _math(nt, 'ADD', 0.35, _math(nt, 'MULTIPLY', _noise(nt, pos, 0.08, 3).outputs['Fac'], 1.0)))
    S.over(L['clay'], clay_w)
    # water margins: wet mud + gravel bars
    S.over(L['mud'], _math(nt, 'MULTIPLY', _math(nt, 'MAXIMUM', _math(nt, 'MULTIPLY', bank, 0.8), _smooth(nt, moist, 0.85, 1.0)), 0.9))
    S.over(L['bank'], _smooth(nt, bank, 0.45, 0.9))
    # talus / scree below outcrops, then exposed rock on cliffs + rock land use
    S.over(L['talus'], _math(nt, 'MULTIPLY', talus, _math(nt, 'SUBTRACT', 1.0, _math(nt, 'MULTIPLY', canopy, 0.6))))
    rock_w = _math(nt, 'MAXIMUM', _smooth(nt, slope, 0.3, 0.5), _math(nt, 'MULTIPLY', _smooth(nt, rock_lu, 0.3, 0.8), _smooth(nt, slope, 0.08, 0.25)))
    S.over(L['rock'], rock_w, 0.9)
    # rock lichen / moss: pale grey-green lichen blotches on dry rock, moss on moist rock
    lich = _noise(nt, pos, 0.9, 4)
    lichen_w = _math(nt, 'MULTIPLY', rock_w, _math(nt, 'MULTIPLY', _smooth(nt, lich.outputs['Fac'], 0.52, 0.68), 0.7))
    col = _mix(nt, lichen_w, S.col, (0.13, 0.135, 0.11))
    moss_w = _math(nt, 'MULTIPLY', rock_w, _math(nt, 'MULTIPLY', _smooth(nt, moist, 0.5, 0.9), _smooth(nt, nz.outputs['Z'], 0.5, 0.9)))
    col = _mix(nt, _math(nt, 'MULTIPLY', moss_w, 0.8), col, (0.05, 0.07, 0.025))

    # --- macro variation: 250 m and 50 m value / hue drifts (breaks uniform fields and forest floors)
    m1 = _noise(nt, pos, 0.004, 2); m2 = _noise(nt, pos, 0.02, 2)
    hs = nt.nodes.new('ShaderNodeHueSaturation')
    nt.links.new(col, hs.inputs['Color'])
    hue = nt.nodes.new('ShaderNodeMapRange'); hue.inputs['To Min'].default_value = 0.485; hue.inputs['To Max'].default_value = 0.515
    nt.links.new(m1.outputs['Fac'], hue.inputs['Value']); nt.links.new(hue.outputs['Result'], hs.inputs['Hue'])
    val = nt.nodes.new('ShaderNodeMapRange'); val.inputs['To Min'].default_value = 0.86; val.inputs['To Max'].default_value = 1.14
    nt.links.new(m2.outputs['Fac'], val.inputs['Value']); nt.links.new(val.outputs['Result'], hs.inputs['Value'])
    col = hs.outputs['Color']
    # near the camera the open ground is covered by ground-cover plants (lib_groundcover): darken the
    # soil between clumps (thatch / self-shadow) so it never reads as bare sand; fades out by ~140 m
    cd = nt.nodes.new('ShaderNodeCameraData')
    near = _math(nt, 'SUBTRACT', 1.0, _smooth(nt, cd.outputs['View Distance'], 40.0, 150.0))
    grassy = _math(nt, 'MULTIPLY', _math(nt, 'MAXIMUM', _math(nt, 'MAXIMUM', pasture, fallow), _math(nt, 'MAXIMUM', meadow, _math(nt, 'MULTIPLY', hay, 0.5))), opn)
    col = _mix(nt, _math(nt, 'MULTIPLY', _math(nt, 'MULTIPLY', near, grassy), 0.55), col, _mix(nt, 1.0, col, (0.45, 0.42, 0.35), 'MULTIPLY'))
    # forest floor under canopy slightly darker (canopy shade is also real lighting; keep it subtle)
    col = _mix(nt, _math(nt, 'MULTIPLY', canopy, 0.25), col, _mix(nt, 1.0, col, (0.75, 0.75, 0.75), 'MULTIPLY'))
    # stylized macro albedo from the map painter (very light touch: keeps map identity at world scale)
    if albedo_path and os.path.exists(albedo_path):
        alb = nt.nodes.new('ShaderNodeTexImage'); alb.image = _img(albedo_path); alb.interpolation = 'Cubic'
        uv = nt.nodes.new('ShaderNodeUVMap'); nt.links.new(uv.outputs[0], alb.inputs['Vector'])
        bwm = nt.nodes.new('ShaderNodeRGBToBW'); nt.links.new(alb.outputs['Color'], bwm.inputs[0])
        mv = _math(nt, 'ADD', 0.75, _math(nt, 'MULTIPLY', bwm.outputs[0], 1.4))
        mvc = nt.nodes.new('ShaderNodeCombineColor')
        for i in range(3):
            nt.links.new(mv, mvc.inputs[i])
        col = _mix(nt, 0.12, col, _mix(nt, 1.0, col, mvc.outputs[0], 'MULTIPLY'))
    nt.links.new(col, bsdf.inputs['Base Color'])
    nt.links.new(S.r, bsdf.inputs['Roughness'])
    bsdf.inputs['Specular IOR Level'].default_value = 0.35
    if S.h is not None:
        bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.45; bp.inputs['Distance'].default_value = 0.06
        nt.links.new(S.h, bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], bsdf.inputs['Normal'])
    return m


# ------------------------------------------------------------------ simple PBR (roads etc.)
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


# ------------------------------------------------------------------ shared utility materials
UTIL = {
    # name: (texture, tile m, palette tint (linear), roughness, metallic, bump, weathering)
    'concrete':        ('rough_concrete', 3.0, (0.40, 0.385, 0.35), 0.88, 0.0, 0.25, 0.35),
    'bridge_concrete': ('concrete_wall_006', 4.0, (0.36, 0.345, 0.315), 0.9, 0.0, 0.3, 0.6),
    'rusted_metal':    ('rusty_metal_02', 1.5, (0.20, 0.10, 0.05), 0.75, 0.35, 0.3, 0.0),
    'weathered_wood':  ('weathered_planks', 2.0, (0.13, 0.11, 0.09), 0.9, 0.0, 0.4, 0.0),
    'gravel':          ('gravel_ground_01', 1.8, PAL['gravel'][0], 0.9, 0.0, 0.5, 0.0),
    'dirt':            ('dirt_floor', 2.5, (0.14, 0.09, 0.055), 0.95, 0.0, 0.4, 0.0),
    'red_clay':        ('red_mud_stones', 2.5, PAL['clay'][0], 0.9, 0.0, 0.4, 0.0),
    'rock':            ('rock_face_03', 4.0, PAL['rock'][0], 0.85, 0.0, 0.6, 0.0),
    'mud':             ('brown_mud_02', 2.2, PAL['mud'][0], 0.55, 0.0, 0.4, 0.0),
    'dry_grass':       ('withered_grass', 2.0, PAL['verge'][0], 0.95, 0.0, 0.3, 0.0),
    'forest_floor':    ('forest_leaves_02', 2.6, PAL['forest'][1], 0.9, 0.0, 0.4, 0.0),
    'grass':           ('leafy_grass', 2.8, PAL['pasture'][1], 0.95, 0.0, 0.3, 0.0),
}


def get(name):
    """Shared material by name (world-space box projection, no UVs needed). Created once."""
    mname = f'MAT_A3_{name}'
    m = bpy.data.materials.get(mname)
    if m:
        return m
    aid, size, tint, rough, metal, bump, weather = UTIL[name]
    m = bpy.data.materials.new(mname); m.use_nodes = True
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    vec = _mapping(nt, geo.outputs['Position'], size)
    d = _tex(nt, aid, 'diff', vec, True, box=True)
    h = _tex(nt, aid, 'disp', vec, False, box=True)
    mean = MEAN.get(aid, (0.2, 0.2, 0.2))
    if d is not None:
        det = _mix(nt, 1.0, d.outputs['Color'], (1 / max(mean[0], 1e-3), 1 / max(mean[1], 1e-3), 1 / max(mean[2], 1e-3)), 'MULTIPLY')
        g = nt.nodes.new('ShaderNodeGamma'); g.inputs['Gamma'].default_value = 0.8; nt.links.new(det, g.inputs['Color'])
        col = _mix(nt, 1.0, tint + (1,), g.outputs['Color'], 'MULTIPLY')
    else:
        col = _mix(nt, 0.0, tint, tint)
    # macro weathering (streaks / grime) for concrete
    if weather:
        n = _noise(nt, geo.outputs['Position'], 0.35, 4)
        sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(geo.outputs['Position'], sep.inputs[0])
        streak = _noise(nt, geo.outputs['Position'], 1.2, 2)
        grime = _math(nt, 'MULTIPLY', _smooth(nt, n.outputs['Fac'], 0.45, 0.7), weather)
        col = _mix(nt, grime, col, _mix(nt, 1.0, col, (0.62, 0.6, 0.55), 'MULTIPLY'))
    nt.links.new(col, bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = rough
    bsdf.inputs['Metallic'].default_value = metal
    if h is not None and bump:
        bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = bump; bp.inputs['Distance'].default_value = 0.03
        nt.links.new(h.outputs['Color'], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], bsdf.inputs['Normal'])
    return m


def hdri_world(hdr_path, strength=1.0, rot_deg=0.0, sun_rot=None):
    """Legacy (build_world.setup_world); lib_atmosphere.world() is the current path."""
    sc = bpy.context.scene
    w = sc.world
    nt = w.node_tree
    for n in list(nt.nodes):
        if n.type in ('TEX_SKY', 'BACKGROUND', 'TEX_ENVIRONMENT', 'MAPPING', 'TEX_COORD'):
            nt.nodes.remove(n)
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_WORLD')
    tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Rotation'].default_value = (0, 0, math.radians(rot_deg))
    env = nt.nodes.new('ShaderNodeTexEnvironment'); env.image = _img(hdr_path)
    bg = nt.nodes.new('ShaderNodeBackground'); bg.inputs['Strength'].default_value = strength
    nt.links.new(tc.outputs['Generated'], mp.inputs['Vector']); nt.links.new(mp.outputs[0], env.inputs['Vector'])
    nt.links.new(env.outputs[0], bg.inputs[0]); nt.links.new(bg.outputs[0], out.inputs['Surface'])


# ------------------------------------------------------------------ terrain ecology attributes
def add_terrain_attributes(terrain_objs):
    """Per-vertex ecology attributes from public/world/eco_u8.bin (tools/pipeline/vegetation.py):
       eco_a = (moisture, disturbed, canopy cover, hedge), eco_b = (pasture, hay, plowed, fallow),
       eco_c = (lawn, field angle 0..1 -> 0..pi, tpi (0.5 = flat), talus).
    Used by the terrain material and the ground-cover scatter."""
    import numpy as np
    H, W = 667, 2000
    p = os.path.join(ROOT, 'public/world/eco_u8.bin')
    if not os.path.exists(p):
        print('  no eco_u8.bin: terrain ecology attributes skipped')
        return
    raw = np.fromfile(p, np.uint8).reshape(H, W, 8)
    E = raw.astype(np.float32) / 255
    ft = raw[..., 3]

    def bil(a, x, y):
        x = np.clip(x - 0.5, 0, W - 1.001); y = np.clip(y - 0.5, 0, H - 1.001)
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
        return a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy) + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy
    onehot = np.stack([ft == k for k in (1, 2, 3, 4, 5)], -1).astype(np.float32)
    for ob in terrain_objs:
        me = ob.data
        co = np.zeros(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
        x = co[:, 0] / 2.5 + 1000.0; y = -co[:, 1] / 2.5 + 333.5
        e = bil(E, x, y); oh = bil(onehot, x, y)
        # field angle: nearest (not interpolated) so stripes stay coherent inside a field
        ang = E[np.clip(y.astype(int), 0, H - 1), np.clip(x.astype(int), 0, W - 1), 4]
        for name, data in (('eco_a', np.c_[e[:, 0], e[:, 1], e[:, 2], e[:, 5]]),
                           ('eco_b', oh[:, :4]),
                           ('eco_c', np.c_[oh[:, 4], ang, e[:, 6], e[:, 7]])):
            if name in me.attributes:
                me.attributes.remove(me.attributes[name])
            a = me.attributes.new(name, 'FLOAT_COLOR', 'POINT')
            a.data.foreach_set('color', np.ascontiguousarray(data, np.float32).ravel())
