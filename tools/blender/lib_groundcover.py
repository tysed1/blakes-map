"""Ground cover: CC0 Poly Haven grass / shrubs / ferns / mossy rocks instanced by Geometry
Nodes onto the terrain, driven by the terrain's land-use attributes (lu_a / lu_b) and slope.

Grass is dense near the camera and thins with distance (budgeted per camera); shrubs/ferns/
rocks use world-wide densities. Game export uses the same rules (tools/pipeline/*)."""
import bpy, os, math
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PH = os.path.join(ROOT, 'assets/external/polyhaven')


def _append(aid, names):
    p = os.path.join(PH, aid, f'{aid}_1k.blend')
    with bpy.data.libraries.load(p, link=False) as (df, dt):
        dt.objects = [n for n in df.objects if n in names]
    return [o for o in dt.objects if o is not None]


def load_protos(coll):
    groups = {
        'grass': ('grass_medium_01', lambda n: n.endswith('_LOD1') and 'geonodes' not in n and any(k in n for k in ('_mid_', '_large_', '_tall_', '_small_'))),
        'shrub': ('shrub_02', lambda n: n.endswith('_LOD1')),
        'shrub2': ('shrub_01', lambda n: n.endswith('_LOD1')),
        'fern': ('fern_02', lambda n: True),
        'rock': ('rock_moss_set_01', lambda n: True),
        'rock2': ('rock_moss_set_02', lambda n: True),
    }
    out = {}
    for key, (aid, keep) in groups.items():
        p = os.path.join(PH, aid, f'{aid}_1k.blend')
        with bpy.data.libraries.load(p, link=False) as (df, dt):
            dt.objects = [n for n in df.objects if keep(n) and 'geonodes' not in n and not n.endswith('_geo')]
        objs = [o for o in dt.objects if o is not None and o.type == 'MESH']
        c = bpy.data.collections.new(f'GC_{key}')
        coll.children.link(c)
        for o in objs:
            o.location = (0, 0, 0)
            o.parent = None
            c.objects.link(o)
        c.hide_render = False
        out[key] = (c, objs)
    return out


def scatter_modifier(terrain_objs, protos, cam_locs, grass_radius=260.0):
    """One GN tree reused by all terrain chunks."""
    ng = bpy.data.node_groups.new('GN_GroundCover', 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    N, L = ng.nodes, ng.links
    gi = N.new('NodeGroupInput'); go = N.new('NodeGroupOutput')
    join = N.new('GeometryNodeJoinGeometry')
    L.new(gi.outputs[0], join.inputs[0])
    L.new(join.outputs[0], go.inputs[0])

    def attr(name, comp):
        a = N.new('GeometryNodeInputNamedAttribute'); a.data_type = 'FLOAT_COLOR'; a.inputs['Name'].default_value = name
        sep = N.new('FunctionNodeSeparateColor'); L.new(a.outputs['Attribute'], sep.inputs[0])
        return sep.outputs[comp]

    def math(op, a, b):
        m = N.new('ShaderNodeMath'); m.operation = op
        for i, v in enumerate((a, b)):
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                L.new(v, m.inputs[i])
        return m.outputs[0]

    nrm = N.new('GeometryNodeInputNormal')
    nz = N.new('ShaderNodeSeparateXYZ'); L.new(nrm.outputs[0], nz.inputs[0])
    flat = math('GREATER_THAN', nz.outputs['Z'], 0.82)
    pos3 = N.new('GeometryNodeInputPosition')
    sp = N.new('ShaderNodeSeparateXYZ'); L.new(pos3.outputs[0], sp.inputs[0])
    cp = N.new('ShaderNodeCombineXYZ'); L.new(sp.outputs['X'], cp.inputs['X']); L.new(sp.outputs['Y'], cp.inputs['Y'])
    pos = cp  # planar position (distance budget is horizontal)
    # distance to nearest hero camera (for the grass budget)
    dmin = None
    for (cx, cy) in cam_locs:
        v = N.new('ShaderNodeVectorMath'); v.operation = 'DISTANCE'
        L.new(pos.outputs[0], v.inputs[0]); v.inputs[1].default_value = (cx, cy, 0)
        # flatten z influence
        dmin = v.outputs['Value'] if dmin is None else math('MINIMUM', dmin, v.outputs['Value'])
    near = math('LESS_THAN', dmin, grass_radius) if dmin is not None else None

    def layer(key, density, weight, scale=(0.8, 1.3), zoff=0.0, align=False, seed=0):
        c, objs = protos[key]
        dist = N.new('GeometryNodeDistributePointsOnFaces'); dist.distribute_method = 'RANDOM'
        dist.inputs['Density'].default_value = density
        dist.inputs['Seed'].default_value = seed
        L.new(gi.outputs[0], dist.inputs['Mesh'])
        L.new(weight, dist.inputs['Selection'])
        ci = N.new('GeometryNodeCollectionInfo'); ci.inputs['Collection'].default_value = c
        ci.inputs['Separate Children'].default_value = True; ci.inputs['Reset Children'].default_value = True
        iop = N.new('GeometryNodeInstanceOnPoints'); iop.inputs['Pick Instance'].default_value = True
        rnd_i = N.new('FunctionNodeRandomValue'); rnd_i.data_type = 'INT'; rnd_i.inputs['Max'].default_value = max(0, len(objs) - 1); rnd_i.inputs['Seed'].default_value = seed + 1
        rnd_s = N.new('FunctionNodeRandomValue'); rnd_s.data_type = 'FLOAT'; rnd_s.inputs['Min'].default_value = scale[0]; rnd_s.inputs['Max'].default_value = scale[1]; rnd_s.inputs['Seed'].default_value = seed + 2
        rnd_r = N.new('FunctionNodeRandomValue'); rnd_r.data_type = 'FLOAT_VECTOR'; rnd_r.inputs[0].default_value = (0, 0, 0); rnd_r.inputs[1].default_value = (0.08, 0.08, 6.283); rnd_r.inputs['Seed'].default_value = seed + 3
        L.new(dist.outputs['Points'], iop.inputs['Points']); L.new(ci.outputs[0], iop.inputs['Instance'])
        L.new(rnd_i.outputs[2], iop.inputs['Instance Index'])
        L.new(rnd_s.outputs[1], iop.inputs['Scale']); L.new(rnd_r.outputs[0], iop.inputs['Rotation'])
        L.new(iop.outputs[0], join.inputs[0])

    field = attr('lu_a', 'Red'); meadow = attr('lu_a', 'Green'); dev = attr('lu_a', 'Blue')
    forest = attr('lu_b', 'Red'); bank = attr('lu_b', 'Green'); shoulder = attr('lu_b', 'Blue')
    openg = math('MAXIMUM', math('MAXIMUM', field, meadow), math('MULTIPLY', dev, 0.6))
    not_road = math('LESS_THAN', shoulder, 0.3)
    # grass clumps: open ground near cameras
    if near is not None:
        g = math('MULTIPLY', math('MULTIPLY', math('GREATER_THAN', openg, 0.4), not_road), near)
        layer('grass', 4.0, math('MULTIPLY', g, flat), (2.6, 4.6), seed=10)
    # shrubs: forest edges, meadows, banks
    edge = math('MULTIPLY', math('GREATER_THAN', forest, 0.2), math('LESS_THAN', forest, 0.85))
    sh = math('MAXIMUM', math('MAXIMUM', edge, math('GREATER_THAN', bank, 0.3)), math('GREATER_THAN', meadow, 0.5))
    layer('shrub', 0.02, math('MULTIPLY', sh, not_road), (1.2, 2.4), seed=20)
    layer('shrub2', 0.015, math('MULTIPLY', sh, not_road), (1.2, 2.2), seed=30)
    # ferns: forest floor
    layer('fern', 0.03, math('MULTIPLY', math('GREATER_THAN', forest, 0.6), not_road), (1.0, 1.8), seed=40)
    # rocks: banks + steep + forest floor sparse
    steep = math('LESS_THAN', nz.outputs['Z'], 0.8)
    rk = math('MAXIMUM', math('GREATER_THAN', bank, 0.4), steep)
    layer('rock', 0.012, math('MULTIPLY', rk, not_road), (0.8, 2.5), seed=50)
    layer('rock2', 0.008, math('MULTIPLY', math('GREATER_THAN', forest, 0.5), not_road), (0.6, 1.6), seed=60)
    for t in terrain_objs:
        m = t.modifiers.new('GroundCover', 'NODES'); m.node_group = ng
    return ng
