"""Lighting, sky and atmosphere for the Blender world (graphics ref.png look).

  * HDRI sky (Poly Haven kloppenheim_06_puresky, CC0) - one Mapping node, so render.py --sun
    can keep the HDRI sun aligned with the SUN lamp (calibrated there).
  * SUN: warm low golden-hour key light with a soft penumbra.
  * Ambient term (node group A3_Ambient, inserted after every Principled BSDF): forgiving
    shadows - forest floors, crown undersides and shaded road cuts stay readable instead of
    black (stylized realism). Optional SUN_FILL (--fill > 0): skylight fill shadowed only by
    terrain (shadow linking); off by default because it doubles render time.
  * Compositor aerial perspective: distance haze (mist pass) that shifts distant ridges
    toward desaturated blue, warmer toward the sun side; geometry only (sky keeps its own
    colour). Only ONE Math node in the tree so render.py --mist keeps working
    (it sets the haze strength).
  * Grade: warm highlights / cool lifted shadows, gentle saturation, soft glow.

Hook (build_world.main): LA.setup(ARG) replaces setup_world().
"""
import bpy, math, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HDRI = os.path.join(ROOT, 'assets/external/polyhaven/kloppenheim_06_puresky/kloppenheim_06_puresky_4k.hdr')

SUN_AZ, SUN_EL = 255.0, 18.0          # compass azimuth / elevation (deg): late afternoon, WSW (sun from the left)
SUN_COLOR = (1.0, 0.80, 0.60)
FILL_COLOR = (0.62, 0.74, 1.0)
HAZE_NEAR = (0.62, 0.66, 0.72, 1)     # haze colour mid distance (slightly warm grey-blue)
HAZE_FAR = (0.40, 0.52, 0.74, 1)      # far ridges: cooler, bluer


def sun_dir(az, el):
    from mathutils import Vector
    th = math.radians(90 - az)
    return Vector((math.cos(th) * math.cos(math.radians(el)), math.sin(th) * math.cos(math.radians(el)), math.sin(math.radians(el))))


def _lamp(name, energy, color, angle_deg, az, el):
    L = bpy.data.lights.get(name) or bpy.data.lights.new(name, 'SUN')
    L.energy = energy; L.color = color; L.angle = math.radians(angle_deg)
    ob = bpy.data.objects.get(name) or bpy.data.objects.new(name, L)
    if ob.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(ob)
    ob.rotation_euler = (-sun_dir(az, el)).to_track_quat('-Z', 'Y').to_euler()
    return ob


def world(strength=0.8, rot_deg=0.0):
    sc = bpy.context.scene
    w = bpy.data.worlds.get('World') or bpy.data.worlds.new('World')
    sc.world = w; w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    tc = nt.nodes.new('ShaderNodeTexCoord'); mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Rotation'].default_value = (0, 0, math.radians(rot_deg))
    env = nt.nodes.new('ShaderNodeTexEnvironment'); env.image = bpy.data.images.load(HDRI, check_existing=True)
    bg = nt.nodes.new('ShaderNodeBackground'); bg.inputs['Strength'].default_value = strength
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(tc.outputs['Generated'], mp.inputs['Vector']); nt.links.new(mp.outputs[0], env.inputs['Vector'])
    nt.links.new(env.outputs[0], bg.inputs[0]); nt.links.new(bg.outputs[0], out.inputs['Surface'])
    w.mist_settings.start = 500; w.mist_settings.depth = 15000; w.mist_settings.falloff = 'LINEAR'
    return w


def lights(fill_energy=1.1):
    sun = _lamp('SUN', 6.0, SUN_COLOR, 1.6, SUN_AZ, SUN_EL)
    if fill_energy <= 0:  # default: ambient term instead (a shadowed fill through alpha foliage doubles render time)
        return sun, None
    fill = _lamp('SUN_FILL', fill_energy, FILL_COLOR, 35.0, SUN_AZ + 150, 62)
    # shadow linking: the fill is only shadowed by terrain / rocks / structures, not by foliage
    try:
        blk = bpy.data.collections.get('A3_FILL_BLOCKERS') or bpy.data.collections.new('A3_FILL_BLOCKERS')
        fill.light_linking.blocker_collection = blk
        # a blocker collection references objects without being linked in the scene
        for o in bpy.data.objects:
            if o.type == 'MESH' and o.name.startswith(('TERRAIN_', 'BRIDGE ', 'BACKDROP', 'PROP_points')):
                if o.name not in blk.objects:
                    blk.objects.link(o)
    except Exception as e:  # older Blender: plain fill without linking
        print('  fill light: no shadow linking', e)
    return sun, fill


def compositor(haze=0.85):
    """Aerial perspective on geometry only, edge-correct: the film is rendered transparent and the
    sky comes back from the Environment pass, so silhouettes against the sky are anti-aliased
    correctly (a depth-threshold sky mask leaves dark un-hazed fringes on far ridges)."""
    sc = bpy.context.scene
    vl = sc.view_layers[0]
    vl.use_pass_mist = True; vl.use_pass_z = True; vl.use_pass_environment = True
    sc.render.film_transparent = True
    sc.use_nodes = True
    ct = sc.node_tree
    for n in list(ct.nodes):
        ct.nodes.remove(n)
    N, L = ct.nodes, ct.links

    def mixrgb(op, a, b, fac=1.0):
        m = N.new('CompositorNodeMixRGB'); m.blend_type = op; m.use_clamp = op in ('SUBTRACT', 'DIVIDE')
        if isinstance(fac, float):
            m.inputs[0].default_value = fac
        else:
            L.new(fac, m.inputs[0])
        for sock, v in ((m.inputs[1], a), (m.inputs[2], b)):
            if isinstance(v, tuple):
                sock.default_value = v
            else:
                L.new(v, sock)
        return m.outputs[0]
    rl = N.new('CompositorNodeRLayers'); comp = N.new('CompositorNodeComposite')
    A = rl.outputs['Alpha']
    one_minus_a = mixrgb('SUBTRACT', (1, 1, 1, 1), A)
    # mist of the geometry part of the pixel (background mist = 1): (mist - (1 - A)) / A
    m_geo = mixrgb('DIVIDE', mixrgb('SUBTRACT', rl.outputs['Mist'], one_minus_a), A)
    bw = N.new('CompositorNodeRGBToBW'); L.new(m_geo, bw.inputs[0])
    # haze amount = mist * strength (the ONLY Math node: render.py --mist sets its 2nd input)
    k = N.new('CompositorNodeMath'); k.operation = 'MULTIPLY'; k.inputs[1].default_value = haze; k.use_clamp = True
    L.new(bw.outputs[0], k.inputs[0])
    # haze colour ramps from near (warm grey) to far (blue), premultiplied by coverage
    ramp = N.new('CompositorNodeValToRGB')
    ramp.color_ramp.elements[0].color = HAZE_NEAR; ramp.color_ramp.elements[1].color = HAZE_FAR
    L.new(bw.outputs[0], ramp.inputs['Fac'])
    haze_a = mixrgb('MULTIPLY', ramp.outputs['Image'], A)
    # desaturate with distance (atmospheric perspective), then blend toward the haze colour
    desat = N.new('CompositorNodeHueSat')
    sat_amt = N.new('CompositorNodeMapRange'); sat_amt.inputs['From Min'].default_value = 0; sat_amt.inputs['From Max'].default_value = 1
    sat_amt.inputs['To Min'].default_value = 1.0; sat_amt.inputs['To Max'].default_value = 0.35; sat_amt.use_clamp = True
    L.new(k.outputs[0], sat_amt.inputs['Value'])
    L.new(rl.outputs['Image'], desat.inputs['Image']); L.new(sat_amt.outputs[0], desat.inputs['Saturation'])
    geo = mixrgb('MIX', desat.outputs[0], haze_a, k.outputs[0])
    # sky back in (environment pass) behind the geometry
    img = mixrgb('ADD', geo, mixrgb('MULTIPLY', rl.outputs['Env'], one_minus_a))
    sa = N.new('CompositorNodeSetAlpha'); sa.mode = 'REPLACE_ALPHA'; sa.inputs['Alpha'].default_value = 1.0
    L.new(img, sa.inputs['Image'])
    # soft glow (golden hour bloom)
    gl = N.new('CompositorNodeGlare'); gl.glare_type = 'FOG_GLOW'; gl.quality = 'MEDIUM'; gl.mix = -0.88; gl.threshold = 0.85; gl.size = 8
    L.new(sa.outputs[0], gl.inputs['Image'])
    # grade: warm highlights, cool lifted shadows, rich but controlled saturation
    cb = N.new('CompositorNodeColorBalance'); cb.correction_method = 'LIFT_GAMMA_GAIN'
    cb.lift = (0.985, 1.0, 1.03); cb.gamma = (1.03, 1.0, 0.95); cb.gain = (0.98, 0.93, 0.8)
    hs = N.new('CompositorNodeHueSat'); hs.inputs['Saturation'].default_value = 1.28
    L.new(gl.outputs[0], cb.inputs['Image']); L.new(cb.outputs[0], hs.inputs['Image'])
    L.new(hs.outputs[0], comp.inputs['Image'])


AMBIENT_COLOR = (0.78, 0.86, 1.0)   # sky-tinted ambient


def ambient_group(strength=0.3):
    """Shader node group 'A3_Ambient': Emission(albedo * ambient colour * strength).
    A cheap stylized ambient term (game-engine style): lifts shadowed forest floors, crown
    undersides and road cuts without the cost of an extra shadowed light through alpha foliage.
    Tune globally: bpy.data.node_groups['A3_Ambient'].nodes['AMB'].outputs[0].default_value."""
    g = bpy.data.node_groups.get('A3_Ambient')
    if g:
        return g
    g = bpy.data.node_groups.new('A3_Ambient', 'ShaderNodeTree')
    g.interface.new_socket('Albedo', in_out='INPUT', socket_type='NodeSocketColor')
    g.interface.new_socket('Shader', in_out='OUTPUT', socket_type='NodeSocketShader')
    N, L = g.nodes, g.links
    gi = N.new('NodeGroupInput'); go = N.new('NodeGroupOutput')
    v = N.new('ShaderNodeValue'); v.name = 'AMB'; v.outputs[0].default_value = strength
    mul = N.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs['Factor'].default_value = 1
    mul.inputs['B'].default_value = AMBIENT_COLOR + (1,)
    L.new(gi.outputs['Albedo'], mul.inputs['A'])
    em = N.new('ShaderNodeEmission'); L.new(mul.outputs['Result'], em.inputs['Color']); L.new(v.outputs[0], em.inputs['Strength'])
    L.new(em.outputs[0], go.inputs['Shader'])
    return g


def apply_ambient(mats=None, skip=('MAT_Water', 'WATER', 'Marking', 'Backdrop')):
    """Insert Add(principled, A3_Ambient(base colour)) after every Principled BSDF."""
    g = ambient_group()
    n_done = 0
    for m in (mats or bpy.data.materials):
        if not m.use_nodes or m.get('a3_amb') or any(k in m.name for k in skip):
            continue
        nt = m.node_tree
        for b in [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED']:
            outs = list(b.outputs[0].links)
            if not outs:
                continue
            gn = nt.nodes.new('ShaderNodeGroup'); gn.node_tree = g
            bc = b.inputs['Base Color']
            if bc.is_linked:
                nt.links.new(bc.links[0].from_socket, gn.inputs['Albedo'])
            else:
                gn.inputs['Albedo'].default_value = bc.default_value
            add = nt.nodes.new('ShaderNodeAddShader')
            nt.links.new(b.outputs[0], add.inputs[0]); nt.links.new(gn.outputs[0], add.inputs[1])
            for l in outs:
                to = l.to_socket
                nt.links.remove(l)
                nt.links.new(add.outputs[0], to)
        # the ambient emission must NOT become a light source (millions of emissive triangles would
        # blow up the light tree: memory + render time)
        try:
            m.cycles.emission_sampling = 'NONE'
        except Exception:
            pass
        m['a3_amb'] = True
        n_done += 1
    return n_done


def setup(ARG=lambda k, d=None: d):
    world(float(ARG('--sky', 0.8)), float(ARG('--skyrot', 0)))
    lights(float(ARG('--fill', 0.0)))
    compositor(float(ARG('--haze', 0.85)))
    ambient_group(float(ARG('--ambient', 0.9)))
    print('  ambient on', apply_ambient(), 'materials')
