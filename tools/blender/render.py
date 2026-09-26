"""Render cameras from a built world .blend.
    blender -b exports/blender/world.blend --python tools/blender/render.py -- --cams CAM_A,CAM_B --res 1600x900 --samples 48 --outdir exports/renders
"""
import bpy, os, sys
ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
def ARG(k, d=None):
    return ARGS[ARGS.index(k) + 1] if k in ARGS else d
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sc = bpy.context.scene
w, h = map(int, ARG('--res', '1600x900').split('x'))
sc.render.resolution_x, sc.render.resolution_y = w, h
sc.render.resolution_percentage = 100
sc.cycles.samples = int(ARG('--samples', 48))
sc.cycles.use_adaptive_sampling = True
sc.render.threads_mode = 'AUTO'
outdir = os.path.join(ROOT, ARG('--outdir', 'exports/renders'))
os.makedirs(outdir, exist_ok=True)
cams = ARG('--cams', 'CAM_HollowRidge_Overlook').split(',')
import math
from mathutils import Vector
HDRI_SUN_MATH_DEG = 139.8  # measured: kloppenheim_06 sun direction in HDRI space (atan2(y, x)), elev 4.6 deg
HDRI_MIRROR = 325.0  # empirical: equirect azimuth is mirrored relative to world compass


def set_sun(az_compass, el):
    th = math.radians(90 - az_compass)
    d = Vector((math.cos(th) * math.cos(math.radians(el)), math.sin(th) * math.cos(math.radians(el)), math.sin(math.radians(el))))
    so = bpy.data.objects.get('SUN')
    if so:
        so.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    for n in sc.world.node_tree.nodes:
        if n.type == 'MAPPING':
            n.inputs['Rotation'].default_value[2] = math.radians(HDRI_SUN_MATH_DEG - (90 - (HDRI_MIRROR - az_compass)))


if ARG('--sun'):
    az, el = map(float, ARG('--sun').split(','))
    set_sun(az, el)
if ARG('--sun-energy'):
    bpy.data.lights['SUN'].energy = float(ARG('--sun-energy'))
if ARG('--exposure'):
    sc.view_settings.exposure = float(ARG('--exposure'))
if ARG('--sky'):
    for n in sc.world.node_tree.nodes:
        if n.type == 'BACKGROUND':
            n.inputs['Strength'].default_value = float(ARG('--sky'))
if ARG('--mist'):
    for n in sc.node_tree.nodes:
        if n.type == 'MATH':
            n.inputs[1].default_value = float(ARG('--mist'))
if ARG('--hide'):
    for pre in ARG('--hide').split(','):
        for o in bpy.data.objects:
            if o.name.startswith(pre):
                o.hide_render = True
for c in cams:
    ob = bpy.data.objects[c]
    sc.camera = ob
    if c == 'CAM_Validation_Top':
        sc.render.resolution_x, sc.render.resolution_y = 2000, 667
        for o in bpy.data.objects:
            if o.name == 'BACKDROP':
                o.hide_render = True
    sc.render.filepath = os.path.join(outdir, c + '.jpg')
    bpy.ops.render.render(write_still=True)
    print('rendered', sc.render.filepath)
