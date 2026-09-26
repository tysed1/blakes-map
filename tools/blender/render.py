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
