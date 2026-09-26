"""Render Agent 2's road-level test cameras inside a fully built world .blend (trees,
ground cover, A3 materials). One camera per process (embree is unstable across cameras):
    blender -b exports/a2/world.blend --python tools/blender/render_roadcams.py -- --cam US19_Gap_Driver \
        [--res 960x540] [--samples 16] [--outdir exports/a2/renders/full]
Sun / sky / exposure defaults match the coordinator's render.py call."""
import bpy, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []


def ARG(k, d=None):
    return ARGS[ARGS.index(k) + 1] if k in ARGS else d


import test_roads as TR
sc = bpy.context.scene
name = ARG('--cam')
rz = TR.RoadZ()
cc = bpy.data.collections.get('A2_TEST_CAMERAS') or bpy.data.collections.new('A2_TEST_CAMERAS')
if cc.name not in sc.collection.children:
    sc.collection.children.link(cc)
ob = TR.make_cam(name, TR.CAMS[name], rz, cc)
sc.camera = ob
w, h = map(int, ARG('--res', '960x540').split('x'))
sc.render.resolution_x, sc.render.resolution_y = w, h
sc.render.resolution_percentage = 100
sc.cycles.samples = int(ARG('--samples', 16))
sc.cycles.use_adaptive_sampling = True
# same lighting controls as tools/blender/render.py
sys.argv = [sys.argv[0], '--', '--cams', 'NONE', '--sun', ARG('--sun', '255,18'), '--sun-energy', ARG('--sun-energy', '10'),
            '--sky', ARG('--sky', '0.6'), '--exposure', ARG('--exposure', '0.7')]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'render.py')).read()
src = src.split('for c in cams:')[0]  # lighting setup only
exec(compile(src, 'render.py', 'exec'), {'__name__': 'render_setup'})
sc.camera = ob
outdir = os.path.join(TR.BW.ROOT, ARG('--outdir', 'exports/a2/renders/full'))
os.makedirs(outdir, exist_ok=True)
sc.render.filepath = os.path.join(outdir, name + '.jpg')
bpy.ops.render.render(write_still=True)
print('rendered', sc.render.filepath)
