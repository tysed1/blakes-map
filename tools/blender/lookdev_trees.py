"""Tree look-dev: all species prototypes under world-like lighting (instanced exactly like the world).
blender -b --python tools/blender/lookdev_trees.py -- --out exports/a3/lookdev [--samples 16] [--only A,B] [--cams line,close,forest]"""
import bpy, sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import lib_trees as LT
ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
def ARG(k, d=None):
    return ARGS[ARGS.index(k) + 1] if k in ARGS else d
ROOT = LT.ROOT
outdir = os.path.join(ROOT, ARG('--out', 'exports/a3/lookdev'))
os.makedirs(outdir, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
coll = sc.collection
pc = bpy.data.collections.new('VEG_prototypes'); coll.children.link(pc)
protos = LT.build_prototypes(pc)
pc.hide_render = True
rng = np.random.default_rng(3)
# lineup (row y=0) + a forest patch (y 60..160)
pts, sp = [], []
x = 0
for i, p in enumerate(protos):
    pts.append((x, 0, 0)); sp.append(i); x += 14
n_line = len(pts)
fam = LT.FAMILY_OF
forest = [LT.SPECIES_ID[n] for n in ('WhiteOak_A', 'WhiteOak_B', 'RedMaple_A', 'RedMaple_B', 'TulipPoplar_A', 'TulipPoplar_B', 'Hickory_A', 'WhitePine_A', 'Hemlock_A', 'Hemlock_B')]
under = [LT.SPECIES_ID[n] for n in ('Dogwood', 'Sapling_HW_A', 'Sapling_HW_B', 'Rhododendron_A', 'Laurel_B', 'Brush_A', 'Sapling_Pine')]
for k in range(700):
    px, py = rng.uniform(0, 300), rng.uniform(80, 240)
    pts.append((px, py, 0)); sp.append(int(rng.choice(forest)))
for k in range(500):
    px, py = rng.uniform(0, 300), rng.uniform(75, 240)
    pts.append((px, py, 0)); sp.append(int(rng.choice(under)))
sp = np.array(sp)
scale = np.r_[np.ones(n_line), rng.uniform(0.75, 1.2, len(sp) - n_line)]
tint = LT.instance_tints(sp, rng)
LT.instancer('VEG_points', coll, np.array(pts), sp, scale, rng.random(len(sp)) * 6.28, tint, pc)
bpy.ops.mesh.primitive_plane_add(size=1200, location=(150, 100, 0))
g = bpy.context.active_object
gm = bpy.data.materials.new('ground'); gm.use_nodes = True
next(n for n in gm.node_tree.nodes if n.type == 'BSDF_PRINCIPLED').inputs['Base Color'].default_value = (0.09, 0.08, 0.03, 1)
g.data.materials.append(gm)
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
env = w.node_tree.nodes.new('ShaderNodeTexEnvironment')
env.image = bpy.data.images.load(os.path.join(ROOT, 'assets/external/polyhaven/kloppenheim_06_puresky/kloppenheim_06_puresky_4k.hdr'))
bg = next(n for n in w.node_tree.nodes if n.type == 'BACKGROUND')
w.node_tree.links.new(env.outputs[0], bg.inputs[0]); bg.inputs[1].default_value = 0.6
sun = bpy.data.lights.new('SUN', 'SUN'); sun.energy = 10; sun.color = (1, .78, .55); sun.angle = math.radians(1.5)
so = bpy.data.objects.new('SUN', sun); coll.objects.link(so)
from mathutils import Vector as _V
_az, _el = map(float, ARG('--sun', '255,18').split(','))
_th = math.radians(90 - _az)
_d = _V((math.cos(_th) * math.cos(math.radians(_el)), math.sin(_th) * math.cos(math.radians(_el)), math.sin(math.radians(_el))))
so.rotation_euler = (-_d).to_track_quat('-Z', 'Y').to_euler()


def cam(name, loc, look, lens=35):
    from mathutils import Vector
    cd = bpy.data.cameras.new(name); cd.lens = lens; cd.clip_end = 5000
    co = bpy.data.objects.new(name, cd); coll.objects.link(co)
    co.location = loc
    co.rotation_euler = (Vector(look) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    return co


cams = {
    'line1': cam('line1', (49, -62, 9), (49, 0, 9), 30),
    'line2': cam('line2', (161, -62, 9), (161, 0, 9), 30),
    'line3': cam('line3', (259, -62, 9), (259, 0, 9), 30),
    'close': cam('close', (20, -24, 1.7), (14, 0, 8), 30),
    'forest': cam('forest', (150, 20, 1.7), (150, 100, 6), 30),
    'aerial': cam('aerial', (150, -60, 110), (150, 140, 0), 30),
}
sc.render.engine = 'CYCLES'; sc.cycles.samples = int(ARG('--samples', 16)); sc.cycles.use_denoising = True
sc.cycles.max_bounces = 4; sc.cycles.transparent_max_bounces = 16
sc.render.resolution_x, sc.render.resolution_y = 960, 540
sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Punchy'; sc.view_settings.exposure = 0.7
sc.render.image_settings.file_format = 'JPEG'
for c in ARG('--cams', 'line1,line2,line3,close,forest').split(','):
    sc.camera = cams[c]
    sc.render.filepath = os.path.join(outdir, f'{c}.jpg')
    bpy.ops.render.render(write_still=True)
if ARG('--save'):
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ROOT, ARG('--save')))
