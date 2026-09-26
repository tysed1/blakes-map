"""Tree look-dev: lineup of all species prototypes under the world lighting.
blender -b --python tools/blender/lookdev_trees.py -- out.jpg"""
import bpy, sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import lib_trees as LT
ROOT = LT.ROOT
out = sys.argv[sys.argv.index('--') + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
coll = sc.collection
protos = LT.build_prototypes(coll)
tints = {'A': (0.09, 0.13, 0.035), 'B': (0.3, 0.08, 0.02), 'C': (0.16, 0.14, 0.03), 'D': (0.1, 0.12, 0.04), 'E': (0.07, 0.1, 0.05), 'F': (0.06, 0.09, 0.05)}
x = 0
for ob in protos:
    ob.location = (x, 0, 0)
    ob['tint'] = tints[ob.name[0]]
    x += 13
# instancer attribute workaround for single objects: use object attribute via custom prop -> use 'OBJECT' attribute type
for m in bpy.data.materials:
    if m.name.startswith('MAT_Foliage'):
        for n in m.node_tree.nodes:
            if n.type == 'ATTRIBUTE':
                n.attribute_type = 'OBJECT'
bpy.ops.mesh.primitive_plane_add(size=400, location=(60, 0, 0))
g = bpy.context.active_object
gm = bpy.data.materials.new('ground'); gm.use_nodes = True
next(n for n in gm.node_tree.nodes if n.type == 'BSDF_PRINCIPLED').inputs['Base Color'].default_value = (0.05, 0.07, 0.02, 1)
g.data.materials.append(gm)
w = bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
env = w.node_tree.nodes.new('ShaderNodeTexEnvironment')
env.image = bpy.data.images.load(os.path.join(ROOT, 'assets/external/polyhaven/kloppenheim_06_puresky/kloppenheim_06_puresky_4k.hdr'))
bg = next(n for n in w.node_tree.nodes if n.type == 'BACKGROUND')
w.node_tree.links.new(env.outputs[0], bg.inputs[0]); bg.inputs[1].default_value = 0.5
sun = bpy.data.lights.new('sun', 'SUN'); sun.energy = 3; sun.color = (1, .82, .62); sun.angle = 0.02
so = bpy.data.objects.new('sun', sun); coll.objects.link(so); so.rotation_euler = (math.radians(70), 0, math.radians(-60))
cd = bpy.data.cameras.new('c'); cd.lens = 32
co = bpy.data.objects.new('c', cd); coll.objects.link(co)
co.location = (65, -95, 12); co.rotation_euler = (math.radians(85), 0, 0); cd.lens = 28
sc.camera = co
sc.render.engine = 'CYCLES'; sc.cycles.samples = 32; sc.cycles.use_denoising = True
sc.cycles.transparent_max_bounces = 16
sc.render.resolution_x, sc.render.resolution_y = 1600, 600
sc.view_settings.view_transform = 'AgX'
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
