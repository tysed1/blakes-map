"""Bake the Blender terrain look (A3 material system: fields, litter, rock, clay, verges,
riverbed...) into a top-down albedo texture for the web viewer.
Renders the orthographic CAM_Validation_Top with the Diffuse Colour pass (unlit albedo),
so the web renderer can light it itself.
    blender -b exports/web/terrain.blend --python tools/blender/bake_web_albedo.py -- out.png [W]
"""
import bpy, sys, os
a = sys.argv[sys.argv.index('--') + 1:]
out = a[0]; W = int(a[1]) if len(a) > 1 else 6000
sc = bpy.context.scene
keep_prefix = ('TERRAIN_', 'WATER_Bed', 'WATER_Shore', 'WATER_Boulders', 'CAM_Validation_Top')
for o in bpy.data.objects:
    if o.type == 'MESH' and not o.name.startswith(keep_prefix):
        o.hide_render = True
    if o.type == 'MESH' and o.name.startswith('TERRAIN_'):
        for m in o.modifiers:
            if m.type == 'NODES':
                m.show_render = False   # no ground-cover instances in the bake
sc.camera = bpy.data.objects['CAM_Validation_Top']
sc.render.engine = 'CYCLES'
sc.cycles.samples = 4
sc.cycles.use_denoising = False
sc.render.resolution_x, sc.render.resolution_y = W, int(round(W * 667 / 2000))
sc.render.resolution_percentage = 100
sc.render.film_transparent = False
vl = sc.view_layers[0]
vl.use_pass_diffuse_color = True
sc.use_nodes = True
nt = sc.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
rl = nt.nodes.new('CompositorNodeRLayers')
comp = nt.nodes.new('CompositorNodeComposite')
nt.links.new(rl.outputs['DiffCol'], comp.inputs['Image'])
sc.view_settings.view_transform = 'Standard'
sc.view_settings.look = 'None'
sc.view_settings.exposure = 0
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_mode = 'RGB'
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
print('baked', out)
