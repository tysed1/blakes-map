"""Dump the Blender scene + A3 validation cameras for the web viewer (public/world/cams.json).

    blender -b exports/web/terrain.blend --python tools/blender/dump_cams.py

Web (three.js, Y-up): position (x, z, -y), look direction = camera -Z axis, vertical FOV from
the 36 mm sensor (horizontal fit) at 16:9.
"""
import bpy, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [a for a in sys.argv if a != '--']
import a3_testcams  # noqa: F401  (creates the TC_* cameras)
from mathutils import Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
bpy.context.view_layer.update()
out = {}
for ob in bpy.data.objects:
    if ob.type != 'CAMERA' or ob.name == 'CAM_Validation_Top':
        continue
    m = ob.matrix_world
    p = m.translation; d = (m.to_3x3() @ Vector((0, 0, -1))).normalized()
    fov_h = 2 * math.atan(18 / ob.data.lens)
    out[ob.name] = dict(pos=[p.x, p.z, -p.y], dir=[d.x, d.z, -d.y], fov=math.degrees(2 * math.atan(math.tan(fov_h / 2) / (16 / 9))))
json.dump(out, open(os.path.join(ROOT, 'public/world/cams.json'), 'w'), indent=1)
print('cams:', ', '.join(out))
