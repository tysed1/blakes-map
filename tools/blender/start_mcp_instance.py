"""Blender startup script: find + enable the MCP add-on and start its socket server on a port.

Runs INSIDE Blender (UI mode; the add-on's server needs the event loop, so not with -b):
    blender --python tools/blender/start_mcp_instance.py -- --port 9878
(or set BLENDER_PORT). Works with both add-on generations:
  * newer bundled add-on (module usually `blender_mcp`): scene.blendermcp_auto_start_server
  * older GitHub addon.py (module often `addon`): operator bpy.ops.blendermcp.start_server()
Port 9877 is reserved on this project's machines and is refused.
Normally launched by tools/blender/launch_mcp_blender.py.
"""
import os
import sys
import bpy
import addon_utils

FORBIDDEN = {9877}
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
port = int(argv[argv.index('--port') + 1]) if '--port' in argv else int(os.environ.get('BLENDER_PORT', '9876'))
if port in FORBIDDEN:
    raise SystemExit(f'[blakes-map] port {port} is reserved; pick another (e.g. 9878, 9879)')

# find the MCP add-on whatever its module name is
mod_name = None
for m in addon_utils.modules():
    info = getattr(m, 'bl_info', {}) or {}
    name = str(info.get('name', '')).lower()
    if 'mcp' in name or 'mcp' in m.__name__.lower():
        mod_name = m.__name__
        break
if mod_name is None:
    raise SystemExit('[blakes-map] MCP add-on not installed in this Blender '
                     '(Edit > Preferences > Add-ons > Install addon.py from github.com/ahujasid/blender-mcp)')
addon_utils.enable(mod_name, default_set=True, persistent=True)
print(f'[blakes-map] MCP add-on "{mod_name}" enabled; target port {port}')


def _start():
    sc = bpy.context.scene
    if sc is None:
        return 0.5
    if hasattr(sc, 'blendermcp_port'):
        sc.blendermcp_port = port
    if hasattr(sc, 'blendermcp_auto_start_server'):
        sc.blendermcp_auto_start_server = True   # the add-on's own timer starts it
    srv = getattr(bpy.types, 'blendermcp_server', None)
    if srv is not None and getattr(srv, 'running', False):
        print(f'[blakes-map] MCP server running on localhost:{getattr(srv, "port", port)}')
        return None
    try:
        bpy.ops.blendermcp.start_server()
    except Exception as e:  # operator missing or context not ready yet: retry
        print('[blakes-map] start_server retry:', e)
        return 1.0
    return 1.0   # re-check until running


for s in bpy.data.scenes:
    if hasattr(s, 'blendermcp_port'):
        s.blendermcp_port = port
bpy.app.timers.register(_start, first_interval=1.0, persistent=True)
