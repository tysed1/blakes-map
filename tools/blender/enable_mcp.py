"""Blender startup script: enable the MCP add-on so its socket server
auto-starts (default port 9876, override with BLENDER_PORT)."""
import os
import bpy
import addon_utils

addon_utils.enable("blender_mcp", default_set=True, persistent=True)
port = int(os.environ.get("BLENDER_PORT", "9876"))
for scene in bpy.data.scenes:
    scene.blendermcp_port = port
    scene.blendermcp_auto_start_server = True
print(f"[blakes-map] blender_mcp add-on enabled; server will listen on localhost:{port}")
