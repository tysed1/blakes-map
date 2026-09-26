# Blender tooling

Blender is used as an optional authoring/export backend for the world
(terrain, road meshes, water, building proxies → `.blend` / `.glb`).
Two ways to drive it:

## 1. Headless scripts (no MCP needed)

```bash
blender -b --python tools/blender/<script>.py -- <args>
```

Background mode (`-b`) runs `bpy` scripts and exits. This is what the export
pipeline uses; it works in CI and in cloud sessions.

## 2. Blender MCP bridge (interactive, for Claude Code)

[`blender-mcp`](https://github.com/ahujasid/blender-mcp) exposes a running
Blender to Claude Code as MCP tools (`get_scene_info`, `execute_blender_code`,
`get_viewport_screenshot`, `export_scene`, …).

```
Claude Code ──stdio──> uvx blender-mcp ──TCP :9876──> Blender + "MCP for Blender" add-on
```

Setup (once per machine / container):

```bash
tools/blender/setup.sh        # Blender 4.2 LTS, Xvfb, blender-mcp (uv tool), matching add-on
```

Each session:

```bash
tools/blender/start_mcp.sh [file.blend]   # starts Blender (on Xvfb if no $DISPLAY) with the add-on server
python3 tools/blender/mcp_ping.py         # sanity check: prints scene info
tools/blender/stop_mcp.sh
```

The project `.mcp.json` registers the `blender` MCP server (`uvx blender-mcp`,
telemetry disabled). Claude Code loads MCP servers at session start, so start
Blender first, then start/restart Claude Code in this repo and approve the
project server when prompted.

Notes

* The add-on's socket server needs Blender's event loop, so it cannot run in
  `blender -b`; `start_mcp.sh` uses a virtual display (`xvfb-run`) instead.
* The add-on installed by `setup.sh` is the copy bundled inside the
  `blender-mcp` package, so the add-on/server protocol versions always match.
* Only one Blender instance can own port 9876; set `BLENDER_PORT` to run more.
