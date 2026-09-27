# Shared rules for every refinement agent (copied into each brief)

## Project facts
- Repo root: /home/user/blakes-map (git, branch claude/clever-turing-m6936x). Read CLAUDE.md and
  docs/WORLD_DESIGN_PRINCIPLES.md first. Art target: `graphics ref.png`. Setting: north Georgia, autumn 1974, golden hour.
- Canonical transform: px -> metres X=(px-1000)*2.5, Z=(py-333.5)*2.5, Y=elevation (three.js). Blender Z-up, by=-Z.
- Source of truth: data/manual/* + pipeline code. Never hand-edit generated outputs; never modify assets/maps/source/base_map.webp.
- OUT OF SCOPE: buildings, vehicles, people.

## Blender MCP (live Blender, use it for look-dev and inspection)
- Four Blender 4.2 instances run with the MCP add-on: lead `blender` (9876), N `blender2` (9878), E `blender3` (9879), R `blender4` (9880).
  Use ONLY your own server's tools (`mcp__blenderX__execute_blender_code`, `get_viewport_screenshot`, `get_scene_info`, ...).
- The first call on a server can time out while it warms up: just retry once.
- If your instance died: `python3 tools/blender/launch_mcp_blender.py --port <yours>` (idempotent). NEVER use port 9877. Never close another instance.
- Heavy scenes: exports/web/terrain.blend (223 MB, terrain + water + roads + infra, no trees) can be opened in your instance
  (`bpy.ops.wm.open_mainfile(filepath=...)`); RAM is ~15 GB shared by 4 Blenders + browsers, so don't open world_rv.blend (250 MB+)
  in more than one instance, and close big files when done (`bpy.ops.wm.read_homefile(use_empty=True)`).
- The committed truth is scripts: anything you prototype through MCP must end up in the repo's Python (tools/blender/*, tools/pipeline/*) and be reproducible headless (`blender -b --python ...`).

## Regeneration commands (rerun only what your change affects)
- Core world data (pipeline): `bash tools/build_world.sh` (slow; prefer single stages: `python3 tools/pipeline/<stage>.py`, then `python3 tools/pipeline/export_web.py` which rewrites public/world core files + the vegetation scatter).
- Infra (roads/bridges/rail/furniture/water) web export: `python3 tools/blender/export_web_infra.py` (textures) then `blender -b --factory-startup --python tools/blender/export_web_infra.py -- --stats` (~1 min).
- Trees: `blender -b --python tools/blender/export_web_trees.py`; textures `python3 tools/pipeline/web_tree_textures.py`.
- Ground cover: `blender -b --python tools/blender/export_web_groundcover.py`; `python3 tools/pipeline/web_groundcover_textures.py`; `python3 tools/pipeline/web_groundcover_raster.py`.
- Terrain albedo: `blender -b --python tools/blender/build_world.py -- --no-trees --no-groundcover --out exports/web/terrain.blend` (~1 min) -> `blender -b exports/web/terrain.blend --python tools/blender/bake_web_albedo.py -- exports/web/albedo_bake.png` (~8 min) -> `python3 tools/pipeline/web_albedo.py exports/web/albedo_bake.png data/terrain/albedo_web.jpg` -> copy to public/world/albedo.jpg. Terrain detail: `python3 tools/pipeline/web_terrain_detail.py`.
- Cameras: `blender -b exports/web/terrain.blend --python tools/blender/dump_cams.py`; golden shots: `python3 tools/qa/make_golden_shots.py`.

## QA loop (every change)
1. Top item you own on docs/refine/BOARD.md (impact first).
2. Smallest change that fixes it.
3. `npx tsc --noEmit`; relevant validators (`python3 tools/qa/validate_roads.py`, `python3 tools/qa/validate_terrain.py`).
4. Static build in YOUR OWN dir + port (N: /tmp/qa_n :4174, E: /tmp/qa_e :4175, R: /tmp/qa_r :4176):
   `npx vite build --outDir /tmp/qa_x --emptyOutDir && (npx vite preview --outDir /tmp/qa_x --port 417x --strictPort &)`
   then `node tools/qa/shots_multi.mjs exports/refine/<you>/<iter> http://localhost:417x/ @golden:SHOT_A,SHOT_B` (960x540 default; batch shots in one run; software GL = ~1-3 min per shot)
   then `python3 tools/qa/contact_sheet.py exports/refine/<you>/<iter>.jpg exports/refine/baseline exports/refine/<you>/<iter>` for before/after, and LOOK at it.
   Perf-sensitive change: `node tools/qa/perf.mjs http://localhost:417x/ --samples 8 --w 640 --h 416 --presets high --out /tmp/perf_x.json`.
5. Self-critique vs graphics ref.png and baseline: clearly better, no new artefacts (pops, seams, shimmer, z-fighting, colour shifts, floating objects)? Otherwise revert/iterate. Never ship a sidegrade.
6. Commit ONLY your files (`git add <paths>`; never `git add -A`), message "<N|E|R>: <what>, <metric delta>" + before/after sheet path in the body; end the message with:
   Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
   Claude-Session: https://claude.ai/code/session_01EXc4vTkCY4CT3DQs2Wg9Bb
   Don't push (the lead pushes). Tick the item on BOARD.md (append a line under the item: "DONE <commit> <note>"), keep edits to BOARD.md minimal and re-read before editing.
7. Report each finished batch to the lead in your final message (items, commits, sheets, metrics, open problems).

## Budgets (High preset, M2 target; in this container they're checked as proxies)
main pass <= 350 draw calls and <= 1.5 M triangles typical aerial (<= 2.5 M worst); shadow passes <= 150 calls; <= 40 shader programs; GPU memory <= 900 MB; download <= 40 MB; JS cull/LOD work <= 4 ms/frame; Low: <= 600 k tris, <= 150 calls, no real-time shadows.

## Style
Stylized realism, AAA open-world art direction (RDR2 / Forza Horizon), not photorealism. 1974-correct details only.
