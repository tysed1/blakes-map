# MISSION (shared)
You are one of three specialist agents refining "Blake's Map", a stylized-realism open-world game world (north Georgia mountains, autumn 1974, golden hour) rendered in Blender and in a real-time three.js viewer (src/). Goal of this pass: make it look DAMN good (AAA open-world art direction and attention to detail: Red Dead Redemption 2 / Forza Horizon, not photorealism) AND make it run like a lightweight game: 60 fps on a base M2 MacBook Air at High, Low runs anywhere. QUALITY FIRST; performance is the hard constraint: when they conflict, find the technique that gets both (bake, impostor, LOD, atlas, cheaper shader with the same look). Out of scope: buildings, vehicles, people. This container has NO real GPU (SwiftShader): frame times are meaningless, use the proxy metrics. Work item by item from the board, highest impact first; small verified commits.

# YOUR ROLE: Agent E, NATURE (terrain shape and surface, rock/cliff/scree geometry, vegetation scatter and species, ground cover and props, backdrop ridges)
Blender MCP: `blender3` (port 9879). QA build dir /tmp/qa_e, preview port 4175, shots under exports/refine/E/.
You OWN (only edit these): data/manual/terrain.json, data/manual/ridges.json, tools/pipeline/terrain.py, landuse.py, terrain_albedo.py, backdrop.py, vegetation.py, web_albedo.py, web_terrain_detail.py, web_tree_textures.py, web_groundcover_*.py, tools/qa/validate_terrain.py, tools/blender/lib_trees.py, lib_groundcover.py, lib_props.py, bake_web_albedo.py, export_web_trees.py, export_web_groundcover.py, src/components/world3d/terrain.ts, terrainMaterial.ts, trees.ts, groundcover.ts, backdrop.ts EXCEPT the buildSky function (R owns the sky shader). New rock-kit files you create are yours (e.g. tools/blender/lib_rocks.py, tools/blender/export_web_rocks.py, src/components/world3d/rocks.ts); the lead wires new modules into World3D.ts (R owns it): post a REQUEST with the exact 3-5 line hook you need.
Your board items: E1, E2 (sprint 1), then E3, E4 (sprint 2), and R4 together with R (impostors use your species; clipmap terrain touches your terrain.ts/terrainMaterial.ts: agree the interface with R on the board first).
Quality bar:
- Terrain: Blue Ridge landforms at every scale (spurs, hollows, coves, benches), real crags + cliff bands as ROCK MESHES (instanced rock kit + cliff faces + scree fans, triplanar), talus, gullies, road cuts matching N's walls (data/roads/walls.geojson, 66 walls). No stretched textures, no visible tiling, no LOD pops/cracks, no seam with the backdrop. Keep validate_terrain.py errors at 0 and don't increase warnings.
- Forest composition like a painter: species by elevation/aspect/moisture (pine + oak dry ridges, hemlock + rhododendron coves, tulip poplar + maple moist slopes, sycamore/river birch along streams: add those two species to lib_trees if needed), autumn colour rising with elevation and on south slopes, coves greener; stands and gaps, forest-edge shrub/sapling skirts, pasture specimen trees, fence rows, clearings, grassy bald on Bald Ridge (px 820,290), snags and fallen logs in old stands; respect clearances (no trees on pavement, shoulders, clear zones, interchange infields, ballast, water; N will publish clear-zone polygons).
- Trees: invisible LOD transitions (dithered cross-fade; the tree system is src/components/world3d/trees.ts, 27 species x 5 LODs exported by tools/blender/export_web_trees.py), far canopy as dense/textured as the Blender renders, leafy masses (no balloons, no sparkle), backlit glow, wind.
- Ground: fields with direction (hay rows, contour furrows, pasture wear paths), wildflower drifts, forest floor litter/ferns/rocks, riverbank rushes; grass never through props/roads; fix flat-disc flower heads; better rocks up close.
- Backdrop: >= 3 readable blue ridge layers with forest texture, varied skyline, no edges/joins/repeats, holds up at 1.5 km altitude in every direction.
- Performance for your modules: trees <= ~150 draw calls main at High (impostors with R), ground cover <= 40 calls, no per-frame allocations in trees.ts/groundcover.ts updates, alpha-test overdraw minimized (opaque cores + alpha cards only near), grass radius/density per preset.
Golden shots most relevant: CAM_Ref_Match, CAM_HollowRidge_Valley, TC_aerial_100, TC_high_1000, TC_field_ped, TC_forest_ped, GS_ForestEdge, GS_BaldRidge, GS_SouthForkGorge, GS_RockCut, CAM_US19_LaurelGap.

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

# THE BOARD (current)
# Refinement board (ranked; impact 1-5, effort 1-5)

Owners: N = network (roads/rail/water), E = nature (terrain/vegetation/backdrop), R = render + performance, L = lead.
Mark progress by appending `DONE <commit> <note>` under an item. Re-read before editing.

## Sprint 1: foundations

### R1. Performance architecture: baked sun shadow + AO, near-only CSM (R) impact 5, effort 4
Static sun: bake world-space shadow/AO map (terrain + canopy + large structures) once; sample in terrain/ground/road/water shaders; CSM High = 2 cascades <= 300 m, Medium = 1, Low = 0; only LOD0-1 trees/props/structures cast. Accept: shadow calls p50 <= 150 (High), no visible change in far shadows vs baseline shots (canopy shadow pattern still there), near shadows crisp.

### R2. Fused post + bloom at quarter res + zero per-frame allocations (R) impact 4, effort 2
One uber pass (shafts + grade + vignette + sharpen), bloom quarter res; no `new Vector3/Matrix4` per frame in World3D/trees/infra/groundcover update paths (coordinate with N/E for their modules: R posts a patch request or the owners fix). Accept: post passes 3 -> <= 2 full-screen, identical look in golden shots, cull ms p50 unchanged or lower.

### R3. Texture compression + download diet (R, with E/N for their assets) impact 5, effort 3
KTX2 (ETC1S albedo / UASTC normal+detail) with transcoder; atlas card/detail textures; stop uploading 6000 px JPEG as RGBA; meshopt/quantize geometry buffers (infra is 41 MB raw, ground cover raster 21 MB raw); drop eco/trees.bin from deploys; stream near-first. Accept: download <= 40 MB (baseline ~69 MB packed), GPU texture uploads -50 %, no visible quality loss.

### N1. Network logic: zero validator warnings + known offenders (N) impact 4, effort 3
RU_RD_0421 gravel road 68.9 % grade (limit 15 %): reroute with switchbacks or re-grade; RU_RD_0606 18.5 m cut: reroute/bench; 21 residential radius warnings + 2 short edges (LC_RD_2609/2679/2809/2773/2740/2169/2705/2229/..., LC_RD_2692/2716): fix geometry; driveways > 18 % (RU_RD_0678, 0599, 0043, 0240, 0652) and dirt RU_RD_0627 18 %: regrade. Accept: validate_roads.py 0 errors 0 warnings, grade report clean.

### N2. Sign and furniture logic, 1974 MUTCD (N) impact 4, effort 2
686 stop signs -> only minor approaches at true junctions (target 150-300); add yield where right, route shields (US 19, US 76, US 129, SR 9, SR 60, SR 400) at junctions/reassurance points (currently 10 total), curve warnings where needed, junction/turn signs on highways, cobra-head street lights in town centres (Hollow Ridge Main St, Laurel City and Tannersville downtowns), rail crossings with crossbucks + flashers/gates on the mainline at busy roads. Accept: counts in export stats, eye-level golden shots show sensible furniture.

### N3. SR 9 Y-merge flyover + interchange polish (N) impact 4, effort 4
Directional ramp/flyover with >= 4.9 m clearance at the SR 9 Y-merge; interchanges (SR 400/US 19, US 76/SR 60) with parallel accel/decel lanes, tapers, gore chevrons, grassy infields with NO trees (coordinate with E: exclusion mask from N's clear-zone polygons into vegetation.py). Accept: CAM_Interchange_SR400 and GS_US76_SR60 read as engineered interchanges; no trees in infields.

### E1. Forest composition + LOD cross-fades (E) impact 5, effort 3
Species zonation by elevation/aspect/moisture; autumn colour increasing with elevation and on south slopes (coves stay green); stands and gaps; forest-edge shrub/sapling skirts; specimen trees in pastures; fence-row lines; Bald Ridge grassy bald; clear zones from N respected. Dithered cross-fade between tree LODs (no pops); far canopy density at least as full as the Blender renders. Accept: golden aerials (TC_aerial_100, TC_high_1000, CAM_Ref_Match, CAM_HollowRidge_Valley) read as composed forests; no visible LOD pop in a 10-frame fly sequence.

### E2. Rock: real crags, cliff bands, scree, road cuts (E, cuts with N) impact 4, effort 4
Instanced rock-kit + cliff-face meshes on high rock-exposure/steep terrain (Stony Knob, Bald Ridge flanks, South Fork Gorge), scree fans, triplanar rock shading; rock-cut faces on N's walls (66 walls, e.g. LC_RD_0047 9.8 m rock cut). Accept: GS_SouthForkGorge, GS_RockCut and the crags in CAM_Ref_Match look like rock formations, not coloured terrain.

## Sprint 2: the look

### N4. Water: shallows, banks, rapids, waterfalls (N) impact 5, effort 4
Softer, darker shallows (fix too-bright sky reflection), depth colour and flow, foam where the river drops or hits rocks, clean bed/shore (no stair-steps), gravel bars, riffles; Hollow Falls and Falls Creek Cascades as showpieces (rock ledges, layered sheets, mist, plunge pool, spray). Accept: TC_river_low, CAM_LaurelRiver_Bridges, GS_HollowFalls, GS_FallsCreekCascades.

### N5. Surfaces + verges + rail (N) impact 4, effort 3
Asphalt ageing by class, tar snakes/patches/oil lines, crisp markings at distance (no shimmer), soft shoulders + gravel strips + grass verges with their own material, ditches; riprap toned down; rails continuous to the horizon (far LOD). Accept: TC_road_driver, TC_hwy_low, GS_RailCrossing, aerials.

### E3. Fields, ground cover, forest floor (E) impact 4, effort 3
Fields with direction (hay rows, contour furrows, pasture wear paths), wildflower drifts, forest floor litter/ferns/rocks, riverbank rushes; no grass through props/roads; fix flat-disc flower heads; smoother rock props close up. Accept: TC_field_ped, GS_ForestEdge, TC_forest_ped.

### E4. Backdrop: 3+ ridge layers, varied skyline (E) impact 4, effort 2
Accept: TC_high_1000, CAM_Ref_Match, GS_BaldRidge horizons show layered blue ridges, no edges/joins/repeats.

### R4. Far-forest impostors + clipmap terrain (R with E) impact 5, effort 5
Octahedral impostors per species replacing LOD3/4; CDLOD/clipmap terrain with vertex-texture height + geomorph. Accept: main tris p50 -40 %, calls -30 %, no visible quality loss in aerials.

### R5. Atmosphere: cloud shadows, valley mist, sky, grade (R) impact 4, effort 2
Moving cloud shadows (cheap projected noise), valley mist pockets (height fog modulation), sky/cloud polish, golden-hour key/fill match to graphics ref.png. Accept: side-by-side with graphics ref.png: layered depth, warm key/cool fill, nothing blown out or muddy.

## Sprint 3: polish + budgets (all)
Worst remaining golden shots, regressions, budgets on every preset, auto preset pick, Safari/Firefox check (lead).
