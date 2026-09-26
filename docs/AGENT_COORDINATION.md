# Overnight environment pass: agent coordination

Primary goal: **make the world itself beautiful** (terrain, water, roads, infrastructure,
vegetation, materials, atmosphere). Buildings are out of scope for this pass.
The art target is `graphics ref.png`; the geography source is `assets/maps/source/base_map.webp`.
The priorities are in `docs/WORLD_DESIGN_PRINCIPLES.md`.

## Shared invariants (nobody changes these)
* Coordinate transform `data/world/world.json` (px -> metres, 2.5 m/px, Blender bx=(px-1000)*2.5, by=-(py-333.5)*2.5).
* The overall identity of the map: town, city and river placement, major road flow.
* Pipeline order in `tools/build_world.sh`: water -> roads -> rail -> terrain -> grading -> landuse -> albedo -> backdrop -> QA -> export_web.
  Blender scene: `tools/blender/build_world.py` (reads the data above). The coordinator owns the orchestration code in this file.

## Dependency order
Terrain/Water establishes the landforms -> Roads adapt to them -> Environment Art dresses the result.
Agents may work concurrently on safe tasks; changes to shared outputs go through the files below.

## File ownership
| Agent | Owns (may edit) |
|---|---|
| **A1 Terrain & Water** | `tools/pipeline/terrain.py`, `tools/pipeline/water.py`, `tools/pipeline/backdrop.py`, `data/manual/terrain.json`, `data/manual/waterways.json`, new `tools/blender/lib_water.py`, `tools/qa/validate_terrain.py` |
| **A2 Roads & Infrastructure** | `tools/pipeline/roads*.py`, `grids.py`, `railways.py`, `grading.py`, `tools/lib/engineer.py`, `tools/lib/interchange.py`, `data/manual/roads/*`, `data/manual/interchanges.json`, `data/manual/road_edits.json`, `data/roads/road_types.json`, `tools/qa/validate_roads.py`, new `tools/blender/lib_roads.py`, `tools/blender/lib_infrastructure.py` (poles, lines, guardrails, signs, culverts, fences, walls, crossings) |
| **A3 Vegetation, Materials & Env Art** | `tools/blender/lib_trees.py`, `lib_materials.py`, `lib_groundcover.py`, new `lib_props.py` (logs, debris), `tools/assets/*`, `assets/foliage/*`, `tools/pipeline/landuse.py`, `tools/pipeline/terrain_albedo.py`, tree scatter in `tools/pipeline/export_web.py` (`scatter_trees` only) |
| **Coordinator** | `tools/blender/build_world.py` (orchestration only), `tools/blender/render.py`, `tools/qa/compare_ref.py`, `tools/build_world.sh`, docs |

Hooks into `build_world.py`: each agent exposes `build(root_collection, ctx)` style functions in its
own `lib_*.py`. A minimal call line in `build_world.py` is allowed. Keep the edit small and
mention it in the final report.

## Working rules
* Blender builds and test renders: keep them small (<= 960x540, <= 24 samples) because 4 CPU cores are shared.
  Write test builds and renders to your own folder: `exports/<agent>/...` (never overwrite `exports/blender/world.blend`).
* Commit only your own files: `git add <your paths>`, then commit with a clear message. If `index.lock` exists, wait and retry.
  Push with `git push -u origin claude/clever-turing-m6936x` and pull/rebase on conflicts. Never force-push.
* Validate at driver eye (1.2 m), pedestrian (1.7 m), low cinematic (4-8 m), medium aerial (60-150 m) and high map (1000+ m).
* Buildings: do not work on them.
