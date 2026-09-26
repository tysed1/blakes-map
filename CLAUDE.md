# Blake's Map: world-planning foundation

Read `docs/WORLD_DESIGN_PRINCIPLES.md` first. It is the authoritative priority order:
function and believability beat pixel-faithful tracing of the base map.

* Canonical transform: `data/world/world.json`. Source-image px -> world metres:
  X = (px-1000)*2.5, Z = (py-333.5)*2.5, Y = elevation. All data is stored in
  source px. Python mirror: `tools/lib/coords.py`.
* Hand-authored inputs: `data/manual/` (the editable source of truth).
  The pipeline (`tools/pipeline/*.py`) regenerates `data/{roads,water,buildings,terrain,...}`.
* Art-direction target for the 3D world (Blender phase only): `graphics ref.png`.
* Blender: `tools/blender/` (headless scripts + MCP bridge setup).
* The source map `assets/maps/source/base_map.webp` must never be modified.
