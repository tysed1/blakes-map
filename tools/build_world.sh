#!/usr/bin/env bash
# Rebuild the whole world dataset from the source map + hand-authored inputs (data/manual).
# Order matters: water -> roads (auto graph, grids, authored, engineering) -> rail -> terrain -> grading -> buildings -> QA.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 tools/pipeline/water.py
[ -f tools/.cache/roads_auto.json ] || python3 tools/pipeline/roads_auto.py
python3 tools/pipeline/grids.py
python3 tools/pipeline/roads.py
python3 tools/pipeline/railways.py
python3 tools/pipeline/terrain.py
python3 tools/pipeline/grading.py
python3 tools/qa/validate_terrain.py
# buildings: deferred (tools/pipeline/buildings.py is a draft, not part of the build yet)
python3 tools/pipeline/landuse.py
python3 tools/pipeline/terrain_albedo.py
python3 tools/pipeline/backdrop.py
python3 tools/qa/validate_roads.py
[ -f tools/qa/validate_world.py ] && python3 tools/qa/validate_world.py
[ -f tools/pipeline/export_web.py ] && python3 tools/pipeline/export_web.py
echo "world rebuilt"
