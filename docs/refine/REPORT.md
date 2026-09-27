# World refinement pass: report

Test build for the user. Lead plus three agents: N (roads, rail, water), E (nature), R (render and performance).
All work is on branch `claude/clever-turing-m6936x`. Per-item history is in `docs/refine/BOARD.md`.

## Deliverables
- Artifact (packed build, 41.7 MB): https://claude.ai/artifact/NDcXgk1PNfrSQEY6fBoeup (version 4).
- Local package: `blakes-map-world_part1.zip` + `_part2.zip`.
  - Unzip both into the same folder.
  - Run `python3 -m http.server 8080` inside `blakes-map-world/`.
  - Open http://localhost:8080/.
- Golden shots, baseline vs final (20 shots, 960x540): `exports/refine/final.jpg` and `exports/refine/final/`.
- Perf data: `docs/refine/final_perf.json`. The baseline is in `docs/refine/baseline/`.

## Checks (final tree)
- `tsc --noEmit`: clean.
- `validate_roads`: `{}`, 1 component. Grade report: 0 issues.
- `validate_terrain`: 0 errors. Warnings match the baseline.
- `vegetation.py`, `rocks.py` and `sun_shadow_bake.py` were rerun on the final roads and terrain. Their outputs came out byte-identical, so the committed data is consistent.
- The packed local package was rendered and compared with the QA build: CAM_Ref_Match mean difference is about 1/255.

## Performance
These are proxy numbers from SwiftShader, 12 samples on a fixed fly path. Frame times from SwiftShader are not meaningful.

| High preset | Baseline | Final |
|---|---|---|
| Main draw calls p50 | 277 | 177 |
| Shadow draw calls p50 | 680 | 34 |
| Main triangles p50 / max | 13.3 M / 20.3 M | 2.47 M / 3.71 M |
| All-pass triangles max | 87 M | 4.6 M |
| JS cull per frame p50 / max | 8 ms / 100+ ms spikes | 1.3 / 3.7 ms |
| Download, deployed | ~69 MB | 41.7 MB |
| GPU texture uploads | 680 MB | ~400 MB |

| Preset | Main triangles p50 | Main draw calls p50 | Shadow draw calls p50 |
|---|---|---|---|
| Low | 1.98 M | 159 | 0 |
| Medium | 2.30 M | 173 | 14 |
| Ultra | 3.67 M | 177 | 109 |

- First launch auto-picks Low, Medium or High from a 2 s benchmark.
- Expected result on an M2: heavy views are several times faster than the baseline.

## What changed
- **N (roads, rail, water)**
  - Validators went to zero warnings: 68.9 % and 18.5 m cut offenders fixed, about 100 streets eased.
  - Signs rationalised: STOP 686 → 179, YIELD 61, route shields 45, warning signs, 124 cobra-head street lights, rail gates and flashers.
  - SR 9 wye flyover.
  - Interchange clear zones, 12 painted gores, auxiliary-lane lines at 13 ramps.
  - 1970s traffic signals: 62 intersections (Laurel City 27, Tannersville 35, Hollow Ridge 0), all span-wire, mast-arm and pedestal types.
  - Water:
    - reflections of the banks;
    - tannin-coloured shallows;
    - Falls Creek Cascades whitewater and mist;
    - Hollow Falls as a free plunge with ribbons, lip rocks and foam.
  - Rail: weathered ballast, and tie stripes at far LOD.
  - Asphalt no longer sparkles.
  - Soft shoulders.
  - Concrete town retaining walls.
  - Dead-end caps are removed where they would float.
- **E (nature)**
  - Species zonation, with sycamore and river birch added.
  - Autumn palette driven by elevation, south slopes and forest edges.
  - Bald Ridge made a grassy bald.
  - Dithered LOD cross-fades.
  - Tree triangles cut 85–97 %.
  - Rock kit: crags, cliff bands, gorge walls, talus, road-cut courses.
  - Waterfall terrain steps.
  - Backdrop cut 1.49 M → 0.49 M triangles, with three far ridge layers.
  - Ground-cover and rock cull budgets.
  - The "it9" look is locked.
- **R (render and performance)**
  - Static sun shadow and AO bake, with short real-time cascades per preset.
  - One fused post pass.
  - Per-frame work is allocation-free and culling is staggered.
  - Packed deploy (`.wasm` containers with lossless filters) and KTX2 textures.
  - Valley mist, cloud shadows, golden-hour key and fill, grade, capped bloom.
  - Quality-switch crash fixed.
  - Auto preset.
  - The far billboard layer is culled per cell (−0.4 M triangles on every preset, no visual change).

## Known limits and next steps
- **Triangles over budget.** High p50 is 2.47 M against the 1.5 M target. What's left is tree LOD3 cores, roads, water, the backdrop, and grass at pedestrian height in forest.
- **Far-tree impostors are opt-in** (`?impostors`). The baked-normal sun term comes out darker than on the real trees, so it failed visual parity. Fixing that is the next big performance step.
- **Deploy size.** The deploy is 41.7 MB, just over the 40 MB target. E's KTX2 detail maps were added after R3.
- **Occasional 10–20 ms cull hitches** on this heavily loaded box. They may not show on a real machine.
- **Visual gaps:**
  - a faint bright line along some distant ridges (also in the baseline);
  - terrain detail looks speckled in sparse fields;
  - LOD dither crosshatch is visible in stills;
  - the stair-stepped shorelines were not done.
- **Out of scope:** buildings, vehicles and people.
