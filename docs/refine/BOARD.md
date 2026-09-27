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
