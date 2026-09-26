# Claude Code prompt: 1974 building kit (local, Blender MCP, lead + 3 agents)

Setup before you paste the prompt:
1. Unzip BlakesMap-part1.zip, BlakesMap-part2.zip (the viewer) and BlakesMap-source.zip into the
   SAME place, so you get one `BlakesMap/` folder containing `app/` plus `tools/`, `data/`,
   `docs/`, `src/`, `CLAUDE.md`, `graphics ref.png`, `.mcp.json`, `reference_renders/`.
2. Save your DALL-E sheet(s) as `BlakesMap/assets/reference/building_kit_ref.png`
   (+ `building_kit_ref_2.png` ... if you made the focused sheets).
3. Install Blender 4.2 LTS, `uv` (https://docs.astral.sh/uv/) and Node 18+.
4. Open Blender 3 times (see Phase 0 of the prompt for ports), then start `claude` inside
   `BlakesMap/`, approve the `blender`, `blender2`, `blender3` MCP servers, pick Opus with
   high effort, and paste everything inside the fence below.

```
# MISSION
You are the ART DIRECTOR and TECHNICAL LEAD for "Blake's Map", a stylized-realism open-world game set in the mountains of north Georgia in autumn 1974. The world already exists and looks great: terrain, rivers, engineered roads and bridges, the Georgia Northern railroad, 215,000 leaf-card trees, grass and wildflowers, and a golden-hour atmosphere, in Blender and in a real-time three.js viewer. BUILDINGS are the missing layer. You will lead a team of up to 3 subagents to create a building kit and landmark set so good that a player flying over Hollow Ridge at sunset would believe it is a shipped AAA open-world game: the craft of Red Dead Redemption 2, with a warmer, cleaner, slightly painterly stylized realism that matches our world exactly.

"Good" is not enough here. Every asset must pass the review gates in this prompt. You have full creative authority inside the style bible you write, and full responsibility for consistency across the whole kit. Quality over quantity: if time runs short, ship fewer, better assets rather than many mediocre ones, and say so.

# 0. GROUND TRUTH: WHAT IS IN THIS FOLDER
You are in the project root `BlakesMap/` (a plain folder, not a git repo yet: run `git init` now and commit after every accepted asset family so progress is never lost).
- `app/`: the compiled three.js game viewer + ALL world data in `app/world/` (terrain_u16.bin, vegetation, roads.json, landmarks.json, settlements.json, regions.json, cams.json, manifest.json, albedo.jpg, sky.jpg, trees/, infra/, groundcover/...). Run it with the included launchers or `python -m http.server 8173` in `app/`. This is the game's look: study it by flying it yourself (screenshots via a headless browser if available, or ask the user).
- Source: `CLAUDE.md`, `docs/WORLD_DESIGN_PRINCIPLES.md`, `tools/` (Blender libs, pipeline, QA, deploy), `data/` (buildings/building_types.json, manual/landmarks.json, manual/zones.json, manual/regions.json, roads/road_types.json + roads.geojson, terrain/height_graded_f32.bin 2000x667 float32 metres, world/world.json), `src/` (viewer source: three.js r169, TypeScript, Vite), `.mcp.json` (three Blender MCP servers on ports 9876/9877/9878).
- Art direction: `graphics ref.png` (THE visual target: golden hour, layered ridges, white clapboard houses, red barns, a brick Main Street, a white steeple, a water tower, a truss bridge). `assets/reference/building_kit_ref*.png` = the DALL-E building kit sheet(s) the user generated for massing, palette and mood (concept art: fix anything impossible, badly proportioned or anachronistic).
- `reference_renders/`: how the world renders today. blender_ref_match/, blender_testcams/, blender_aerial/, blender_roads/ (road/HR_Main_Ped.jpg and road/RR_Depot_Driver.jpg show EXACTLY where Hollow Ridge's Main Street and depot buildings will stand), threejs_viewer/ (the real-time look, e.g. hero.jpg, CAM_Ref_Match.jpg, TC_field_ped.jpg).
- `assets/maps/source/base_map.webp`: the concept map (never modify). Crop it at Hollow Ridge px (930-1220, 200-470), Laurel City (0-420, 150-500), Tannersville (1520-2000, 120-520) and upscale x3 to read roof colours, density, the two domed arenas, towers and marinas.
- Not included: the huge world .blend files. For in-context Blender tests you build a small context scene yourself (Phase 2).
- Coordinates: 1 Blender unit = 1 m, Z-up. Map px -> world: X=(px-1000)*2.5, Blender Y=-(py-333.5)*2.5, Z=elevation (sample data/terrain/height_graded_f32.bin, row-major [667][2000]). 1 map px = 2.5 m. In three.js (Y-up) the same point is (X, Z_elev, (py-333.5)*2.5).
- Reusable project code to read before writing your own: tools/blender/lib_trees.py (procedural asset style, deterministic seeds, custom normals, per-vertex attributes), tools/blender/export_web_trees.py (how we ship assets to three.js: LODs, quantized binary + JSON), tools/blender/lib_materials.py (shared material library pattern `get(name)`), tools/blender/lib_atmosphere.py + render.py (the project's sun/HDRI/exposure rig), tools/blender/lib_infrastructure.py (period-correct roadside props already in the world: utility poles and wires, guardrails, signs, fences: do not duplicate them), tools/assets/fetch_polyhaven.py (CC0 texture fetcher; needs a User-Agent header, already handled). src/components/world3d/trees.ts + infra.ts show how the viewer loads, instances, LODs and shades assets.

# 1. THE TEAM (you + at most 3 subagents)
You are the lead. You do NOT bulk-model; you set the standard, build the shared core, make the exemplar, brief the agents, review everything, integrate and ship. Spawn at most 3 subagents with the Agent tool (general-purpose, run in background), each with a SELF-CONTAINED brief (they cannot see this conversation): write each brief to docs/agents/<agent>.md first, then pass it in full.
- Agent R, "Rural & Residential" -> Blender MCP server `blender` (port 9876) -> owns tools/blender/buildings/rural_residential.py, exports/buildings/review/rural/.
- Agent T, "Town, Main Street & Civic" -> MCP `blender2` (9877) -> owns tools/blender/buildings/town.py, exports/buildings/review/town/.
- Agent C, "City, Industry & Rail" -> MCP `blender3` (9878) -> owns tools/blender/buildings/city_industry.py, exports/buildings/review/city/.
- You own: the style bible, tools/blender/lib_buildings.py (core API, materials, modules, LOD + UV + export helpers), tools/blender/buildings/landmarks.py (hero landmarks: you may delegate individual landmarks to the agent whose family they belong to), tools/blender/build_building_kit.py, tools/blender/validate_kit.py, tools/blender/review_render.py, tools/blender/context_scene.py, tools/blender/export_web_buildings.py, data/buildings/kit_catalog.json, the viewer preview integration.
- Rules for everyone: never edit another owner's files (ask the lead to change the core); one Blender instance per agent (never drive another agent's port); every change reproducible from scripts (MCP is for building, inspecting and screenshots; the committed truth is the Python); report blockers immediately instead of working around the core.
- Agents submit work to you per asset family via a short report + review renders. You reply ACCEPT or REVISE with specific, numbered notes. Keep a log in docs/agents/review_log.md.

# 2. PHASE 0: ENVIRONMENT + MCP (blocking; do not model before this passes)
Detect the OS first.
- Windows/macOS (normal case): install the MCP add-on into Blender 4.2 (Edit > Preferences > Add-ons > Install, using addon.py from https://github.com/ahujasid/blender-mcp, same version as `uvx blender-mcp`), enable it, then launch THREE Blender windows. In each: N-panel > BlenderMCP tab > set Port (9876, 9877, 9878) > "Connect to MCP server"/"Start". Blender executable is usually "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" (Windows) or /Applications/Blender.app/Contents/MacOS/Blender (macOS); headless scripts use it with `-b --python`.
- Linux: tools/blender/setup.sh then `BLENDER_PORT=9876 tools/blender/start_mcp.sh`, `BLENDER_PORT=9877 ...`, `BLENDER_PORT=9878 ...` (xvfb-run when there is no display).
- Verify EACH server from Claude Code: get_scene_info; execute_blender_code that prints bpy.app.version_string (must be 4.2.x) and creates a 1 m cube `MCP_TEST_<port>`; get_viewport_screenshot and confirm the cube is visible; delete it. Also `python tools/blender/mcp_ping.py` with BLENDER_PORT set per port. Check Poly Haven via get_polyhaven_status (optional).
- If a server fails: fix it (port conflict, add-on not started, uv missing, stale Blender) and retry. If you can only get 1 or 2 servers working, run fewer parallel agents (the remaining ones use headless `blender -b` scripts only) and tell the user. Report the final MCP status.
- Also check `npm install` in the root works (for the viewer preview in Phase 5) and that Python 3 has numpy + Pillow outside Blender (for texture processing).

# 3. PHASE 1: RESEARCH + STYLE BIBLE (lead) -> docs/BUILDING_KIT_STYLE.md
Study every reference listed in section 0 (look at the images, don't skim filenames). Then write the bible. It is the law for all agents:
1. The look in one paragraph, and what makes our world's style ours: stylized realism, soft golden light, controlled saturation, clean readable silhouettes, generous bevels that catch the low sun, believable wear, no noise for noise's sake.
2. Era rules for 1974 north Georgia. Allowed: clapboard, board-and-batten, asbestos-cement shingle siding, red and buff brick, painted cinder block, permastone, tin (5V-crimp, corrugated, standing seam), asphalt shingle in greys, greens and browns, wood-sash and steel windows, aluminium storm doors, jalousie windows, window AC units, TV antennas, chain-link, 1950s-70s commercial modernism, mansard fronts, International Style towers. Forbidden: vinyl siding, modern windows, satellite dishes, modern signage or lighting, anything after 1974, real brands or logos.
3. The palette: 20-30 named colours with linear-RGB values, matched to graphics ref.png, the DALL-E sheet and the world grade. Include how saturated each family may be.
4. Material families and the SHARED MATERIAL LIBRARY: MAT_BLD_* names, texel density (~256 px/m trims, 512 px/m hero close-ups), texture sets (albedo sRGB, normal, ORM packed), trim-sheet layouts (window/door/cornice/fascia/flashing strips), tiling sets (brick running + common bond, clapboard 10 cm reveal, board-and-batten, block, shingles, tin, concrete, stucco, glass). Fetch CC0 sources with tools/assets/fetch_polyhaven.py; author the rest procedurally in Blender and BAKE to textures (the web can't run Blender node trees).
5. Proportion rules (storey heights, door/window sizes, sill heights, porch depths, roof pitches, eave overhangs, chimney heights, foundation heights), plus the regional vernacular: I-house, dogtrot, bungalow, mill-village cottage, ranch, split-level, gambrel and gable barns, poultry houses, country store, service station, motor court, main-street commercial block, Baptist church.
6. CRAFT RULES (the difference between "asset" and "exceptional asset"):
   - Silhouette first: every building must be identifiable as a black shape at 300 m (roof form, chimney, porch, steeple, sign pole, water tank).
   - Three-level detail hierarchy: primary masses, secondary (porches, dormers, awnings, cornices, chimneys, steps), tertiary (trim, gutters, downspouts, flashing, vents, meters, AC units, antennas). Tertiary detail comes from trims, normals and small modular meshes, never from dense sculpting.
   - Chamfer/bevel every visible hard edge (1-3 cm small, 3-6 cm structural) so the golden sun catches edges; use weighted normals; no razor-sharp CG edges.
   - Construction logic: everything must look buildable and maintained by 1974 means (rafter tails, gutters that drain to downspouts, flashing where roofs meet walls, foundation vents, piers on slopes, steps to every door, lintels over openings, expansion joints on block).
   - Weathering with a cause: rain streaks under sills and roof edges, splash-back dirt at the base, sun-faded paint on the south/west faces, rust at fasteners and seams on tin, moss and lichen on north-facing roofs and shaded brick, soot above chimneys, worn paint at door handles and steps. Wear intensity comes from building age + owner prosperity.
   - Controlled imperfection: slight roof sag on old barns, shutters a few degrees off, a missing tin panel, an open window; never on hero civic buildings unless the story says so.
   - Storytelling: each building has an owner. The props, repairs, colours and condition tell who (a proud widow's farmhouse, a struggling mill cottage, a busy feed store).
   - Variation without chaos: every generator exposes seeded variation (proportions, colours from the palette, details), but a row of houses must still look like one town.
7. Review rubric (section 6 of this prompt) copied into the bible.
Show the user a short summary of the bible when it is done; then continue without waiting unless they object.

# 4. PHASE 2: CORE TECH + THE EXEMPLAR (lead, before spawning agents)
1. tools/blender/lib_buildings.py, the core everyone builds on:
   - Material library: get_material(name) building MAT_BLD_* from baked textures with shared node groups (wear mask from AO/curvature + world-Z dirt gradient + per-object seed colour variation via object info so one material can colour hundreds of buildings from the palette).
   - Module toolkit: wall runs with openings on a 0.5 m grid, window/door/storefront modules (sash, 2-over-2, 6-over-6, picture window, jalousie, storefront with transom, garage door, church lancet/arched), roofs (gable, hip, gambrel, shed, flat with parapet, mansard, saltbox, pyramid, dome segments) with real overhangs, fascia, soffit and rafter tails, porches (posts, rails, steps, deck, ceiling), chimneys, dormers, cornices, awnings (fixed metal + canvas stripes), gutters/downspouts, foundations (piers, block, slab) extending 1.5 m below grade, stairs, parapets, signage frames (blank), rooftop units.
   - Generator API: build(spec: dict, seed) -> object with LOD0/1/2 + collision proxy, where spec = footprint, storeys, roof, facade materials, trim colour, openings rhythm, porch/awning/chimney/dormer flags, wear, story tags. Deterministic.
   - LOD tools: LOD1 automatic (collapse tertiary modules, keep silhouette, bake small details into normal maps), LOD2 = box massing + roof + baked facade (albedo+normal) in a small per-asset atlas. Bake helpers.
   - UV/texel checks, weighted normals, naming helpers, custom properties (class, catalog id, footprint, height, wear, seed).
2. tools/blender/review_render.py: a standard review rig matching the game: sun azimuth ~255 deg, elevation ~18 deg, warm colour, Nishita/HDRI sky using assets/external/polyhaven/kloppenheim_06_puresky/kloppenheim_06_puresky_2k.hdr at the strength and exposure used in tools/blender/render.py, AgX/Filmic comparable to the viewer's ACES look, soft AO, subtle haze. Shots per asset: 3/4 aerial at 35 degrees (like the DALL-E sheet), front elevation eye-level 1.7 m, a back-3/4, a 300 m "silhouette" squint shot (desaturated, blurred), and a clay (white material) shot. Include a 1.8 m human and a 1974 sedan (5.3 x 2.0 m) for scale on the ground plane. Output to exports/buildings/review/<family>/<asset>_vNN.png plus a contact sheet per family.
3. tools/blender/context_scene.py: builds a context scene of Hollow Ridge from data/terrain/height_graded_f32.bin (a 1.2 x 1 km patch around px 1050,330) textured with app/world/albedo.jpg, the Main Street (US 19) + Depot Street + Georgia Northern alignments drawn from data/roads/roads.geojson and app/world/rail.json as simple ribbons, a sprinkle of simple trees (instances of low-poly crowns in the world palette) and the review lighting, plus the two cameras from app/world/cams.json that frame the valley (CAM_HollowRidge_Valley, CAM_Ref_Match; the JSON is in three.js Y-up: Blender = (x, -z, y)). Used to judge buildings in their real place and light.
4. tools/blender/validate_kit.py: automated checks (scale: door 2.0-2.2 m, storey heights per bible; origin at front-base centre; front faces -Y; applied transforms; foundation depth; no n-gons, zero-area faces, flipped normals, loose verts; materials only from the MAT_BLD library; texture sizes; tri budgets per LOD; naming; required custom properties). Prints a table and exits non-zero on failure.
5. THE EXEMPLAR: build the Hollow Ridge First Baptist Church (white clapboard, tall steeple, the town's signature, footprint ~20 x 42 m, eaves ~12 m, spire ~26-30 m) to the full standard using the core, and one white clapboard farmhouse with a wraparound porch. Review them with the rubric, iterate until both score >= 9, and put their review renders into the agent briefs as "this is the bar".
6. Commit.

# 5. PHASE 3: PARALLEL PRODUCTION (3 agents)
Write the three briefs (docs/agents/R.md, T.md, C.md) containing: the mission paragraph, their family list below with counts and sizes, the ownership and MCP port rules, the style bible (path + the non-negotiables copied in), the core API reference, the exemplar renders, the technical spec (section 7), the per-asset loop and the rubric (section 6), and the submission format. Then spawn them in the background, all at once.
Per-asset loop every agent follows: blockout from spec (massing and silhouette only) -> REVIEW GATE 1 (silhouette at 300 m, proportions, scale vs human/car, era read; the lead signs off on the first of each type) -> modules and secondary detail -> materials from the library -> tertiary detail and weathering with a cause -> LODs -> validate_kit.py -> review renders -> self-score with the rubric (be harsh; compare against the exemplar and graphics ref.png) -> submit.
A. Agent R, Rural & Residential (building_types: house, house_large, trailer, barn, shed; + rural church):
   - Houses: 1920s bungalow, mill-village cottage (for Laurel Mill), 1940s minimal-traditional, 1950s brick ranch, 1960s ranch with carport, 1970s split-level, dogtrot log house with tin roof. >= 8 generator presets x 3 colourways.
   - Large houses: white clapboard farmhouse with wraparound porch, I-house, Victorian cottage with gingerbread, two-storey brick colonial. >= 5.
   - Mobile homes: 1960s rounded single-wide, 1970s boxy two-tone single-wide, double-wide, with/without skirting and add-on porch. >= 4.
   - Farm: gambrel dairy barn, gable hay/tobacco barn with side sheds, open equipment shed, poultry house (long and low with curtain sides and fans; very north Georgia), corn crib, concrete-stave silo with dome, smokehouse, wellhouse, outhouse, hog pen, fences (split-rail, barbed wire, board). >= 10.
   - Sheds/garages: garden shed, 1- and 2-bay detached garage, carport. >= 4.
   - Farmsteads assembled from the kit: Pickens Farm (px 790,130), Dockery Farm (px 945,630); hamlet sets: Harmony Grove (px 1395,225: small white church + cemetery + 3-4 houses), Laurel Mill (px 665,340: water-powered mill + mill-village row).
   - Dressing props for homes and farms: mailbox, porch swing, rocking chairs, clothesline, propane tank, TV antenna, window AC, wood pile, hay bales, farm gate, 1950s tractor (optional, if time).
B. Agent T, Town, Main Street & Civic (commercial_small, motel, downtown_lowrise, civic, church):
   - Main Street kit: narrow 2-3 storey brick storefront modules with >= 6 cornice, upper-window and storefront/awning variants that tile into continuous party-wall blocks along Hollow Ridge's Main Street (US 19, 14 m wide road, sidewalks), a corner bank (limestone trim), art-deco movie theatre (blank marquee), hardware store with false front, drug store with soda fountain windows, 1920s two-storey hotel, Masonic lodge, WPA-style post office.
   - Highway commercial: country general store with porch and one round-top gas pump (Pickens Crossroads), 1960s box service station + 1970s canopy station (invented generic brand), drive-in burger stand, laundromat, barber shop, feed & seed store, bait shop, 1-storey L-shaped motor court + 2-storey motor lodge with pole signs (blank neon).
   - Civic: 1950s brick school with gym, fire station (2-bay brick), town hall/police, library, clinic, VFW hall; brick Baptist church with columned portico, small Methodist church, cinder-block country chapel, cemetery props.
   - Hollow Ridge landmarks (with the lead): LMK_LM_HR_SCHOOL (~52 x 32 m, rot 22 deg, px 1076,250), LMK_LM_HR_DEPOT (Georgia Northern combination depot at px 1060,392 on Depot Street / Railroad Avenue: board-and-batten, mustard and brown, bay window, freight room, long platform, baggage carts), Hollow Ridge water tower (steel legs, blank tank, on high ground north of Main Street).
   - Town props: parking meters, benches, trash cans, newspaper boxes, pay phone, soda machine, barber pole, bank clock, blank billboard, blank neon sign frames.
C. Agent C, City, Industry & Rail (urban_midrise, tower, industrial, arena):
   - Towers: >= 6 designs of 1960s-70s International Style (dark bronze glass curtain wall, white concrete bands, setbacks, one rooftop sign frame) + one 1920s-30s setback brick tower; stackable floor modules for 40-95 m; facade richness from textures and a few deep reveals, not geometry.
   - Midrises: 1920s brick office, 1960s concrete/glass office, hospital block, apartment building, brutalist parking deck. >= 6.
   - Downtown sets: Laurel City (older, more brick, courthouse with clock cupola on Court Street) and Tannersville (riverfront, more 1960s-70s glass, riverfront warehouses, marina/boat docks and boathouses along the Tanner). They must not look cloned.
   - Industry: corrugated warehouses with loading docks (3 sizes), sawtooth-roof textile/carpet mill with boiler house and brick chimney, feed mill + concrete grain elevator, sawmill with log deck and conical sawdust burner (Sawmill Road), poultry processing plant, water treatment building, substation enclosure, oil/propane depot tanks.
   - Rail: Laurel City Yard (px 95,410) and Tannersville Yard (px 1840,480) buildings: roundhouse with turntable, yard office, sand tower, freight house, a rail-served warehouse.
   - Arenas (landmarks, with the lead): LMK_LM_LC_COLISEUM (px 152,229) and LMK_LM_TV_ARENA (px 1864,302), both ~70 m diameter, 16-22 m high, shallow white dome over a red-brick drum (the map shows white domes with red rings), entry canopies, plazas. Sister designs, clearly distinct.
The lead can rebalance work between agents at any time based on progress.

# 6. REVIEW RUBRIC + GATES (applied by agents to themselves and by the lead to everything)
Score 1-10 each; ACCEPT only if every line >= 8 and the average >= 8.5 (hero landmarks >= 9):
1. Silhouette and readability at 300 m (squint test).
2. Proportion and scale (vs human/car, vs the bible, vs real 1974 buildings).
3. Era authenticity (1974 north Georgia; zero anachronisms).
4. Style match: sits in graphics ref.png and reference_renders/ without standing out (palette, saturation, value range, level of detail, edge softness).
5. Material read: you can tell brick from block from clapboard from tin at eye level and at 100 m.
6. Weathering with a cause and storytelling (who owns it, how old, how cared for).
7. Construction logic (it could be built and maintained).
8. Variation vs coherence (seeds give variety; a street still reads as one town).
9. Technical: passes validate_kit.py, within budget, clean topology, good LODs (no popping silhouettes), correct pivot/orientation.
10. The "wow" test: would this hold up in a hero screenshot at sunset from 20 m away?
Required evidence per submission: the review contact sheet, the 300 m silhouette shot, the clay shot, the context-scene shot for anything that belongs in Hollow Ridge, the validator table, and a 3-line self-critique. The lead REVISEs with numbered notes; max 4 revision rounds per asset before the lead personally intervenes or cuts it.

# 7. TECHNICAL SPEC (hard requirements)
- Metres, Z-up, front (street/door side) faces -Y; origin at ground level, centre of the front facade base; all transforms applied; foundations extend 1.5 m below grade.
- Naming: BLD_<class>_<style>_<variant> (e.g. BLD_house_bungalow_A); landmarks LMK_<landmark id> (e.g. LMK_LM_HR_CHURCH); props PRP_<name>; kit modules KIT_<family>_<part>; materials MAT_BLD_<name> (shared; no per-asset duplicates); LODs <asset>_LOD0/_LOD1/_LOD2; collision <asset>_COL.
- LOD0 triangle budgets: prop <= 500, house <= 3k, large house/farm/commercial <= 6k, main-street block unit <= 8k, midrise <= 10k, tower <= 15k, landmark <= 40k. LOD1 25-35 % of LOD0 with the same silhouette; LOD2 <= 300 tris (massing + baked facade).
- Textures: shared trim sheets and tiling sets at <= 2048^2 (albedo sRGB, normal OpenGL, ORM); hero landmarks may have one unique 2k set; LOD2 per-asset atlases <= 512^2. Every texture bakes out of Blender (no node-only looks). Provide a per-asset window emissive mask (for lit windows at dusk later).
- Glass: tinted, lightly reflective, with fake interiors (interior mapping or a painted interior card) so windows never read as black holes or mirrors.
- Web: the game renders with three.js r169 MeshStandardMaterial, ACES Filmic tone mapping, golden sun + HDRI env, cascaded shadows. Materials must survive that: no reliance on SSS, refraction or procedural nodes; check albedo ranges (no pure white > 0.85 linear, no pure black < 0.03).
- Reproducibility: `blender -b --python tools/blender/build_building_kit.py -- --out assets/buildings/building_kit.blend` rebuilds the entire kit from scripts (all families + landmarks + props), marks everything as Blender Assets with catalogs by class, and renders asset previews.
- Exports:
  * exports/buildings/glb/<asset>.glb (LODs as separate nodes).
  * tools/blender/export_web_buildings.py (model it on export_web_trees.py) -> app/world/buildings/geo.bin + geo.json + textures (quantized per-asset per-LOD buffers, shared material table).
  * data/buildings/kit_catalog.json: per asset: id, class (building_types.json key), replace_with category, style, era_decade, footprint_m [w, d], height_m, storeys, roof, lod_tris, materials, colour presets, wear range, zone tags (downtown, town_center, town, rural, farm, industrial, highway, riverfront), suitable settlements/landmarks, frontage road types (from road_types.json), description. Every building_types.json replace_with category must be covered.

# 8. PHASE 4: INTEGRATION + IN-GAME PROOF (lead)
1. Run build_building_kit.py headless from scratch, then validate_kit.py (must pass), then export_web_buildings.py.
2. In-game preview (dev only, clearly marked as a preview; the real placement pass comes later): in src/, add a small module src/components/world3d/buildingsPreview.ts that loads app/world/buildings and places (a) every landmark at its real px position/rotation from app/world/landmarks.json on the terrain, (b) a demonstration Hollow Ridge Main Street: both sides of US 19 between the church and the depot filled with main-street blocks, plus a few houses on the side streets, oriented to the roads in app/world/roads.json, (c) Pickens Farm. Build the viewer with `npm install` and `npm run dev` (copy or link app/world to public/world for dev), fly it, take screenshots from cams CAM_HollowRidge_Valley, CAM_Ref_Match, TC_aerial_100 and a Main Street eye-level shot, and compare against graphics ref.png. Fix what breaks the illusion (scale, colour, brightness, LOD pops, shadow acne, floating foundations).
3. Performance check in the viewer (P key shows fps, triangles, draw calls): buildings added to the Hollow Ridge view should cost < 1.5 M triangles and < 150 draw calls at the High preset.

# 9. DON'TS
- Don't modify base_map.webp, terrain/roads/water/vegetation data, or the existing world look; don't commit the preview as final placement.
- No real brands, logos or legible real-world text; no post-1974 anything.
- Don't let agents touch each other's files or Blender instances.
- Don't accept "good enough". Don't pad counts with near-duplicates; variants must differ meaningfully.

# 10. FINAL REPORT (to the user)
MCP status and fixes; the style bible summary; asset counts per class and the landmark list; the accepted contact sheets (paths) and the in-game screenshots; budgets and validator output; rebuild commands; the review log highlights (what got rejected and why); honest known gaps; recommended next steps (full placement pass using kit_catalog.json + data/manual/zones.json + roads, dusk lit windows, signage decals with invented period names, a few enterable interiors).
```
