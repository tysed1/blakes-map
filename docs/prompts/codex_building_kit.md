# Codex prompt: Blender MCP check + 1974 building kit

Before running: save the DALL·E sheet(s) as `assets/reference/building_kit_ref.png`
(+ optional `building_kit_ref_2.png` … `_4.png`) in the repo.

```
You are the lead environment artist + technical artist on "Blake's Map", a stylized-realism open-world game set in north Georgia (southern Appalachia), autumn 1974. The world (terrain, rivers, roads, bridges, rail, 215k trees, ground cover, golden-hour atmosphere) is already built in Blender and in a three.js viewer. Buildings are the one missing layer. Your job: verify the Blender MCP bridge, then build a production-quality, performance-budgeted BUILDING KIT and LANDMARK SET in Blender that matches the game's look exactly, driven by the reference sheet(s) in assets/reference/building_kit_ref*.png. Work in the repo at the project root on a new git branch `buildings-kit` (never push to main). Take your time; quality and consistency matter more than speed.

=====================================================================
0. READ FIRST (project truth; do not skip)
=====================================================================
- CLAUDE.md and docs/WORLD_DESIGN_PRINCIPLES.md (priority: functional logic > believable engineering > visual quality > ~90% map faithfulness; setting 1974 north Georgia).
- `graphics ref.png` (repo root): the art-direction target. Golden-hour light, stylized realism, clean silhouettes, rich but controlled colour, lived-in but not grimy.
- assets/reference/building_kit_ref*.png: the DALL-E kit sheet(s). Use them for massing, palette, materials and mood. They are concept art: fix any impossible geometry, wrong proportions, or anachronism (nothing after 1974).
- Look at how the world already renders so buildings sit in it seamlessly:
  exports/a3/renders/r15/CAM_Ref_Match.jpg, exports/a3/renders/tc10/*.jpg, exports/a2/renders/road/*.jpg (HR_Main_Ped.jpg, RR_Depot_Driver.jpg = where Hollow Ridge buildings will stand), exports/web/qa1/*.png (the three.js look).
- data/buildings/building_types.json: the 14 building classes the placement pipeline already uses (house, house_large, trailer, barn, shed, commercial_small, motel, downtown_lowrise, urban_midrise, tower, industrial, civic, church, arena), their height ranges and `replace_with` categories. Your kit MUST cover every one of these `replace_with` categories, and your catalog uses them as keys.
- data/manual/landmarks.json + public/world/landmarks.json (named landmarks with footprints), public/world/settlements.json, public/world/regions.json, data/manual/zones.json, data/roads/road_types.json (road widths: main_street 14 m, residential 7.5 m, highway 10.2 m, urban_street 11 m).
- assets/maps/source/base_map.webp (the concept map; NEVER modify it). Crop it around Hollow Ridge (px 930-1220, 200-470), Laurel City (px 0-420, 150-500) and Tannersville (px 1520-2000, 120-520) to see roof colours, density and the two round arenas.
- tools/blender/README.md (MCP setup), tools/blender/lib_materials.py (shared material library, `LM.get(name)`), tools/blender/lib_atmosphere.py + tools/blender/render.py (the project's lighting/camera rig), tools/blender/lib_trees.py and tools/blender/export_web_trees.py (the pattern to copy for web export: named prototypes, LODs, compact binary + JSON).
- Coordinates: 1 Blender unit = 1 metre, Z-up. World: source px -> metres X=(px-1000)*2.5, Blender Y=-(py-333.5)*2.5, Z = elevation. A map pixel is 2.5 m.

=====================================================================
1. VERIFY THE BLENDER MCP CONNECTION (gate: do not model until this passes)
=====================================================================
Blender 4.2 LTS is the project version (`blender --version`). The bridge is blender-mcp: Codex -> `uvx blender-mcp` (stdio) -> TCP localhost:9876 -> Blender with the "MCP for Blender" add-on.
1. If not installed: run `tools/blender/setup.sh` (installs Blender 4.2 LTS if missing, Xvfb, `uv tool install blender-mcp`, and the matching add-on). Make sure your MCP config registers the server: command `uvx`, args `["blender-mcp"]`, env `DISABLE_TELEMETRY=true` (see the project's .mcp.json).
2. Start Blender with the add-on server: `tools/blender/start_mcp.sh` (uses xvfb-run when there is no display; only one Blender may own port 9876; use BLENDER_PORT for another).
3. Socket check: `python3 tools/blender/mcp_ping.py` must print scene info with status success; `python3 tools/blender/mcp_ping.py --code "import bpy; print(bpy.app.version_string)"` must print 4.2.x.
4. MCP tool check from inside Codex: call get_scene_info, then execute_blender_code creating a 1x1x1 m test cube named `MCP_TEST`, then get_viewport_screenshot and confirm you can see it, then delete it. Also confirm Poly Haven access with get_polyhaven_status (optional; the repo already has CC0 Poly Haven textures in assets/external/polyhaven/).
5. If any step fails: read the error, check the port, restart Blender/the MCP server, re-run. Report exactly what was wrong and how you fixed it. If MCP truly cannot work, fall back to headless scripts (`blender -b --python tools/blender/<script>.py -- args`) for everything, and say so. The kit must be reproducible from scripts either way (see section 5).

=====================================================================
2. STYLE BIBLE (write it, then obey it): docs/BUILDING_KIT_STYLE.md
=====================================================================
Derive from the reference sheet(s) + graphics ref.png + the existing renders:
- Era rules, 1974 north Georgia: what exists and what does not (no vinyl siding, no modern windows, no satellite dishes, no modern signage; OK: clapboard, board-and-batten, brick, cinder block, asbestos shingle, tin roofs, aluminium storm doors, window AC units, TV antennas, chain-link, 1950s-70s commercial modernism, International Style towers).
- Palette: named colours with linear RGB values (sun-bleached white, faded barn red, weathered red brick and buff brick, mustard, harvest gold, avocado, burnt orange, galvanized steel, rust, asphalt-shingle greys, dark bronze glass). Keep saturation controlled; the world grade is warm.
- Materials: a small shared set built with PBR textures (the repo already has CC0 Poly Haven concrete_wall_006, rough_concrete, weathered_planks, rusty_metal_02 in assets/external/polyhaven/; fetch more CC0 sets such as red/buff brick, clay and asphalt roof shingles, corrugated and painted metal, painted wood siding, plaster and stone with tools/assets/fetch_polyhaven.py or the Poly Haven MCP tools, same 1k/2k resolutions) plus procedural wear (edge wear, rust streaks, dirt at the base, moss on north roof edges, paint fade). One trim sheet + one tiling set per material family so hundreds of buildings share a handful of materials.
- Proportions: storey height 2.7-3.0 m residential, 4.0-4.5 m commercial ground floor, 3.6 m upper commercial floors; door 2.1 m; window sills 0.8-0.9 m; porch depth 2.4-3.0 m; roof pitches (8/12 farmhouse, 4/12 ranch, 12/12 church); plinths/foundations 0.3-0.8 m (houses on brick piers on slopes).
- Wear level: lived-in, cared-for small town; poorer on the rural fringes (mill cottages, trailers, tobacco sheds), prosperous downtown banks and churches.

=====================================================================
3. KIT ARCHITECTURE (how to build it)
=====================================================================
Build a SYSTEM, not one-offs:
- Snap grid 0.5 m. Modular parts per family (wall bays 1/2/3 m, corner pieces, window and door modules, storefront bays, cornices, porches, awnings, chimneys, gutters, roof pieces by pitch) in a kit library collection `KIT_parts`.
- A Python generator `tools/blender/lib_buildings.py` (mirror the style of lib_trees.py) that assembles variants from parameters: footprint w x d, storeys, roof type/pitch, facade material + trim colour, porch/awning/chimney toggles, window rhythm, wear amount, seed. Deterministic per seed. This is how we get hundreds of distinct but consistent buildings.
- Hero LANDMARKS are hand-authored (can still use kit parts) with extra detail.
- Add small DRESSING props that sell the era, as separate instanced assets: window AC unit, TV antenna, propane tank, mailbox, porch swing, rocking chairs, soda machine, round-top gas pump, 1970s pay phone, trash barrels, pallets, oil drums, clothesline, chain-link fence segment, picket fence segment, blank neon sign frame, blank billboard, water meter, electric meter/weatherhead (utility poles and lines already exist in the world; do not remake them).

=====================================================================
4. WHAT TO BUILD (map-driven; check counts and sizes against the data)
=====================================================================
A. Class coverage (every building_types.json class; aim for >= the variant counts listed; each variant = a distinct generator preset + 3 colour/material presets):
- house (1-1.5 storey, 5-7.5 m): 1920s bungalow, mill-village cottage, 1940s minimal-traditional, 1950s brick ranch, 1960s ranch with carport, 1970s split-level. >= 8 variants.
- house_large (7.5-9.5 m): white clapboard farmhouse with wraparound porch, I-house, Victorian cottage with gingerbread, two-storey brick colonial. >= 5.
- trailer (3.5-4 m): single-wide (1960s rounded, 1970s boxy two-tone), double-wide, with skirting/no skirting, on blocks, with add-on porch. >= 4.
- barn / rural (7-10 m): gambrel dairy barn, gable tobacco/hay barn with side sheds, open equipment shed, poultry house (long, low, curtain sides; very north Georgia), corn crib, concrete-stave silo, smokehouse, wellhouse. >= 8.
- shed (3-4 m): garden shed, detached garage (1 and 2 bay), carport. >= 4.
- commercial_small (4.5-7 m): general store with porch and gas pump, service station (1960s box + 1970s canopy variant, generic invented brand, no real logos), hardware store with false front, drive-in burger stand, laundromat, barber shop, feed & seed store, bait shop. >= 8.
- motel (4-7 m): 1-storey L/U motor court with walkway and office, 2-storey motor lodge with exterior stairs, pole sign with blank neon. >= 3.
- downtown_lowrise (8-16 m, 2-4 floors): narrow brick storefronts with 6+ cornice/awning/upper-window variants that tile into a continuous Main Street block (party walls), corner bank (limestone trim), art-deco movie theatre (blank marquee), 1920s hotel, Masonic lodge, department store, post office (WPA style), courthouse with clock cupola (Laurel City's Courthouse Road / Court Street exist). >= 12.
- urban_midrise (18-36 m, 5-10 floors): 1920s brick office, 1960s concrete/glass office, hospital block, apartment building, brutalist parking deck. >= 6.
- tower (40-95 m, 11+ floors): 1960s-70s International Style towers (dark bronze glass curtain wall, white concrete bands, some with setbacks, one with a rooftop sign frame), one 1920s-30s setback brick tower. Laurel City and Tannersville each have ~8-12 towers in the map: make >= 6 tower designs with stackable floor modules so heights can vary 40-95 m. Keep them cheap (facade detail from texture, not geometry).
- industrial (7-13 m): corrugated warehouse with loading docks (3 sizes), sawtooth-roof textile/carpet mill with boiler house + brick chimney, feed mill + concrete grain elevator, sawmill with log deck and conical sawdust burner (Sawmill Road, Laurel Mill hamlet), poultry processing plant, freight depot/warehouse beside sidings, rail yard buildings (yard office, sand tower, engine house / roundhouse with turntable for the Laurel City and Tannersville yards), water treatment building, electrical substation equipment enclosure. >= 12.
- civic (8-14 m): brick school (1950s flat-roof with ribbon windows) + gym, fire station (2-bay brick), police/town hall, library, small hospital/clinic, VFW hall, National Guard armory. >= 7.
- church (9-12 m + spire): white clapboard country church with steeple, brick Baptist church with columned portico and tall steeple, small Methodist church, cinder-block country chapel, plus church cemetery props (headstones, iron fence). >= 4.
- arena: see landmarks.
B. Named LANDMARKS (hand-authored hero assets; sizes from the landmark data, 1 px = 2.5 m):
- Laurel City Civic Coliseum (LM_LC_COLISEUM, px 152,229, radius 14 px = ~70 m diameter, 16-22 m high): round 1960s coliseum, shallow white dome, red/brown brick drum with vertical window slots, entry canopies, surrounding parking. On the map it reads white dome + red ring.
- Tannersville Memorial Arena (LM_TV_ARENA, px 1864,302, same size class): sister design but distinct: white dome, stronger red band, riverside plaza.
- Hollow Ridge First Baptist Church (LM_HR_CHURCH, footprint 8x17 px = ~20 x 42 m, rot -4 deg): the town's white steepled church at the heart of Main Street, 12 m eaves + ~24-30 m steeple total, visible from the whole valley. This is the town's signature; make it beautiful.
- Hollow Ridge School (LM_HR_SCHOOL, 21x13 px = ~52 x 32 m, rot 22 deg): 1950s brick school with gym and ball field backstop.
- Hollow Ridge Depot (LM_HR_DEPOT, px 1060,392, on the Georgia Northern Mainline, Depot Street / Railroad Avenue): board-and-batten combination depot, mustard/brown paint, bay window, freight room, long platform, baggage carts.
- Hollow Ridge water tower (new; on high ground north of Main Street): steel tank on legs, blank tank (the town name can be added later as a decal).
- Laurel Mill (hamlet at px 665,340): the old water-powered grist/lumber mill + mill-village houses.
- Pickens Farm (px 790,130) and Dockery Farm (px 945,630): complete farmsteads (farmhouse, barn, silo, sheds, poultry house, fences).
- Pickens Crossroads (px 800,140): crossroads general store + gas pump + church.
- Harmony Grove (px 1395,225): small church + cemetery + a few houses.
- Rail: Laurel City Yard (px 95,410) and Tannersville Yard (px 1840,480) buildings (roundhouse, yard office, freight house).
- Downtowns: a Laurel City and a Tannersville "downtown set" (towers + midrises + lowrise blocks) with enough variety that two cities don't look cloned (Laurel City: slightly older, more brick; Tannersville: riverfront, more 1960s-70s glass).

=====================================================================
5. TECHNICAL SPEC (hard requirements)
=====================================================================
- Units metres, Z-up, +Y = back, the FRONT (street/door side) faces -Y. Origin = ground level at the centre of the front facade base, so placement can snap a building to a lot and face it to the road. Apply all transforms. Real-world scale (check against a 1.8 m human and a 1974 sedan 5.3 x 2.0 m).
- Ground contact: include a foundation/plinth that extends 1.5 m below ground so buildings sit on slopes without floating (the world has real terrain).
- Naming: `BLD_<class>_<style>_<variant>` (e.g. BLD_house_bungalow_A), landmarks `LMK_<landmark id>` (e.g. LMK_LM_HR_CHURCH), props `PRP_<name>`, parts `KIT_<family>_<part>`. Materials `MAT_BLD_<name>`, shared across assets (no per-asset duplicate materials).
- LODs per asset: LOD0 (full), LOD1 (~25-35% tris, porches/trim simplified), LOD2 (<= 300 tris: box massing + roof + baked facade texture), named `<asset>_LOD0/1/2`. Budgets for LOD0: house <= 3k tris, farm/commercial <= 6k, downtown lowrise <= 8k, midrise <= 10k, tower <= 15k (facades from textures), landmark <= 40k, prop <= 500. Consistent texel density ~256 px/m on trim sheets; hero landmarks may use one unique 2k set each.
- Textures: <= 2048^2, sRGB albedo, non-colour ORM (occlusion/roughness/metal) and normal; web-friendly (we ship to three.js). Glass: simple PBR, no refraction (fake interiors via a parallax/interior texture or dark tinted glass with a subtle reflection; also provide an emissive mask per window set so we can do lit windows at dusk later).
- Clean topology for export: no n-gons in final meshes, no zero-area faces, no inverted normals, manifold where practical, UVs in 0-1 for unique maps, trim-sheet UVs for shared.
- Collision: `<asset>_COL` simple convex/box proxy.
- Reproducible: everything is built by scripts (lib_buildings.py generator + a `tools/blender/build_building_kit.py` entry that rebuilds the whole library headless: `blender -b --python tools/blender/build_building_kit.py -- --out assets/buildings/building_kit.blend`). Use MCP for iteration and visual checks, but commit the scripts, not just the .blend.
- Outputs:
  * assets/buildings/building_kit.blend (assets marked as Blender Assets with previews, organized in catalogs by class), git-LFS or gitignored if > 50 MB (say which).
  * exports/buildings/glb/<asset>.glb per asset (LODs as separate nodes, Draco off; the web uses its own binary format like export_web_trees.py).
  * tools/blender/export_web_buildings.py modelled on tools/blender/export_web_trees.py: writes public/world/buildings/geo.bin + geo.json + textures (compact quantized buffers per asset per LOD, material table).
  * data/buildings/kit_catalog.json: one entry per asset: id, class (building_types key), replace_with category, style, era_decade, footprint_m [w, d], height_m, storeys, roof, lods (tri counts), materials, colour presets, zone tags (downtown / town_center / town / rural / industrial / farm / highway), which settlements/landmarks it suits, allowed road types for frontage, and a short description. The later placement pass picks from this catalog by zone + footprint + road type.
- Do NOT place buildings into the world scene, do not edit terrain/roads/water/vegetation data or the web viewer (other passes own those). Do not use real brand names or logos anywhere (invent period-plausible generic ones only if needed, e.g. "Mountain Oil", "Hollow Ridge Feed & Seed"). Nothing from after 1974.

=====================================================================
6. VALIDATION LOOP (repeat until it is excellent)
=====================================================================
For every asset family:
1. Lineup render in Blender with the project's look: sun ~255 deg azimuth, 18 deg elevation, warm (render.py/lib_atmosphere.py settings; HDRI assets/external/polyhaven/kloppenheim_06_puresky), a 35-degree aerial camera matching the reference sheet framing, a 1.8 m human and a 1974 sedan for scale, plus an eye-level 1.7 m shot. Save to exports/buildings/review/<family>_vNN.png.
2. Compare side by side with building_kit_ref*.png and graphics ref.png (write a short critique: silhouette, proportions, palette, material read, wear, era accuracy, repetition). Fix and re-render. Keep the best version per family.
3. Context test: append the Hollow Ridge church, depot, a Main Street block of 6 lowrises, 3 houses and a barn into a COPY of exports/web/terrain.blend (or a small test scene with the same lighting) near their map positions (use the transform in section 0), render from the existing cameras CAM_HollowRidge_Valley and CAM_Ref_Match, and check they sit in the world's light, haze and colour (not too saturated, not too clean, not too dark). Delete the test scene afterwards; nothing gets committed into the world files.
4. Budget check: print a table of every asset: tris per LOD, material count, texture sizes; flag anything over budget and fix it.
5. Automated checks: scale (door heights, storey heights), origin at front-base, front faces -Y, no n-gons, no missing textures, all names follow the convention, every building_types.json class covered.

=====================================================================
7. REPORT
=====================================================================
When done, report: MCP status and anything you had to fix; the style bible summary; the full asset list (counts per class, landmarks); budgets table; paths to the contact sheets / review renders (best versions + the in-world context renders); how to rebuild (commands); known gaps and what you would do next (e.g. lit windows at dusk, signage decals with town names, interiors for a few hero buildings). Commit on branch `buildings-kit` with clear messages; do not merge.
```
