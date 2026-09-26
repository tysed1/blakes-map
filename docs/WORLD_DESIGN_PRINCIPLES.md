# World design principles (authoritative)

The base map (`assets/maps/source/base_map.webp`) is **concept art for the world,
not survey data**. It defines the general shape, direction, density and identity
of the world: where towns, rivers, ridges and major routes are. It does not define
exact pixels.

Priority order when these conflict:

1. **Functional geographic logic.** Water flows downhill, roads connect, grades are drivable.
2. **Believable engineering and world design.** A real civil engineer, planner or
   environment artist would approve it.
3. **Visual quality.** Stylized realism, per `graphics ref.png` (Blender/art phase only).
4. **Approximate faithfulness to the reference.** Aim for about 90% faithful and
   100% believable, not 100% traced and broken.

Setting: 1974, southern Appalachia / north Georgia. Invent missing information
conservatively and consistently with that setting.

## Roads

* Every road can be built, driven and maintained. Endpoints are logical; a dead end
  has a reason (driveway, farm, logging, service, cul-de-sac, abandoned).
* No orphan fragments, near-miss endpoints, crossings without a junction or grade
  separation, duplicate parallel traces, or impossible angles or radii.
* Rural junctions may be irregular and terrain-driven but must still work.
  Urban junctions become more regular as density increases.
* Freeways and interchanges are engineered separately: ramps with proper
  tapers, merge lengths, radii, directional logic and overpasses. A confusing
  interchange in the reference is redesigned into the closest believable equivalent.
* Roads follow valleys, use passes, contour slopes, and use switchbacks only when
  needed. Terrain is cut or filled for roads rather than roads following bad terrain.
* Every water crossing is classified as bridge, culvert, ford or causeway.

## Water

Drainage behaves like a real system: monotonic downhill flow, logical confluences,
channels in valleys, and no unexplained split-and-rejoin. Routes may be adjusted
to keep hydrological sense.

## Buildings and settlements

Every building has road access, a sensible orientation (facing its road), usable
land, a buildable slope and sensible spacing. Hollow Ridge grew organically along
the main road, creek and rail line; the cities grow more structured toward their
cores, with a believable rural-to-urban transition.

## Rail

Gradual curves, low grades, intentional road crossings, bridges where needed, and
yards connected to the mainline. No decorative fragments.

## Land use

Farms on flatter usable land, forest on steep ridges, industry on road and rail,
commerce on busy routes. The reference's colours are not unquestionable truth.

## Process

Automated validators (`tools/qa/`) plus manual visual review at world, region,
town, intersection and road-level scale. Before declaring a system complete, ask
"would a real engineer / planner / environment artist approve this?" and fix
whatever fails.
