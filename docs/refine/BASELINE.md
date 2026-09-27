# Baseline (before the refinement pass)

Mode: SwiftShader in a cloud container: frame times are NOT meaningful; proxies below are.
Probe: `node tools/qa/perf.mjs http://localhost:4173/ --samples 12 --w 640 --h 416` (fixed 6-waypoint spline). Raw: docs/refine/baseline/perf.json.

| preset | main calls p50 / max | shadow calls p50 | main tris p50 / max | all-pass tris max | cull ms p50 / max | programs |
|---|---|---|---|---|---|---|
| low | 255 / 387 | 85 | 6.17 M / 9.34 M | 16.6 M | 14.7 / 69 | 55 |
| medium | 245 / 423 | 398 | 9.77 M / 14.4 M | 35.6 M | 8.4 / 13.5 | 55 |
| high | 277 / 424 | 680 | 13.3 M / 20.3 M | 86.9 M | 8.2 / 16.6 | 55 |
| ultra | 288 / 429 | 684 | 18.0 M / 27.9 M | 121 M | 23 / 38.6 | 55 |

Load (all layers ready) 7 s local; download 109.8 MB (raw, unpacked static build; artifact packing ~69 MB); GPU uploads: textures 680 MB, buffers 213 MB; JS heap 399 MB.

Validators: roads 0 errors, 21 radius + 2 short-edge warnings; grading issues: RU_RD_0421 68.9 % grade (gravel, limit 15 %), RU_RD_0606 18.5 m cut. Terrain: 0 errors; warnings perched_river 9, spike_raw 5, spike_graded 19, zone_slope 7.
Infra counts: 686 stop signs, 10 route shields, 18 curve signs, 8 crossbucks, 0 street lights, 1392 guardrail posts, 660 delineators, 430 poles.

## Versus budgets (High)
main tris 13.3 M vs 1.5 M (x9), worst 20.3 M vs 2.5 M; all-pass 87 M vs 4 M (x22: shadow casters); shadow calls 680 vs 150; programs 55 vs 40; cull 8 ms vs 4 ms; download 110 MB vs 40 MB; texture uploads 680 MB (uncompressed).
Biggest suspects: tree LOD0-2 counts and shadow casting of LOD0-2 across 4 cascades; grass blade slots; far LOD3/4 still geometry; 6000 px albedo and many 1-2k textures uncompressed.

Golden shots: exports/refine/baseline/*.png (20 shots, tools/qa/golden_shots.json); contact sheet docs/refine/baseline/golden.jpg.
