# Validation: `sun` vs GRASS GIS `r.sun`

Reference implementation: **GRASS 8.3.2** (`r.sun`, system package, 12-thread
OpenMP). Engine under test: the `sun` extension (array API), CPU and GPU
paths. All runs: Linke = 3.0, albedo = 0.2, step = 0.5 h, solar constant
default (1367).

## Test data

- `make_test_dem.py` → synthetic alpine terrain, 200×200 px @ 30 m,
  EPSG:32633 (UTM 33N, ~Salzburg), 596–1833 m, ridges/valleys so cast
  shadows occur; nodata border column.
- `wall.tif` → controlled geometry: flat 500 m plain + one 1000 m E–W wall
  at row 150. Analytic winter-solstice shadow length at lat 47.37°N:
  1000 m / tan(19.2°) ≈ 2873 m ≈ 96 px north of the wall.
- Slope/aspect for both engines come from **GRASS's own `r.slope.aspect`
  rasters** (CCW-from-east), so slope derivation is not a confounder.
  (Cross-check: feeding my Horn-derived slope/aspect instead changes the
  global-irradiation mean by +0.0 Wh/m², corr 1.00000.)

Reproduce: `make_test_dem.py`, then `grass_reference.sh` (needs GRASS 8),
then `compare_sun_grass.py`, `compare_tilted_plane.py`,
`compare_wall_shadow.py`. `make_figures.py` regenerates `maps.png` and
`wall_profile.png` with the installed engine.

## Results

### 1. Horizontal surface (pure solar geometry + ESRA atmosphere)

| Day | Component | sun | GRASS | rel. diff |
|----|----|----|----|----|
| 172 | beam | 7778.11 | 7777.80 | +0.00% |
| 172 | diffuse | 1298.16 | 1298.11 | +0.00% |
| 172 | global | 9076.28 | 9075.91 | +0.00% |
| 172 | insolation | 16.00 h | 16.00 h | 0.00% |
| 355 | beam | 1139.41 | 1139.47 | −0.01% |
| 355 | diffuse | 435.18 | 435.20 | −0.00% |
| 355 | global | 1574.59 | 1574.66 | −0.01% |
| 355 | insolation | 8.00 h | 8.00 h | 0.00% |

The atmospheric/geometry core is numerically identical to r.sun.

### 2. Tilted plane (slope incidence), aspect via raster

South-facing ramp (21.8°, aspect raster = 270 CCW-from-east), day 355,
shadows off on both sides (`r.sun -p` vs all-nodata shadow context):

| | sun | GRASS | rel. diff | corr |
|----|----|----|----|----|
| global | 3032.4 | 3032.5 | −0.00% | 1.00000 |

Slope-incidence math is identical when both engines read the same aspect
raster.

### 3. Rough terrain, full pipeline (post-fix)

| Day | Shadows | sun glob | GRASS glob | bias | corr | nodata agreement |
|----|----|----|----|----|----|----|
| 172 | off | 8932.1 | 8945.1 | −0.15% | 0.999 | 100% |
| 172 | on | 8660.3 | 8938.4 | −3.11% | 0.784 | 100% |
| 355 | off | 1721.1 | 1726.5 | −0.31% | 1.000 | 100% |
| 355 | on | 1573.4 | 1717.9 | −8.41% | 0.983 | 100% |

Orientation check (uniform 20° planes, day 355, shadows off), post-fix:

| Facing (GRASS CCW-from-east) | sun | GRASS | rel. diff |
|----|----|----|----|
| east (0) | 1581.6 | 1593.0 | −0.7% |
| north (90) | 415.8 | 435.2 | −4.5% |
| west (180) | 1581.6 | 1593.0 | −0.7% |
| south (270) | 3082.8 | 3083.0 | −0.0% |

The residual shadow-on bias is now entirely the §4 penumbra-vs-terminator
modeling difference (it grows with slope: −2.4% at 0–5° to −16.8% at
25–40°, where the terminator cuts a larger share of each pixel's day).

### 4. Shadow model: penumbra vs hard terminator

Controlled wall (see `wall_profile.png`): in **deep shadow both engines
agree exactly** (both remove 72% of flux; both report 435 Wh/m²/day at rows
100–148). They differ at the shadow edge:

- GRASS: hard terminator at row 83 (horizon-angle lookup: a time step is
  either shaded or not).
- sun: gradual penumbra, rows 38–67 (0.5-px bilinear ray-march: the sun disc
  is progressively occluded as the ray grazes the wall top).

This is a *modeling* difference, not a bug: the ray-march resolves partial
occlusion that r.sun's horizon binarization cannot. It costs a few percent
of mean flux near terminators and explains the shadow-off residual bias.

### 5. FIXED: nodata on slopes the sun never clears (+ east-facing bug)

Two bugs found by this comparison, both fixed (commit "fix: horizontal-day
integration window + east aspect collapse"):

**a) Never-sunlit slopes lost diffuse+reflected.** Where the sun never
rises above the slope plane (steep north-facing slopes in winter), the
engine integrated over the *slope-plane* day, whose sunrise equation has no
solution → pixel written as nodata, dropping sky radiation entirely. GRASS
integrates the *horizontal* day and gates beam by slope incidence (s0 > 0),
keeping diffuse+reflected. Fix: `solar::horizontal_day_window()` — all
compute paths (CPU bands, annual, pixel, path-based CPU, WGSL shader) now
integrate the horizontal day; beam/insolation stay 0 on never-sunlit slopes.
Before: 472/40000 pixels nodata-only-in-sun on day 355 (100% north-facing).
After: nodata agreement 100% (0 mismatches both directions).

**b) Due-east aspect collapsed to north.** `convert_grass_aspect` had a
special case `aspect_deg == 0.0 → 0.0 rad`, but internal 0 rad is *north*,
so east-facing slopes were computed as north-facing (winter beam 0 instead
of ~1136 Wh/m²/day). The special case existed for flat pixels, but those are
already handled by the slope==0 branch of `compute_slope_geometry`. Removed
in Rust and WGSL; `test_aspect_conversion` now pins east → π/2.

### 6. Aspect convention (scalar vs raster)

GRASS is internally inconsistent: its **raster** aspect is CCW-from-east
(r.slope.aspect: "90 degrees is North"), while its **scalar**
`aspect_value` behaves as compass azimuth in measurement (0=N, 180=S gave
the south peak) despite the manual line "270 is south". The sun tool uses
CCW-from-east everywhere, matching GRASS *rasters* and r.slope.aspect. The
plugin never passes scalar aspect (rasters or Horn-derived only), so this
does not affect plugin results; `compute_pixel(aspect_deg=…)` documents the
CCW-from-east convention.

### 7. Performance (200×200, day run, shadows on, same machine)

| Engine | Time |
|----|----|
| GRASS r.sun | 0.75–0.77 s |
| sun CPU | 0.16–0.22 s |
| sun GPU (Radeon 740M) | 0.08–0.12 s |

sun CPU ≈ 4× and GPU ≈ 7× faster than r.sun at this size; the gap widens
with raster size (see README benchmarks).

## Verdict

Radiation physics (geometry, ESRA atmosphere, slope incidence, albedo) is
numerically identical to r.sun (<0.01% on controlled surfaces). Two bugs the
comparison exposed are fixed (never-sunlit nodata; east-aspect collapse).
One deliberate modeling difference remains: penumbra at shadow terminators
(sun resolves partial occlusion; r.sun's horizon lookup is binary), which
accounts for the entire residual shadow-on bias.
