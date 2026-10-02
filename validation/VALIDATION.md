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
`compare_wall_shadow.py`.

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

### 3. Rough terrain, full pipeline

| Day | Shadows | sun glob | GRASS glob | bias | corr | nodata agreement |
|----|----|----|----|----|----|----|
| 172 | off | 8884.2 | 8945.1 | −0.68% | 0.985 | 100% |
| 172 | on | 8612.3 | 8938.4 | −3.65% | 0.745 | 100% |
| 355 | off | 1696.3 | 1742.6 | −2.66% | 0.999 | 98.8% |
| 355 | on | 1546.8 | 1733.9 | −10.79% | 0.982 | 98.8% |

With shadows disabled on both sides the engines agree within ~1–3% (the
residual is the penumbra/terminator difference below). With shadows on, the
winter bias grows to ~11% — decomposed in §4 and §5.

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

### 5. BUG FOUND: nodata on slopes the sun never clears

Where the sun never rises above the slope plane (steep **north-facing**
slopes in winter), `compute_sunrise_sunset` returns None and the pixel is
written as nodata — dropping diffuse and reflected radiation too. r.sun
keeps those pixels: beam = 0 but diffuse + reflected are integrated over the
day. Evidence:

- Test DEM day 355: 472/40000 pixels (1.2%) nodata in sun, valid in GRASS;
  **100% of them north-facing** (compass 315–45°); GRASS values there ≈
  414 Wh/m²/day (pure diffuse).
- Uniform 20° plane, day 355: sun returns nodata for east and north aspects
  (n=0 valid pixels); GRASS returns 435 (north, diffuse-only) and 1593
  (east, morning beam).
- `compute_pixel` raises "No sunrise …" for the same inputs.

Physically the sky is still visible from such slopes, so diffuse/reflected
must survive. Fix: when the slope-plane sunrise is undefined but the
horizontal day is not, integrate diffuse/reflected over the horizontal
sunrise–sunset window with beam = insol = 0.

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
numerically identical to r.sun (<0.01% on controlled surfaces). Two
deliberate/known divergences remain: penumbra at shadow terminators (sun is
finer-grained), and the §5 nodata bug on never-sunlit slopes (to be fixed).
