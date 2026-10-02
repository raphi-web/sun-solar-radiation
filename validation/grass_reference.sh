#!/bin/bash
# Run GRASS r.sun reference computations for the sun-vs-GRASS validation.
# Requires: GRASS 8.x (GISBASE below), and the test rasters produced by
# make_test_dem.py / compare_wall_shadow.py (wall.tif) to exist.
#
# Outputs GeoTIFFs to $OUT for every configuration the comparison scripts read:
#   dem run: slope/aspect maps, shadowed + shadow-free daily outputs (172, 355)
#   flat run: horizontal reference (172, 355)
#   ramp run: south-facing tilted plane via GRASS's own slope/aspect raster (355)
#   wall run: controlled shadow geometry (355)
set -e
export GISBASE=${GISBASE:-/usr/lib/grass83}
export PATH=$GISBASE/bin:$PATH
GD=/tmp/sun_vs_grass/grassdata
OUT=/tmp/sun_vs_grass/grass_out
mkdir -p "$OUT"

if [ ! -d "$GD/cmp" ]; then
  echo "==> creating GRASS location from DEM"
  grass --text -c /tmp/sun_vs_grass/dem.tif -e "$GD/cmp" > /dev/null 2>&1
fi

grass "$GD/cmp/PERMANENT" --exec bash -c "
set -e
r.in.gdal input=/tmp/sun_vs_grass/dem.tif output=dem --o -o > /dev/null 2>&1
r.slope.aspect elevation=dem slope=slope aspect=aspect --o > /dev/null 2>&1

# shadowed + shadow-free daily runs
for day in 172 355; do
  r.sun elevation=dem slope=slope aspect=aspect linke_value=3.0 albedo_value=0.2 \
    day=\$day step=0.5 beam_rad=b\$day diff_rad=d\$day refl_rad=r\$day glob_rad=g\$day insol_time=i\$day --o > /dev/null 2>&1
  r.sun -p elevation=dem slope=slope aspect=aspect linke_value=3.0 albedo_value=0.2 \
    day=\$day step=0.5 glob_rad=pg\$day beam_rad=pb\$day insol_time=pi\$day --o > /dev/null 2>&1
done

# horizontal reference (flat DEM, no slope)
r.mapcalc \"flat = 1000.0\" --o > /dev/null 2>&1
for day in 172 355; do
  r.sun -p elevation=flat slope_value=0 aspect_value=0 linke_value=3.0 albedo_value=0.2 \
    day=\$day step=0.5 beam_rad=fb\$day diff_rad=fd\$day glob_rad=fg\$day insol_time=fi\$day --o > /dev/null 2>&1
done

# south-facing ramp (tilted-plane physics via GRASS's own aspect raster)
r.mapcalc \"ramp = 1000.0 - (row() * 30.0 * 0.4)\" --o > /dev/null 2>&1
r.slope.aspect elevation=ramp slope=ramp_s aspect=ramp_a --o > /dev/null 2>&1
r.sun -p elevation=ramp slope=ramp_s aspect=ramp_a linke_value=3.0 albedo_value=0.2 \
  day=355 step=0.5 glob_rad=rampg --o > /dev/null 2>&1

# controlled wall shadow
r.in.gdal input=/tmp/sun_vs_grass/wall.tif output=wall --o > /dev/null 2>&1
r.slope.aspect elevation=wall slope=wall_s aspect=wall_a --o > /dev/null 2>&1
r.sun elevation=wall slope=wall_s aspect=wall_a linke_value=3.0 albedo_value=0.2 \
  day=355 step=0.5 glob_rad=wglob_sh --o > /dev/null 2>&1
r.sun -p elevation=wall slope=wall_s aspect=wall_a linke_value=3.0 albedo_value=0.2 \
  day=355 step=0.5 glob_rad=wglob_ns --o > /dev/null 2>&1

# export everything the comparison scripts need
for m in slope aspect b172 d172 r172 g172 i172 b355 d355 r355 g355 i355 \
         pg172 pb172 pi172 pg355 pb355 pi355 \
         fb172 fd172 fg172 fi172 fb355 fd355 fg355 fi355 \
         rampg ramp_s ramp_a wglob_sh wglob_ns wall_s wall_a; do
  r.out.gdal -f input=\$m output=$OUT/\$m.tif format=GTiff type=Float32 --o > /dev/null 2>&1
done
r.out.gdal -f input=ramp output=$OUT/ramp_elev.tif format=GTiff type=Float32 --o > /dev/null 2>&1
echo 'GRASS reference outputs written to $OUT'
"
