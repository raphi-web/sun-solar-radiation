
#!/bin/bash
set -e
export GISBASE=/usr/lib/grass83
export PATH=$GISBASE/bin:$PATH
GD=/tmp/sun_vs_grass/grassdata

grass "$GD/cmp/PERMANENT" --exec bash -c '
set -e
# -s = cast shadows from terrain ON (matches the sun tool behaviour)
r.sun -s elevation=dem slope=slope aspect=aspect linke_value=3.0 albedo_value=0.2 \
  day=172 step=0.5 \
  beam_rad=b172s diff_rad=d172s refl_rad=r172s glob_rad=g172s insol_time=i172s --o > /dev/null 2>&1
echo "--- day 172 (with shadows) done ---"
r.sun -s elevation=dem slope=slope aspect=aspect linke_value=3.0 albedo_value=0.2 \
  day=355 step=0.5 \
  beam_rad=b355s diff_rad=d355s refl_rad=r355s glob_rad=g355s insol_time=i355s --o > /dev/null 2>&1
echo "--- day 355 (with shadows) done ---"
mkdir -p /tmp/sun_vs_grass/grass_out
for m in b172s d172s r172s g172s i172s b355s d355s r355s g355s i355s; do
  r.out.gdal input=$m output=/tmp/sun_vs_grass/grass_out/$m.tif format=GTiff type=Float32 --o > /dev/null 2>&1
done
ls /tmp/sun_vs_grass/grass_out/*s.tif | wc -l
'
