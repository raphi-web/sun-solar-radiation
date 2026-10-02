
#!/bin/bash
set -e
export GISBASE=/usr/lib/grass83
export PATH=$GISBASE/bin:$PATH
GD=/tmp/sun_vs_grass/grassdata

grass "$GD/cmp/PERMANENT" --exec bash -c '
set -e
r.sun elevation=dem slope=slope aspect=aspect linke_value=3.0 albedo_value=0.2 \
  day=172 step=0.5 \
  beam_rad=b172 diff_rad=d172 refl_rad=r172 glob_rad=g172 insol_time=i172 --o 2>&1 | grep -v "^ *[0-9]*%" | tail -1
echo "--- day 172 done ---"
r.sun elevation=dem slope=slope aspect=aspect linke_value=3.0 albedo_value=0.2 \
  day=355 step=0.5 \
  beam_rad=b355 diff_rad=d355 refl_rad=r355 glob_rad=g355 insol_time=i355 --o 2>&1 | grep -v "^ *[0-9]*%" | tail -1
echo "--- day 355 done ---"
mkdir -p /tmp/sun_vs_grass/grass_out
for m in b172 d172 r172 g172 i172 b355 d355 r355 g355 i355 slope aspect; do
  r.out.gdal input=$m output=/tmp/sun_vs_grass/grass_out/$m.tif format=GTiff type=Float32 --o > /dev/null 2>&1
done
ls /tmp/sun_vs_grass/grass_out/ | tr "\n" " "
echo ""
'
