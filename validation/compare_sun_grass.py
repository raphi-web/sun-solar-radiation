
"""sun (array API, CPU + GPU) vs GRASS r.sun on the same DEM & parameters."""
import importlib.util, sys, time
from pathlib import Path
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

# load the extension from the plugin dir
pkg = Path("/home/raphi/Dokumente/Programming/Python/sun_qgis-plugin/sun_qgis")
spec = importlib.util.spec_from_file_location("sun_core", str(pkg / "core.py"))
core = importlib.util.module_from_spec(spec); sys.modules["sun_core"] = core
spec.loader.exec_module(core)
sun = core.load_sun(pkg)

def read(path):
    ds = gdal.Open(path)
    a = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    ds = None
    return a

dem = read("/tmp/sun_vs_grass/dem.tif")
slope_g = read("/tmp/sun_vs_grass/grass_out/slope.tif")
aspect_g = read("/tmp/sun_vs_grass/grass_out/aspect.tif")
nrows, ncols = dem.shape
UNDEF = -9999.0

# ---- latitudes per row (UTM 33N -> WGS84), same as the plugin pipeline ----
from osgeo import osr
ds = gdal.Open("/tmp/sun_vs_grass/dem.tif")
gt = ds.GetGeoTransform(); wkt = ds.GetProjection(); ds = None
srs = osr.SpatialReference(); srs.ImportFromWkt(wkt)
srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
tgt = osr.SpatialReference(); tgt.ImportFromEPSG(4326)
tgt.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
tr = osr.CoordinateTransformation(srs, tgt)
x_c = gt[0] + (ncols/2.0)*gt[1]
lats = np.array([tr.TransformPoint(x_c, gt[3] + (r+0.5)*gt[5])[1] for r in range(nrows)], dtype=np.float32)

def flat(a): return np.ascontiguousarray(np.nan_to_num(a, nan=UNDEF).astype(np.float32)).ravel()

elev_f = flat(dem)
slope_f = flat(slope_g)
aspect_f = flat(aspect_g)

def run_sun(day, gpu):
    t0 = time.perf_counter()
    out = sun.compute_raster_bands(
        elevation=elev_f, ncols=ncols, nrows=nrows, row_lat=lats,
        day=day, step=0.5, linke_value=3.0, albedo_value=0.2,
        dx_m=30.0, dy_m=30.0,
        slope=slope_f, aspect=aspect_f,           # GRASS's own slope/aspect!
        outputs=["glob","beam","diff","refl","insol"],
        gpu=gpu, row_offset=0, full_nrows=nrows, quiet=True)
    dt = time.perf_counter() - t0
    return {k: np.asarray(v).reshape(nrows, ncols).astype(np.float64) for k,v in out.items()}, dt

def compare(name, mine, grass):
    m = mine.ravel(); g = grass.ravel()
    nodata_m = m == UNDEF; nodata_g = (g == UNDEF) | np.isnan(g)
    valid = ~nodata_m & ~nodata_g
    mm, gg = m[valid], g[valid]
    # r.sun writes NULL where shadowed-out/nodata -> NaN through gdal; align both
    both_nan = np.isnan(mm) & np.isnan(gg)
    mm, gg = mm[~both_nan], gg[~both_nan]
    diff = mm - gg
    rel = np.abs(diff) / np.maximum(np.abs(gg), 1.0)
    corr = np.corrcoef(mm, gg)[0,1]
    return dict(name=name, n=valid.sum(),
                nodata_agree=float((nodata_m == nodata_g).mean()),
                bias=float(diff.mean()), rmse=float(np.sqrt((diff**2).mean())),
                mean_abs_rel=float(rel.mean()), p95_rel=float(np.percentile(rel,95)),
                corr=float(corr), mine_mean=float(mm.mean()), grass_mean=float(gg.mean()))

results = []
for day in (172, 355):
    grass_maps = {k: read(f"/tmp/sun_vs_grass/grass_out/{p}{day}.tif")
                  for k,p in [("glob","g"),("beam","b"),("diff","d"),("refl","r"),("insol","i")]}
    cpu, t_cpu = run_sun(day, False)
    gpu, t_gpu = run_sun(day, True)
    for comp, gk in [("glob","glob"),("beam","beam"),("diff","diff"),("refl","refl"),("insol","insol")]:
        results.append((f"day{day}", comp, compare(f"cpu-{comp}", cpu[comp], grass_maps[gk])))
        results.append((f"day{day}", comp, compare(f"gpu-{comp}", gpu[comp], grass_maps[gk])))
    print(f"day {day}: sun CPU {t_cpu:.2f}s, GPU {t_gpu:.2f}s")

print()
hdr = f"{'day':6} {'comp':6} {'engine':8} {'n':>7} {'nodata-agree':>12} {'bias':>9} {'RMSE':>9} {'meanRel%':>9} {'p95Rel%':>8} {'corr':>7} {'sun-mean':>9} {'grass-mean':>10}"
print(hdr); print("-"*len(hdr))
for day, comp, r in results:
    eng = r["name"].split("-")[0]
    print(f"{day:6} {comp:6} {eng:8} {r['n']:>7} {r['nodata_agree']*100:>11.2f}% "
          f"{r['bias']:>9.2f} {r['rmse']:>9.2f} {r['mean_abs_rel']*100:>8.2f}% "
          f"{r['p95_rel']*100:>7.2f}% {r['corr']:>7.5f} {r['mine_mean']:>9.1f} {r['grass_mean']:>10.1f}")

import json
json.dump(results, open("/tmp/sun_vs_grass/comparison.json","w"), indent=1)
print("\nsaved /tmp/sun_vs_grass/comparison.json")
