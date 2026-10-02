
"""Ramp test: south-facing plane, aspect from GRASS's OWN raster (both engines
read the identical raster). This is the apples-to-apples tilted-plane test —
the scalar aspect_value convention difference does not apply here."""
import importlib.util, sys
from pathlib import Path
import numpy as np
from osgeo import gdal, osr
gdal.UseExceptions()

pkg = Path("/home/raphi/Dokumente/Programming/Python/sun_qgis-plugin/sun_qgis")
spec = importlib.util.spec_from_file_location("sun_core", str(pkg / "core.py"))
core = importlib.util.module_from_spec(spec); sys.modules["sun_core"] = core
spec.loader.exec_module(core)
sun = core.load_sun(pkg)

def read(p):
    ds = gdal.Open(p); a = ds.GetRasterBand(1).ReadAsArray().astype(np.float64); ds=None; return a

# ramp is EPSG:32633 in the same location; get its georeference
ds = gdal.Open("/tmp/sun_vs_grass/grass_out/ramp_elev.tif")
gt = ds.GetGeoTransform(); wkt = ds.GetProjection()
nrows, ncols = ds.RasterYSize, ds.RasterXSize; ds=None
print(f"ramp grid {ncols}x{nrows}, gt0={gt[0]} gt3={gt[3]} px={gt[1]}")

srs = osr.SpatialReference(); srs.ImportFromWkt(wkt); srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
tgt = osr.SpatialReference(); tgt.ImportFromEPSG(4326); tgt.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
tr = osr.CoordinateTransformation(srs, tgt)
x_c = gt[0]+(ncols/2.0)*gt[1]
lats = np.array([tr.TransformPoint(x_c, gt[3]+(r+0.5)*gt[5])[1] for r in range(nrows)], dtype=np.float32)

ramp_e = read("/tmp/sun_vs_grass/grass_out/ramp_elev.tif")
ramp_s = read("/tmp/sun_vs_grass/grass_out/ramp_s.tif")
ramp_a = read("/tmp/sun_vs_grass/grass_out/ramp_a.tif")
ramp_g = read("/tmp/sun_vs_grass/grass_out/rampg.tif")
UNDEF=-9999.0
def flat_a(a): return np.ascontiguousarray(np.nan_to_num(a, nan=UNDEF).astype(np.float32)).ravel()

print(f"slope raster ~ {np.nanmean(ramp_s[10:-10,10:-10]):.1f} deg, aspect raster ~ {np.nanmean(ramp_a[10:-10,10:-10]):.0f} (270=south CCW-from-east)")

out = sun.compute_raster_bands(elevation=flat_a(ramp_e), ncols=ncols, nrows=nrows, row_lat=lats,
    day=355, step=0.5, linke_value=3.0, albedo_value=0.2, dx_m=30.0, dy_m=30.0,
    slope=flat_a(ramp_s), aspect=flat_a(ramp_a),
    outputs=["glob","beam","insol"], gpu=False, row_offset=0, full_nrows=nrows, quiet=True)
mine = np.asarray(out["glob"]).reshape(nrows,ncols)
ok = (mine!=UNDEF)&~np.isnan(mine)&~np.isnan(ramp_g)&(ramp_g!=UNDEF)
ms, gs = mine[ok].mean(), ramp_g[ok].mean()
print(f"\nSOUTH-FACING RAMP, day 355 (winter), shadows off both sides:")
print(f"  sun glob   = {ms:.1f} Wh/m2/day")
print(f"  grass glob = {gs:.1f} Wh/m2/day")
print(f"  rel diff   = {(ms-gs)/gs*100:+.2f}%   corr = {np.corrcoef(mine[ok], ramp_g[ok])[0,1]:.5f}")
