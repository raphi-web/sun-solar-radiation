
"""Controlled wall-shadow geometry: how far north does each engine's winter
shadow reach? Wall at row 150 (1000 m tall), sun due south => shadow to the
north (rows < 150). Compare the shadow footprint of sun vs r.sun."""
import importlib.util, sys
from pathlib import Path
import numpy as np
from osgeo import gdal, osr
gdal.UseExceptions()
pkg=Path("/home/raphi/Dokumente/Programming/Python/sun_qgis-plugin/sun_qgis")
spec=importlib.util.spec_from_file_location("sun_core",str(pkg/"core.py"))
core=importlib.util.module_from_spec(spec); sys.modules["sun_core"]=core; spec.loader.exec_module(core)
sun=core.load_sun(pkg)
def read(p):
    ds=gdal.Open(p); a=ds.GetRasterBand(1).ReadAsArray().astype(np.float64); ds=None; return a
ds=gdal.Open("/tmp/sun_vs_grass/wall.tif"); gt=ds.GetGeoTransform(); wkt=ds.GetProjection()
nrows,ncols=ds.RasterYSize,ds.RasterXSize; ds=None
srs=osr.SpatialReference(); srs.ImportFromWkt(wkt); srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
tgt=osr.SpatialReference(); tgt.ImportFromEPSG(4326); tgt.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
tr=osr.CoordinateTransformation(srs,tgt); x_c=gt[0]+(ncols/2.0)*gt[1]
lats=np.array([tr.TransformPoint(x_c,gt[3]+(r+0.5)*gt[5])[1] for r in range(nrows)],dtype=np.float32)
wall=read("/tmp/sun_vs_grass/wall.tif"); UNDEF=-9999.0
def fa(a): return np.ascontiguousarray(np.nan_to_num(a,nan=UNDEF).astype(np.float32)).ravel()
# my engine, shadows ON, GRASS-derived slope/aspect
ws=read("/tmp/sun_vs_grass/grass_out/wall_s.tif"); wa=read("/tmp/sun_vs_grass/grass_out/wall_a.tif")
out=sun.compute_raster_bands(elevation=fa(wall),ncols=ncols,nrows=nrows,row_lat=lats,day=355,step=0.5,
    linke_value=3.0,albedo_value=0.2,dx_m=30.0,dy_m=30.0,slope=fa(ws),aspect=fa(wa),
    outputs=["glob","insol"],gpu=False,row_offset=0,full_nrows=nrows,quiet=True)
mine=np.asarray(out["glob"]).reshape(nrows,ncols)
mine_ins=np.asarray(out["insol"]).reshape(nrows,ncols)
g_sh=read("/tmp/sun_vs_grass/grass_out/wglob_sh.tif")
g_ns=read("/tmp/sun_vs_grass/grass_out/wglob_ns.tif")

col=100  # middle column, away from wall ends
print("row  elev   sun-glob  grass-glob(shadow)  grass-glob(noshadow)")
for row in [40,60,80,100,120,140,149,150,151,160,180]:
    print(f"{row:>3} {wall[row,col]:>7.0f} {mine[row,col]:>9.0f} {g_sh[row,col]:>18.0f} {g_ns[row,col]:>20.0f}")

# quantify: relative to the no-shadow reference, how much flux is removed per row?
print("\nShadow depth (1 - shadow/noshadow) along col 100, north of wall:")
print(f"{'row':>4} {'sun':>7} {'grass':>7}")
for row in [60,80,100,120,140,148]:
    ds_ = 1 - mine[row,col]/g_ns[row,col] if g_ns[row,col] else 0
    dg_ = 1 - g_sh[row,col]/g_ns[row,col] if g_ns[row,col] else 0
    print(f"{row:>4} {ds_*100:>6.0f}% {dg_*100:>6.0f}%")

# insolation hours north of wall
print("\nInsolation hours (max possible ~8h at winter solstice, lat 47.4):")
print(f"{'row':>4} {'sun':>6} {'grass?':>8}")
gi=read("/tmp/sun_vs_grass/grass_out/wglob_sh.tif")  # placeholder
for row in [80,100,120,140,160]:
    print(f"{row:>4} {mine_ins[row,col]:>6.2f}")
