
import numpy as np
from osgeo import gdal, osr

np.random.seed(42)
ncols = nrows = 200
px = 30.0  # metres

x = np.linspace(0, 1, ncols)
y = np.linspace(0, 1, nrows)
xx, yy = np.meshgrid(x, y)

# Multi-octave ridged terrain: a main valley along a diagonal, ridges either side
base = 900.0 + 700.0 * np.abs(np.sin(3.1 * (xx * 0.7 + yy * 0.3) + 0.4)) ** 1.5
ridge = 250.0 * np.sin(6.2 * xx + 1.3) * np.cos(5.1 * yy + 0.7)
detail = 60.0 * np.sin(17.0 * xx) * np.sin(14.0 * yy) + 40.0 * np.cos(23.0 * yy - 9.0 * xx)
valley = -180.0 * np.exp(-((xx - yy) ** 2) / 0.02)   # valley trench along diagonal
elev = (base + ridge + detail + valley).astype(np.float32)

# a nodata border strip (col 0) to compare nodata handling too
elev[:, 0] = -9999.0

path = "/tmp/sun_vs_grass/dem.tif"
import os; os.makedirs("/tmp/sun_vs_grass", exist_ok=True)
drv = gdal.GetDriverByName("GTiff")
ds = drv.Create(path, ncols, nrows, 1, gdal.GDT_Float32)
# UTM 33N origin near Salzburg: E 400000, N 5250000
ds.SetGeoTransform((400000.0, px, 0.0, 5250000.0, 0.0, -px))
srs = osr.SpatialReference(); srs.ImportFromEPSG(32633)
ds.SetProjection(srs.ExportToWkt())
band = ds.GetRasterBand(1)
band.SetNoDataValue(-9999.0)
band.WriteArray(elev)
ds = band = None
v = elev[elev > -9000]
print(f"DEM written: {path}")
print(f"  {ncols}x{nrows} @ {px:.0f} m, EPSG:32633")
print(f"  elev {v.min():.0f}..{v.max():.0f} m, mean {v.mean():.0f} m")

# Controlled shadow geometry: flat 500 m plain + one 1000 m E-W wall at
# row 150 (cols 20-179), same grid and georeference as dem.tif.
wall = np.full((nrows, ncols), 500.0, dtype=np.float32)
wall[150, 20:180] = 1500.0
wpath = "/tmp/sun_vs_grass/wall.tif"
ds = drv.Create(wpath, ncols, nrows, 1, gdal.GDT_Float32)
ds.SetGeoTransform((400000.0, px, 0.0, 5250000.0, 0.0, -px))
ds.SetProjection(srs.ExportToWkt())
band = ds.GetRasterBand(1)
band.SetNoDataValue(-9999.0)
band.WriteArray(wall)
ds = band = None
print(f"wall written: {wpath} (1000 m wall at row 150)")
