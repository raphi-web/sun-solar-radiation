"""Regenerate the validation figures (maps.png, wall_profile.png).

Needs the GRASS reference outputs from grass_reference.sh in
/tmp/sun_vs_grass/grass_out and the installed engine
(pip install sun-solar-radiation).
"""
from pathlib import Path

import matplotlib
import numpy as np
from osgeo import gdal, osr

import sun

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

gdal.UseExceptions()
OUT_DIR = Path(__file__).resolve().parent
WORK = Path("/tmp/sun_vs_grass")
UNDEF = -9999.0


def read(p):
    ds = gdal.Open(str(p))
    a = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    ds = None
    return a


def flat(a):
    return np.ascontiguousarray(np.nan_to_num(a, nan=UNDEF).astype(np.float32)).ravel()


def masked(a):
    a = a.copy()
    a[(a == UNDEF) | ~np.isfinite(a)] = np.nan
    return a


def row_latitudes(path):
    ds = gdal.Open(str(path))
    gt, wkt = ds.GetGeoTransform(), ds.GetProjection()
    nrows, ncols = ds.RasterYSize, ds.RasterXSize
    ds = None
    src = osr.SpatialReference()
    src.ImportFromWkt(wkt)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(src, dst)
    x_c = gt[0] + (ncols / 2.0) * gt[1]
    lats = [tr.TransformPoint(x_c, gt[3] + (r + 0.5) * gt[5])[1] for r in range(nrows)]
    return np.array(lats, dtype=np.float32), nrows, ncols


def glob(dem, slope, aspect, lats, nrows, ncols, day):
    out = sun.compute_raster_bands(
        elevation=flat(dem), ncols=ncols, nrows=nrows, row_lat=lats, day=day, step=0.5,
        linke_value=3.0, albedo_value=0.2, dx_m=30.0, dy_m=30.0,
        slope=flat(slope), aspect=flat(aspect), outputs=["glob"], gpu=False,
        row_offset=0, full_nrows=nrows, quiet=True,
    )
    return np.asarray(out["glob"]).reshape(nrows, ncols)


# ── rough-terrain maps, day 355 ───────────────────────────────────────────
lats, nrows, ncols = row_latitudes(WORK / "dem.tif")
dem = read(WORK / "dem.tif")
mine = glob(dem, read(WORK / "grass_out/slope.tif"), read(WORK / "grass_out/aspect.tif"),
            lats, nrows, ncols, 355)
grass = read(WORK / "grass_out/g355.tif")

mn, gn = masked(mine), masked(grass)
only_sun = np.isnan(mn) & ~np.isnan(gn)
only_grass = ~np.isnan(mn) & np.isnan(gn)
print(f"nodata only in sun: {int(only_sun.sum())}, only in GRASS: "
      f"{int(only_grass.sum())}, in both: {int((np.isnan(mn) & np.isnan(gn)).sum())} "
      f"(DEM nodata column + r.slope.aspect's undefined border ring)")

fig = plt.figure(figsize=(15, 4.4), dpi=110)
ax0 = fig.add_subplot(1, 4, 1)
im0 = ax0.imshow(masked(dem), cmap="terrain")
ax0.set_title("Test DEM (200×200, UTM33N) [m]", fontsize=9)
ax0.axis("off")
plt.colorbar(im0, ax=ax0, fraction=0.046)
vmax = np.nanmax([np.nanpercentile(mn, 99), np.nanpercentile(gn, 99)])
for i, (arr, title) in enumerate(((mn, "sun: global day-355"),
                                  (gn, "GRASS r.sun: global day-355")), start=2):
    ax = fig.add_subplot(1, 4, i)
    im = ax.imshow(arr, cmap="inferno", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
d = mn - gn
dv = np.nanpercentile(np.abs(d), 99)
ax3 = fig.add_subplot(1, 4, 4)
im3 = ax3.imshow(d, cmap="RdBu_r", vmin=-dv, vmax=dv)
ax3.set_title("sun − GRASS (Wh/m²)", fontsize=9)
ax3.axis("off")
plt.colorbar(im3, ax=ax3, fraction=0.046)
plt.tight_layout()
plt.savefig(OUT_DIR / "maps.png", bbox_inches="tight")
plt.close()

# ── controlled wall shadow profile, day 355 ───────────────────────────────
wlats, wrows, wcols = row_latitudes(WORK / "wall.tif")
wmine = glob(read(WORK / "wall.tif"), read(WORK / "grass_out/wall_s.tif"),
             read(WORK / "grass_out/wall_a.tif"), wlats, wrows, wcols, 355)
wg_sh = read(WORK / "grass_out/wglob_sh.tif")
wg_ns = read(WORK / "grass_out/wglob_ns.tif")

col, rows = 100, np.arange(30, 152)
fig2, ax = plt.subplots(figsize=(7.5, 4.4), dpi=110)
ax.plot(rows, wg_ns[rows, col], "k--", lw=1.4, label="GRASS no-shadow (reference)")
ax.plot(rows, wg_sh[rows, col], "C1-", lw=1.8, label="GRASS r.sun (shadow on)")
ax.plot(rows, masked(wmine)[rows, col], "C0-", lw=1.8, label="sun (ray-march shadow)")
ax.axvline(150, color="0.4", ls=":", lw=1)
ax.text(150, 1560, " wall (1000 m)", fontsize=8, color="0.3", rotation=90, va="top")
ax.set_xlabel("grid row (wall at 150, shadow falls north → lower rows)")
ax.set_ylabel("global irradiation day-355 [Wh/m²/day]")
ax.set_title("Controlled wall shadow: winter solstice, sun due south", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
ax.set_xlim(30, 152)
ax.set_ylim(0, 1750)
plt.tight_layout()
plt.savefig(OUT_DIR / "wall_profile.png", bbox_inches="tight")
plt.close()
print(f"figures written to {OUT_DIR}")
