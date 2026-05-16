#!/usr/bin/env python3
"""Mask raster correctness test (CPU + GPU).

Requires that ``tests/compare_with_grass.py`` has already produced the
unmasked baselines (``tests/output/cpu_glob.tif`` and a DEM at
``tests/output/dummy_elevation.tif``). The mask is a 100×100 binary raster
that switches off the bottom half plus a 10×10 hole inside the top half.

Asserts:
  * outside the mask, CPU and GPU emit the UNDEFZ nodata sentinel
  * inside the mask, the CPU result is bit-exact equal to the unmasked CPU run
  * inside the mask, GPU agrees with CPU within f32 noise (< 5 Wh/m²/day)

Run:  python3 tests/test_mask.py
"""

import os
import subprocess
import sys

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

UNDEFZ = -9999.0

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "output")
SUN_BIN = os.path.join(PROJECT_ROOT, "target", "release", "sun")

ELEV = os.path.join(OUT_DIR, "dummy_elevation.tif")
MASK = os.path.join(OUT_DIR, "mask.tif")

DAY = 172
STEP = 0.5


def out(name: str) -> str:
    return os.path.join(OUT_DIR, name)


def load(path: str) -> np.ndarray:
    ds = gdal.Open(path)
    return ds.GetRasterBand(1).ReadAsArray().astype(np.float64)


def make_mask():
    """Top half = 1 (compute), bottom half = 0 (skip); 10×10 hole at (20:30, 40:50)."""
    src = gdal.Open(ELEV)
    ncols, nrows = src.RasterXSize, src.RasterYSize
    gt = src.GetGeoTransform()
    proj = src.GetProjection()

    mask = np.ones((nrows, ncols), dtype=np.float32)
    mask[nrows // 2:, :] = 0.0
    mask[20:30, 40:50] = 0.0

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(MASK, ncols, nrows, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    ds.GetRasterBand(1).WriteArray(mask)
    ds.FlushCache()
    return int((mask == 1).sum()), int((mask == 0).sum())


def run_rust(label: str, *, gpu: bool):
    args = [
        SUN_BIN,
        "--elevation", ELEV,
        "--mask", MASK,
        "--day", str(DAY),
        "--step", str(STEP),
        "--glob-rad",  out(f"{label}_glob_masked.tif"),
        "--beam-rad",  out(f"{label}_beam_masked.tif"),
        "--diff-rad",  out(f"{label}_diff_masked.tif"),
        "--refl-rad",  out(f"{label}_refl_masked.tif"),
        "--insol-time", out(f"{label}_insol_masked.tif"),
    ]
    if gpu:
        args.append("--gpu")
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{label} run failed:\n{r.stderr}")


def main():
    if not os.path.exists(ELEV) or not os.path.exists(out("cpu_glob.tif")):
        sys.exit("Run tests/compare_with_grass.py first to generate the DEM "
                 "and the unmasked baseline.")

    on, off = make_mask()
    print(f"[mask]  {on} compute pixels, {off} skipped")

    print("[run]   cpu (masked)")
    run_rust("cpu", gpu=False)
    print("[run]   gpu (masked)")
    run_rust("gpu", gpu=True)

    mask     = load(MASK)
    cpu_full = load(out("cpu_glob.tif"))
    cpu_msk  = load(out("cpu_glob_masked.tif"))
    gpu_msk  = load(out("gpu_glob_masked.tif"))

    outside = mask == 0
    inside  = mask == 1

    assert np.all(cpu_msk[outside] == UNDEFZ), "CPU output not UNDEFZ outside mask"
    assert np.all(gpu_msk[outside] == UNDEFZ), "GPU output not UNDEFZ outside mask"
    print(f"[ok]    CPU/GPU correctly emit UNDEFZ for {outside.sum()} masked-off pixels")

    cpu_diff = np.abs(cpu_msk[inside] - cpu_full[inside])
    assert cpu_diff.max() == 0.0, (
        f"CPU masked output should be bit-exact for active pixels "
        f"(max diff = {cpu_diff.max()})")
    print(f"[ok]    CPU masked output bit-exact vs unmasked")

    gpu_diff = np.abs(gpu_msk[inside] - cpu_full[inside])
    assert gpu_diff.max() < 5.0, (
        f"GPU mask shouldn't change values (got {gpu_diff.max()})")
    print(f"[ok]    GPU vs CPU(unmasked) inside mask: "
          f"max diff = {gpu_diff.max():.4f} Wh/m²/day")

    gpu_vs_cpu = np.abs(gpu_msk[inside] - cpu_msk[inside])
    print(f"[ok]    GPU vs CPU(masked) inside mask:   "
          f"max diff = {gpu_vs_cpu.max():.4f}  "
          f"RMSE = {np.sqrt(np.mean(gpu_vs_cpu ** 2)):.4f}")

    print("\nAll mask assertions passed.")


if __name__ == "__main__":
    main()
