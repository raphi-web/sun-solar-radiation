#!/usr/bin/env python3
"""End-to-end accuracy check: Rust CPU vs Rust GPU vs GRASS r.sun.

Drives the full pipeline:
  1. Build the sun binary if needed
  2. Generate a dummy elevation raster
  3. Derive slope & aspect in GRASS (r.slope.aspect) and export as GeoTIFF
  4. Run Rust CPU & GPU using those slope/aspect rasters
  5. Run GRASS r.sun on the same DEM with the same slope/aspect
  6. Compare pixel-by-pixel statistics across all three

Using GRASS-derived slope/aspect as a common input isolates the radiation
model from terrain derivation, so any divergence reflects only the
radiation calculation itself (not Horn vs r.slope.aspect differences).

Run from the project root:    python3 tests/compare_with_grass.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

# ── Parameters (must match across all three runs) ──────────────────────────
DAY = 172              # Summer solstice
STEP = 0.5             # Time step [h]
LINKE_VALUE = 3.0
ALBEDO_VALUE = 0.2
SOLAR_CONSTANT = 1367.0

# ── Paths (relative to project root) ───────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "output")
SUN_BIN = os.path.join(PROJECT_ROOT, "target", "release", "sun")

ELEV = os.path.join(OUT_DIR, "dummy_elevation.tif")
SLOPE_TIF = os.path.join(OUT_DIR, "grass_slope.tif")
ASPECT_TIF = os.path.join(OUT_DIR, "grass_aspect.tif")


def out(name: str) -> str:
    return os.path.join(OUT_DIR, name)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# ── 1. Build & run the Rust binary ─────────────────────────────────────────


def ensure_binary():
    if not os.path.isfile(SUN_BIN):
        print("[build] cargo build --release ...")
        r = run(["cargo", "build", "--release"], cwd=PROJECT_ROOT)
        if r.returncode != 0:
            sys.exit(f"build failed:\n{r.stderr}")


def run_rust(label: str, *, gpu: bool, mask: str | None = None):
    args = [
        SUN_BIN,
        "--elevation", ELEV,
        "--slope", SLOPE_TIF,
        "--aspect", ASPECT_TIF,
        "--day", str(DAY),
        "--step", str(STEP),
        "--linke-value", str(LINKE_VALUE),
        "--albedo-value", str(ALBEDO_VALUE),
        "--solar-constant", str(SOLAR_CONSTANT),
        "--glob-rad",  out(f"{label}_glob.tif"),
        "--beam-rad",  out(f"{label}_beam.tif"),
        "--diff-rad",  out(f"{label}_diff.tif"),
        "--refl-rad",  out(f"{label}_refl.tif"),
        "--insol-time", out(f"{label}_insol.tif"),
    ]
    if gpu:
        args.append("--gpu")
    if mask is not None:
        args += ["--mask", mask]

    print(f"[run]   {label} ({'GPU' if gpu else 'CPU'})"
          f"{' + mask' if mask else ''}")
    r = run(args, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        sys.exit(f"  {label} run failed:\n{r.stderr}")


def create_dummy_dem():
    print(f"[dem]   {ELEV}")
    r = run(
        [SUN_BIN, "--create-dummy", "--day", str(DAY), "--elevation", ELEV],
        cwd=PROJECT_ROOT,
    )
    if r.returncode != 0:
        sys.exit(f"  DEM creation failed:\n{r.stderr}")


# ── 2. GRASS reference run ─────────────────────────────────────────────────


def run_grass_reference(tmpdir: str):
    location = os.path.join(tmpdir, "grassdb", "comparison")
    mapset = os.path.join(location, "PERMANENT")

    print("[grass] creating temporary location …")
    r = run(["grass", "-c", ELEV, location, "-e"])
    if r.returncode != 0:
        sys.exit(f"  grass -c failed:\n{r.stderr}")

    def in_grass(*cmd):
        return run(
            ["grass", mapset, "--exec", *cmd],
            env={**os.environ,
                 "GRASS_MESSAGE_FORMAT": "plain",
                 "GRASS_VERBOSE": "0"},
        )

    print("[grass] r.in.gdal")
    r = in_grass("r.in.gdal", f"input={ELEV}", "output=elevation", "--overwrite")
    if r.returncode != 0:
        sys.exit(f"  r.in.gdal failed:\n{r.stderr}")

    in_grass("g.region", "raster=elevation")

    print("[grass] r.slope.aspect")
    r = in_grass(
        "r.slope.aspect",
        "elevation=elevation",
        "slope=slope",
        "aspect=aspect",
        "--overwrite",
    )
    if r.returncode != 0:
        sys.exit(f"  r.slope.aspect failed:\n{r.stderr}")

    print("[grass] r.out.gdal (slope, aspect)")
    in_grass("r.out.gdal", "input=slope", f"output={SLOPE_TIF}",
             "format=GTiff", "type=Float32", "--overwrite")
    in_grass("r.out.gdal", "input=aspect", f"output={ASPECT_TIF}",
             "format=GTiff", "type=Float32", "--overwrite")

    print("[grass] r.sun")
    r = in_grass(
        "r.sun",
        "elevation=elevation",
        "slope=slope",
        "aspect=aspect",
        f"day={DAY}",
        f"step={STEP}",
        f"linke_value={LINKE_VALUE}",
        f"albedo_value={ALBEDO_VALUE}",
        f"solar_constant={SOLAR_CONSTANT}",
        "beam_rad=grass_beam",
        "diff_rad=grass_diff",
        "glob_rad=grass_glob",
        "refl_rad=grass_refl",
        "--overwrite",
        "-p",  # no shadow
    )
    if r.returncode != 0:
        sys.exit(f"  r.sun failed:\n{r.stderr}\nstdout:\n{r.stdout}")

    print("[grass] r.out.gdal (× 4)")
    for grass_name, fname in (
        ("grass_beam", "grass_beam.tif"),
        ("grass_diff", "grass_diff.tif"),
        ("grass_glob", "grass_glob.tif"),
        ("grass_refl", "grass_refl.tif"),
    ):
        in_grass("r.out.gdal", f"input={grass_name}",
                 f"output={out(fname)}", "format=GTiff",
                 "type=Float32", "--overwrite")


# ── 3. Comparison ──────────────────────────────────────────────────────────


def load(path: str) -> np.ndarray:
    ds = gdal.Open(path)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float64)
    nd = band.GetNoDataValue()
    if nd is not None:
        arr = np.where(arr == nd, np.nan, arr)
    return arr


def compare(label: str, a_name: str, b_name: str,
            a_path: str, b_path: str):
    a, b = load(a_path), load(b_path)
    m = ~(np.isnan(a) | np.isnan(b))
    diff = a[m] - b[m]
    rel = diff / (np.abs(b[m]) + 1e-10) * 100.0
    print(f"  {label:8s}"
          f"  {a_name} mean={np.mean(a[m]):8.2f}"
          f"  {b_name} mean={np.mean(b[m]):8.2f}"
          f"  RMSE={np.sqrt(np.mean(diff**2)):7.3f}"
          f"  max|rel|={np.max(np.abs(rel)):.4f}%")


def report():
    print("\n=== Rust CPU vs GRASS ===")
    for label in ("beam", "diff", "refl", "glob"):
        compare(label, "rust", "grass",
                out(f"cpu_{label}.tif"), out(f"grass_{label}.tif"))

    print("\n=== Rust GPU vs GRASS ===")
    for label in ("beam", "diff", "refl", "glob"):
        compare(label, "rust", "grass",
                out(f"gpu_{label}.tif"), out(f"grass_{label}.tif"))

    print("\n=== Rust GPU vs Rust CPU ===")
    for label in ("beam", "diff", "refl", "glob"):
        compare(label, "gpu", "cpu",
                out(f"gpu_{label}.tif"), out(f"cpu_{label}.tif"))


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ensure_binary()
    create_dummy_dem()

    # Run GRASS first: produces slope/aspect rasters that the Rust runs reuse,
    # so all three implementations integrate over identical terrain inputs.
    tmpdir = tempfile.mkdtemp(prefix="grass_compare_")
    try:
        run_grass_reference(tmpdir)
        run_rust("cpu", gpu=False)
        run_rust("gpu", gpu=True)
        report()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print(f"r.sun comparison  |  day={DAY} step={STEP}h "
          f"TL={LINKE_VALUE} albedo={ALBEDO_VALUE}")
    print("=" * 60)
    main()
