"""A/B compare the horizon-precompute CPU path against the ray-march CPU path.

Builds the synthetic 100x100 alpine DEM via ``sun.create_dummy``, runs
``compute_annual_potential`` three times (ray-march reference, horizon n_az=16,
horizon n_az=32), and reports timing + accuracy.

Outputs go to /dev/shm and are removed afterwards.
"""

from __future__ import annotations

import os
import time

import numpy as np
import rasterio

import sun

TMP = "/dev/shm"
DEM = f"{TMP}/horizon_dem.tif"
OUT_REF = f"{TMP}/horizon_ref.tif"
OUT_HZ16 = f"{TMP}/horizon_hz16.tif"
OUT_HZ32 = f"{TMP}/horizon_hz32.tif"


def read_band(path: str) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float64)
        nd = ds.nodatavals[0]
    if nd is not None:
        arr = np.where(arr == nd, np.nan, arr)
    return arr


def run(out_path: str, **kwargs) -> float:
    t0 = time.perf_counter()
    sun.compute_annual_potential(
        elevation=DEM,
        out_path=out_path,
        day_step=10,
        step=0.5,
        quiet=True,
        **kwargs,
    )
    return time.perf_counter() - t0


def diffs(ref: np.ndarray, alt: np.ndarray) -> dict:
    valid = ~(np.isnan(ref) | np.isnan(alt))
    d = alt[valid] - ref[valid]
    return {
        "n_valid": int(valid.sum()),
        "mean_signed": float(np.mean(d)),
        "mean_abs": float(np.mean(np.abs(d))),
        "max_abs": float(np.max(np.abs(d))),
        "rel_mean": float(np.mean(np.abs(d)) / np.mean(np.abs(ref[valid]))),
    }


def main() -> None:
    sun.create_dummy(DEM)

    t_ref = run(OUT_REF, use_horizon=False)
    runs = {}
    for n in (16, 32, 64, 128):
        out = f"{TMP}/horizon_hz{n}.tif"
        runs[n] = (run(out, use_horizon=True, horizon_n_az=n), out)

    ref = read_band(OUT_REF)
    samples = {n: (t, read_band(p)) for n, (t, p) in runs.items()}
    results = {n: (t, diffs(ref, arr)) for n, (t, arr) in samples.items()}

    print(f"DEM:                100x100 synthetic alpine (sun.create_dummy)")
    print(f"Settings:           day_step=10, step=0.5")
    print()
    cols = f"{'mode':<14}{'time [s]':>10}{'speedup':>9}{'signed':>11}{'mean abs':>11}{'max abs':>10}{'rel mean':>10}"
    print(cols)
    print("-" * len(cols))
    print(f"{'ray-march':<14}{t_ref:>10.3f}{'1.00x':>9}{'—':>11}{'—':>11}{'—':>10}{'—':>10}")
    for n in sorted(results):
        t, d = results[n]
        print(
            f"{'horizon n=' + str(n):<14}{t:>10.3f}{t_ref/t:>8.2f}x"
            f"{d['mean_signed']:>11.3f}{d['mean_abs']:>11.3f}{d['max_abs']:>10.3f}"
            f"{d['rel_mean']*100:>9.3f}%"
        )
    print()
    print(f"Valid pixels: {results[16][1]['n_valid']}")

    cleanup = [DEM, OUT_REF] + [p for _, p in runs.values()]
    for p in cleanup:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
