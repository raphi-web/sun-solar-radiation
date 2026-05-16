"""A/B compare horizon-precompute vs ray-march on the real test-tile DSM.

Uses /home/raphi/Dokumente/Programming/Python/Housing/test-site/test-tile/als.tif
(243x259 ALS DSM at 1 m, ~40 m relief). Runs the ray-march reference once
and the horizon path at several n_az values, reporting timing and diff
against the reference.
"""

from __future__ import annotations

import os
import time

import numpy as np
import rasterio

import sun

DSM = "/home/raphi/Dokumente/Programming/Python/Housing/test-site/test-tile/als.tif"
TMP = "/dev/shm"
OUT_REF = f"{TMP}/horizon_real_ref.tif"

DAY_STEP = 14  # matches make_solar.py


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
        elevation=DSM,
        out_path=out_path,
        day_step=DAY_STEP,
        step=0.5,
        panel_efficiency=1.0,
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
    print(f"DSM:            {DSM}")
    with rasterio.open(DSM) as ds:
        nr, nc = ds.shape
        print(f"Shape:          {nr}x{nc} = {nr*nc} pixels")
    print(f"Settings:       day_step={DAY_STEP}, step=0.5")
    print()

    print("Running ray-march reference...")
    t_ref = run(OUT_REF, use_horizon=False)
    ref = read_band(OUT_REF)

    results = {}
    for n in (16, 32, 64, 128):
        out = f"{TMP}/horizon_real_hz{n}.tif"
        print(f"Running horizon n={n}...")
        t = run(out, use_horizon=True, horizon_n_az=n)
        arr = read_band(out)
        results[n] = (t, out, diffs(ref, arr))

    print()
    cols = f"{'mode':<14}{'time [s]':>10}{'speedup':>9}{'signed':>11}{'mean abs':>11}{'max abs':>10}{'rel mean':>10}"
    print(cols)
    print("-" * len(cols))
    print(f"{'ray-march':<14}{t_ref:>10.3f}{'1.00x':>9}{'—':>11}{'—':>11}{'—':>10}{'—':>10}")
    for n in sorted(results):
        t, _, d = results[n]
        print(
            f"{'horizon n=' + str(n):<14}{t:>10.3f}{t_ref/t:>8.2f}x"
            f"{d['mean_signed']:>11.3f}{d['mean_abs']:>11.3f}{d['max_abs']:>10.3f}"
            f"{d['rel_mean']*100:>9.3f}%"
        )
    print()
    print(f"Valid pixels: {results[16][2]['n_valid']}")

    cleanup = [OUT_REF] + [p for _, p, _ in results.values()]
    for p in cleanup:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
