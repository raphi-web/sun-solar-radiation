#!/usr/bin/env python3
"""Speed comparison: Rust CPU vs WebGPU compute path.

Calls ``sun.compute_raster()`` 100× with ``gpu=False`` and 100× with ``gpu=True``,
measuring pure compute time (no subprocess / no binary startup cost).

Prerequisites:
    pip install --user maturin
    maturin build --release && pip install --force-reinstall target/wheels/sun-*.whl
    # or: maturin develop --release   (inside a virtualenv)

Run:
    python3 tests/speed_test.py [iterations]
"""

import contextlib
import os
import statistics
import sys
import time

import sun


@contextlib.contextmanager
def silence_stderr():
    """Redirect the OS-level stderr fd (so Rust eprintln! output is also gone)."""
    saved_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull)
        os.close(saved_fd)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "output")

DEM = os.path.join(OUT_DIR, "dummy_elevation.tif")


# Per-iteration outputs (overwritten each call; we don't read them)
def outs(label: str) -> dict:
    return {
        "glob_rad": os.path.join(OUT_DIR, f"speed_{label}_glob.tif"),
        "beam_rad": os.path.join(OUT_DIR, f"speed_{label}_beam.tif"),
        "diff_rad": os.path.join(OUT_DIR, f"speed_{label}_diff.tif"),
        "refl_rad": os.path.join(OUT_DIR, f"speed_{label}_refl.tif"),
        "insol_time": os.path.join(OUT_DIR, f"speed_{label}_insol.tif"),
    }


DAY = 172
STEP = 0.5


def ensure_dem():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(DEM):
        print(f"[setup] creating dummy DEM at {DEM}")
        sun.create_dummy(DEM)


def time_runs(label: str, iterations: int, *, gpu: bool) -> list[float]:
    out = outs(label)
    samples: list[float] = []
    with silence_stderr():
        # Warm-up: first GPU call pays device init + pipeline compile.
        sun.compute_raster(DEM, DAY, step=STEP, gpu=gpu, **out)

        for _ in range(iterations):
            t0 = time.perf_counter()
            sun.compute_raster(DEM, DAY, step=STEP, gpu=gpu, **out)
            samples.append(time.perf_counter() - t0)

    for i in range(0, iterations, 20):
        chunk = samples[i : i + 20]
        print(
            f"  [{label}] {i + len(chunk):>3}/{iterations}  "
            f"median {statistics.median(chunk) * 1e3:.2f} ms"
        )
    return samples


def stats(label: str, samples: list[float]) -> None:
    ms = [s * 1e3 for s in samples]
    print(f"\n  {label}")
    print(f"    min    : {min(ms):8.3f} ms")
    print(f"    median : {statistics.median(ms):8.3f} ms")
    print(f"    mean   : {statistics.mean(ms):8.3f} ms")
    print(f"    stdev  : {statistics.stdev(ms):8.3f} ms")
    print(f"    max    : {max(ms):8.3f} ms")
    print(f"    total  : {sum(ms):8.3f} ms  ({len(ms)} runs)")


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    ensure_dem()

    print("=" * 60)
    print(
        f"r.sun speed test  |  {iterations} iterations  |  "
        f"100×100 DEM  |  day={DAY} step={STEP}h"
    )
    print("(one warm-up call per backend; not counted in samples)")
    print("=" * 60)

    print("\n[cpu] timing …")
    cpu = time_runs("cpu", iterations, gpu=False)

    print("\n[gpu] timing …")
    gpu = time_runs("gpu", iterations, gpu=True)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    stats("CPU", cpu)
    stats("GPU", gpu)

    cpu_med = statistics.median(cpu)
    gpu_med = statistics.median(gpu)
    if gpu_med < cpu_med:
        print(f"\n  → GPU is {cpu_med / gpu_med:.2f}× faster than CPU (median)")
    else:
        print(f"\n  → CPU is {gpu_med / cpu_med:.2f}× faster than GPU (median)")
        print(
            "    (Expected on small rasters: per-call GPU dispatch + buffer "
            "upload/readback overhead dominates.)"
        )


if __name__ == "__main__":
    main()
