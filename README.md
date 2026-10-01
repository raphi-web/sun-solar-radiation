# sun

GPU-accelerated solar radiation modeling for digital elevation models.

## Features

- **Fast computation**: GPU acceleration via wgpu, with CPU fallback
- **Multiple output modes**:
  - Daily irradiation (beam, diffuse, reflected, global)
  - Sunshine duration
  - Annual solar potential
- **Flexible inputs**: DEM with optional slope, aspect, Linke turbidity, albedo, and mask rasters
- **QGIS integration**: Available as a QGIS plugin for interactive use

## Installation

### From PyPI

```bash
pip install sun-solar-radiation
```

### From source

Requires Rust 1.70+ and Python 3.12+.

```bash
git clone <repository-url>
cd sun
pip install maturin
maturin build --release
pip install target/wheels/sun_solar_radiation-*.whl
```

## Usage

### Python API

```python
import sun

# Compute daily irradiation for a single point
result = sun.compute_pixel(
    lat_deg=47.5,
    elevation_m=800.0,
    day=172,
    slope_deg=15.0,
    aspect_deg=180.0,  # south-facing
    linke=3.0,
    albedo=0.2
)

print(f"Global radiation: {result.global_rad:.1f} Wh/m²/day")
print(f"Sunshine duration: {result.insol_time:.1f} hours")

# Process a DEM raster
sun.compute_raster(
    elevation="dem.tif",
    day=172,
    glob_rad="output_global.tif",
    insol_time="output_sunshine.tif",
    gpu=True  # use GPU acceleration
)

# Compute annual solar potential
sun.compute_annual_potential(
    elevation="dem.tif",
    out_path="annual_potential.tif",
    day_start=1,
    day_end=365,
    day_step=10,
    panel_efficiency=0.20,
    gpu=True
)
```

### Command Line

```bash
# Process a DEM for day 172 (summer solstice)
sun --elevation dem.tif --day 172 --glob-rad output.tif --gpu

# Compute annual potential
sun --elevation dem.tif --annual --out-path annual.tif --gpu
```

## Performance

Typical runtimes on a modern laptop (Intel i7 + integrated GPU):

| DEM size | Mode | CPU time | GPU time |
|----------|------|----------|----------|
| 100×100 | Daily | 0.1s | 0.05s |
| 1000×1000 | Daily | 8s | 0.3s |
| 1600×1600 | Daily | 20s | 0.8s |
| 1000×1000 | Annual | 5min | 15s |

## Algorithm

Implements the GRASS GIS r.sun model for clear-sky solar radiation:
- Accounts for terrain shadowing and slope/aspect effects
- Computes beam, diffuse, and reflected radiation components
- Uses Linke turbidity coefficient for atmospheric attenuation
- Supports spatially varying inputs (slope, aspect, albedo, Linke, mask)

## Requirements

- Python 3.12+
- GDAL 3.8+ (for raster I/O)
- For GPU mode: Vulkan-compatible GPU (optional, falls back to CPU)

## License

MIT License - see LICENSE file for details.

## Citation

Based on the r.sun model from GRASS GIS:
- Hofierka, J., Suri, M. (2002): The solar radiation model for use in complex terrain.
