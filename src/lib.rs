// PyO3 0.22 proc-macros expand to a few `.into()` calls that clippy flags as
// useless conversions; silence them crate-wide rather than per-function.
#![allow(clippy::useless_conversion)]

pub mod arrays;
#[cfg(feature = "gdal-io")]
pub mod annual;
#[cfg(feature = "gdal-io")]
pub mod gpu;
pub mod gpu_array;
pub mod horizon;
pub mod radiation;
pub mod shadow;
/// r.sun Rust port — Library crate
///
/// Exposes the r.sun solar irradiation model as both a Rust library and
/// a Python extension module (via PyO3).
///
/// # Python usage
/// ```python
/// import sun
///
/// # Single-pixel computation
/// result = sun.compute_pixel(lat_deg=47.5, elevation_m=800.0, day=172)
/// print(f"Global: {result.global_rad:.0f} Wh/m²/day")
///
/// # Full raster computation
/// sun.compute_raster(elevation="dem.tif", day=172, glob_rad="glob.tif")
/// ```
pub mod solar;
pub(crate) mod terrain;

#[cfg(feature = "gdal-io")]
use gdal::raster::Buffer;
#[cfg(feature = "gdal-io")]
use gdal::spatial_ref::{AxisMappingStrategy, CoordTransform, SpatialRef};
#[cfg(feature = "gdal-io")]
use gdal::{Dataset, DriverManager};
use pyo3::prelude::*;
#[cfg(feature = "gdal-io")]
use rayon::prelude::*;
#[cfg(feature = "gdal-io")]
use std::path::Path;
#[cfg(feature = "gdal-io")]
use std::sync::atomic::{AtomicUsize, Ordering};

use radiation::{DailyIrradiation, RadiationParams, integrate_daily};
use solar::{DEG2RAD, com_declin, com_sol_const, compute_slope_geometry, horizontal_day_window};
#[cfg(feature = "gdal-io")]
use solar::RAD2DEG;

/// Nodata sentinel value matching GRASS r.sun convention.
pub const UNDEFZ: f32 = -9999.0;

// ── Shared helpers ──────────────────────────────────────────────────────────

/// Convert GRASS GIS aspect (degrees, 0=E CCW) to r.sun radians (CW from North).
///
/// GRASS aspect convention: 0=East, 90=North, 180=West, 270=South (CCW from East).
/// r.sun internal convention: compass bearing, North=0, East=90°, South=180°.
pub fn convert_grass_aspect(aspect_deg: f64) -> f64 {
    // GRASS raster convention: degrees CCW from East (0=E, 90=N, 180=W,
    // 270=S). Internal convention: radians CW-ish with 0=N, PI/2=E, PI=S,
    // 3PI/2=W. NOTE: aspect 0 (due East) must NOT be special-cased — flat
    // pixels (r.slope.aspect also emits 0 there) are handled by the
    // slope==0 branch in compute_slope_geometry, which ignores aspect.
    // A historical `if aspect_deg == 0.0 { return 0.0; }` here silently
    // computed East-facing slopes as North-facing (GRASS parity test caught
    // it: East winter beam was 0 instead of ~1136 Wh/m²/day).
    let converted = if aspect_deg < 90.0 {
        90.0 - aspect_deg
    } else {
        450.0 - aspect_deg
    };
    converted * DEG2RAD
}

/// Compute daily solar irradiation for a single geographic point.
///
/// Returns `None` if the sun never rises over this slope on the given day.
///
/// # Arguments
/// * `lat_deg`     - geographic latitude [°]
/// * `elevation_m` - terrain elevation [m]
/// * `slope_deg`   - terrain slope [°]
/// * `aspect_deg`  - terrain aspect [°, GRASS convention: 0=E CCW]
/// * `linke`       - Linke atmospheric turbidity factor (typical 2–5)
/// * `albedo`      - ground albedo (0–1, typical 0.2)
/// * `day`         - day of year (1–365)
/// * `step`        - time integration step [hours]
/// * `solar_constant` - solar constant [W/m²], default 1367
#[allow(clippy::too_many_arguments)]
pub fn compute_pixel_irradiation(
    lat_deg: f64,
    elevation_m: f64,
    slope_deg: f64,
    aspect_deg: f64,
    linke: f64,
    albedo: f64,
    day: i32,
    step: f64,
    solar_constant: f64,
) -> Option<DailyIrradiation> {
    if !(1..=365).contains(&day) || step <= 0.0 {
        return None;
    }

    let lat = lat_deg * DEG2RAD;
    let declination = com_declin(day);
    let g_norm_extra = com_sol_const(day, solar_constant);
    let sindecl = declination.sin();
    let cosdecl = declination.cos();

    let slope_rad = slope_deg * DEG2RAD;
    let aspect_rad = convert_grass_aspect(aspect_deg);

    let geom = compute_slope_geometry(slope_rad, aspect_rad, lat, sindecl, cosdecl);
    // Horizontal-day window (GRASS parity): never-sunlit slopes keep
    // diffuse+reflected; beam/insolation stay 0 via the s0 > 0 gate.
    let (sunrise, sunset) = horizontal_day_window(lat, sindecl, cosdecl)?;

    let params = RadiationParams {
        g_norm_extra,
        linke,
        albedo,
        cbh: 1.0,
        cdh: 1.0,
        elevation: elevation_m,
        solar_altitude: 0.0,
        slope: slope_rad,
        aspect: aspect_rad,
        solar_azimuth: 0.0,
    };

    Some(integrate_daily(
        &geom,
        &params,
        sunrise,
        sunset,
        step,
        lat,
        declination,
        None,
    ))
}

/// Read elevation band as `Vec<f32>` with band-declared nodata (and NaN) mapped
/// to [`UNDEFZ`], so downstream guards (Horn neighborhood check, per-pixel
/// `elev == UNDEFZ`) catch the input file's actual nodata sentinel rather than
/// only the hard-coded `-9999`.
#[cfg(feature = "gdal-io")]
pub(crate) fn read_elev_normalized(
    ds: &Dataset,
    ncols: usize,
    nrows: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let band = ds.rasterband(1)?;
    let nodata = band.no_data_value();
    let buf: Buffer<f32> = band.read_as((0, 0), (ncols, nrows), (ncols, nrows), None)?;
    let mut data = buf.data().to_vec();
    let nodata_f32 = nodata.map(|v| v as f32);
    for v in data.iter_mut() {
        if v.is_nan() || nodata_f32.is_some_and(|nd| *v == nd) {
            *v = UNDEFZ;
        }
    }
    Ok(data)
}

/// Read a single-band raster as `Vec<f32>`, or fill with `constant` if `path` is `None`.
#[cfg(feature = "gdal-io")]
pub(crate) fn read_raster_or_constant(
    path: &Option<String>,
    ncols: usize,
    nrows: usize,
    constant: f32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    match path {
        None => Ok(vec![constant; ncols * nrows]),
        Some(p) => {
            let ds = Dataset::open(Path::new(p))?;
            let band = ds.rasterband(1)?;
            let buf: Buffer<f32> = band.read_as((0, 0), (ncols, nrows), (ncols, nrows), None)?;
            Ok(buf.data().to_vec())
        }
    }
}

/// Resolve slope and aspect inputs with precedence: raster path > scalar > derive-from-DEM.
///
/// Each axis is resolved independently — passing only `slope=` or only
/// `aspect_value=` is fine. If both are absent for a given axis, that axis is
/// derived from `elev_data` via Horn's 3×3 finite-difference. Output aspect is
/// always in GRASS convention (0=E CCW), matching the existing raster path.
#[cfg(feature = "gdal-io")]
pub(crate) fn resolve_slope_aspect(
    slope_path: &Option<String>,
    slope_value: Option<f64>,
    aspect_path: &Option<String>,
    aspect_value: Option<f64>,
    elev_data: &[f32],
    ncols: usize,
    nrows: usize,
    gt: &[f64; 6],
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let need_derive_slope = slope_path.is_none() && slope_value.is_none();
    let need_derive_aspect = aspect_path.is_none() && aspect_value.is_none();

    let derived = if need_derive_slope || need_derive_aspect {
        Some(terrain::horn_slope_aspect(elev_data, ncols, nrows, gt))
    } else {
        None
    };

    let slope = match (slope_path, slope_value) {
        (Some(_), _) => read_raster_or_constant(slope_path, ncols, nrows, 0.0)?,
        (None, Some(v)) => vec![v as f32; ncols * nrows],
        (None, None) => derived.as_ref().unwrap().0.clone(),
    };
    let aspect = match (aspect_path, aspect_value) {
        (Some(_), _) => read_raster_or_constant(aspect_path, ncols, nrows, 0.0)?,
        (None, Some(v)) => vec![v as f32; ncols * nrows],
        (None, None) => derived.as_ref().unwrap().1.clone(),
    };
    Ok((slope, aspect))
}

/// Pixel-row-centre latitudes in WGS84 degrees, one entry per raster row.
///
/// For geographic CRSes the geo-transform already yields latitude directly.
/// For projected CRSes we sample the column-centre x at each row-centre y and
/// reproject to EPSG:4326. Latitude varies primarily with the raster's Y axis
/// within typical tile sizes, so a single per-row sample is accurate enough
/// while keeping the reprojection cost at O(nrows) rather than O(npixels).
#[cfg(feature = "gdal-io")]
pub(crate) fn compute_row_latitudes(ds: &Dataset) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let (ncols, nrows) = ds.raster_size();
    let gt = ds.geo_transform()?;
    let mut src = ds.spatial_ref()?;
    src.set_axis_mapping_strategy(AxisMappingStrategy::TraditionalGisOrder);

    if src.is_geographic() {
        let col_centre = ncols as f64 / 2.0;
        return Ok((0..nrows)
            .map(|row| gt[3] + col_centre * gt[4] + (row as f64 + 0.5) * gt[5])
            .collect());
    }

    let mut tgt = SpatialRef::from_epsg(4326)?;
    tgt.set_axis_mapping_strategy(AxisMappingStrategy::TraditionalGisOrder);
    let tr = CoordTransform::new(&src, &tgt)?;

    let col_centre = ncols as f64 / 2.0;
    let mut xs: Vec<f64> = (0..nrows)
        .map(|row| gt[0] + col_centre * gt[1] + (row as f64 + 0.5) * gt[2])
        .collect();
    let mut ys: Vec<f64> = (0..nrows)
        .map(|row| gt[3] + col_centre * gt[4] + (row as f64 + 0.5) * gt[5])
        .collect();
    tr.transform_coords(&mut xs, &mut ys, &mut [])?;
    Ok(ys)
}

/// Write a `Vec<f32>` as a single-band Float32 GeoTIFF with nodata = `UNDEFZ`.
#[cfg(feature = "gdal-io")]
pub(crate) fn write_raster(
    driver: &gdal::Driver,
    path: &str,
    ncols: usize,
    nrows: usize,
    geo_transform: &[f64; 6],
    projection: &str,
    data: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut out_ds = driver.create_with_band_type::<f32, _>(path, ncols, nrows, 1)?;
    out_ds.set_geo_transform(geo_transform)?;
    out_ds.set_projection(projection)?;
    let mut out_band = out_ds.rasterband(1)?;
    out_band.set_no_data_value(Some(UNDEFZ as f64))?;
    let mut buffer = Buffer::new((ncols, nrows), data.to_vec());
    out_band.write((0, 0), (ncols, nrows), &mut buffer)?;
    Ok(())
}

/// Create a synthetic 100×100 elevation raster over central Austria for testing.
///
/// Location: 14.0–15.0°E, 47.0–48.0°N (0.01°/pixel ≈ 1 km).
/// Elevation: synthetic alpine terrain, 520–1080 m.
#[cfg(feature = "gdal-io")]
pub fn create_dummy_elevation(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::f64::consts::PI;

    let ncols = 100usize;
    let nrows = 100usize;
    let x_origin = 14.0_f64;
    let y_origin = 48.0_f64;
    let pixel_size = 0.01_f64;

    let mut elev_data: Vec<f32> = Vec::with_capacity(ncols * nrows);
    for row in 0..nrows {
        for col in 0..ncols {
            let fx = col as f64 / (ncols - 1) as f64;
            let fy = row as f64 / (nrows - 1) as f64;
            let base = 600.0_f64;
            let ridge = 400.0 * (PI * fx).sin() * (PI * fy).sin();
            let detail = 80.0 * (4.0 * PI * fx).sin() * (3.0 * PI * fy).cos();
            elev_data.push((base + ridge + detail) as f32);
        }
    }

    let driver = DriverManager::get_driver_by_name("GTiff")?;
    let mut ds = driver.create_with_band_type::<f32, _>(path, ncols, nrows, 1)?;
    let geo_transform = [x_origin, pixel_size, 0.0, y_origin, 0.0, -pixel_size];
    ds.set_geo_transform(&geo_transform)?;
    let srs = gdal::spatial_ref::SpatialRef::from_epsg(4326)?;
    ds.set_projection(&srs.to_wkt()?)?;
    let mut band = ds.rasterband(1)?;
    let mut buffer = Buffer::new((ncols, nrows), elev_data);
    band.write((0, 0), (ncols, nrows), &mut buffer)?;
    Ok(())
}

/// Run the full r.sun raster computation.
///
/// Reads elevation (and optionally slope/aspect/linke/albedo/mask) rasters,
/// integrates solar irradiation for each pixel, and writes output GeoTIFFs.
///
/// When `mask_path` is provided, pixels with mask value 0 (false) are skipped
/// and written as UNDEFZ; all non-zero mask values are treated as true.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "gdal-io")]
pub fn run_raster_computation(
    elev_path: &str,
    slope_path: &Option<String>,
    aspect_path: &Option<String>,
    linke_path: &Option<String>,
    albedo_path: &Option<String>,
    mask_path: &Option<String>,
    slope_value: Option<f64>,
    aspect_value: Option<f64>,
    linke_value: f64,
    albedo_value: f64,
    day: i32,
    step: f64,
    solar_constant: f64,
    out_glob: &Option<String>,
    out_beam: &Option<String>,
    out_diff: &Option<String>,
    out_refl: &Option<String>,
    out_insol: &Option<String>,
    quiet: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let declination = com_declin(day);
    let g_norm_extra = com_sol_const(day, solar_constant);
    let sindecl = declination.sin();
    let cosdecl = declination.cos();

    if !quiet {
        eprintln!(
            "Day: {day}  |  Declination: {:.2}°  |  G0: {g_norm_extra:.1} W/m²",
            -declination * RAD2DEG
        );
    }

    let elev_ds = Dataset::open(Path::new(elev_path))?;
    let (ncols, nrows) = elev_ds.raster_size();
    let geo_transform = elev_ds.geo_transform()?;
    let projection = elev_ds.projection();
    let row_lat_deg = compute_row_latitudes(&elev_ds)?;

    if !quiet {
        eprintln!("Raster: {ncols}×{nrows} pixels");
    }

    let elev_data = read_elev_normalized(&elev_ds, ncols, nrows)?;
    let elev_data = elev_data.as_slice();

    let (slope_data, aspect_data) = resolve_slope_aspect(
        slope_path,
        slope_value,
        aspect_path,
        aspect_value,
        elev_data,
        ncols,
        nrows,
        &geo_transform,
    )?;
    let linke_data = read_raster_or_constant(linke_path, ncols, nrows, linke_value as f32)?;
    let albedo_data = read_raster_or_constant(albedo_path, ncols, nrows, albedo_value as f32)?;
    // Mask default is 1.0 (compute every pixel); any non-zero pixel value means "compute".
    let mask_data = read_raster_or_constant(mask_path, ncols, nrows, 1.0)?;

    // Shared shadow-march context: grid metadata + precomputed max valid elevation.
    let dx = geo_transform[1].abs();
    let dy = geo_transform[5].abs();
    let max_z = elev_data
        .iter()
        .copied()
        .filter(|&v| v != UNDEFZ)
        .fold(f32::NEG_INFINITY, f32::max) as f64;

    let npixels = ncols * nrows;
    let mut o_beam = vec![UNDEFZ; npixels];
    let mut o_diff = vec![UNDEFZ; npixels];
    let mut o_refl = vec![UNDEFZ; npixels];
    let mut o_glob = vec![UNDEFZ; npixels];
    let mut o_insol = vec![UNDEFZ; npixels];

    let report_interval = (nrows / 20).max(1);
    let valid = AtomicUsize::new(0);
    let rows_done = AtomicUsize::new(0);

    if !quiet {
        eprintln!("Using {} CPU threads", rayon::current_num_threads());
    }

    o_beam
        .par_chunks_mut(ncols)
        .zip(o_diff.par_chunks_mut(ncols))
        .zip(o_refl.par_chunks_mut(ncols))
        .zip(o_glob.par_chunks_mut(ncols))
        .zip(o_insol.par_chunks_mut(ncols))
        .enumerate()
        .for_each(|(row, ((((rb, rd), rr), rg), ri))| {
            let mut local_valid = 0usize;
            let lat = row_lat_deg[row] * DEG2RAD;
            for col in 0..ncols {
                let idx = row * ncols + col;
                let elev = elev_data[idx];
                if elev == UNDEFZ || elev < -1000.0 {
                    continue;
                }
                // Mask: 0 = skip (output stays UNDEFZ). Non-zero = compute.
                let m = mask_data[idx];
                if m == 0.0 || m == UNDEFZ {
                    continue;
                }

                if slope_data[idx] == UNDEFZ || aspect_data[idx] == UNDEFZ {
                    continue;
                }

                let slope_deg = slope_data[idx] as f64;
                let aspect_deg = aspect_data[idx] as f64;
                let linke = linke_data[idx] as f64;
                let albedo = albedo_data[idx] as f64;

                let slope_rad = slope_deg * DEG2RAD;
                let aspect_rad = convert_grass_aspect(aspect_deg);

                let geom = compute_slope_geometry(slope_rad, aspect_rad, lat, sindecl, cosdecl);
                // Horizontal-day window (GRASS parity): never-sunlit slopes
                // keep diffuse+reflected; beam/insolation gated by s0 > 0.
                let (sunrise, sunset) = match horizontal_day_window(lat, sindecl, cosdecl) {
                    Some(s) => s,
                    None => continue, // polar night
                };

                let params = RadiationParams {
                    g_norm_extra,
                    linke,
                    albedo,
                    cbh: 1.0,
                    cdh: 1.0,
                    elevation: elev as f64,
                    solar_altitude: 0.0,
                    slope: slope_rad,
                    aspect: aspect_rad,
                    solar_azimuth: 0.0,
                };

                let shadow_ctx = shadow::ShadowContext {
                    elev: elev_data,
                    ncols,
                    nrows,
                    dx,
                    dy,
                    max_z,
                    row,
                    col,
                    eye_z: elev as f64,
                    horizon: None,
                };
                let result = integrate_daily(
                    &geom,
                    &params,
                    sunrise,
                    sunset,
                    step,
                    lat,
                    declination,
                    Some(&shadow_ctx),
                );

                rb[col] = result.beam as f32;
                rd[col] = result.diffuse as f32;
                rr[col] = result.reflected as f32;
                rg[col] = result.global as f32;
                ri[col] = result.insol_time as f32;
                local_valid += 1;
            }
            valid.fetch_add(local_valid, Ordering::Relaxed);
            if !quiet {
                let done = rows_done.fetch_add(1, Ordering::Relaxed) + 1;
                if done % report_interval == 0 {
                    eprint!("\rProgress: {:.0}%   ", 100.0 * done as f64 / nrows as f64);
                }
            }
        });

    let valid = valid.load(Ordering::Relaxed);
    if !quiet {
        eprintln!("\rProgress: 100%   ");
        eprintln!("Processed {valid} valid pixels");

        let valid_glob: Vec<f32> = o_glob.iter().copied().filter(|&v| v != UNDEFZ).collect();
        if !valid_glob.is_empty() {
            let mean = valid_glob.iter().sum::<f32>() / valid_glob.len() as f32;
            let min = valid_glob.iter().copied().fold(f32::INFINITY, f32::min);
            let max = valid_glob.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("Global irradiation: min={min:.0}  mean={mean:.0}  max={max:.0} Wh/m²/day");
        }
    }

    let driver = DriverManager::get_driver_by_name("GTiff")?;
    if let Some(p) = out_beam {
        write_raster(
            &driver,
            p,
            ncols,
            nrows,
            &geo_transform,
            &projection,
            &o_beam,
        )?;
    }
    if let Some(p) = out_diff {
        write_raster(
            &driver,
            p,
            ncols,
            nrows,
            &geo_transform,
            &projection,
            &o_diff,
        )?;
    }
    if let Some(p) = out_refl {
        write_raster(
            &driver,
            p,
            ncols,
            nrows,
            &geo_transform,
            &projection,
            &o_refl,
        )?;
    }
    if let Some(p) = out_glob {
        write_raster(
            &driver,
            p,
            ncols,
            nrows,
            &geo_transform,
            &projection,
            &o_glob,
        )?;
    }
    if let Some(p) = out_insol {
        write_raster(
            &driver,
            p,
            ncols,
            nrows,
            &geo_transform,
            &projection,
            &o_insol,
        )?;
    }

    Ok(())
}

// ── PyO3 Python bindings ────────────────────────────────────────────────────

/// Daily solar irradiation result for a single pixel.
#[pyclass(get_all)]
#[derive(Clone)]
pub struct IrradiationResult {
    /// Beam (direct) irradiation [Wh/m²/day]
    pub beam: f64,
    /// Diffuse irradiation [Wh/m²/day]
    pub diffuse: f64,
    /// Reflected irradiation [Wh/m²/day]
    pub reflected: f64,
    /// Global (beam + diffuse + reflected) irradiation [Wh/m²/day]
    pub global_rad: f64,
    /// Duration of direct sunshine [h/day]
    pub insol_time: f64,
}

#[pymethods]
impl IrradiationResult {
    fn __repr__(&self) -> String {
        format!(
            "IrradiationResult(beam={:.1}, diffuse={:.1}, reflected={:.1}, \
             global={:.1}, insol_time={:.2}h)",
            self.beam, self.diffuse, self.reflected, self.global_rad, self.insol_time
        )
    }
}

/// Compute daily solar irradiation for a single geographic point.
///
/// Parameters
/// ----------
/// lat_deg : float
///     Geographic latitude [°, positive = North]
/// elevation_m : float
///     Terrain elevation [m]
/// day : int
///     Day of year [1–365]
/// slope_deg : float, optional
///     Terrain slope [°], default 0 (flat)
/// aspect_deg : float, optional
///     Terrain aspect [°, GRASS convention: 0=E CCW, 270=S], default 270
/// linke : float, optional
///     Linke atmospheric turbidity factor (2–5), default 3.0
/// albedo : float, optional
///     Ground albedo (0–1), default 0.2
/// step : float, optional
///     Time integration step [hours], default 0.5
/// solar_constant : float, optional
///     Solar constant [W/m²], default 1367.0
///
/// Returns
/// -------
/// IrradiationResult
///     Named struct with .beam, .diffuse, .reflected, .global_rad, .insol_time
///
/// Raises
/// ------
/// ValueError
///     If day is out of range or the sun never rises for the given parameters.
#[pyfunction]
#[pyo3(signature = (lat_deg, elevation_m, day, *, slope_deg=0.0, aspect_deg=270.0,
                    linke=3.0, albedo=0.2, step=0.5, solar_constant=1367.0))]
#[allow(clippy::too_many_arguments)]
fn compute_pixel(
    lat_deg: f64,
    elevation_m: f64,
    day: i32,
    slope_deg: f64,
    aspect_deg: f64,
    linke: f64,
    albedo: f64,
    step: f64,
    solar_constant: f64,
) -> PyResult<IrradiationResult> {
    compute_pixel_irradiation(
        lat_deg,
        elevation_m,
        slope_deg,
        aspect_deg,
        linke,
        albedo,
        day,
        step,
        solar_constant,
    )
    .map(|r| IrradiationResult {
        beam: r.beam,
        diffuse: r.diffuse,
        reflected: r.reflected,
        global_rad: r.global,
        insol_time: r.insol_time,
    })
    .ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "No sunrise for lat={lat_deg}°, day={day} (polar night or invalid slope orientation)"
        ))
    })
}

/// Compute daily solar irradiation for a raster.
///
/// Reads the elevation raster (and optional slope/aspect/linke/albedo rasters),
/// runs the r.sun model for each pixel, and writes output GeoTIFFs.
///
/// Parameters
/// ----------
/// elevation : str
///     Path to input elevation raster [m] (GeoTIFF)
/// day : int
///     Day of year [1–365]
/// slope : str, optional
///     Path to slope raster [°]. If ``None`` and ``slope_value`` is also
///     ``None`` (the default), slope is derived from ``elevation`` via Horn's
///     3×3 finite-difference. Edge ring and any pixel touching DEM nodata is
///     emitted as nodata.
/// aspect : str, optional
///     Path to aspect raster [°, GRASS CCW-from-East]. Same precedence rule
///     as ``slope``: path > ``aspect_value`` > derive-from-DEM.
/// linke : str, optional
///     Path to Linke turbidity raster; uses ``linke_value`` if not given
/// albedo : str, optional
///     Path to albedo raster; uses ``albedo_value`` if not given
/// mask : str, optional
///     Path to a binary mask GeoTIFF. Pixels with value 0 are skipped (output
///     is set to nodata); any non-zero pixel value is treated as "compute".
/// slope_value : float, optional
///     Constant slope [°]. Default ``None`` → derive from DEM. Pass ``0.0``
///     explicitly to treat the whole tile as flat.
/// aspect_value : float, optional
///     Constant aspect [°, GRASS CCW-from-East, 270=South]. Default ``None``
///     → derive from DEM.
/// linke_value : float
///     Constant Linke turbidity, default 3.0
/// albedo_value : float
///     Constant albedo, default 0.2
/// step : float
///     Integration time step [hours], default 0.5
/// solar_constant : float
///     Solar constant [W/m²], default 1367.0
/// glob_rad : str, optional
///     Output path for global irradiation [Wh/m²/day]
/// beam_rad : str, optional
///     Output path for beam irradiation [Wh/m²/day]
/// diff_rad : str, optional
///     Output path for diffuse irradiation [Wh/m²/day]
/// refl_rad : str, optional
///     Output path for reflected irradiation [Wh/m²/day]
/// insol_time : str, optional
///     Output path for insolation time [h/day]
/// gpu : bool, optional
///     If True, run the WebGPU compute shader instead of the scalar CPU loop.
///     Defaults to False.
/// quiet : bool, optional
///     If True, suppress all progress/info messages on stderr. Defaults to False.
#[cfg(feature = "gdal-io")]
#[pyfunction]
#[pyo3(signature = (elevation, day, *, slope=None, aspect=None, linke=None, albedo=None,
                    mask=None,
                    slope_value=None, aspect_value=None, linke_value=3.0, albedo_value=0.2,
                    step=0.5, solar_constant=1367.0,
                    glob_rad=None, beam_rad=None, diff_rad=None, refl_rad=None, insol_time=None,
                    gpu=false, quiet=false))]
#[allow(clippy::too_many_arguments)]
fn compute_raster(
    py: Python<'_>,
    elevation: &str,
    day: i32,
    slope: Option<String>,
    aspect: Option<String>,
    linke: Option<String>,
    albedo: Option<String>,
    mask: Option<String>,
    slope_value: Option<f64>,
    aspect_value: Option<f64>,
    linke_value: f64,
    albedo_value: f64,
    step: f64,
    solar_constant: f64,
    glob_rad: Option<String>,
    beam_rad: Option<String>,
    diff_rad: Option<String>,
    refl_rad: Option<String>,
    insol_time: Option<String>,
    gpu: bool,
    quiet: bool,
) -> PyResult<()> {
    // Release the GIL for the whole computation: it runs for seconds/minutes
    // and only touches Rust + native memory. Without this, callers embedding
    // us in a GUI (QGIS plugin) freeze: their threads cannot run, so progress
    // reports queue up and arrive in one burst at the end.
    // Errors are mapped to String inside the closure because Box<dyn Error>
    // is not Send, which allow_threads requires.
    let run = py.allow_threads(|| {
        let result = if gpu {
            gpu::compute_raster_gpu(
                elevation,
                &slope,
                &aspect,
                &linke,
                &albedo,
                &mask,
                slope_value,
                aspect_value,
                linke_value,
                albedo_value,
                day,
                step,
                solar_constant,
                &glob_rad,
                &beam_rad,
                &diff_rad,
                &refl_rad,
                &insol_time,
                quiet,
            )
        } else {
            run_raster_computation(
                elevation,
                &slope,
                &aspect,
                &linke,
                &albedo,
                &mask,
                slope_value,
                aspect_value,
                linke_value,
                albedo_value,
                day,
                step,
                solar_constant,
                &glob_rad,
                &beam_rad,
                &diff_rad,
                &refl_rad,
                &insol_time,
                quiet,
            )
        };
        result.map_err(|e| e.to_string())
    });
    run.map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Compute annual solar potential per m² (GPU).
///
/// Samples global irradiation on a DOY grid (day_start..=day_end stepped by
/// day_step), accumulates ``glob × day_step`` into annual Wh/m², then writes
/// ``(annual_wh / 1000) × panel_efficiency`` (kWh/m²/year) as a single GeoTIFF.
/// Pixels with mask=0 (or outside data bounds) are written as nodata.
///
/// The GPU device, pipeline, and per-tile buffers are reused across all
/// sampled days — only the per-day uniform is rewritten — so this is
/// significantly faster than calling ``compute_raster`` in a Python loop.
///
/// Parameters
/// ----------
/// elevation : str
///     Path to input elevation raster [m] (GeoTIFF)
/// out_path : str
///     Output GeoTIFF path for annual potential [kWh/m²/year]
/// slope, aspect, linke, albedo, mask : str, optional
///     Same semantics as ``compute_raster``. Mask=0 ⇒ pixel skipped (nodata).
/// slope_value, aspect_value : float, optional
///     Scalar fallback. If both the raster path and the scalar are ``None``
///     (the default for slope/aspect), the value is derived from
///     ``elevation`` via Horn's 3×3 finite-difference — edge ring and any
///     pixel touching DEM nodata is emitted as nodata.
/// linke_value, albedo_value : float
///     Scalar fallbacks when the corresponding raster is None.
/// day_start, day_end : int
///     Inclusive DOY range, default 1..=365.
/// day_step : int
///     DOY sampling stride, default 10. Each sample is weighted by day_step
///     to approximate the full-year integral.
/// step : float
///     Sub-daily integration step [hours], default 0.5.
/// solar_constant : float
///     Solar constant [W/m²], default 1367.
/// panel_efficiency : float
///     Multiplicative efficiency factor applied after the Wh→kWh conversion,
///     default 1.0.
/// gpu : bool, optional
///     If True, run the WebGPU implementation. If False (default), run the
///     rayon-parallel CPU implementation. Both produce equivalent rasters.
/// use_horizon : bool, optional
///     CPU path only. If True, precompute a per-pixel azimuth-binned horizon
///     map up front and replace each per-timestep cast-shadow ray-march with
///     an O(1) lookup. Default False (matches the canonical ray-march path).
/// horizon_n_az : int, optional
///     Number of azimuth bins for the horizon precompute. Default 64
///     (bin width 5.625°). Linear inter-bin interpolation means residual
///     bias halves with each doubling of n_az; 64 holds bias under ~2%
///     of annual yield on smooth terrain. Bump to 128 for steep urban
///     DSMs with sharp building edges if needed.
/// quiet : bool
///     Suppress progress messages on stderr.
#[cfg(feature = "gdal-io")]
#[pyfunction]
#[pyo3(signature = (elevation, out_path, *, slope=None, aspect=None, linke=None, albedo=None,
                    mask=None,
                    slope_value=None, aspect_value=None, linke_value=3.0, albedo_value=0.2,
                    day_start=1, day_end=365, day_step=10,
                    step=0.5, solar_constant=1367.0, panel_efficiency=1.0,
                    gpu=false, use_horizon=false, horizon_n_az=64, quiet=false))]
#[allow(clippy::too_many_arguments)]
fn compute_annual_potential(
    py: Python<'_>,
    elevation: &str,
    out_path: &str,
    slope: Option<String>,
    aspect: Option<String>,
    linke: Option<String>,
    albedo: Option<String>,
    mask: Option<String>,
    slope_value: Option<f64>,
    aspect_value: Option<f64>,
    linke_value: f64,
    albedo_value: f64,
    day_start: i32,
    day_end: i32,
    day_step: i32,
    step: f64,
    solar_constant: f64,
    panel_efficiency: f64,
    gpu: bool,
    use_horizon: bool,
    horizon_n_az: usize,
    quiet: bool,
) -> PyResult<()> {
    // Release the GIL for the whole (long) computation — see compute_raster.
    // Box<dyn Error> is mapped to String inside the closure: it is not Send.
    let run = py.allow_threads(|| {
        let result = if gpu {
            annual::compute_annual_potential_gpu(
                elevation,
                &slope,
                &aspect,
                &linke,
                &albedo,
                &mask,
                slope_value,
                aspect_value,
                linke_value,
                albedo_value,
                day_start,
                day_end,
                day_step,
                step,
                solar_constant,
                panel_efficiency,
                out_path,
                use_horizon,
                horizon_n_az,
                quiet,
            )
        } else {
            annual::compute_annual_potential_cpu(
                elevation,
                &slope,
                &aspect,
                &linke,
                &albedo,
                &mask,
                slope_value,
                aspect_value,
                linke_value,
                albedo_value,
                day_start,
                day_end,
                day_step,
                step,
                solar_constant,
                panel_efficiency,
                out_path,
                use_horizon,
                horizon_n_az,
                quiet,
            )
        };
        result.map_err(|e| e.to_string())
    });
    run.map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Create a synthetic 100×100 test elevation raster (central Austria, WGS84).
///
/// Parameters
/// ----------
/// path : str
///     Output GeoTIFF path
#[cfg(feature = "gdal-io")]
#[pyfunction]
fn create_dummy(path: &str) -> PyResult<()> {
    create_dummy_elevation(path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

// ── Array API (no GDAL — I/O happens in Python) ─────────────────────────────
//
// Python reads the rasters (tiled, see the plugin's raster_io.py) and passes
// flat f32 numpy arrays; these functions compute one row band and hand back
// numpy arrays that Python writes to GeoTIFF. Because the extension never
// opens a file, it has no GDAL dependency and builds on any platform.

use arrays::{BandInputs, DailyWants};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};

/// Borrow a numpy array as `&[f32]`, validating contiguity and length.
///
/// The returned slice borrows from the `PyReadonlyArray` handle — callers
/// must keep the handle alive for as long as the slice is used (it is NOT
/// valid for the whole `'py` lifetime).
fn slice_of<'a>(
    arr: &'a PyReadonlyArray1<'a, f32>,
    expected: usize,
    name: &str,
) -> PyResult<&'a [f32]> {
    let s = arr.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be a contiguous float32 array (got a non-contiguous \
             or wrongly-typed ndarray; use np.ascontiguousarray(x, dtype=np.float32))"
        ))
    })?;
    if s.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} has {} elements, expected {expected}",
            s.len()
        )));
    }
    Ok(s)
}

/// Resolve an optional raster input: borrow the caller's array, or fill a
/// band-sized constant buffer when absent. The `Some(handle)` variant keeps
/// the numpy handle alive so the borrowed slice stays valid.
fn resolve_input<'a>(
    arr: Option<PyReadonlyArray1<'a, f32>>,
    fill: f32,
    npixels: usize,
    name: &str,
) -> PyResult<(Option<PyReadonlyArray1<'a, f32>>, Vec<f32>)> {
    match arr {
        Some(a) => {
            slice_of(&a, npixels, name)?;
            Ok((Some(a), Vec::new()))
        }
        None => Ok((None, vec![fill; npixels])),
    }
}

/// Effective slice for an optional input: the borrowed array or the fill-in.
fn input_slice<'a, 'b>(
    owned: &'b Option<PyReadonlyArray1<'a, f32>>,
    fallback: &'b [f32],
) -> &'b [f32] {
    match owned {
        // Safety of lifetime: slice_of validated length; as_slice borrows the
        // handle which lives in `owned` for 'b.
        Some(a) => a.as_slice().unwrap_or(fallback),
        None => fallback,
    }
}

/// Compute daily irradiation for ONE row band from numpy arrays (no file I/O).
///
/// Parameters
/// ----------
/// elevation : ndarray[float32]
///     Flat band elevation, length ncols*nrows. Nodata/NaN must already be
///     normalized to -9999 by the caller.
/// ncols, nrows : int
///     Band geometry.
/// row_lat : ndarray[float32]
///     Pixel-centre latitude [deg] per band row, length nrows.
/// day : int
///     Day of year 1-365.
/// step : float
///     Sub-daily integration step [hours].
/// linke_value, albedo_value : float
///     Constant fallbacks used when the matching raster array is None.
/// dx_m, dy_m : float
///     Pixel size along X / Y in map units (used by the shadow ray-march).
/// slope, aspect, linke, albedo, mask : ndarray[float32], optional
///     Flat band-sized arrays. `slope`/`aspect` are DERIVED from the DEM via
///     Horn's 3×3 when omitted; linke/albedo fall back to the scalar values;
///     mask defaults to all-compute.
/// shadow_context_elev : ndarray[float32], optional
///     FULL-grid flat elevation (ncols * full_nrows) used as the cast-shadow
///     context. Omit only when the band IS the whole grid.
/// row_offset : int
///     This band's first row in the full grid.
/// full_nrows : int
///     Total rows in the full grid.
/// outputs : list[str]
///     Any of "beam", "diff", "refl", "glob", "insol".
/// gpu : bool
///     Reserved for the WebGPU band path; currently CPU-only.
///
/// Returns
/// -------
/// dict[str, ndarray[float32]]
///     One entry per requested output, each flat of length ncols*nrows, with
///     skipped pixels set to -9999.
#[pyfunction]
#[pyo3(signature = (elevation, ncols, nrows, row_lat, day, *, step=0.5,
                    linke_value=3.0, albedo_value=0.2, solar_constant=1367.0,
                    dx_m=1.0, dy_m=1.0,
                    slope=None, aspect=None, linke=None, albedo=None, mask=None,
                    shadow_context_elev=None, row_offset=0, full_nrows=0,
                    outputs=None, gpu=false, quiet=true))]
#[allow(clippy::too_many_arguments)]
fn compute_raster_bands<'py>(
    py: Python<'py>,
    elevation: PyReadonlyArray1<'py, f32>,
    ncols: usize,
    nrows: usize,
    row_lat: PyReadonlyArray1<'py, f32>,
    day: i32,
    step: f64,
    linke_value: f64,
    albedo_value: f64,
    solar_constant: f64,
    dx_m: f64,
    dy_m: f64,
    slope: Option<PyReadonlyArray1<'py, f32>>,
    aspect: Option<PyReadonlyArray1<'py, f32>>,
    linke: Option<PyReadonlyArray1<'py, f32>>,
    albedo: Option<PyReadonlyArray1<'py, f32>>,
    mask: Option<PyReadonlyArray1<'py, f32>>,
    shadow_context_elev: Option<PyReadonlyArray1<'py, f32>>,
    row_offset: usize,
    full_nrows: usize,
    outputs: Option<Vec<String>>,
    gpu: bool,
    quiet: bool,
) -> PyResult<std::collections::HashMap<String, Bound<'py, PyArray1<f32>>>> {
    let npixels = ncols.checked_mul(nrows).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("ncols*nrows overflows")
    })?;
    let full_nrows = if full_nrows == 0 { nrows } else { full_nrows };

    // Parse the output selection BEFORE doing any heavy work so a typo fails fast.
    let requested = outputs.unwrap_or_else(|| vec!["glob".to_string()]);
    if requested.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "outputs must contain at least one of beam/diff/refl/glob/insol",
        ));
    }
    let mut wants = DailyWants::default();
    for name in &requested {
        match name.as_str() {
            "beam" => wants.beam = true,
            "diff" => wants.diffuse = true,
            "refl" => wants.reflected = true,
            "glob" => wants.global = true,
            "insol" => wants.insol = true,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown output {other:?}; expected one of beam, diff, refl, glob, insol"
                )))
            }
        }
    }
    if !wants.beam && !wants.diffuse && !wants.reflected && !wants.global && !wants.insol {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "no valid output selected",
        ));
    }

    let elev = slice_of(&elevation, npixels, "elevation")?;
    let lats = slice_of(&row_lat, nrows, "row_lat")?;

    // slope/aspect: caller array > derive-from-DEM (never a constant fill).
    let derived = if slope.is_none() || aspect.is_none() {
        Some(py.allow_threads(|| {
            arrays::slope_aspect_from_elev(elev, ncols, nrows, dx_m, dy_m)
        }))
    } else {
        None
    };

    let (slope_h, slope_fill) = resolve_input(slope, UNDEFZ, npixels, "slope")?;
    let (aspect_h, aspect_fill) = resolve_input(aspect, UNDEFZ, npixels, "aspect")?;
    let slope_s: &[f32] = match (&slope_h, &derived) {
        (Some(h), _) => h.as_slice().unwrap_or(&slope_fill),
        (None, Some((d, _))) => d,
        (None, None) => unreachable!(),
    };
    let aspect_s: &[f32] = match (&aspect_h, &derived) {
        (Some(h), _) => h.as_slice().unwrap_or(&aspect_fill),
        (None, Some((_, d))) => d,
        (None, None) => unreachable!(),
    };

    let (linke_h, linke_fill) = resolve_input(linke, linke_value as f32, npixels, "linke")?;
    let (albedo_h, albedo_fill) = resolve_input(albedo, albedo_value as f32, npixels, "albedo")?;
    let (mask_h, mask_fill) = resolve_input(mask, 1.0, npixels, "mask")?;
    let linke_s = input_slice(&linke_h, &linke_fill);
    let albedo_s = input_slice(&albedo_h, &albedo_fill);
    let mask_s = input_slice(&mask_h, &mask_fill);

    // Shadow context defaults to the band itself (single-band runs).
    let shadow_s: &[f32] = match &shadow_context_elev {
        Some(a) => slice_of(a, ncols * full_nrows, "shadow_context_elev")?,
        None => elev,
    };

    let inp = BandInputs {
        elevation: elev,
        slope: slope_s,
        aspect: aspect_s,
        linke: linke_s,
        albedo: albedo_s,
        mask: mask_s,
        row_lat_deg: lats,
        shadow_elev: shadow_s,
        ncols,
        nrows,
        row_offset,
        full_nrows,
        dx: dx_m,
        dy: dy_m,
    };
    inp.validate().map_err(pyo3::exceptions::PyValueError::new_err)?;

    // The heavy loop runs with the GIL released so the embedding GUI stays
    // responsive and progress reports can be parsed concurrently. The GPU
    // path (wgpu dispatch + readback inside pollster::block_on) also runs
    // GIL-free. GPU errors are RuntimeError — they're environmental (no
    // adapter), not argument errors.
    let out = py
        .allow_threads(|| {
            if gpu {
                let g = gpu_array::compute_daily_band_gpu(
                    &inp, day, step, solar_constant, quiet,
                )
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
                // The shader computes all five components; filter to wants.
                Ok::<_, pyo3::PyErr>(arrays::DailyBandOut {
                    beam: wants.beam.then_some(g.beam),
                    diffuse: wants.diffuse.then_some(g.diffuse),
                    reflected: wants.reflected.then_some(g.reflected),
                    global: wants.global.then_some(g.global),
                    insol: wants.insol.then_some(g.insol),
                })
            } else {
                arrays::compute_daily_band_cpu(&inp, day, step, solar_constant, wants, quiet)
                    .map_err(pyo3::exceptions::PyValueError::new_err)
            }
        })?;

    let mut result = std::collections::HashMap::new();
    if let Some(v) = out.beam {
        result.insert("beam".to_string(), v.into_pyarray_bound(py));
    }
    if let Some(v) = out.diffuse {
        result.insert("diff".to_string(), v.into_pyarray_bound(py));
    }
    if let Some(v) = out.reflected {
        result.insert("refl".to_string(), v.into_pyarray_bound(py));
    }
    if let Some(v) = out.global {
        result.insert("glob".to_string(), v.into_pyarray_bound(py));
    }
    if let Some(v) = out.insol {
        result.insert("insol".to_string(), v.into_pyarray_bound(py));
    }
    Ok(result)
}

/// Compute annual PV potential [kWh/m²/yr] for ONE row band from numpy arrays.
///
/// Same array/geometry semantics as [`compute_raster_bands`]; returns a single
/// flat float32 array of length ncols*nrows (nodata = -9999).
#[pyfunction]
#[pyo3(signature = (elevation, ncols, nrows, row_lat, *, day_start=1, day_end=365, day_step=10,
                    step=0.5, solar_constant=1367.0, panel_efficiency=1.0,
                    linke_value=3.0, albedo_value=0.2, dx_m=1.0, dy_m=1.0,
                    slope=None, aspect=None, linke=None, albedo=None, mask=None,
                    shadow_context_elev=None, row_offset=0, full_nrows=0,
                    gpu=false, use_horizon=false, horizon_n_az=64, quiet=true))]
#[allow(clippy::too_many_arguments)]
fn compute_annual_bands<'py>(
    py: Python<'py>,
    elevation: PyReadonlyArray1<'py, f32>,
    ncols: usize,
    nrows: usize,
    row_lat: PyReadonlyArray1<'py, f32>,
    day_start: i32,
    day_end: i32,
    day_step: i32,
    step: f64,
    solar_constant: f64,
    panel_efficiency: f64,
    linke_value: f64,
    albedo_value: f64,
    dx_m: f64,
    dy_m: f64,
    slope: Option<PyReadonlyArray1<'py, f32>>,
    aspect: Option<PyReadonlyArray1<'py, f32>>,
    linke: Option<PyReadonlyArray1<'py, f32>>,
    albedo: Option<PyReadonlyArray1<'py, f32>>,
    mask: Option<PyReadonlyArray1<'py, f32>>,
    shadow_context_elev: Option<PyReadonlyArray1<'py, f32>>,
    row_offset: usize,
    full_nrows: usize,
    gpu: bool,
    use_horizon: bool,
    horizon_n_az: usize,
    quiet: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if day_step <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "day_step must be positive",
        ));
    }
    if !(1..=365).contains(&day_start) || !(1..=365).contains(&day_end) || day_end < day_start {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "day_start/day_end must lie in 1..=365 with day_end >= day_start",
        ));
    }
    if use_horizon && horizon_n_az < 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "horizon_n_az must be >= 4 when use_horizon=true",
        ));
    }

    let npixels = ncols.checked_mul(nrows).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("ncols*nrows overflows")
    })?;
    let full_nrows = if full_nrows == 0 { nrows } else { full_nrows };
    let days: Vec<i32> = (day_start..=day_end).step_by(day_step as usize).collect();

    let elev = slice_of(&elevation, npixels, "elevation")?;
    let lats = slice_of(&row_lat, nrows, "row_lat")?;

    let derived = if slope.is_none() || aspect.is_none() {
        Some(py.allow_threads(|| {
            arrays::slope_aspect_from_elev(elev, ncols, nrows, dx_m, dy_m)
        }))
    } else {
        None
    };

    let (slope_h, slope_fill) = resolve_input(slope, UNDEFZ, npixels, "slope")?;
    let (aspect_h, aspect_fill) = resolve_input(aspect, UNDEFZ, npixels, "aspect")?;
    let slope_s: &[f32] = match (&slope_h, &derived) {
        (Some(h), _) => h.as_slice().unwrap_or(&slope_fill),
        (None, Some((d, _))) => d,
        (None, None) => unreachable!(),
    };
    let aspect_s: &[f32] = match (&aspect_h, &derived) {
        (Some(h), _) => h.as_slice().unwrap_or(&aspect_fill),
        (None, Some((_, d))) => d,
        (None, None) => unreachable!(),
    };

    let (linke_h, linke_fill) = resolve_input(linke, linke_value as f32, npixels, "linke")?;
    let (albedo_h, albedo_fill) = resolve_input(albedo, albedo_value as f32, npixels, "albedo")?;
    let (mask_h, mask_fill) = resolve_input(mask, 1.0, npixels, "mask")?;
    let linke_s = input_slice(&linke_h, &linke_fill);
    let albedo_s = input_slice(&albedo_h, &albedo_fill);
    let mask_s = input_slice(&mask_h, &mask_fill);
    let shadow_s: &[f32] = match &shadow_context_elev {
        Some(a) => slice_of(a, ncols * full_nrows, "shadow_context_elev")?,
        None => elev,
    };

    let inp = BandInputs {
        elevation: elev,
        slope: slope_s,
        aspect: aspect_s,
        linke: linke_s,
        albedo: albedo_s,
        mask: mask_s,
        row_lat_deg: lats,
        shadow_elev: shadow_s,
        ncols,
        nrows,
        row_offset,
        full_nrows,
        dx: dx_m,
        dy: dy_m,
    };
    inp.validate().map_err(pyo3::exceptions::PyValueError::new_err)?;

    // Horizon precompute spans the FULL grid, so it is only valid when this
    // call covers the whole raster; otherwise the per-pixel ray-march (which
    // already indexes the full grid via shadow_context_elev) is used.
    let horizon_map = if use_horizon && row_offset == 0 && nrows == full_nrows {
        Some(py.allow_threads(|| {
            crate::horizon::compute_horizon_map_cpu(
                shadow_s,
                ncols,
                full_nrows,
                dx_m,
                dy_m,
                mask_s,
                horizon_n_az,
                quiet,
            )
        }))
    } else {
        None
    };

    let potential = py
        .allow_threads(|| {
            if gpu {
                gpu_array::compute_annual_band_gpu(
                    &inp,
                    &days,
                    day_step,
                    step,
                    solar_constant,
                    panel_efficiency,
                    horizon_map.as_ref(),
                    quiet,
                )
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)
            } else {
                arrays::compute_annual_band_cpu(
                    &inp,
                    &days,
                    day_step,
                    step,
                    solar_constant,
                    panel_efficiency,
                    horizon_map.as_ref(),
                    quiet,
                )
                .map_err(pyo3::exceptions::PyValueError::new_err)
            }
        })?;

    Ok(potential.into_pyarray_bound(py))
}

/// Derive slope [deg] and aspect [deg, GRASS CCW-from-East] from a flat f32
/// elevation band via Horn's 3×3 finite difference.
///
/// Returns `(slope, aspect)` as flat float32 arrays of length ncols*nrows.
/// The band's edge ring, and any pixel whose 3×3 neighbourhood touches
/// nodata, is emitted as -9999.
#[pyfunction]
#[pyo3(signature = (elevation, ncols, nrows, *, dx_m=1.0, dy_m=1.0))]
fn horn_slope_aspect<'py>(
    py: Python<'py>,
    elevation: PyReadonlyArray1<'py, f32>,
    ncols: usize,
    nrows: usize,
    dx_m: f64,
    dy_m: f64,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let npixels = ncols.checked_mul(nrows).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("ncols*nrows overflows")
    })?;
    let elev = slice_of(&elevation, npixels, "elevation")?;
    let (slope, aspect) = py.allow_threads(|| {
        arrays::slope_aspect_from_elev(elev, ncols, nrows, dx_m, dy_m)
    });
    Ok((
        slope.into_pyarray_bound(py),
        aspect.into_pyarray_bound(py),
    ))
}

/// r.sun solar irradiation model — Python bindings
///
/// A port of GRASS GIS r.sun to Rust, exposed as a Python extension module.
/// Implements the ESRA clear-sky model for beam, diffuse, and reflected
/// solar irradiation.
///
/// Functions
/// ---------
/// compute_pixel(lat_deg, elevation_m, day, ...)
///     Compute irradiation for a single geographic point.
/// compute_raster(elevation, day, ...)
///     Compute irradiation maps from GeoTIFF rasters.
/// create_dummy(path)
///     Create a synthetic test DEM.
#[pymodule]
fn sun(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IrradiationResult>()?;
    m.add_function(wrap_pyfunction!(compute_pixel, m)?)?;
    #[cfg(feature = "gdal-io")]
    {
        m.add_function(wrap_pyfunction!(compute_raster, m)?)?;
        m.add_function(wrap_pyfunction!(compute_annual_potential, m)?)?;
        m.add_function(wrap_pyfunction!(create_dummy, m)?)?;
    }
    // Array API (no GDAL): band-in / band-out, I/O lives in Python.
    m.add_function(wrap_pyfunction!(compute_raster_bands, m)?)?;
    m.add_function(wrap_pyfunction!(compute_annual_bands, m)?)?;
    m.add_function(wrap_pyfunction!(horn_slope_aspect, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_last_run_stats, m)?)?;
    Ok(())
}

/// Probe whether a usable GPU adapter (Vulkan/Metal/DX12 via wgpu) is
/// present. Returns False instead of raising, so callers can fall back to
/// the CPU path cleanly.
#[pyfunction]
fn gpu_available() -> bool {
    gpu_array::gpu_available()
}

/// Submission counters of the most recent GPU band call:
/// {"submissions": int, "max_submit_seconds": float}. Diagnostics/tests.
#[pyfunction]
fn gpu_last_run_stats(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let s = gpu_array::last_run_stats();
    let d = pyo3::types::PyDict::new_bound(py);
    d.set_item("submissions", s.submissions)?;
    d.set_item("max_submit_seconds", s.max_submit_seconds)?;
    Ok(d)
}
