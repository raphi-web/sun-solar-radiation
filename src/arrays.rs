//! Array-in / array-out compute cores — no GDAL, no file I/O.
//!
//! These are the functions the QGIS plugin calls: Python reads rasters
//! (tiled, via `raster_io.py`), passes flat f32 buffers, and writes the
//! returned band arrays back to GeoTIFFs. Keeping I/O out of the extension
//! means it has no native GDAL dependency, so it builds on any platform.
//!
//! # Band semantics
//!
//! A *band* is a contiguous row range of the full grid. Because the cast-
//! shadow ray-march spans the whole DEM (`shadow.rs`), [`BandInputs`] carries
//! both the band's own elevation and the FULL-grid elevation used as the
//! shadow context, plus `row_offset` so shadow rays index the full grid.
//! Computing a band with the correct offset must give results identical to
//! the same rows of a full-grid run — that equivalence is what makes tiling
//! safe.
use rayon::prelude::*;

use crate::radiation::{RadiationParams, integrate_daily};
use crate::shadow::ShadowContext;
use crate::solar::{DEG2RAD, com_declin, com_sol_const, compute_slope_geometry, horizontal_day_window};
use crate::terrain::horn_slope_aspect;
use crate::{UNDEFZ, convert_grass_aspect};

/// Inputs for one row band, all row-major flat f32 slices.
///
/// Lengths: `elevation`, `slope`, `aspect`, `linke`, `albedo`, `mask` and
/// `row_lat_deg` are band-sized (`ncols * nrows`, except `row_lat_deg` which
/// is `nrows`); `shadow_elev` is full-grid sized (`ncols * full_nrows`).
pub struct BandInputs<'a> {
    pub elevation: &'a [f32],
    pub slope: &'a [f32],
    pub aspect: &'a [f32],
    pub linke: &'a [f32],
    pub albedo: &'a [f32],
    pub mask: &'a [f32],
    /// Pixel-centre latitude [deg] for each band row.
    pub row_lat_deg: &'a [f32],
    /// Full-grid elevation for cast-shadow ray marching.
    pub shadow_elev: &'a [f32],
    pub ncols: usize,
    /// Rows in THIS band.
    pub nrows: usize,
    /// This band's first row in the full grid.
    pub row_offset: usize,
    pub full_nrows: usize,
    /// Pixel size along X / Y in map units (metres for projected CRS).
    pub dx: f64,
    pub dy: f64,
}

impl BandInputs<'_> {
    /// Validate buffer lengths; returns an error message rather than
    /// panicking, since callers come from Python.
    pub fn validate(&self) -> Result<(), String> {
        let band_px = self.ncols * self.nrows;
        let full_px = self.ncols * self.full_nrows;
        if self.ncols == 0 || self.nrows == 0 {
            return Err("empty band".into());
        }
        for (name, len) in [
            ("elevation", self.elevation.len()),
            ("slope", self.slope.len()),
            ("aspect", self.aspect.len()),
            ("linke", self.linke.len()),
            ("albedo", self.albedo.len()),
            ("mask", self.mask.len()),
        ] {
            if len != band_px {
                return Err(format!(
                    "{name} has {len} elements, expected ncols*nrows = {band_px}"
                ));
            }
        }
        if self.row_lat_deg.len() != self.nrows {
            return Err(format!(
                "row_lat_deg has {} entries, expected nrows = {}",
                self.row_lat_deg.len(),
                self.nrows
            ));
        }
        if self.shadow_elev.len() != full_px {
            return Err(format!(
                "shadow_elev has {} elements, expected ncols*full_nrows = {full_px}",
                self.shadow_elev.len()
            ));
        }
        if self.row_offset + self.nrows > self.full_nrows {
            return Err(format!(
                "band rows {}..{} exceed full_nrows {}",
                self.row_offset,
                self.row_offset + self.nrows,
                self.full_nrows
            ));
        }
        Ok(())
    }

    /// Max valid elevation over the FULL grid — the ray-march early-exit
    /// threshold. Must be full-grid, not band-local, or shadows from peaks
    /// outside the band would be missed.
    pub fn full_max_z(&self) -> f64 {
        self.shadow_elev
            .iter()
            .copied()
            .filter(|&v| v != UNDEFZ)
            .fold(f32::NEG_INFINITY, f32::max) as f64
    }
}

/// Which output components to compute for a daily band.
#[derive(Clone, Copy, Debug, Default)]
pub struct DailyWants {
    pub beam: bool,
    pub diffuse: bool,
    pub reflected: bool,
    pub global: bool,
    pub insol: bool,
}

/// Computed outputs for one band. Each `Vec` is `ncols * nrows` long, with
/// skipped pixels set to [`UNDEFZ`]. Only requested components are present.
#[derive(Default)]
pub struct DailyBandOut {
    pub beam: Option<Vec<f32>>,
    pub diffuse: Option<Vec<f32>>,
    pub reflected: Option<Vec<f32>>,
    pub global: Option<Vec<f32>>,
    pub insol: Option<Vec<f32>>,
}

/// Compute daily irradiation for one row band (CPU, rayon-parallel).
///
/// Mirrors the per-pixel loop of `run_raster_computation` in lib.rs, but
/// reads/writes caller-owned buffers and indexes shadows into the full grid.
#[allow(clippy::too_many_arguments)]
pub fn compute_daily_band_cpu(
    inp: &BandInputs<'_>,
    day: i32,
    step: f64,
    solar_constant: f64,
    wants: DailyWants,
    quiet: bool,
) -> Result<DailyBandOut, String> {
    inp.validate()?;
    if !(1..=365).contains(&day) {
        return Err(format!("day must be 1-365, got {day}"));
    }
    if step <= 0.0 {
        return Err(format!("step must be > 0, got {step}"));
    }

    let ncols = inp.ncols;
    let nrows = inp.nrows;
    let npixels = ncols * nrows;

    let declination = com_declin(day);
    let g_norm_extra = com_sol_const(day, solar_constant);
    let sindecl = declination.sin();
    let cosdecl = declination.cos();
    let max_z = inp.full_max_z();

    // All five outputs are always allocated (the rayon loop zips five
    // disjoint per-row chunk iterators); unrequested ones are dropped before
    // returning. Band memory is bounded by the caller's band size.
    let mut o_beam = vec![UNDEFZ; npixels];
    let mut o_diff = vec![UNDEFZ; npixels];
    let mut o_refl = vec![UNDEFZ; npixels];
    let mut o_glob = vec![UNDEFZ; npixels];
    let mut o_insol = vec![UNDEFZ; npixels];

    o_beam
        .par_chunks_mut(ncols)
        .zip(o_diff.par_chunks_mut(ncols))
        .zip(o_refl.par_chunks_mut(ncols))
        .zip(o_glob.par_chunks_mut(ncols))
        .zip(o_insol.par_chunks_mut(ncols))
        .enumerate()
        .for_each(|(band_row, ((((rb, rd), rr), rg), ri))| {
            let lat = inp.row_lat_deg[band_row] as f64 * DEG2RAD;
            // Full-grid row: shadows index the whole DEM, not just this band.
            let full_row = inp.row_offset + band_row;
            for col in 0..ncols {
                let idx = band_row * ncols + col;
                let elev = inp.elevation[idx];
                if elev == UNDEFZ || elev < -1000.0 {
                    continue;
                }
                let m = inp.mask[idx];
                if m == 0.0 || m == UNDEFZ {
                    continue;
                }
                if inp.slope[idx] == UNDEFZ || inp.aspect[idx] == UNDEFZ {
                    continue;
                }

                let slope_rad = inp.slope[idx] as f64 * DEG2RAD;
                let aspect_rad = convert_grass_aspect(inp.aspect[idx] as f64);

                let geom = compute_slope_geometry(slope_rad, aspect_rad, lat, sindecl, cosdecl);
                let (sunrise, sunset) = match horizontal_day_window(lat, sindecl, cosdecl) {
                    Some(s) => s,
                    None => continue, // polar night
                };

                let params = RadiationParams {
                    g_norm_extra,
                    linke: inp.linke[idx] as f64,
                    albedo: inp.albedo[idx] as f64,
                    cbh: 1.0,
                    cdh: 1.0,
                    elevation: elev as f64,
                    solar_altitude: 0.0,
                    slope: slope_rad,
                    aspect: aspect_rad,
                    solar_azimuth: 0.0,
                };

                let shadow_ctx = ShadowContext {
                    elev: inp.shadow_elev,
                    ncols,
                    nrows: inp.full_nrows,
                    dx: inp.dx,
                    dy: inp.dy,
                    max_z,
                    row: full_row,
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
            }
        });

    if !quiet {
        let done = inp.row_offset + nrows;
        eprint!(
            "\rProgress: {:.0}%   ",
            100.0 * done as f64 / inp.full_nrows as f64
        );
    }

    Ok(DailyBandOut {
        beam: wants.beam.then_some(o_beam),
        diffuse: wants.diffuse.then_some(o_diff),
        reflected: wants.reflected.then_some(o_refl),
        global: wants.global.then_some(o_glob),
        insol: wants.insol.then_some(o_insol),
    })
}

/// Derive slope [deg] and aspect [deg, GRASS CCW-from-East] from an elevation
/// band via Horn's 3×3 finite difference.
///
/// NOTE: the edge ring of the BAND is emitted as nodata, so when called per
/// band the interior rows adjacent to a band seam lose one row. Callers
/// processing a large DEM in bands should either pass overlapping bands or
/// accept a one-pixel seam; for the QGIS plugin the DEM is read whole and
/// this is called once, so there is no seam.
pub fn slope_aspect_from_elev(elev: &[f32], ncols: usize, nrows: usize, dx: f64, dy: f64) -> (Vec<f32>, Vec<f32>) {
    // horn_slope_aspect takes a GDAL-style geotransform; only |gt[1]| and
    // |gt[5]| are read, so build a minimal one from the pixel sizes.
    let gt = [0.0, dx, 0.0, 0.0, 0.0, -dy];
    horn_slope_aspect(elev, ncols, nrows, &gt)
}

/// Compute annual PV potential [kWh/m²/yr] for one row band (CPU).
///
/// Accumulates global irradiation over the sampled DOY grid weighted by
/// `day_step`, then scales by `panel_efficiency / 1000`.
#[allow(clippy::too_many_arguments)]
pub fn compute_annual_band_cpu(
    inp: &BandInputs<'_>,
    days: &[i32],
    day_step: i32,
    step: f64,
    solar_constant: f64,
    panel_efficiency: f64,
    horizon: Option<&crate::horizon::HorizonMap>,
    quiet: bool,
) -> Result<Vec<f32>, String> {
    inp.validate()?;
    if days.is_empty() {
        return Err("empty day sampling list".into());
    }
    if day_step <= 0 {
        return Err(format!("day_step must be positive, got {day_step}"));
    }

    let ncols = inp.ncols;
    let nrows = inp.nrows;
    let npixels = ncols * nrows;
    let max_z = inp.full_max_z();

    let day_consts: Vec<(f64, f64, f64, f64)> = days
        .iter()
        .map(|&d| {
            let decl = com_declin(d);
            let g0 = com_sol_const(d, solar_constant);
            (decl, decl.sin(), decl.cos(), g0)
        })
        .collect();

    let day_step_f = day_step as f64;
    let kwh_factor = panel_efficiency / 1000.0;

    let mut potential = vec![UNDEFZ; npixels];

    potential
        .par_chunks_mut(ncols)
        .enumerate()
        .for_each(|(band_row, out_row)| {
            let lat = inp.row_lat_deg[band_row] as f64 * DEG2RAD;
            let full_row = inp.row_offset + band_row;
            for col in 0..ncols {
                let idx = band_row * ncols + col;
                let elev = inp.elevation[idx];
                if elev == UNDEFZ || elev < -1000.0 {
                    continue;
                }
                let m = inp.mask[idx];
                if m == 0.0 || m == UNDEFZ {
                    continue;
                }
                if inp.slope[idx] == UNDEFZ || inp.aspect[idx] == UNDEFZ {
                    continue;
                }

                let slope_rad = inp.slope[idx] as f64 * DEG2RAD;
                let aspect_rad = convert_grass_aspect(inp.aspect[idx] as f64);

                let shadow_ctx = ShadowContext {
                    elev: inp.shadow_elev,
                    ncols,
                    nrows: inp.full_nrows,
                    dx: inp.dx,
                    dy: inp.dy,
                    max_z,
                    row: full_row,
                    col,
                    eye_z: elev as f64,
                    horizon,
                };

                // Riemann sum across the sampled days, weighted by day_step.
                // Skipped days (polar night) contribute zero — matching the
                // path-based implementation's accumulator semantics.
                let mut acc = 0.0f64;
                for &(declination, sindecl, cosdecl, g_norm_extra) in &day_consts {
                    let geom = compute_slope_geometry(slope_rad, aspect_rad, lat, sindecl, cosdecl);
                    // Horizontal-day window: see compute_daily_band_cpu. Keeps
                    // diffuse+reflected on never-sunlit slopes (GRASS parity).
                    let (sunrise, sunset) = match horizontal_day_window(lat, sindecl, cosdecl) {
                        Some(s) => s,
                        None => continue, // polar night
                    };
                    let params = RadiationParams {
                        g_norm_extra,
                        linke: inp.linke[idx] as f64,
                        albedo: inp.albedo[idx] as f64,
                        cbh: 1.0,
                        cdh: 1.0,
                        elevation: elev as f64,
                        solar_altitude: 0.0,
                        slope: slope_rad,
                        aspect: aspect_rad,
                        solar_azimuth: 0.0,
                    };
                    let r = integrate_daily(
                        &geom,
                        &params,
                        sunrise,
                        sunset,
                        step,
                        lat,
                        declination,
                        Some(&shadow_ctx),
                    );
                    acc += r.global * day_step_f;
                }
                out_row[col] = (acc * kwh_factor) as f32;
            }
        });

    if !quiet {
        let done = inp.row_offset + nrows;
        eprint!(
            "\rProgress: {:.0}%   ",
            100.0 * done as f64 / inp.full_nrows as f64
        );
    }

    Ok(potential)
}
