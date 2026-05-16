/// Per-pixel horizon map — precomputed maximum terrain elevation angle per
/// azimuth bin. Replaces the per-timestep cast-shadow ray-march with a single
/// buffer read + compare.
///
/// The horizon along a given direction depends only on the DEM (not on day or
/// time), so this is computed once and reused across every (day, time-step)
/// pair the annual sweep evaluates. For an annual run with 36 sampled days ×
/// ~30 daylight steps that's ~1000 shadow evaluations per pixel collapsed
/// into a single O(n_az) precompute.
use std::f64::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::UNDEFZ;
use crate::shadow::sample_bilinear;

/// Dense per-pixel horizon. Stored as max elevation angle [radians] per
/// azimuth bin, row-major over pixels then azimuth:
/// `data[(row * ncols + col) * n_az + bin]`.
///
/// Values are clamped at 0 — a pixel with no terrain above its own eye level
/// along a given direction stores 0 there, which correctly classifies any
/// sun above the geometric horizon as visible.
pub struct HorizonMap {
    pub n_az: usize,
    pub ncols: usize,
    pub nrows: usize,
    pub data: Vec<f32>,
}

impl HorizonMap {
    #[inline]
    pub fn shaded(&self, row: usize, col: usize, sun_az_compass: f64, sun_alt: f64) -> bool {
        if sun_alt <= 0.0 {
            return true;
        }
        let n = self.n_az as f64;
        // Bin centres lie at (a + 0.5) * 2π/N; map sun_az to a fractional bin
        // index relative to those centres.
        let two_pi = 2.0 * PI;
        let az = sun_az_compass.rem_euclid(two_pi);
        if az.is_nan() {
            return false;
        }
        let bin_f = az * n / two_pi - 0.5;
        let i0_signed = bin_f.floor() as isize;
        let t = (bin_f - i0_signed as f64) as f32;
        let i0 = i0_signed.rem_euclid(self.n_az as isize) as usize;
        let i1 = (i0 + 1) % self.n_az;
        let base = (row * self.ncols + col) * self.n_az;
        let h0 = self.data[base + i0];
        let h1 = self.data[base + i1];
        let horizon_alt = h0 * (1.0 - t) + h1 * t;
        (sun_alt as f32) < horizon_alt
    }
}

/// Build the horizon map for every pixel where `mask != 0` and elevation is
/// valid. Skipped pixels keep horizon = 0 across all bins — those pixels will
/// not be evaluated downstream anyway.
///
/// Algorithm: for each azimuth bin centre, march in pixel space at 0.5-pixel
/// steps using the same bilinear DEM probe as `shadow::ray_blocked`, and
/// record the maximum apparent elevation angle of terrain along that
/// direction. Early-exits when no remaining terrain can beat the current max
/// (uses the precomputed `max_z`).
pub fn compute_horizon_map_cpu(
    elev: &[f32],
    ncols: usize,
    nrows: usize,
    dx: f64,
    dy: f64,
    mask: &[f32],
    n_az: usize,
    quiet: bool,
) -> HorizonMap {
    assert!(n_az >= 4, "n_az must be >= 4");
    assert_eq!(elev.len(), ncols * nrows);
    assert_eq!(mask.len(), ncols * nrows);

    let max_z = elev
        .iter()
        .copied()
        .filter(|&v| v != UNDEFZ)
        .fold(f32::NEG_INFINITY, f32::max) as f64;

    // Sample horizon at each bin centre. The lookup linearly interpolates
    // between adjacent bin centres, reconstructing horizon as a continuous
    // function — that's accurate as long as the true horizon doesn't have
    // sub-bin features (1-pixel-wide ridges between sample directions). For
    // urban DSMs and natural terrain this holds because real obstructions
    // span many azimuths from any rooftop pixel.
    let two_pi = 2.0 * PI;
    let bin_dirs: Vec<(f64, f64, f64)> = (0..n_az)
        .map(|a| {
            let az = (a as f64 + 0.5) * two_pi / n_az as f64;
            let dirx_w = az.sin();
            let diry_w = -az.cos();
            let scale = dirx_w.abs().max(diry_w.abs()).max(1e-12);
            let step_px_x = dirx_w / scale * 0.5;
            let step_px_y = diry_w / scale * 0.5;
            let step_world =
                ((step_px_x * dx).powi(2) + (step_px_y * dy).powi(2)).sqrt();
            (step_px_x, step_px_y, step_world)
        })
        .collect();

    let max_steps = 2 * (ncols + nrows) + 4;
    let mut data = vec![0.0f32; ncols * nrows * n_az];

    let rows_done = AtomicUsize::new(0);
    let report_interval = (nrows / 20).max(1);

    data.par_chunks_mut(ncols * n_az)
        .enumerate()
        .for_each(|(row, row_slab)| {
            for col in 0..ncols {
                let idx = row * ncols + col;
                let elev_v = elev[idx];
                let m = mask[idx];
                if elev_v == UNDEFZ || elev_v < -1000.0 || m == 0.0 || m == UNDEFZ {
                    // Skipped pixels: leave horizon at 0 (will not be queried).
                    continue;
                }
                let eye_z = elev_v as f64 + 1e-3;
                let pix_base = col * n_az;

                for (a, &(spx, spy, sw)) in bin_dirs.iter().enumerate() {
                    let mut x = col as f64 + 0.5;
                    let mut y = row as f64 + 0.5;
                    let mut max_tan = 0.0f64;

                    for step in 1..=max_steps {
                        x += spx;
                        y += spy;
                        if x < 0.0
                            || y < 0.0
                            || x > ncols as f64
                            || y > nrows as f64
                        {
                            break;
                        }
                        let horiz_dist = step as f64 * sw;
                        let max_possible = (max_z - eye_z) / horiz_dist;
                        if max_possible <= max_tan {
                            break;
                        }
                        let z = sample_bilinear(elev, ncols, nrows, x, y);
                        if z.is_nan() {
                            break;
                        }
                        let tan_a = (z - eye_z) / horiz_dist;
                        if tan_a > max_tan {
                            max_tan = tan_a;
                        }
                    }

                    row_slab[pix_base + a] = max_tan.atan() as f32;
                }
            }
            if !quiet {
                let done = rows_done.fetch_add(1, Ordering::Relaxed) + 1;
                if done % report_interval == 0 {
                    eprint!(
                        "\rHorizon precompute: {:.0}%   ",
                        100.0 * done as f64 / nrows as f64
                    );
                }
            }
        });

    if !quiet {
        eprintln!("\rHorizon precompute: 100% ({} bins/pixel)   ", n_az);
    }

    HorizonMap {
        n_az,
        ncols,
        nrows,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shadow::{ShadowContext, ray_blocked};

    /// DEM with a 30 m wall at columns 25..28, all rows. Observer sits at
    /// col=0, mid-row — wall is due east. dx=dy=1 m. The wall is several
    /// pixels wide both axially and in cross-azimuth so it subtends multiple
    /// horizon bins, removing the 1-pixel-peak discretization pitfall.
    fn build_test_dem() -> (Vec<f32>, usize, usize) {
        let ncols = 30;
        let nrows = 11;
        let mut elev = vec![0.0f32; ncols * nrows];
        for row in 0..nrows {
            for col in 25..28 {
                elev[row * ncols + col] = 30.0;
            }
        }
        (elev, ncols, nrows)
    }

    #[test]
    fn horizon_points_at_obstruction() {
        let (elev, ncols, nrows) = build_test_dem();
        let mask = vec![1.0f32; ncols * nrows];
        let hm = compute_horizon_map_cpu(&elev, ncols, nrows, 1.0, 1.0, &mask, 16, true);

        // Observer at row=5, col=0. Wall starts at col=25 → distance 25 m,
        // height 30 m → atan(30/25) ≈ 50.2°. Bin 3 (centre 78.75°) and bin 4
        // (centre 101.25°) bracket due east; both should record the wall.
        let base = (5 * ncols + 0) * hm.n_az;
        for &bin in &[3usize, 4] {
            let h = hm.data[base + bin];
            assert!(
                h > 0.6,
                "expected horizon ~atan(30/25)≈0.876 rad at bin {bin}, got {h}"
            );
        }

        // Due west bin (11) faces away from the wall — should be flat.
        let west_horizon = hm.data[base + 11];
        assert!(
            west_horizon < 1e-3,
            "expected ~0 horizon at west bin (no obstruction), got {west_horizon}"
        );
    }

    /// Behavioural checks on the lookup: sun far above the highest horizon is
    /// never shaded; low sun in the direction of the wall is shaded; same
    /// low sun in the opposite direction is not.
    ///
    /// We don't assert pixel-perfect agreement with the ray-march — the
    /// interpolated lookup is slightly over-conservative at azimuths that
    /// fall on the angular boundary of an obstacle (the bin that contains
    /// the obstacle leaks horizon into its neighbour via linear interp).
    /// That bias is benign for solar work (very small Wh/m² shift) and is
    /// validated end-to-end via Python A/B against `use_horizon=False`.
    #[test]
    fn horizon_lookup_basic_behaviour() {
        let (elev, ncols, nrows) = build_test_dem();
        let mask = vec![1.0f32; ncols * nrows];
        let hm = compute_horizon_map_cpu(&elev, ncols, nrows, 1.0, 1.0, &mask, 16, true);

        let ctx = ShadowContext {
            elev: &elev,
            ncols,
            nrows,
            dx: 1.0,
            dy: 1.0,
            max_z: 30.0,
            row: 5,
            col: 0,
            eye_z: 1e-3,
            horizon: Some(&hm),
        };

        let east = std::f64::consts::FRAC_PI_2;
        let west = 3.0 * std::f64::consts::FRAC_PI_2;
        let alt_low = 10.0_f64.to_radians();
        let alt_high = 80.0_f64.to_radians();

        assert!(
            ray_blocked(&ctx, east, alt_low),
            "low sun east of observer must be blocked by the wall"
        );
        assert!(
            !ray_blocked(&ctx, east, alt_high),
            "high sun east of observer should clear the wall"
        );
        assert!(
            !ray_blocked(&ctx, west, alt_low),
            "low sun west of observer (open horizon) should not be blocked"
        );

        // Also sanity-check stored values: every bin must be in [0, π/2].
        for v in &hm.data {
            assert!(*v >= 0.0 && *v <= std::f64::consts::FRAC_PI_2 as f32);
        }
    }
}
