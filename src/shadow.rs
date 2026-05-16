/// Terrain cast-shadow ray-marching.
///
/// Given a source pixel and a sun direction (compass azimuth + altitude),
/// march a ray from the pixel surface toward the sun, sampling the DEM with
/// bilinear interpolation at half-pixel intervals. Returns true as soon as a
/// terrain probe rises above the ray altitude; returns false if the ray exits
/// the grid, clears the precomputed grid maximum, or hits nodata.
use crate::UNDEFZ;
use crate::horizon::HorizonMap;

/// Bilinear DEM probe. Averages the four surrounding pixel centres weighted
/// by sub-cell offsets. Smooths peaks down across vertical walls (returns
/// fictitious in-between heights), which empirically gives the best mean
/// agreement with r.sun on urban DSMs among bilinear/nearest/max-corner.
#[inline]
pub(crate) fn sample_bilinear(elev: &[f32], ncols: usize, nrows: usize, x: f64, y: f64) -> f64 {
    let fx = (x - 0.5).clamp(0.0, (ncols - 1) as f64);
    let fy = (y - 0.5).clamp(0.0, (nrows - 1) as f64);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(ncols - 1);
    let y1 = (y0 + 1).min(nrows - 1);
    let tx = fx - x0 as f64;
    let ty = fy - y0 as f64;

    let z00 = elev[y0 * ncols + x0];
    let z10 = elev[y0 * ncols + x1];
    let z01 = elev[y1 * ncols + x0];
    let z11 = elev[y1 * ncols + x1];

    if z00 == UNDEFZ || z10 == UNDEFZ || z01 == UNDEFZ || z11 == UNDEFZ {
        return f64::NAN;
    }

    let a = z00 as f64 * (1.0 - tx) + z10 as f64 * tx;
    let b = z01 as f64 * (1.0 - tx) + z11 as f64 * tx;
    a * (1.0 - ty) + b * ty
}

/// Per-raster shadow context. `elev` is row-major, `ncols * nrows` long.
/// `dx`/`dy` are pixel sizes in meters; `max_z` is the max valid elevation in
/// the grid (an early-exit threshold for the ray-march).
///
/// If `horizon` is `Some`, [`ray_blocked`] skips the per-call ray-march and
/// reads the precomputed horizon angle for `(row, col)` instead.
pub struct ShadowContext<'a> {
    pub elev: &'a [f32],
    pub ncols: usize,
    pub nrows: usize,
    pub dx: f64,
    pub dy: f64,
    pub max_z: f64,
    pub row: usize,
    pub col: usize,
    pub eye_z: f64,
    pub horizon: Option<&'a HorizonMap>,
}

/// Returns true if terrain blocks the line of sight from `(row, col, eye_z)`
/// toward the sun at compass-bearing azimuth `sun_az_compass` and altitude
/// `sun_alt` (both radians).
pub fn ray_blocked(ctx: &ShadowContext, sun_az_compass: f64, sun_alt: f64) -> bool {
    if sun_alt <= 0.0 {
        return true;
    }
    if let Some(h) = ctx.horizon {
        return h.shaded(ctx.row, ctx.col, sun_az_compass, sun_alt);
    }

    // Pixel-space direction. Compass: 0=N, 90=E, 180=S, 270=W.
    // East = +x_pixel, North = -y_pixel (row index grows south).
    let dirx_w = sun_az_compass.sin();
    let diry_w = -sun_az_compass.cos();

    // 0.5-pixel step along the dominant axis.
    let scale = dirx_w.abs().max(diry_w.abs()).max(1e-12);
    let step_px_x = dirx_w / scale * 0.5;
    let step_px_y = diry_w / scale * 0.5;

    let step_world =
        ((step_px_x * ctx.dx).powi(2) + (step_px_y * ctx.dy).powi(2)).sqrt();
    let rise_per_step = step_world * sun_alt.tan();

    let mut x = ctx.col as f64 + 0.5;
    let mut y = ctx.row as f64 + 0.5;
    // Tiny clearance avoids self-blocking from bilinear ties on the source pixel.
    let mut ray_z = ctx.eye_z + 1e-3;

    let max_steps = 2 * (ctx.ncols + ctx.nrows) + 4;

    for _ in 0..max_steps {
        x += step_px_x;
        y += step_px_y;
        ray_z += rise_per_step;

        if x < 0.0 || y < 0.0 || x > ctx.ncols as f64 || y > ctx.nrows as f64 {
            return false;
        }
        if ray_z >= ctx.max_z {
            return false;
        }

        let z = sample_bilinear(ctx.elev, ctx.ncols, ctx.nrows, x, y);
        if z.is_nan() {
            return false;
        }
        if z > ray_z {
            return true;
        }
    }
    false
}
