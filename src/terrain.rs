/// Horn (1981) 3×3 finite-difference slope and aspect from an elevation grid.
///
/// Aspect is emitted in the GRASS convention (0=E, 90=N, 180=W, 270=S; CCW
/// from East) so the result feeds directly into `convert_grass_aspect` in the
/// existing r.sun pipeline. Slope is in degrees.
///
/// Edge ring and any pixel whose 3×3 neighborhood touches `UNDEFZ` is set to
/// `UNDEFZ` in both outputs — this matches the standard whitebox/GRASS
/// behavior and the existing per-pixel UNDEFZ guard in the radiation loop.
use crate::UNDEFZ;

const RAD2DEG: f64 = 180.0 / std::f64::consts::PI;

/// Compute slope (deg) and aspect (deg, GRASS CCW-from-East) for every
/// interior pixel of `elev`. `gt` is a GDAL geo-transform: `gt[1]` is pixel
/// width along X, `gt[5]` is pixel height along Y (typically negative).
pub(crate) fn horn_slope_aspect(
    elev: &[f32],
    ncols: usize,
    nrows: usize,
    gt: &[f64; 6],
) -> (Vec<f32>, Vec<f32>) {
    let npixels = ncols * nrows;
    let mut slope = vec![UNDEFZ; npixels];
    let mut aspect = vec![UNDEFZ; npixels];

    if ncols < 3 || nrows < 3 {
        return (slope, aspect);
    }

    let dx = gt[1].abs();
    let dy = gt[5].abs();
    if dx == 0.0 || dy == 0.0 {
        return (slope, aspect);
    }

    let inv_8dx = 1.0 / (8.0 * dx);
    let inv_8dy = 1.0 / (8.0 * dy);

    for row in 1..nrows - 1 {
        for col in 1..ncols - 1 {
            let idx = row * ncols + col;

            let z = [
                elev[(row - 1) * ncols + (col - 1)], // a (NW)
                elev[(row - 1) * ncols + col],       // b (N)
                elev[(row - 1) * ncols + (col + 1)], // c (NE)
                elev[row * ncols + (col - 1)],       // d (W)
                elev[row * ncols + (col + 1)],       // f (E)
                elev[(row + 1) * ncols + (col - 1)], // g (SW)
                elev[(row + 1) * ncols + col],       // h (S)
                elev[(row + 1) * ncols + (col + 1)], // i (SE)
            ];
            if z.iter().any(|&v| v == UNDEFZ) {
                continue;
            }
            let (a, b, c, d, f, g, h, i) = (
                z[0] as f64,
                z[1] as f64,
                z[2] as f64,
                z[3] as f64,
                z[4] as f64,
                z[5] as f64,
                z[6] as f64,
                z[7] as f64,
            );

            // Horn's weighted central differences.
            let dzdx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) * inv_8dx;
            // Row index grows southward when gt[5] < 0, so dzdy here is rise/run
            // from south to north (positive = uphill northward).
            let dzdy = ((a + 2.0 * b + c) - (g + 2.0 * h + i)) * inv_8dy;

            let slope_rad = (dzdx * dzdx + dzdy * dzdy).sqrt().atan();
            slope[idx] = (slope_rad * RAD2DEG) as f32;

            // GRASS aspect is the *downhill* bearing measured CCW from East
            // (270 = south-facing). The gradient (dzdx, dzdy) points uphill,
            // so the downhill vector is its negation. Flat → 0 by convention.
            aspect[idx] = if dzdx == 0.0 && dzdy == 0.0 {
                0.0
            } else {
                let a = (-dzdy).atan2(-dzdx) * RAD2DEG;
                let a = if a < 0.0 { a + 360.0 } else { a };
                a as f32
            };
        }
    }

    (slope, aspect)
}
