/// Solar radiation model for r.sun port
///
/// Implements the ESRA (European Solar Radiation Atlas) clear-sky model
/// for beam, diffuse, and reflected radiation, matching the formulas used
/// in GRASS GIS r.sun (rsunlib.c).
use std::f64::consts::PI;

const PI2: f64 = PI * 2.0;
const RAD2DEG: f64 = 180.0 / PI;

/// Parameters controlling the radiation model for a single calculation step.
pub struct RadiationParams {
    /// Extraterrestrial solar irradiance [W/m²] (from com_sol_const)
    pub g_norm_extra: f64,
    /// Linke atmospheric turbidity coefficient [-] (typical 1–7, clear sky ~2–3)
    pub linke: f64,
    /// Ground albedo coefficient [-] (0–1, typical ~0.2)
    pub albedo: f64,
    /// Real-sky beam radiation coefficient (cloud factor) [-]
    pub cbh: f64,
    /// Real-sky diffuse radiation coefficient (haze factor) [-]
    pub cdh: f64,
    /// Terrain elevation [m] (for air mass pressure correction)
    pub elevation: f64,
    /// Solar altitude [radians]
    pub solar_altitude: f64,
    /// Terrain slope [radians]
    pub slope: f64,
    /// Terrain aspect [radians, CW from North / compass bearing]
    pub aspect: f64,
    /// Solar azimuth [radians, CW from North / compass bearing]
    pub solar_azimuth: f64,
}

/// Result of the radiation calculation for a single time step.
pub struct RadiationStep {
    /// Beam (direct) irradiance on slope [W/m²]
    pub beam: f64,
    /// Beam (direct) irradiance on horizontal surface [W/m²]
    pub beam_horiz: f64,
    /// Diffuse irradiance on slope [W/m²]
    pub diffuse: f64,
    /// Reflected irradiance on slope [W/m²]
    pub reflected: f64,
}

/// Compute optical air mass with Kasten-Young formula including atmospheric
/// refraction correction and elevation pressure correction.
///
/// Matches GRASS rsunlib.c `brad()` computation exactly:
///   elevationCorr = exp(-z / 8434.5)
///   drefract = 0.061359 * (0.1594 + h*(1.123 + 0.065656*h))
///                       / (1 + h*(28.9344 + 277.3971*h))
///   m = elevationCorr / (sin(h + drefract) + 0.50572 * (h_deg_refracted + 6.07995)^-1.6364)
fn optical_air_mass(solar_altitude_rad: f64, elevation_m: f64) -> f64 {
    let elevation_corr = (-elevation_m / 8434.5).exp();
    // Atmospheric refraction correction (radians)
    let temp1 = 0.1594 + solar_altitude_rad * (1.123 + 0.065656 * solar_altitude_rad);
    let temp2 = 1.0 + solar_altitude_rad * (28.9344 + 277.3971 * solar_altitude_rad);
    let drefract = 0.061359 * temp1 / temp2;
    let h0refract = solar_altitude_rad + drefract;
    let h0refract_deg = h0refract * RAD2DEG;
    elevation_corr / (h0refract.sin() + 0.50572 * (h0refract_deg + 6.07995_f64).powf(-1.6364))
}

/// Compute Rayleigh optical thickness as a function of optical air mass.
///
/// Louche et al. (1986) polynomial fit:
///   m ≤ 20: delta_R = 1 / (6.6296 + 1.7513*m - 0.1202*m² + 0.0065*m³ - 0.00013*m⁴)
///   m >  20: delta_R = 1 / (10.4 + 0.718*m)
fn rayleigh_optical_thickness(am: f64) -> f64 {
    if am <= 20.0 {
        let am2 = am * am;
        let am3 = am2 * am;
        let am4 = am3 * am;
        1.0 / (6.6296 + 1.7513 * am - 0.1202 * am2 + 0.0065 * am3 - 0.00013 * am4)
    } else {
        1.0 / (10.4 + 0.718 * am)
    }
}

/// Compute beam (direct) radiation on slope and horizontal surface.
///
/// Equivalent to `brad()` in r.sun's rsunlib.c.
///
/// # Arguments
/// * `s0` - cosine of incidence angle on slope (from `slope_incidence()`)
/// * `params` - radiation and geometry parameters
///
/// # Returns
/// (beam_on_slope, beam_on_horizontal) both in [W/m²]
pub fn beam_radiation(s0: f64, params: &RadiationParams) -> (f64, f64) {
    let sin_h = params.solar_altitude.sin().max(0.0);
    let am = optical_air_mass(params.solar_altitude, params.elevation);
    let delta_r = rayleigh_optical_thickness(am);
    let tn = (-0.8662 * params.linke * am * delta_r).exp();

    // Beam on horizontal: bh = cbh * G0 * sin_h * Tn
    let beam_horiz = params.cbh * params.g_norm_extra * tn * sin_h;

    // Beam on slope: br = bh * s0 / sin_h = cbh * G0 * Tn * s0 (for slope)
    //               br = bh                                      (for flat)
    let beam_slope = if params.slope > 1e-6 {
        params.cbh * params.g_norm_extra * tn * s0
    } else {
        beam_horiz
    };

    (beam_slope, beam_horiz)
}

/// Compute diffuse and reflected radiation on slope.
///
/// Equivalent to `drad()` in r.sun's rsunlib.c.
///
/// Uses the ESRA 3-term polynomial diffuse model:
///   tn = -0.015843 + TL*(0.030543 + 0.0003797*TL)   [turbidity transmittance]
///   A1..A3 = Linke-turbidity-dependent coefficients
///   fd = A1 + A2*sin_h + A3*sin_h²                   [diffuse distribution function]
///   dh = cdh * G0 * fd * tn                           [diffuse on horizontal]
///
/// For slope: Muneer-Hay anisotropic sky model from rsunlib.c drad()
/// For flat:  dr = dh,  rr = 0
///
/// # Arguments
/// * `s0` - cosine of incidence angle on slope (0 if in shadow)
/// * `beam_horiz` - beam irradiance on horizontal [W/m²] (0 if in shadow)
/// * `is_shadow` - true if pixel is in terrain cast shadow
/// * `params` - radiation and geometry parameters
///
/// # Returns
/// (diffuse_on_slope, reflected_on_slope) both in [W/m²]
pub fn diffuse_radiation(
    s0: f64,
    beam_horiz: f64,
    is_shadow: bool,
    params: &RadiationParams,
) -> (f64, f64) {
    let sin_h = params.solar_altitude.sin().max(0.0);
    let tl = params.linke;

    // ESRA diffuse transmittance and distribution (from GRASS rsunlib.c drad())
    let tn = -0.015843 + tl * (0.030543 + 0.0003797 * tl);
    let a1b = 0.26463 + tl * (-0.061581 + 0.0031408 * tl);
    let a1 = if a1b * tn < 0.0022 { 0.0022 / tn } else { a1b };
    let a2 = 2.04020 + tl * (0.018945 - 0.011161 * tl);
    let a3 = -1.3025 + tl * (0.039231 + 0.0085079 * tl);
    let fd = a1 + a2 * sin_h + a3 * sin_h * sin_h;
    let dh = (params.cdh * params.g_norm_extra * fd * tn).max(0.0);

    let gh = beam_horiz + dh;

    if params.slope > 1e-6 {
        // Slope: Muneer-Hay anisotropic diffuse model (GRASS rsunlib.c drad())
        let cosslope = params.slope.cos();
        let sinslope = params.slope.sin();
        // Sky view factor for the tilted surface
        let r_sky = (1.0 + cosslope) / 2.0;
        // Muneer horizon-brightening factor
        let fg = sinslope - params.slope * cosslope - PI * (0.5 * params.slope).sin().powi(2);
        // Beam clearness index for circumsolar anisotropy
        let g0_sin_h = (params.g_norm_extra * sin_h).max(1e-10);
        let kb = (beam_horiz / g0_sin_h).clamp(0.0, 1.0);

        let fx = if is_shadow || s0 <= 0.0 {
            // Slope in shadow: diffuse-only sky contribution
            r_sky + fg * 0.252271
        } else if params.solar_altitude >= 0.1 {
            // Normal illumination: Hay-Davies circumsolar + isotropic sky
            ((0.00263 - kb * (0.712 + 0.6883 * kb)) * fg + r_sky) * (1.0 - kb)
                + kb * s0 / sin_h.max(1e-10)
        } else {
            // Very low sun angle: azimuth-dependent horizon formula
            let a_ln_raw = params.solar_azimuth - params.aspect;
            let a_ln = if a_ln_raw > PI {
                a_ln_raw - PI2
            } else if a_ln_raw < -PI {
                a_ln_raw + PI2
            } else {
                a_ln_raw
            };
            ((0.00263 - kb * (0.712 + 0.6883 * kb)) * fg + r_sky) * (1.0 - kb)
                + kb * sinslope * a_ln.cos() / (0.1 - 0.008 * params.solar_altitude)
        };

        let dr = (dh * fx).max(0.0);
        // Ground-reflected irradiance: rr = alb * (bh+dh) * (1-cos(β)) / 2
        let rr = (params.albedo * gh * (1.0 - cosslope) / 2.0).max(0.0);
        (dr, rr)
    } else {
        // Flat terrain: dr = dh, rr = 0 (GRASS convention for slope == 0)
        (dh, 0.0)
    }
}

/// Compute all radiation components for a single time step.
///
/// Returns `None` if the sun is below the horizon (solar_altitude ≤ 0).
pub fn compute_radiation(
    s0: f64,
    is_shadow: bool,
    params: &RadiationParams,
) -> Option<RadiationStep> {
    if params.solar_altitude <= 0.0 {
        return None;
    }

    let (beam_slope, beam_horiz) = if !is_shadow && s0 > 0.0 {
        beam_radiation(s0, params)
    } else {
        (0.0, 0.0)
    };

    let (diffuse_slope, reflected_slope) = diffuse_radiation(s0, beam_horiz, is_shadow, params);

    Some(RadiationStep {
        beam: beam_slope,
        beam_horiz,
        diffuse: diffuse_slope,
        reflected: reflected_slope,
    })
}

/// Integrate daily radiation for a single pixel using Mode 2 (daily irradiation).
///
/// Iterates from sunrise to sunset in time_step increments and accumulates:
///   beam_irradiation [Wh/m²/day] = Σ beam_irradiance * Δt
///   diff_irradiation [Wh/m²/day] = Σ diff_irradiance * Δt
///   refl_irradiation [Wh/m²/day] = Σ refl_irradiance * Δt
///   insol_time [h/day] = number of time steps with s0 > 0
///
/// # Arguments
/// * `geom` - precomputed slope geometry
/// * `base_params` - radiation parameters (solar_altitude/azimuth updated per step)
/// * `sunrise_angle` - sunrise hour angle [radians]
/// * `sunset_angle` - sunset hour angle [radians]
/// * `time_step` - integration time step [hours]
/// * `latitude` - geographic latitude [radians]
/// * `declination` - solar declination [radians] (r.sun sign convention)
pub fn integrate_daily(
    geom: &crate::solar::SlopeGeometry,
    base_params: &RadiationParams,
    sunrise_angle: f64,
    sunset_angle: f64,
    time_step: f64,
    latitude: f64,
    declination: f64,
    shadow: Option<&crate::shadow::ShadowContext>,
) -> DailyIrradiation {
    use crate::solar::{HOURANGLE, slope_incidence};

    let step_rad = time_step * HOURANGLE;

    // Start at the center of the first step after sunrise
    let sr_step_no = (sunrise_angle / step_rad).floor() as i64;
    let first_angle = if sunrise_angle - sr_step_no as f64 * step_rad > 0.5 * step_rad {
        (sr_step_no as f64 + 1.5) * step_rad
    } else {
        (sr_step_no as f64 + 0.5) * step_rad
    };

    // Hoist all per-pixel-constant coefficients out of the time-step loop.
    // These depend only on linke/albedo/slope/elevation — none on solar_altitude.
    let pre = PixelPrecomp::new(base_params);

    // Cache lat trig once for the per-step solar_position calls.
    let sinlat = (-latitude).sin();
    let coslat = (-latitude).cos();
    let sindecl = declination.sin();
    let cosdecl = declination.cos();

    let mut beam_total = 0.0_f64;
    let mut diff_total = 0.0_f64;
    let mut refl_total = 0.0_f64;
    let mut insol_total = 0.0_f64;

    let mut time_angle = first_angle;

    while time_angle <= sunset_angle {
        let pos = solar_position_cached(sinlat, coslat, sindecl, cosdecl, time_angle);

        if pos.altitude > 0.0 {
            let s0 = slope_incidence(geom, time_angle);
            // Convert solar azimuth from CCW-from-South to compass bearing
            // (matches GRASS solarAzimuth). Needed for both the cast-shadow
            // ray-march and the low-sun diffuse branch.
            let solar_az_compass = (PI - pos.azimuth).rem_euclid(PI2);

            let is_shadow = match shadow {
                Some(ctx) => crate::shadow::ray_blocked(ctx, solar_az_compass, pos.altitude),
                None => false,
            };

            let (beam_s, beam_h) = if !is_shadow && s0 > 0.0 {
                beam_step(s0, pos.altitude, &pre)
            } else {
                (0.0, 0.0)
            };

            let (diff_s, refl_s) =
                diffuse_step(s0, beam_h, pos.altitude, solar_az_compass, is_shadow, &pre);

            if !is_shadow && s0 > 0.0 {
                insol_total += time_step;
                beam_total += beam_s * time_step;
            }
            diff_total += diff_s * time_step;
            refl_total += refl_s * time_step;
        }

        time_angle += step_rad;
    }

    DailyIrradiation {
        beam: beam_total,
        diffuse: diff_total,
        reflected: refl_total,
        global: beam_total + diff_total + refl_total,
        insol_time: insol_total,
    }
}

/// Per-pixel-constant coefficients precomputed once outside the time-step loop.
///
/// In the original code, `beam_radiation` and `diffuse_radiation` recomputed
/// these from scratch at every time step (typically 24–48 per day per pixel).
/// All of them depend only on quantities that are constant within one day at
/// one pixel: linke turbidity, albedo, slope, elevation, and the
/// extraterrestrial constant.
struct PixelPrecomp {
    // Beam-path:
    elevation_corr: f64, // exp(-elev / 8434.5) — input to Kasten-Young air mass
    cbh_g0: f64,         // cbh * G0 — pulled outside the per-step Tn product
    linke: f64,
    has_slope: bool,
    // Diffuse-path (ESRA Linke polynomial — depends only on TL):
    a1: f64,
    a2: f64,
    a3: f64,
    cdh_g0_tn: f64, // cdh * G0 * tn_diff — outer factor on fd
    // Slope geometry for the diffuse model:
    sinslope: f64,
    r_sky: f64, // (1 + cos β) / 2
    fg: f64,    // Muneer horizon-brightening factor
    albedo_factor: f64, // albedo * (1 - cos β) / 2 — outer factor on reflected
    aspect: f64,
    g_norm_extra: f64,
}

impl PixelPrecomp {
    fn new(p: &RadiationParams) -> Self {
        let tl = p.linke;
        let tn_diff = -0.015843 + tl * (0.030543 + 0.0003797 * tl);
        let a1b = 0.26463 + tl * (-0.061581 + 0.0031408 * tl);
        let a1 = if a1b * tn_diff < 0.0022 {
            0.0022 / tn_diff
        } else {
            a1b
        };
        let a2 = 2.04020 + tl * (0.018945 - 0.011161 * tl);
        let a3 = -1.3025 + tl * (0.039231 + 0.0085079 * tl);

        let has_slope = p.slope > 1e-6;
        let cosslope = p.slope.cos();
        let sinslope = p.slope.sin();
        let r_sky = (1.0 + cosslope) / 2.0;
        let fg = sinslope - p.slope * cosslope - PI * (0.5 * p.slope).sin().powi(2);

        Self {
            elevation_corr: (-p.elevation / 8434.5).exp(),
            cbh_g0: p.cbh * p.g_norm_extra,
            linke: tl,
            has_slope,
            a1,
            a2,
            a3,
            cdh_g0_tn: p.cdh * p.g_norm_extra * tn_diff,
            sinslope,
            r_sky,
            fg,
            albedo_factor: p.albedo * (1.0 - cosslope) / 2.0,
            aspect: p.aspect,
            g_norm_extra: p.g_norm_extra,
        }
    }
}

/// Per-step beam radiation kernel using hoisted coefficients.
/// Returns (beam_on_slope, beam_on_horizontal).
#[inline]
fn beam_step(s0: f64, solar_altitude: f64, pre: &PixelPrecomp) -> (f64, f64) {
    let sin_h = solar_altitude.sin().max(0.0);
    // Inlined optical_air_mass (skipping the `exp` — already in elevation_corr).
    let temp1 = 0.1594 + solar_altitude * (1.123 + 0.065656 * solar_altitude);
    let temp2 = 1.0 + solar_altitude * (28.9344 + 277.3971 * solar_altitude);
    let drefract = 0.061359 * temp1 / temp2;
    let h0refract = solar_altitude + drefract;
    let h0refract_deg = h0refract * RAD2DEG;
    let am = pre.elevation_corr
        / (h0refract.sin() + 0.50572 * (h0refract_deg + 6.07995_f64).powf(-1.6364));
    let delta_r = rayleigh_optical_thickness(am);
    let tn = (-0.8662 * pre.linke * am * delta_r).exp();

    let cbh_g0_tn = pre.cbh_g0 * tn;
    let beam_horiz = cbh_g0_tn * sin_h;
    let beam_slope = if pre.has_slope { cbh_g0_tn * s0 } else { beam_horiz };
    (beam_slope, beam_horiz)
}

/// Per-step diffuse + reflected kernel using hoisted coefficients.
#[inline]
fn diffuse_step(
    s0: f64,
    beam_horiz: f64,
    solar_altitude: f64,
    solar_az_compass: f64,
    is_shadow: bool,
    pre: &PixelPrecomp,
) -> (f64, f64) {
    let sin_h = solar_altitude.sin().max(0.0);
    let fd = pre.a1 + pre.a2 * sin_h + pre.a3 * sin_h * sin_h;
    let dh = (pre.cdh_g0_tn * fd).max(0.0);
    let gh = beam_horiz + dh;

    if pre.has_slope {
        let g0_sin_h = (pre.g_norm_extra * sin_h).max(1e-10);
        let kb = (beam_horiz / g0_sin_h).clamp(0.0, 1.0);

        let fx = if is_shadow || s0 <= 0.0 {
            pre.r_sky + pre.fg * 0.252271
        } else if solar_altitude >= 0.1 {
            ((0.00263 - kb * (0.712 + 0.6883 * kb)) * pre.fg + pre.r_sky) * (1.0 - kb)
                + kb * s0 / sin_h.max(1e-10)
        } else {
            let a_ln_raw = solar_az_compass - pre.aspect;
            let a_ln = if a_ln_raw > PI {
                a_ln_raw - PI2
            } else if a_ln_raw < -PI {
                a_ln_raw + PI2
            } else {
                a_ln_raw
            };
            ((0.00263 - kb * (0.712 + 0.6883 * kb)) * pre.fg + pre.r_sky) * (1.0 - kb)
                + kb * pre.sinslope * a_ln.cos() / (0.1 - 0.008 * solar_altitude)
        };

        let dr = (dh * fx).max(0.0);
        let rr = (pre.albedo_factor * gh).max(0.0);
        (dr, rr)
    } else {
        (dh, 0.0)
    }
}

/// Branch-free `solar_position` for the inner loop: latitude/declination
/// trig are passed in (precomputed once per pixel by `integrate_daily`)
/// instead of being recomputed on every time step.
#[inline]
fn solar_position_cached(
    sinlat: f64,
    coslat: f64,
    sindecl: f64,
    cosdecl: f64,
    time_angle: f64,
) -> crate::solar::SolarPosition {
    let sin_h = sinlat * sindecl + coslat * cosdecl * time_angle.cos();
    let altitude = sin_h.clamp(-1.0, 1.0).asin();

    let cos_h = (1.0 - sin_h * sin_h).sqrt().max(1e-10);
    let cos_a = ((sindecl - sinlat * sin_h) / (coslat * cos_h)).clamp(-1.0, 1.0);
    let mut azimuth = cos_a.acos();
    if time_angle > 0.0 {
        azimuth = PI2 - azimuth;
    }
    crate::solar::SolarPosition {
        altitude,
        sin_alt: sin_h,
        azimuth,
    }
}

/// Daily-integrated solar irradiation results for a single pixel.
#[derive(Clone, Default)]
pub struct DailyIrradiation {
    /// Beam (direct) irradiation [Wh/m²/day]
    pub beam: f64,
    /// Diffuse irradiation [Wh/m²/day]
    pub diffuse: f64,
    /// Reflected irradiation [Wh/m²/day]
    pub reflected: f64,
    /// Global (total) irradiation [Wh/m²/day]
    pub global: f64,
    /// Insolation time [h/day]
    pub insol_time: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solar::*;

    fn make_noon_params(linke: f64, albedo: f64, lat_deg: f64, day: i32) -> RadiationParams {
        let lat = lat_deg * DEG2RAD;
        let decl = com_declin(day);
        let g0 = com_sol_const(day, 1367.0);
        let pos = solar_position(lat, decl, 0.0); // noon
        // At noon: solar azimuth = 180° (South) in compass bearing
        let solar_az_compass = (PI - pos.azimuth).rem_euclid(super::PI2);
        RadiationParams {
            g_norm_extra: g0,
            linke,
            albedo,
            cbh: 1.0,
            cdh: 1.0,
            elevation: 200.0,
            solar_altitude: pos.altitude,
            slope: 0.0,
            aspect: 0.0,
            solar_azimuth: solar_az_compass,
        }
    }

    #[test]
    fn test_beam_radiation_reasonable() {
        // At lat 48°N, summer solstice noon, clear sky (TL=3)
        // Expect beam on horizontal > 600 W/m²
        let params = make_noon_params(3.0, 0.2, 48.0, 172);
        let s0 = params.solar_altitude.sin();
        let (b_slope, b_horiz) = beam_radiation(s0, &params);
        assert!(b_horiz > 600.0, "Beam horizontal {b_horiz:.1} W/m²");
        assert!(b_horiz < 1200.0, "Beam horizontal {b_horiz:.1} W/m²");
        assert!((b_slope - b_horiz).abs() < 1.0, "Flat: slope≈horiz");
    }

    #[test]
    fn test_diffuse_radiation_reasonable() {
        // Diffuse should be smaller than beam but positive
        let params = make_noon_params(3.0, 0.2, 48.0, 172);
        let sin_h = params.solar_altitude.sin();
        let (_b_slope, b_horiz) = beam_radiation(sin_h, &params);
        let (d_slope, r_slope) = diffuse_radiation(sin_h, b_horiz, false, &params);

        assert!(d_slope > 10.0, "Diffuse on flat {d_slope:.1} W/m²");
        assert!(d_slope < b_horiz, "Diffuse < beam");
        assert!(r_slope >= 0.0, "Reflected >= 0");
        println!("Noon diffuse: {d_slope:.1} W/m², beam: {b_horiz:.1} W/m²");
    }

    #[test]
    fn test_daily_integration() {
        // Daily global radiation at 48°N, summer solstice, flat terrain
        let lat = 48.0 * DEG2RAD;
        let day = 172;
        let decl = com_declin(day);
        let sindecl = decl.sin();
        let cosdecl = decl.cos();

        let geom = compute_slope_geometry(0.0, 0.0, lat, sindecl, cosdecl);
        let (sr, ss) = compute_sunrise_sunset(&geom).unwrap();

        let params = RadiationParams {
            g_norm_extra: com_sol_const(day, 1367.0),
            linke: 3.0,
            albedo: 0.2,
            cbh: 1.0,
            cdh: 1.0,
            elevation: 300.0,
            solar_altitude: 0.0,
            slope: 0.0,
            aspect: 0.0,
            solar_azimuth: 0.0,
        };

        let result = integrate_daily(&geom, &params, sr, ss, 0.5, lat, decl, None);

        println!(
            "Day {}: beam={:.0}, diff={:.0}, refl={:.0}, glob={:.0} Wh/m²/day, insol={:.1}h",
            day, result.beam, result.diffuse, result.reflected, result.global, result.insol_time
        );
        // Expect global irradiation > 6000 Wh/m²/day for clear sky summer solstice
        assert!(
            result.global > 6000.0,
            "Global irradiation {:.0} Wh/m²/day too low",
            result.global
        );
        assert!(
            result.global < 15000.0,
            "Global irradiation {:.0} Wh/m²/day too high",
            result.global
        );
        assert!(
            result.insol_time > 12.0,
            "Insolation time {:.1}h too short",
            result.insol_time
        );
    }
}
