/// Solar geometry calculations for r.sun port
/// Implements the ESRA-based solar model matching GRASS GIS r.sun conventions
use std::f64::consts::PI;
pub const PI2: f64 = PI * 2.0;
pub const PIHALF: f64 = PI * 0.5;
pub const DEG2RAD: f64 = PI / 180.0;
pub const RAD2DEG: f64 = 180.0 / PI;
/// Hour angle step in radians per hour (15°/hour)
pub const HOURANGLE: f64 = PI / 12.0;

/// Undefined/no-data sentinel value (matches GRASS UNDEFZ)
pub const UNDEFZ: f32 = -9999.0;

/// Compute solar declination for a given day of year.
///
/// Returns the declination in radians using r.sun's sign convention
/// (stored as negative astronomical declination).
///
/// Formula from r.sun.cpp `com_declin`:
///   d1 = 2π * day / 365.25
///   decl = asin(0.3978 * sin(d1 - 1.4 + 0.0355 * sin(d1 - 0.0489)))
///   returns -decl  (r.sun stores negative of astronomical declination)
pub fn com_declin(day: i32) -> f64 {
    let d1 = PI2 * day as f64 / 365.25;
    let decl = (0.3978 * (d1 - 1.4 + 0.0355 * (d1 - 0.0489).sin()).sin()).asin();
    -decl
}

/// Compute the extraterrestrial solar irradiance [W/m²] accounting for
/// Earth-Sun distance variation throughout the year.
///
/// Based on ESRA formula:
///   G0 = solar_constant * (1 + 0.0334 * cos(2π*day/365.25 - 0.048869))
pub fn com_sol_const(day: i32, solar_constant: f64) -> f64 {
    let d1 = PI2 * day as f64 / 365.25;
    solar_constant * (1.0 + 0.0334 * (d1 - 0.048869).cos())
}

/// Slope geometry derived from terrain slope and aspect, combined with
/// the geographic position and solar declination.
///
/// This struct represents the "equivalent latitude" concept from ESRA:
/// a slope can be treated as a horizontal surface at an effective latitude
/// and longitude offset.
#[derive(Clone, Default)]
pub struct SlopeGeometry {
    /// sin of the effective (equivalent) latitude for this slope
    pub sin_phi_l: f64,
    /// Effective (equivalent) latitude [radians]
    pub latid_l: f64,
    /// Effective longitude offset [radians]
    pub longit_l: f64,
    /// Whether to apply a 12-hour shift to align illumination timing
    pub shift12hrs: bool,
    /// Precomputed: cos(latid_l) * cos(declination)
    pub lum_c31_l: f64,
    /// Precomputed: sin_phi_l * sin(declination)
    pub lum_c33_l: f64,
    /// Precomputed: longit_l + (PI if shift12hrs else 0). Hoists a per-step
    /// branch out of `slope_incidence` and `compute_sunrise_sunset`.
    pub longit_l_adj: f64,
}

/// Compute slope geometry parameters from terrain slope, aspect, and solar geometry.
///
/// Uses r.sun's sign convention where:
///   sinlat = sin(-latitude), coslat = cos(-latitude)
///   declination is stored as -astronomical_declination
///
/// For flat terrain (slope=0): effective lat = geographic lat, longit_l = 0
/// For a slope: the effective lat/lon represent where a horizontal surface
/// would receive the same radiation as the slope.
pub fn compute_slope_geometry(
    slope: f64,    // terrain slope [radians]
    aspect: f64,   // terrain aspect [radians], GRASS convention (0=N, 90=E but stored as 0=E CCW)
    latitude: f64, // geographic latitude [radians]
    sindecl: f64,  // sin(declination) with r.sun sign convention
    cosdecl: f64,  // cos(declination)
) -> SlopeGeometry {
    // r.sun convention: uses -latitude in trig functions
    let sinlat = (-latitude).sin(); // = -sin(lat)
    let coslat = (-latitude).cos(); // = cos(lat)

    // Flat terrain: aspect is meaningless. Bypass the unstable atan(small/small)
    // path that determines shift12hrs — the sign of tan_lam_l for slope≈0 depends
    // on floating-point precision and is unreliable on the GPU (f32) side.
    if slope.abs() < 1e-9 {
        let sin_phi_l = sinlat;
        let latid_l = sin_phi_l.clamp(-1.0, 1.0).asin();
        return SlopeGeometry {
            sin_phi_l,
            latid_l,
            longit_l: 0.0,
            shift12hrs: false,
            lum_c31_l: latid_l.cos() * cosdecl,
            lum_c33_l: sin_phi_l * sindecl,
            longit_l_adj: 0.0,
        };
    }

    // Compute direction cosines for the slope normal
    // cos_u = sin(slope) [slope normal z-component factor]
    // sin_u = cos(slope)
    let cos_u = (PIHALF - slope).cos(); // = sin(slope)
    let sin_u = (PIHALF - slope).sin(); // = cos(slope)

    // cos_v = cos(PI/2 + aspect) = -sin(aspect)
    // sin_v = sin(PI/2 + aspect) = cos(aspect)
    let cos_v = (PIHALF + aspect).cos(); // = -sin(aspect)
    let sin_v = (PIHALF + aspect).sin(); // = cos(aspect)

    // Effective latitude (sin of equivalent latitude for the slope normal)
    let sin_phi_l = -coslat * cos_u * sin_v + sinlat * sin_u;
    let latid_l = sin_phi_l.clamp(-1.0, 1.0).asin();

    // Effective longitude offset
    let q1 = sinlat * cos_u * sin_v + coslat * sin_u;
    let (tan_lam_l, longit_l) = if q1.abs() > 1e-10 {
        let tl = -cos_u * cos_v / q1;
        (tl, tl.atan())
    } else {
        (f64::INFINITY, PIHALF)
    };

    // Determine if a 12-hour shift is needed for correct illumination timing
    // isBestAM: the computed effective longitude implies morning illumination
    // shouldBeBestAM: based on aspect, the slope faces east (should be AM)
    let is_best_am = tan_lam_l > 0.0;
    let should_be_best_am = aspect > 0.0 && aspect <= PI;
    let shift12hrs = should_be_best_am != is_best_am;

    // Precomputed coefficients for the incidence formula:
    // s0 = lum_c31_l * cos(omega - longit_l_adj) + lum_c33_l
    let lum_c31_l = latid_l.cos() * cosdecl;
    let lum_c33_l = sin_phi_l * sindecl;

    let longit_l_adj = if shift12hrs { longit_l + PI } else { longit_l };

    SlopeGeometry {
        sin_phi_l,
        latid_l,
        longit_l,
        shift12hrs,
        lum_c31_l,
        lum_c33_l,
        longit_l_adj,
    }
}

/// Compute slope/horizon sunrise and sunset hour angles [radians].
///
/// Returns (sunrise_angle, sunset_angle) where:
///   - 0 = solar noon
///   - negative = before noon (morning)
///   - positive = after noon (evening)
///
/// Returns None if sun never rises over this slope on this day (polar night
/// or fully shaded slope).
pub fn compute_sunrise_sunset(geom: &SlopeGeometry) -> Option<(f64, f64)> {
    let cos_latid_l = geom.latid_l.cos();
    if cos_latid_l.abs() < 1e-10 {
        return None;
    }

    // When lum_c31_l = 0, the slope never receives direct beam
    if geom.lum_c31_l.abs() < 1e-10 {
        return None;
    }

    // cos(omega_s - longit_l) = -lum_c33_l / lum_c31_l
    let cos_sr = -geom.lum_c33_l / geom.lum_c31_l;

    if cos_sr.abs() > 1.0 {
        if cos_sr < -1.0 {
            // Sun always illuminates this slope (midnight sun case)
            return Some((-PI + geom.longit_l, PI + geom.longit_l));
        } else {
            // Sun never illuminates this slope
            return None;
        }
    }

    let sr_angle = cos_sr.acos(); // [0, PI]

    let sunrise = -sr_angle + geom.longit_l_adj; // morning (negative hour angle)
    let sunset = sr_angle + geom.longit_l_adj; // evening (positive hour angle)

    Some((sunrise, sunset))
}

/// Convert hour angle [radians] to local solar time [hours, 0-24]
#[inline]
pub fn hour_angle_to_time(omega: f64) -> f64 {
    12.0 + omega * 12.0 / PI
}

/// Compute the cosine of the solar incidence angle on the slope
/// (equivalent to the illumination factor for beam radiation).
///
/// For flat terrain: s0 = sin(solar_altitude)
/// For a slope: s0 = cos(angle_between_solar_beam_and_slope_normal)
///
/// Uses the precomputed slope geometry coefficients.
/// Returns 0.0 if the slope is in shadow (sun behind the slope).
#[inline]
pub fn slope_incidence(geom: &SlopeGeometry, time_angle: f64) -> f64 {
    let s0 = geom.lum_c31_l * (time_angle - geom.longit_l_adj).cos() + geom.lum_c33_l;
    s0.max(0.0)
}

/// Solar position at a given time.
pub struct SolarPosition {
    /// Solar altitude angle [radians], negative means below horizon
    pub altitude: f64,
    /// sin(altitude), i.e., the cosine of the solar zenith angle
    pub sin_alt: f64,
    /// Solar azimuth [radians] measured from south, negative=east, positive=west
    pub azimuth: f64,
}

/// Compute solar position (altitude and azimuth) for given lat/lon and time.
///
/// Uses r.sun's sign convention:
///   sinlat = sin(-latitude), coslat = cos(-latitude)
///   declination = -astronomical_declination
pub fn solar_position(latitude: f64, declination: f64, time_angle: f64) -> SolarPosition {
    let sinlat = (-latitude).sin();
    let coslat = (-latitude).cos();
    let sindecl = declination.sin();
    let cosdecl = declination.cos();

    // sin(altitude) = sin(lat)*sin(decl) + cos(lat)*cos(decl)*cos(omega)
    // With r.sun sign conventions this gives the correct astronomical formula
    let sin_h = sinlat * sindecl + coslat * cosdecl * time_angle.cos();
    let altitude = sin_h.clamp(-1.0, 1.0).asin();

    // Compute solar azimuth (measured from south, positive=west)
    let cos_h = (1.0 - sin_h * sin_h).sqrt().max(1e-10);
    let cos_a = (sindecl - sinlat * sin_h) / (coslat * cos_h);
    let cos_a = cos_a.clamp(-1.0, 1.0);
    let mut azimuth = cos_a.acos();
    if time_angle > 0.0 {
        // Afternoon: azimuth is in the west half
        azimuth = PI2 - azimuth;
    }

    SolarPosition {
        altitude,
        sin_alt: sin_h,
        azimuth,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_declination_solstice() {
        // Summer solstice around day 172
        let decl = com_declin(172);
        // Astronomical declination should be ~+23.4°, r.sun stores negative
        let astro_decl_deg = -decl * RAD2DEG;
        assert!(
            (astro_decl_deg - 23.4).abs() < 0.5,
            "Summer solstice declination: {astro_decl_deg:.2}°"
        );
    }

    #[test]
    fn test_declination_equinox() {
        // Spring equinox around day 80
        let decl = com_declin(80);
        let astro_decl_deg = -decl * RAD2DEG;
        assert!(
            astro_decl_deg.abs() < 2.0,
            "Spring equinox declination: {astro_decl_deg:.2}°"
        );
    }

    #[test]
    fn test_flat_slope_geometry() {
        // Flat terrain: effective lat = geographic lat
        let lat = 48.0 * DEG2RAD;
        let decl = com_declin(172); // summer solstice
        let sindecl = decl.sin();
        let cosdecl = decl.cos();

        let geom = compute_slope_geometry(0.0, 0.0, lat, sindecl, cosdecl);

        // For flat terrain, effective latitude should equal geographic latitude
        let effective_lat_deg = geom.latid_l * RAD2DEG;
        // latid_l should be approximately -lat due to r.sun's sign conventions
        assert!(
            (effective_lat_deg + 48.0).abs() < 0.1,
            "Flat terrain effective lat: {effective_lat_deg:.2}°"
        );

        // For flat terrain, longit_l should be 0
        assert!(
            geom.longit_l.abs() < 1e-10,
            "Flat terrain longit_l: {}",
            geom.longit_l
        );
    }

    #[test]
    fn test_flat_incidence_equals_sin_altitude() {
        // For flat terrain, the slope incidence should equal sin(solar altitude)
        let lat = 48.0 * DEG2RAD;
        let decl = com_declin(172);
        let sindecl = decl.sin();
        let cosdecl = decl.cos();

        let geom = compute_slope_geometry(0.0, 0.0, lat, sindecl, cosdecl);

        // At solar noon (omega=0)
        let omega = 0.0_f64;
        let s0 = slope_incidence(&geom, omega);
        let pos = solar_position(lat, decl, omega);

        assert!(
            (s0 - pos.sin_alt).abs() < 1e-10,
            "Flat incidence s0={s0:.6} != sin_alt={:.6}",
            pos.sin_alt
        );
    }

    #[test]
    fn test_sunrise_time_summer() {
        // At lat 48°N on summer solstice, sunrise should be around 4h AM
        let lat = 48.0 * DEG2RAD;
        let decl = com_declin(172);
        let sindecl = decl.sin();
        let cosdecl = decl.cos();

        let geom = compute_slope_geometry(0.0, 0.0, lat, sindecl, cosdecl);
        let (sr, ss) = compute_sunrise_sunset(&geom).expect("Sun should rise");

        let sr_time = hour_angle_to_time(sr);
        let ss_time = hour_angle_to_time(ss);

        assert!(
            (sr_time - 4.08).abs() < 0.5,
            "Summer sunrise at 48°N: {sr_time:.2}h"
        );
        assert!(
            (ss_time - 19.92).abs() < 0.5,
            "Summer sunset at 48°N: {ss_time:.2}h"
        );
    }
}
