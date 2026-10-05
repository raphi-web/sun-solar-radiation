// r.sun WebGPU compute shader
//
// One GPU thread per raster pixel. Each thread runs the full sunrise-to-sunset
// time integration independently, producing beam, diffuse, reflected, global
// irradiation and insolation time for that pixel.
//
// All math mirrors rsunlib.c and src/radiation.rs exactly (in f32).

const PI: f32     = 3.14159265358979323846;
const PI2: f32    = 6.28318530717958647692;
const PIHALF: f32 = 1.57079632679489661923;
const DEG2RAD: f32 = 0.01745329251994329577;
const RAD2DEG: f32 = 57.29577951308232522583;
const HOURANGLE: f32 = 0.26179938779914943654;   // PI / 12
const UNDEFZ: f32 = -9999.0;

// ── Uniforms (day-level constants) ──────────────────────────────────────────

struct Uniforms {
    sindecl:     f32,   // sin(declination) — r.sun sign convention
    cosdecl:     f32,   // cos(declination)
    g_norm_extra: f32,  // extraterrestrial irradiance [W/m²]
    step_rad:    f32,   // integration step [radians of hour angle]
    ncols:       u32,   // full-grid width (and per-tile width — cols never tile)
    nrows:       u32,   // rows in *this* tile
    // Pixels covered along the X dispatch axis. Used to fold gid.y back into
    // the linear pixel index when the host issues a 2D dispatch to stay under
    // wgpu's 65535 per-dimension workgroup-count limit.
    dispatch_x_pixels: u32,
    full_nrows:  u32,   // full-grid height (for cast-shadow ray-march)
    row_offset:  u32,   // this tile's first row in the full grid
    max_z:       f32,   // max valid elevation in the full grid (ray-march early exit)
    dx:          f32,   // pixel x size [m]
    dy:          f32,   // pixel y size [m]
    // 0 → ray-march each shadow test; >0 → look up horizon[idx * n_az + bin].
    n_az:        u32,
    // Optional pixel window inside this tile: when pixel_count > 0 only
    // pixels [pixel_start, pixel_start + pixel_count) are computed. Lets the
    // host split a tile into short GPU submissions (driver watchdogs kill
    // long ones). 0 = whole tile (CLI paths leave it 0).
    pixel_start: u32,
    pixel_count: u32,
    _pad2:       u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

// ── Storage bindings ─────────────────────────────────────────────────────────

// Full-grid elevation (length = u.ncols * u.full_nrows). The cast-shadow
// ray-march needs to step outside the current tile, so this binding holds the
// whole raster — unlike the per-tile slope/aspect/linke/albedo/mask buffers.
@group(0) @binding(1) var<storage, read>       elev_buf:   array<f32>;
@group(0) @binding(2) var<storage, read>       slope_buf:  array<f32>;
@group(0) @binding(3) var<storage, read>       aspect_buf: array<f32>;
@group(0) @binding(4) var<storage, read>       linke_buf:  array<f32>;
@group(0) @binding(5) var<storage, read>       albedo_buf: array<f32>;
@group(0) @binding(6) var<storage, read>       mask_buf:   array<f32>;
@group(0) @binding(7) var<storage, read_write> out_beam:   array<f32>;
@group(0) @binding(8) var<storage, read_write> out_diff:   array<f32>;
@group(0) @binding(9) var<storage, read_write> out_refl:   array<f32>;
@group(0) @binding(10) var<storage, read_write> out_glob:  array<f32>;
@group(0) @binding(11) var<storage, read_write> out_insol: array<f32>;
// Per-row latitude (WGS84°). Length = u.nrows; indexed by the pixel's row in
// the current tile. Lets the shader stay CRS-agnostic — the host reprojects
// once on the CPU.
@group(0) @binding(12) var<storage, read>       row_lat_buf: array<f32>;
// Per-pixel horizon table — length = u.ncols * u.nrows * u.n_az. Indexed
// `idx * u.n_az + bin` where idx is the tile-local pixel and bin is the
// azimuth bucket. When u.n_az == 0 this binding is a dummy (host binds a
// 4-byte buffer) and the shader uses the ray-march path instead.
@group(0) @binding(13) var<storage, read>       horizon_buf: array<f32>;

// ── Slope geometry ────────────────────────────────────────────────────────────

struct SlopeGeometry {
    sin_phi_l: f32,
    latid_l:   f32,
    longit_l:  f32,
    shift12hrs: bool,
    lum_c31_l: f32,
    lum_c33_l: f32,
}

// Convert GRASS aspect (0=E CCW, degrees) to r.sun radians (CW from North).
fn convert_grass_aspect(aspect_deg: f32) -> f32 {
    // Same as the Rust side: NO special case for 0 (due East). Flat pixels
    // are handled by the slope<1e-6 branch below, which ignores aspect.
    var c: f32;
    if aspect_deg < 90.0 {
        c = 90.0 - aspect_deg;
    } else {
        c = 450.0 - aspect_deg;
    }
    return c * DEG2RAD;
}

fn compute_slope_geometry(slope: f32, aspect: f32, lat: f32,
                          sindecl: f32, cosdecl: f32) -> SlopeGeometry {
    let sinlat = sin(-lat);
    let coslat = cos(-lat);

    // Flat terrain: aspect is meaningless. Bypass the unstable atan(small/small)
    // path — in f32 the sign of tan_lam_l flips around slope=0 and corrupts
    // shift12hrs.
    if slope < 1e-6 {
        let sin_phi_l = sinlat;
        let latid_l   = asin(clamp(sin_phi_l, -1.0, 1.0));
        let lum_c31_l = cos(latid_l) * cosdecl;
        let lum_c33_l = sin_phi_l * sindecl;
        return SlopeGeometry(sin_phi_l, latid_l, 0.0, false, lum_c31_l, lum_c33_l);
    }

    let cos_u = cos(PIHALF - slope);   // = sin(slope)
    let sin_u = sin(PIHALF - slope);   // = cos(slope)
    let cos_v = cos(PIHALF + aspect);  // = -sin(aspect)
    let sin_v = sin(PIHALF + aspect);  // = cos(aspect)

    let sin_phi_l = -coslat * cos_u * sin_v + sinlat * sin_u;
    let latid_l   = asin(clamp(sin_phi_l, -1.0, 1.0));

    let q1 = sinlat * cos_u * sin_v + coslat * sin_u;
    var tan_lam_l: f32;
    var longit_l:  f32;
    if abs(q1) > 1e-10 {
        tan_lam_l = -cos_u * cos_v / q1;
        longit_l  = atan(tan_lam_l);
    } else {
        tan_lam_l = 1e30;
        longit_l  = PIHALF;
    }

    let is_best_am      = tan_lam_l > 0.0;
    let should_be_am    = aspect > 0.0 && aspect <= PI;
    let shift12hrs      = should_be_am != is_best_am;

    let lum_c31_l = cos(latid_l) * cosdecl;
    let lum_c33_l = sin_phi_l * sindecl;

    return SlopeGeometry(sin_phi_l, latid_l, longit_l, shift12hrs,
                         lum_c31_l, lum_c33_l);
}

// ── Sunrise / sunset ──────────────────────────────────────────────────────────

struct SunriseSunset {
    sunrise: f32,
    sunset:  f32,
    valid:   bool,
}

fn compute_sunrise_sunset(g: SlopeGeometry) -> SunriseSunset {
    if abs(cos(g.latid_l)) < 1e-10 || abs(g.lum_c31_l) < 1e-10 {
        return SunriseSunset(0.0, 0.0, false);
    }

    let cos_sr = -g.lum_c33_l / g.lum_c31_l;

    if abs(cos_sr) > 1.0 {
        if cos_sr < -1.0 {
            // Midnight sun: sun always above horizon for this slope
            return SunriseSunset(-PI + g.longit_l, PI + g.longit_l, true);
        }
        return SunriseSunset(0.0, 0.0, false);
    }

    let sr_angle = acos(cos_sr);
    var ladj = g.longit_l;
    if g.shift12hrs { ladj += PI; }

    return SunriseSunset(-sr_angle + ladj, sr_angle + ladj, true);
}

// ── Solar position ────────────────────────────────────────────────────────────

struct SolarPos {
    altitude: f32,
    azimuth:  f32,
    sin_alt:  f32,
}

fn solar_position(lat: f32, sindecl: f32, cosdecl: f32,
                  time_angle: f32) -> SolarPos {
    let sinlat = sin(-lat);
    let coslat = cos(-lat);

    let sin_h = sinlat * sindecl + coslat * cosdecl * cos(time_angle);
    let altitude = asin(clamp(sin_h, -1.0, 1.0));

    let cos_h = max(sqrt(max(1.0 - sin_h * sin_h, 0.0)), 1e-10);
    let cos_a = clamp((sindecl - sinlat * sin_h) / (coslat * cos_h), -1.0, 1.0);
    var azimuth = acos(cos_a);
    if time_angle > 0.0 { azimuth = PI2 - azimuth; }

    return SolarPos(altitude, azimuth, sin_h);
}

// ── Atmospheric air mass (with refraction) ────────────────────────────────────

fn optical_air_mass(alt_rad: f32, elevation_m: f32) -> f32 {
    let elev_corr = exp(-elevation_m / 8434.5);
    let temp1 = 0.1594 + alt_rad * (1.123 + 0.065656 * alt_rad);
    let temp2 = 1.0    + alt_rad * (28.9344 + 277.3971 * alt_rad);
    let drefract = 0.061359 * temp1 / temp2;
    let h0r      = alt_rad + drefract;
    let h0r_deg  = h0r * RAD2DEG;
    return elev_corr / (sin(h0r) + 0.50572 * pow(h0r_deg + 6.07995, -1.6364));
}

// ── Rayleigh optical thickness ────────────────────────────────────────────────

fn rayleigh_optical_thickness(am: f32) -> f32 {
    if am <= 20.0 {
        let am2 = am * am;
        let am3 = am2 * am;
        let am4 = am3 * am;
        return 1.0 / (6.6296 + 1.7513*am - 0.1202*am2 + 0.0065*am3 - 0.00013*am4);
    }
    return 1.0 / (10.4 + 0.718 * am);
}

// ── Beam radiation ────────────────────────────────────────────────────────────
// Returns vec2(beam_on_slope, beam_on_horizontal) [W/m²]

fn beam_radiation(s0: f32, alt: f32, elev_m: f32,
                  g0: f32, linke: f32, slope: f32) -> vec2<f32> {
    let sin_h     = max(sin(alt), 0.0);
    let am        = optical_air_mass(alt, elev_m);
    let delta_r   = rayleigh_optical_thickness(am);
    let tn        = exp(-0.8662 * linke * am * delta_r);

    let beam_horiz = g0 * tn * sin_h;              // cbh = 1
    var beam_slope: f32;
    if slope > 1e-6 {
        beam_slope = g0 * tn * s0;
    } else {
        beam_slope = beam_horiz;
    }
    return vec2<f32>(beam_slope, beam_horiz);
}

// ── Diffuse + reflected radiation ─────────────────────────────────────────────
// Returns vec2(diffuse_on_slope, reflected) [W/m²]
// Implements GRASS rsunlib.c drad() exactly.

fn diffuse_radiation(s0: f32, beam_horiz: f32, is_shadow: bool,
                     alt: f32, slope: f32, aspect: f32, solar_azimuth: f32,
                     g0: f32, linke: f32, albedo: f32) -> vec2<f32> {
    let sin_h = max(sin(alt), 0.0);
    let tl    = linke;

    // ESRA turbidity transmittance + diffuse distribution function (drad)
    let tn  = -0.015843 + tl * (0.030543 + 0.0003797 * tl);
    let a1b = 0.26463   + tl * (-0.061581 + 0.0031408 * tl);
    var a1: f32;
    if a1b * tn < 0.0022 { a1 = 0.0022 / tn; } else { a1 = a1b; }
    let a2 = 2.04020 + tl * (0.018945 - 0.011161 * tl);
    let a3 = -1.3025 + tl * (0.039231 + 0.0085079 * tl);
    let fd = a1 + a2 * sin_h + a3 * sin_h * sin_h;
    let dh = max(g0 * fd * tn, 0.0);               // cdh = 1

    let gh = beam_horiz + dh;

    if slope > 1e-6 {
        let cosslope = cos(slope);
        let sinslope = sin(slope);
        let r_sky = (1.0 + cosslope) / 2.0;
        let hs    = sin(0.5 * slope);
        let fg    = sinslope - slope * cosslope - PI * hs * hs;
        let g0sh  = max(g0 * sin_h, 1e-10);
        let kb    = clamp(beam_horiz / g0sh, 0.0, 1.0);

        var fx: f32;
        if is_shadow || s0 <= 0.0 {
            fx = r_sky + fg * 0.252271;
        } else if alt >= 0.1 {
            fx = ((0.00263 - kb * (0.712 + 0.6883 * kb)) * fg + r_sky) * (1.0 - kb)
                 + kb * s0 / max(sin_h, 1e-10);
        } else {
            var a_ln = solar_azimuth - aspect;
            if a_ln >  PI { a_ln -= PI2; }
            if a_ln < -PI { a_ln += PI2; }
            fx = ((0.00263 - kb * (0.712 + 0.6883 * kb)) * fg + r_sky) * (1.0 - kb)
                 + kb * sinslope * cos(a_ln) / (0.1 - 0.008 * alt);
        }

        let dr = max(dh * fx, 0.0);
        let rr = max(albedo * gh * (1.0 - cosslope) / 2.0, 0.0);
        return vec2<f32>(dr, rr);
    }

    // Flat terrain: dr = dh, rr = 0
    return vec2<f32>(max(dh, 0.0), 0.0);
}

// ── Cast-shadow ray-march ─────────────────────────────────────────────────────
// Mirrors src/shadow.rs: 0.5-pixel DDA along the sun azimuth, bilinear DEM
// probe (smooths peaks down across vertical walls; empirically the closest
// match to r.sun on urban DSMs among bilinear/nearest/max-corner). Early-exits
// on grid escape, ray clearing max_z, or stepping into nodata.

fn sample_bilinear(x: f32, y: f32) -> f32 {
    let nx = u.ncols;
    let ny = u.full_nrows;
    let fx = clamp(x - 0.5, 0.0, f32(nx - 1u));
    let fy = clamp(y - 0.5, 0.0, f32(ny - 1u));
    let x0 = u32(floor(fx));
    let y0 = u32(floor(fy));
    let x1 = min(x0 + 1u, nx - 1u);
    let y1 = min(y0 + 1u, ny - 1u);
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);

    let z00 = elev_buf[y0 * nx + x0];
    let z10 = elev_buf[y0 * nx + x1];
    let z01 = elev_buf[y1 * nx + x0];
    let z11 = elev_buf[y1 * nx + x1];

    if z00 == UNDEFZ || z10 == UNDEFZ || z01 == UNDEFZ || z11 == UNDEFZ {
        return UNDEFZ;
    }

    let a = z00 * (1.0 - tx) + z10 * tx;
    let b = z01 * (1.0 - tx) + z11 * tx;
    return a * (1.0 - ty) + b * ty;
}

// Mirror of src/horizon.rs::HorizonMap::shaded — linear interpolation
// between the two adjacent bin samples. Bin centres are at
// (a + 0.5) * 2π/N, so the lookup wraps with rem_euclid semantics.
fn horizon_shaded(idx: u32, sun_az_compass: f32, sun_alt: f32) -> bool {
    if sun_alt <= 0.0 { return true; }
    let n = f32(u.n_az);
    let az = sun_az_compass - PI2 * floor(sun_az_compass / PI2);
    let bin_f = az * n / PI2 - 0.5;
    let i0_f  = floor(bin_f);
    let t     = bin_f - i0_f;
    let n_i   = i32(u.n_az);
    var i0    = i32(i0_f) % n_i;
    if i0 < 0 { i0 += n_i; }
    let i0_u = u32(i0);
    let i1_u = (i0_u + 1u) % u.n_az;
    let base = idx * u.n_az;
    let h0 = horizon_buf[base + i0_u];
    let h1 = horizon_buf[base + i1_u];
    let horizon_alt = h0 * (1.0 - t) + h1 * t;
    return sun_alt < horizon_alt;
}

fn ray_blocked(full_row: u32, col: u32, eye_z: f32,
               sun_az_compass: f32, sun_alt: f32) -> bool {
    if sun_alt <= 0.0 { return true; }

    let dirx_w = sin(sun_az_compass);          // east component
    let diry_w = -cos(sun_az_compass);         // image-y grows south, so −cos
    let scale  = max(max(abs(dirx_w), abs(diry_w)), 1e-12);
    let step_px_x = dirx_w / scale * 0.5;
    let step_px_y = diry_w / scale * 0.5;

    let sx = step_px_x * u.dx;
    let sy = step_px_y * u.dy;
    let step_world    = sqrt(sx * sx + sy * sy);
    let rise_per_step = step_world * tan(sun_alt);

    var x = f32(col) + 0.5;
    var y = f32(full_row) + 0.5;
    // Tiny clearance avoids bilinear ties from the source pixel itself.
    var ray_z = eye_z + 1e-3;

    let max_steps = 2u * (u.ncols + u.full_nrows) + 4u;
    let nx_f = f32(u.ncols);
    let ny_f = f32(u.full_nrows);

    for (var i: u32 = 0u; i < max_steps; i = i + 1u) {
        x = x + step_px_x;
        y = y + step_px_y;
        ray_z = ray_z + rise_per_step;

        if x < 0.0 || y < 0.0 || x > nx_f || y > ny_f { return false; }
        if ray_z >= u.max_z { return false; }

        let z = sample_bilinear(x, y);
        if z == UNDEFZ { return false; }
        if z > ray_z { return true; }
    }
    return false;
}

// ── Main compute kernel ───────────────────────────────────────────────────────

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var idx = gid.y * u.dispatch_x_pixels + gid.x;
    if u.pixel_count != 0u {
        if idx >= u.pixel_count { return; }
        idx = idx + u.pixel_start;
    }
    let total = u.ncols * u.nrows;
    if idx >= total { return; }

    // Per-tile pixel coords + full-grid row (used to index the full elevation
    // buffer for both the source pixel and the cast-shadow ray-march).
    let row      = idx / u.ncols;
    let col      = idx % u.ncols;
    let full_row = row + u.row_offset;
    let full_idx = full_row * u.ncols + col;

    // ── Nodata + mask guard ───────────────────────────────────────────────────
    // Mask = 0 (or nodata) → skip pixel, emit UNDEFZ. Any non-zero value =
    // "compute". This matches the CPU path in lib.rs.
    let mask = mask_buf[idx];
    let elev = elev_buf[full_idx];
    if elev <= -1000.0 || elev == UNDEFZ || mask == 0.0 || mask == UNDEFZ {
        out_beam[idx]  = UNDEFZ;
        out_diff[idx]  = UNDEFZ;
        out_refl[idx]  = UNDEFZ;
        out_glob[idx]  = UNDEFZ;
        out_insol[idx] = UNDEFZ;
        return;
    }

    // Latitude is precomputed per row on the CPU (WGS84°), so this shader is
    // CRS-agnostic — no geo-transform math here.
    let lat = row_lat_buf[row] * DEG2RAD;

    // Per-pixel parameters
    let slope_deg  = slope_buf[idx];
    let aspect_deg = aspect_buf[idx];
    let linke_v    = linke_buf[idx];
    let albedo_v   = albedo_buf[idx];

    if slope_deg == UNDEFZ || aspect_deg == UNDEFZ {
        out_beam[idx]  = UNDEFZ;
        out_diff[idx]  = UNDEFZ;
        out_refl[idx]  = UNDEFZ;
        out_glob[idx]  = UNDEFZ;
        out_insol[idx] = UNDEFZ;
        return;
    }

    let slope_rad  = slope_deg * DEG2RAD;
    let aspect_rad = convert_grass_aspect(aspect_deg);

    // ── Slope geometry + integration window ─────────────────────────────────
    let geom = compute_slope_geometry(slope_rad, aspect_rad, lat,
                                      u.sindecl, u.cosdecl);
    // Integrate over the HORIZONTAL day (GRASS r.sun parity). The slope-plane
    // window degenerates for never-sunlit orientations (north-facing in
    // winter) and partial-day ones (east/west), dropping their diffuse and
    // reflected radiation. beam/insolation stay gated by s0 > 0 below.
    let flat_geom = compute_slope_geometry(0.0, 0.0, lat, u.sindecl, u.cosdecl);
    let sr = compute_sunrise_sunset(flat_geom);
    if (!sr.valid) {
        // Polar night: the sun never rises on the horizontal either.
        out_beam[idx]  = UNDEFZ;
        out_diff[idx]  = UNDEFZ;
        out_refl[idx]  = UNDEFZ;
        out_glob[idx]  = UNDEFZ;
        out_insol[idx] = UNDEFZ;
        return;
    }

    // ── Find first integration step (centred within the step) ─────────────────
    let sr_step_no = floor(sr.sunrise / u.step_rad);
    var first_angle: f32;
    if sr.sunrise - sr_step_no * u.step_rad > 0.5 * u.step_rad {
        first_angle = (sr_step_no + 1.5) * u.step_rad;
    } else {
        first_angle = (sr_step_no + 0.5) * u.step_rad;
    }

    // ── Time integration loop ─────────────────────────────────────────────────
    var beam_total: f32 = 0.0;
    var diff_total: f32 = 0.0;
    var refl_total: f32 = 0.0;
    var insol_total: f32 = 0.0;

    let step_h = u.step_rad / HOURANGLE;   // step size in hours

    var time_angle = first_angle;
    var iters: u32  = 0u;

    while time_angle <= sr.sunset && iters < 200u {
        iters += 1u;

        let pos = solar_position(lat, u.sindecl, u.cosdecl, time_angle);

        if pos.altitude > 0.0 {
            // Slope incidence cosine
            var ladj = geom.longit_l;
            if geom.shift12hrs { ladj += PI; }
            let s0 = max(geom.lum_c31_l * cos(time_angle - ladj) + geom.lum_c33_l, 0.0);

            // Solar azimuth: convert from my CCW-from-South to compass (CW-from-North)
            // compass = (PI - my_azimuth) mod PI2
            var az_compass = PI - pos.azimuth;
            az_compass = az_compass - PI2 * floor(az_compass / PI2);

            // Shadow test: precomputed horizon lookup if available, else
            // a per-call cast-shadow ray-march against the full-grid DEM.
            var is_shadow: bool;
            if u.n_az > 0u {
                is_shadow = horizon_shaded(idx, az_compass, pos.altitude);
            } else {
                is_shadow = ray_blocked(full_row, col, elev, az_compass, pos.altitude);
            }

            // Beam radiation
            var b_slope: f32 = 0.0;
            var b_horiz: f32 = 0.0;
            if !is_shadow && s0 > 0.0 {
                let bv = beam_radiation(s0, pos.altitude, elev, u.g_norm_extra, linke_v, slope_rad);
                b_slope = bv.x;
                b_horiz = bv.y;
            }

            // Diffuse + reflected radiation
            let dv = diffuse_radiation(s0, b_horiz, is_shadow,
                                       pos.altitude, slope_rad, aspect_rad, az_compass,
                                       u.g_norm_extra, linke_v, albedo_v);

            if !is_shadow && s0 > 0.0 {
                insol_total += step_h;
                beam_total  += b_slope * step_h;
            }
            diff_total += dv.x * step_h;
            refl_total += dv.y * step_h;
        }

        time_angle += u.step_rad;
    }

    out_beam[idx]  = beam_total;
    out_diff[idx]  = diff_total;
    out_refl[idx]  = refl_total;
    out_glob[idx]  = beam_total + diff_total + refl_total;
    out_insol[idx] = insol_total;
}
