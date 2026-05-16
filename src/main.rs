/// r.sun solar irradiation model — CLI entry point
///
/// Thin wrapper around the `sun` library crate.
/// All computation logic lives in `lib.rs`.
use clap::Parser;
use sun::{
    create_dummy_elevation,
    gpu::compute_raster_gpu,
    run_raster_computation,
    solar::RAD2DEG,
    solar::{com_declin, com_sol_const},
};

/// r.sun solar irradiation model (Rust port of GRASS GIS r.sun)
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Input elevation raster [m] (GeoTIFF)
    #[arg(long)]
    elevation: Option<String>,

    /// Input slope raster [degrees]
    #[arg(long)]
    slope: Option<String>,

    /// Input aspect raster [degrees, GRASS convention: 0=E CCW]
    #[arg(long)]
    aspect: Option<String>,

    /// Input Linke turbidity raster
    #[arg(long)]
    linke: Option<String>,

    /// Input albedo raster
    #[arg(long)]
    albedo: Option<String>,

    /// Input mask raster (GeoTIFF). Non-zero pixels are computed;
    /// zero pixels are written as nodata.
    #[arg(long)]
    mask: Option<String>,

    /// Constant slope value [degrees]. Omit to derive from DEM via Horn 3×3.
    #[arg(long)]
    slope_value: Option<f64>,

    /// Constant aspect value [degrees, GRASS: 270=south]. Omit to derive from DEM.
    #[arg(long)]
    aspect_value: Option<f64>,

    /// Constant Linke turbidity coefficient
    #[arg(long, default_value = "3.0")]
    linke_value: f64,

    /// Constant ground albedo
    #[arg(long, default_value = "0.2")]
    albedo_value: f64,

    /// Solar constant [W/m²]
    #[arg(long, default_value = "1367.0")]
    solar_constant: f64,

    /// Day of year [1–365]
    #[arg(long, required = true)]
    day: i32,

    /// Time step for daily integration [decimal hours]
    #[arg(long, default_value = "0.5")]
    step: f64,

    /// Output global irradiation raster [Wh/m²/day]
    #[arg(long)]
    glob_rad: Option<String>,

    /// Output beam irradiation raster [Wh/m²/day]
    #[arg(long)]
    beam_rad: Option<String>,

    /// Output diffuse irradiation raster [Wh/m²/day]
    #[arg(long)]
    diff_rad: Option<String>,

    /// Output reflected irradiation raster [Wh/m²/day]
    #[arg(long)]
    refl_rad: Option<String>,

    /// Output insolation time raster [h/day]
    #[arg(long)]
    insol_time: Option<String>,

    /// Create and use a dummy test raster (100×100 pixels over Austria)
    #[arg(long, default_value = "false")]
    create_dummy: bool,

    /// Use WebGPU acceleration for the pixel computation
    #[arg(long, default_value = "false")]
    gpu: bool,

    /// Suppress all progress and informational output to stderr
    #[arg(long, short, default_value = "false")]
    quiet: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = Args::parse();

    // If --create-dummy is given (or no --elevation provided), write a synthetic
    // DEM. The output path is the user-supplied --elevation, or "dummy_elevation.tif"
    // in the current directory as a fallback.
    if args.create_dummy || args.elevation.is_none() {
        let dummy_path = args
            .elevation
            .clone()
            .unwrap_or_else(|| "dummy_elevation.tif".to_string());
        if !args.quiet {
            eprintln!("Creating dummy elevation raster at {dummy_path}...");
        }
        create_dummy_elevation(&dummy_path)?;
        args.elevation = Some(dummy_path);
    }

    let elev_path = args.elevation.as_deref().expect("elevation required");

    if !(1..=365).contains(&args.day) {
        return Err(format!("Day must be 1–365, got {}", args.day).into());
    }
    if args.step <= 0.0 || args.step > 24.0 {
        return Err(format!("Step must be 0–24h, got {}", args.step).into());
    }

    let declination = com_declin(args.day);
    let g_norm_extra = com_sol_const(args.day, args.solar_constant);
    if !args.quiet {
        eprintln!(
            "Day: {}  |  Declination: {:.2}°  |  G0: {:.1} W/m²",
            args.day,
            -declination * RAD2DEG,
            g_norm_extra
        );
    }

    if args.gpu {
        if !args.quiet {
            eprintln!("Using WebGPU acceleration…");
        }
        compute_raster_gpu(
            elev_path,
            &args.slope,
            &args.aspect,
            &args.linke,
            &args.albedo,
            &args.mask,
            args.slope_value,
            args.aspect_value,
            args.linke_value,
            args.albedo_value,
            args.day,
            args.step,
            args.solar_constant,
            &args.glob_rad,
            &args.beam_rad,
            &args.diff_rad,
            &args.refl_rad,
            &args.insol_time,
            args.quiet,
        )?;
    } else {
        run_raster_computation(
            elev_path,
            &args.slope,
            &args.aspect,
            &args.linke,
            &args.albedo,
            &args.mask,
            args.slope_value,
            args.aspect_value,
            args.linke_value,
            args.albedo_value,
            args.day,
            args.step,
            args.solar_constant,
            &args.glob_rad,
            &args.beam_rad,
            &args.diff_rad,
            &args.refl_rad,
            &args.insol_time,
            args.quiet,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gdal::Dataset;
    use std::path::Path;
    use sun::convert_grass_aspect;

    #[test]
    fn test_aspect_conversion() {
        use std::f64::consts::PI;
        let south = convert_grass_aspect(270.0);
        assert!((south - PI).abs() < 1e-10, "South aspect: {south:.4} rad");

        let east = convert_grass_aspect(0.0);
        assert_eq!(east, 0.0, "East aspect");

        let north = convert_grass_aspect(90.0);
        assert!((north - PI * 2.0).abs() < 0.01, "North aspect: {north:.4}");
    }

    #[test]
    fn test_create_dummy_elevation() {
        let tmp = "/tmp/test_dummy_elevation.tif";
        create_dummy_elevation(tmp).unwrap();

        let ds = Dataset::open(Path::new(tmp)).unwrap();
        let (ncols, nrows) = ds.raster_size();
        assert_eq!((ncols, nrows), (100, 100));

        let gt = ds.geo_transform().unwrap();
        assert!((gt[0] - 14.0).abs() < 1e-6);
        assert!((gt[3] - 48.0).abs() < 1e-6);
    }

    #[test]
    fn test_pixel_computation() {
        use sun::compute_pixel_irradiation;
        // Flat terrain at 47.5°N, summer solstice
        let result = compute_pixel_irradiation(47.5, 800.0, 0.0, 270.0, 3.0, 0.2, 172, 0.5, 1367.0)
            .expect("Should compute");
        assert!(
            result.global > 7000.0,
            "Global {:.0} Wh/m²/day",
            result.global
        );
        assert!(result.beam > result.diffuse, "Beam should exceed diffuse");
        println!(
            "47.5°N day 172: beam={:.0} diff={:.0} glob={:.0} insol={:.1}h",
            result.beam, result.diffuse, result.global, result.insol_time
        );
    }
}
