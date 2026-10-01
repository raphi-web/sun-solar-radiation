/// Annual solar potential per m² — GPU-accelerated
///
/// Samples global irradiation on a regular DOY grid (`day_start..=day_end`
/// stepped by `day_step`), accumulates `glob × day_step` into an annual Wh/m²
/// sum, then converts to kWh/m²/year with a panel-efficiency factor and writes
/// a single GeoTIFF. Pixels masked off (or outside data bounds) are written
/// as the standard r.sun nodata sentinel `UNDEFZ`.
///
/// Reuses the GPU device,
/// pipeline, and per-tile buffers across all sampled days — only the uniform
/// buffer is rewritten between days.
use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::radiation::{RadiationParams, integrate_daily};
use crate::solar::{
    DEG2RAD, com_declin, com_sol_const, compute_slope_geometry, compute_sunrise_sunset,
};
use crate::{
    UNDEFZ, compute_row_latitudes, convert_grass_aspect, read_elev_normalized,
    read_raster_or_constant, resolve_slope_aspect, write_raster,
};

use gdal::Dataset;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy, Debug)]
struct Uniforms {
    sindecl: f32,
    cosdecl: f32,
    g_norm_extra: f32,
    step_rad: f32,
    ncols: u32,
    nrows: u32, // tile rows
    dispatch_x_pixels: u32,
    full_nrows: u32, // full-grid height — shadow ray-march span
    row_offset: u32, // tile's first row in the full grid
    max_z: f32,      // max valid elevation (ray-march early exit)
    dx: f32,         // pixel x size [m]
    dy: f32,         // pixel y size [m]
    // 0 → fall back to the per-call ray-march. >0 → use the horizon-lookup
    // path with this many azimuth bins (matches the CPU side).
    n_az: u32,
    // Pad to 64 bytes — WGSL uniform structs round up to 16-byte alignment.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[allow(clippy::too_many_arguments)]
pub fn compute_annual_potential_gpu(
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
    day_start: i32,
    day_end: i32,
    day_step: i32,
    step: f64,
    solar_constant: f64,
    panel_efficiency: f64,
    out_path: &str,
    use_horizon: bool,
    horizon_n_az: usize,
    quiet: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    pollster::block_on(compute_annual_potential_gpu_async(
        elev_path,
        slope_path,
        aspect_path,
        linke_path,
        albedo_path,
        mask_path,
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
    ))
}

#[allow(clippy::too_many_arguments)]
async fn compute_annual_potential_gpu_async(
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
    day_start: i32,
    day_end: i32,
    day_step: i32,
    step: f64,
    solar_constant: f64,
    panel_efficiency: f64,
    out_path: &str,
    use_horizon: bool,
    horizon_n_az: usize,
    quiet: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if day_step <= 0 {
        return Err("day_step must be positive".into());
    }
    if !(1..=365).contains(&day_start) || !(1..=365).contains(&day_end) || day_end < day_start {
        return Err("day_start/day_end must lie in 1..=365 with day_end >= day_start".into());
    }
    if use_horizon && horizon_n_az < 4 {
        return Err("horizon_n_az must be >= 4 when use_horizon=true".into());
    }

    let days: Vec<i32> = (day_start..=day_end).step_by(day_step as usize).collect();
    if days.is_empty() {
        return Err("empty day sampling range".into());
    }

    // ── Raster metadata + input arrays ───────────────────────────────────────
    let elev_ds = Dataset::open(Path::new(elev_path))?;
    let (ncols, nrows) = elev_ds.raster_size();
    let geo_transform = elev_ds.geo_transform()?;
    let projection = elev_ds.projection();
    let npixels = ncols * nrows;
    let row_lat_deg: Vec<f32> = compute_row_latitudes(&elev_ds)?
        .into_iter()
        .map(|v| v as f32)
        .collect();

    let elev_data = read_elev_normalized(&elev_ds, ncols, nrows)?;
    let (slope_data, aspect_data) = resolve_slope_aspect(
        slope_path,
        slope_value,
        aspect_path,
        aspect_value,
        &elev_data,
        ncols,
        nrows,
        &geo_transform,
    )?;
    let linke_data = read_raster_or_constant(linke_path, ncols, nrows, linke_value as f32)?;
    let albedo_data = read_raster_or_constant(albedo_path, ncols, nrows, albedo_value as f32)?;
    let mask_data = read_raster_or_constant(mask_path, ncols, nrows, 1.0)?;

    // Precompute per-pixel horizon map on CPU if requested. This is the same
    // map used by the CPU annual path; we then upload slices to the GPU per tile.
    let dx_f = geo_transform[1].abs();
    let dy_f = geo_transform[5].abs();
    let horizon_map = if use_horizon {
        if !quiet {
            eprintln!("GPU annual  Precomputing horizon map ({horizon_n_az} azimuth bins)...");
        }
        Some(crate::horizon::compute_horizon_map_cpu(
            &elev_data,
            ncols,
            nrows,
            dx_f,
            dy_f,
            &mask_data,
            horizon_n_az,
            quiet,
        ))
    } else {
        None
    };
    let n_az_u32 = if use_horizon { horizon_n_az as u32 } else { 0 };

    if !quiet {
        eprintln!(
            "GPU annual  Raster: {ncols}×{nrows} = {npixels} pixels  |  \
             Sampling {} day(s): step {day_step}, range {day_start}..={day_end}",
            days.len()
        );
    }

    // ── GPU init ─────────────────────────────────────────────────────────────
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or("No GPU adapter found. Try installing Vulkan/OpenGL drivers.")?;

    if !quiet {
        eprintln!("GPU annual  Adapter: {}", adapter.get_info().name);
    }

    let adapter_limits = adapter.limits();
    let max_buf_binding = adapter_limits.max_storage_buffer_binding_size;
    let max_buffer_size = adapter_limits.max_buffer_size;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("r.sun annual device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 13,
                    max_storage_buffer_binding_size: max_buf_binding,
                    max_buffer_size,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await?;

    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
    let step_rad = (step * std::f64::consts::PI / 12.0) as f32;
    let workgroup_size = 64u32;
    let max_dim = 65535u32;

    // ── Tile sizing — same row-banding strategy as gpu.rs ────────────────────
    let bytes_per_row = (ncols * std::mem::size_of::<f32>()) as u64;
    let binding_budget = (max_buf_binding as u64).saturating_mul(95) / 100;
    let tile_rows = ((binding_budget / bytes_per_row.max(1)) as usize)
        .max(1)
        .min(nrows);
    // Total row-band tiles; used for `\rProgress: N%` reporting below.
    let n_tiles = nrows.div_ceil(tile_rows);

    // ── Bind group layout + pipeline (shared across days & tiles) ───────────
    let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("annual_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage_ro(1),
            storage_ro(2),
            storage_ro(3),
            storage_ro(4),
            storage_ro(5),
            storage_ro(6),
            storage_rw(7),
            storage_rw(8),
            storage_rw(9),
            storage_rw(10),
            storage_rw(11),
            storage_ro(12),
            storage_ro(13),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("annual_pipeline_layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("r.sun annual"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // ── Final output (kWh/m²/year), assembled tile by tile ───────────────────
    let mut potential = vec![UNDEFZ; npixels];

    // Per-day constants pre-computed once; reused across tiles.
    let day_consts: Vec<(f32, f32, f32)> = days
        .iter()
        .map(|&d| {
            let decl = com_declin(d);
            let g0 = com_sol_const(d, solar_constant) as f32;
            (decl.sin() as f32, decl.cos() as f32, g0)
        })
        .collect();

    let day_step_f = day_step as f32;
    let kwh_factor = (panel_efficiency / 1000.0) as f32;

    // Shadow ray-march needs the full DEM and its global bounds.
    let dx = dx_f as f32;
    let dy = dy_f as f32;
    let max_z = elev_data
        .iter()
        .copied()
        .filter(|&v| v != UNDEFZ)
        .fold(f32::NEG_INFINITY, f32::max);

    // Full-grid elevation uploaded once and shared across every tile/day.
    let buf_elev_full = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("elevation_full"),
        contents: bytemuck::cast_slice(&elev_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    if !quiet {
        // Initial checkpoint: setup/pipeline build can take a moment before
        // the first day runs; show the bar at 0% rather than blank.
        eprint!("\rProgress: 0%   ");
    }

    let mut row_start = 0usize;
    let mut tile_idx = 0usize;
    while row_start < nrows {
        let row_end = (row_start + tile_rows).min(nrows);
        let tile_h = row_end - row_start;
        let tile_pixels = ncols * tile_h;
        let tile_byte_size = (tile_pixels * std::mem::size_of::<f32>()) as u64;
        let pixel_offset = row_start * ncols;
        let pixel_range = pixel_offset..pixel_offset + tile_pixels;

        let total_workgroups = (tile_pixels as u32).div_ceil(workgroup_size);
        let (num_workgroups_x, num_workgroups_y) = if total_workgroups <= max_dim {
            (total_workgroups, 1u32)
        } else {
            let x = max_dim;
            let y = total_workgroups.div_ceil(x);
            (x, y)
        };
        let dispatch_x_pixels = num_workgroups_x * workgroup_size;

        // ── Per-tile buffers (created once; reused across all days) ─────────
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("annual_uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let make_input = |data: &[f32], label: &str| -> wgpu::Buffer {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&data[pixel_range.clone()]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let buf_slope = make_input(&slope_data, "slope");
        let buf_aspect = make_input(&aspect_data, "aspect");
        let buf_linke = make_input(&linke_data, "linke");
        let buf_albedo = make_input(&albedo_data, "albedo");
        let buf_mask = make_input(&mask_data, "mask");
        let buf_row_lat = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("row_lat"),
            contents: bytemuck::cast_slice(&row_lat_deg[row_start..row_end]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // Per-tile horizon buffer: slice of the precomputed map for this tile's
        // pixel range. When use_horizon=false a dummy 4-byte buffer satisfies
        // the binding; the shader sees n_az=0 and uses the ray-march path instead.
        let buf_horizon = match &horizon_map {
            Some(hm) => {
                let hm_start = pixel_offset * hm.n_az;
                let hm_end = (pixel_offset + tile_pixels) * hm.n_az;
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("horizon"),
                    contents: bytemuck::cast_slice(&hm.data[hm_start..hm_end]),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            }
            None => device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("horizon_dummy"),
                contents: bytemuck::bytes_of(&0u32),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        };

        let make_output = |label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: tile_byte_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        // Shader writes all five outputs; only read back glob, but the
        // bind group still needs the other four to satisfy the layout.
        let buf_out_beam = make_output("out_beam");
        let buf_out_diff = make_output("out_diff");
        let buf_out_refl = make_output("out_refl");
        let buf_out_glob = make_output("out_glob");
        let buf_out_insol = make_output("out_insol");

        let stage_glob = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stage_glob"),
            size: tile_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("annual_bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_elev_full.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_slope.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_aspect.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_linke.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_albedo.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_mask.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buf_out_beam.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buf_out_diff.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buf_out_refl.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buf_out_glob.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buf_out_insol.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: buf_row_lat.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: buf_horizon.as_entire_binding(),
                },
            ],
        });

        // Tile-local annual accumulator (Wh/m²).
        let mut annual_wh = vec![0.0f32; tile_pixels];
        let mut day_glob = vec![0.0f32; tile_pixels];

        if !quiet {
            eprintln!(
                "GPU annual  Tile {tile_idx}: rows {row_start}..{row_end}, dispatch \
                 {num_workgroups_x}×{num_workgroups_y} workgroups × {} days",
                days.len()
            );
        }

        // ── Day loop — only the uniform buffer changes per day ──────────────
        for (di, &(sindecl, cosdecl, g_norm_extra)) in day_consts.iter().enumerate() {
            let uniforms = Uniforms {
                sindecl,
                cosdecl,
                g_norm_extra,
                step_rad,
                ncols: ncols as u32,
                nrows: tile_h as u32,
                dispatch_x_pixels,
                full_nrows: nrows as u32,
                row_offset: row_start as u32,
                max_z,
                dx,
                dy,
                n_az: n_az_u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("annual_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("r.sun annual pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
            }
            encoder.copy_buffer_to_buffer(&buf_out_glob, 0, &stage_glob, 0, tile_byte_size);
            queue.submit(std::iter::once(encoder.finish()));

            {
                let slice = stage_glob.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                device.poll(wgpu::Maintain::Wait);
                let data = slice.get_mapped_range();
                day_glob.copy_from_slice(bytemuck::cast_slice(&data));
                drop(data);
                stage_glob.unmap();
            }

            // Accumulate: treat UNDEFZ pixels as zero contribution, then
            // weight by day_step (Riemann sum across the year).
            for (acc, &v) in annual_wh.iter_mut().zip(day_glob.iter()) {
                if v != UNDEFZ {
                    *acc += v * day_step_f;
                }
            }

            if !quiet && (di + 1) % 10 == 0 {
                eprintln!(
                    "GPU annual  Tile {tile_idx}: day {}/{} done",
                    di + 1,
                    days.len()
                );
            }
            // Overall progress: tiles×days grid position. Same `\rProgress: N%`
            // scheme as the CPU paths so stderr parsers see one format.
            if !quiet {
                let total_units = n_tiles * days.len();
                let done_units = tile_idx * days.len() + di + 1;
                eprint!(
                    "\rProgress: {:.0}%   ",
                    100.0 * done_units as f64 / total_units as f64
                );
            }
        }

        // ── Wh → kWh × efficiency; mask-off pixels become UNDEFZ ────────────
        let tile_mask = &mask_data[pixel_range.clone()];
        let tile_elev = &elev_data[pixel_range.clone()];
        let tile_slope = &slope_data[pixel_range.clone()];
        let tile_aspect = &aspect_data[pixel_range.clone()];
        let out_slice = &mut potential[pixel_range.clone()];
        for i in 0..tile_pixels {
            let m = tile_mask[i];
            let e = tile_elev[i];
            if m == 0.0
                || m == UNDEFZ
                || e == UNDEFZ
                || e < -1000.0
                || tile_slope[i] == UNDEFZ
                || tile_aspect[i] == UNDEFZ
            {
                out_slice[i] = UNDEFZ;
            } else {
                out_slice[i] = annual_wh[i] * kwh_factor;
            }
        }

        row_start = row_end;
        tile_idx += 1;
    }

    if !quiet {
        let valid: Vec<f32> = potential.iter().copied().filter(|&v| v != UNDEFZ).collect();
        if !valid.is_empty() {
            let mean = valid.iter().sum::<f32>() / valid.len() as f32;
            let min = valid.iter().copied().fold(f32::INFINITY, f32::min);
            let max = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "GPU annual  Potential: min={min:.1}  mean={mean:.1}  max={max:.1} kWh/m²/year  \
                 ({} valid pixels)",
                valid.len()
            );
        }
    }

    let driver = gdal::DriverManager::get_driver_by_name("GTiff")?;
    write_raster(
        &driver,
        out_path,
        ncols,
        nrows,
        &geo_transform,
        &projection,
        &potential,
    )?;

    Ok(())
}

/// Annual solar potential per m² — CPU (rayon-parallel) equivalent of
/// [`compute_annual_potential_gpu`].
///
/// Same inputs, same output semantics, and same Riemann-sum-over-days strategy
/// as the GPU version. Each row is processed in parallel; for each unmasked
/// pixel we iterate the sampled days, accumulate `glob × day_step` into an
/// annual Wh/m² total, then convert to kWh/m²/year via `panel_efficiency`.
/// Pixels with mask=0 or invalid elevation are written as `UNDEFZ`.
#[allow(clippy::too_many_arguments)]
pub fn compute_annual_potential_cpu(
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
    day_start: i32,
    day_end: i32,
    day_step: i32,
    step: f64,
    solar_constant: f64,
    panel_efficiency: f64,
    out_path: &str,
    use_horizon: bool,
    horizon_n_az: usize,
    quiet: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if day_step <= 0 {
        return Err("day_step must be positive".into());
    }
    if !(1..=365).contains(&day_start) || !(1..=365).contains(&day_end) || day_end < day_start {
        return Err("day_start/day_end must lie in 1..=365 with day_end >= day_start".into());
    }

    let days: Vec<i32> = (day_start..=day_end).step_by(day_step as usize).collect();
    if days.is_empty() {
        return Err("empty day sampling range".into());
    }

    // ── Raster metadata + input arrays ───────────────────────────────────────
    let elev_ds = Dataset::open(Path::new(elev_path))?;
    let (ncols, nrows) = elev_ds.raster_size();
    let geo_transform = elev_ds.geo_transform()?;
    let projection = elev_ds.projection();
    let npixels = ncols * nrows;
    let row_lat_deg = compute_row_latitudes(&elev_ds)?;

    let elev_data = read_elev_normalized(&elev_ds, ncols, nrows)?;
    let (slope_data, aspect_data) = resolve_slope_aspect(
        slope_path,
        slope_value,
        aspect_path,
        aspect_value,
        &elev_data,
        ncols,
        nrows,
        &geo_transform,
    )?;
    let linke_data = read_raster_or_constant(linke_path, ncols, nrows, linke_value as f32)?;
    let albedo_data = read_raster_or_constant(albedo_path, ncols, nrows, albedo_value as f32)?;
    let mask_data = read_raster_or_constant(mask_path, ncols, nrows, 1.0)?;

    if !quiet {
        eprintln!(
            "CPU annual  Raster: {ncols}×{nrows} = {npixels} pixels  |  \
             Sampling {} day(s): step {day_step}, range {day_start}..={day_end}  |  \
             {} threads",
            days.len(),
            rayon::current_num_threads()
        );
    }

    // Per-day constants pre-computed once.
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

    let dx = geo_transform[1].abs();
    let dy = geo_transform[5].abs();
    let max_z = elev_data
        .iter()
        .copied()
        .filter(|&v| v != UNDEFZ)
        .fold(f32::NEG_INFINITY, f32::max) as f64;

    // Optional horizon precompute: builds a per-pixel azimuth-binned horizon
    // map so each shadow test inside the day loop becomes an O(1) lookup
    // instead of an O(N) ray-march. Cost is roughly one extra full pass over
    // the DEM (n_az rays per pixel), amortized across every (day, time-step).
    let horizon_map = if use_horizon {
        if horizon_n_az < 4 {
            return Err("horizon_n_az must be >= 4".into());
        }
        Some(crate::horizon::compute_horizon_map_cpu(
            &elev_data,
            ncols,
            nrows,
            dx,
            dy,
            &mask_data,
            horizon_n_az,
            quiet,
        ))
    } else {
        None
    };

    let mut potential = vec![UNDEFZ; npixels];
    let valid = AtomicUsize::new(0);
    let rows_done = AtomicUsize::new(0);
    let report_interval = (nrows / 20).max(1);

    let horizon_ref = horizon_map.as_ref();
    potential
        .par_chunks_mut(ncols)
        .enumerate()
        .for_each(|(row, out_row)| {
            let lat = row_lat_deg[row] * DEG2RAD;
            let mut local_valid = 0usize;
            for col in 0..ncols {
                let idx = row * ncols + col;
                let elev = elev_data[idx];
                if elev == UNDEFZ || elev < -1000.0 {
                    continue;
                }
                let m = mask_data[idx];
                if m == 0.0 || m == UNDEFZ {
                    continue;
                }
                if slope_data[idx] == UNDEFZ || aspect_data[idx] == UNDEFZ {
                    continue;
                }

                let slope_rad = slope_data[idx] as f64 * DEG2RAD;
                let aspect_rad = convert_grass_aspect(aspect_data[idx] as f64);
                let linke = linke_data[idx] as f64;
                let albedo = albedo_data[idx] as f64;

                let shadow_ctx = crate::shadow::ShadowContext {
                    elev: &elev_data,
                    ncols,
                    nrows,
                    dx,
                    dy,
                    max_z,
                    row,
                    col,
                    eye_z: elev as f64,
                    horizon: horizon_ref,
                };

                let mut annual_wh = 0.0f64;
                for &(declination, sindecl, cosdecl, g_norm_extra) in &day_consts {
                    let geom = compute_slope_geometry(slope_rad, aspect_rad, lat, sindecl, cosdecl);
                    let (sunrise, sunset) = match compute_sunrise_sunset(&geom) {
                        Some(s) => s,
                        None => continue,
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
                    annual_wh += result.global * day_step_f;
                }

                out_row[col] = (annual_wh * kwh_factor) as f32;
                local_valid += 1;
            }
            valid.fetch_add(local_valid, Ordering::Relaxed);
            if !quiet {
                let done = rows_done.fetch_add(1, Ordering::Relaxed) + 1;
                if done % report_interval == 0 {
                    eprint!(
                        "\rCPU annual  Progress: {:.0}%   ",
                        100.0 * done as f64 / nrows as f64
                    );
                }
            }
        });

    if !quiet {
        eprintln!("\rCPU annual  Progress: 100%   ");
        eprintln!(
            "CPU annual  Processed {} valid pixels",
            valid.load(Ordering::Relaxed)
        );
        let valid_p: Vec<f32> = potential.iter().copied().filter(|&v| v != UNDEFZ).collect();
        if !valid_p.is_empty() {
            let mean = valid_p.iter().sum::<f32>() / valid_p.len() as f32;
            let min = valid_p.iter().copied().fold(f32::INFINITY, f32::min);
            let max = valid_p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "CPU annual  Potential: min={min:.1}  mean={mean:.1}  max={max:.1} kWh/m²/year"
            );
        }
    }

    let driver = gdal::DriverManager::get_driver_by_name("GTiff")?;
    write_raster(
        &driver,
        out_path,
        ncols,
        nrows,
        &geo_transform,
        &projection,
        &potential,
    )?;

    Ok(())
}
