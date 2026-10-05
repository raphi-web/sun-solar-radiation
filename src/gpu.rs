/// WebGPU-accelerated r.sun raster computation
///
/// One GPU thread per pixel. The full sunrise-to-sunset time integration
/// runs in the WGSL compute shader (src/shader.wgsl).
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::solar::{RAD2DEG, com_declin, com_sol_const};
use crate::{
    compute_row_latitudes, read_elev_normalized, read_raster_or_constant, resolve_slope_aspect,
    write_raster,
};

use gdal::Dataset;
use std::path::Path;

/// Uniform buffer layout — must match the WGSL `Uniforms` struct exactly.
/// Total: 16 × 4 bytes = 64 bytes (multiple of 16)
#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy, Debug)]
struct Uniforms {
    sindecl: f32,
    cosdecl: f32,
    g_norm_extra: f32,
    step_rad: f32,
    ncols: u32,
    nrows: u32, // tile rows
    // Number of pixels covered along the X dispatch axis (= num_workgroups_x *
    // workgroup_size). Used by the shader to fold gid.y back into the linear
    // pixel index when the dispatch is 2D.
    dispatch_x_pixels: u32,
    full_nrows: u32, // full-grid height — shadow ray-march span
    row_offset: u32, // this tile's first row in the full grid
    max_z: f32,      // max valid elevation (ray-march early exit)
    dx: f32,         // pixel x size [m]
    dy: f32,         // pixel y size [m]
    // 0 → ray-march each shadow test; >0 → look up horizon[idx * n_az + bin].
    n_az: u32,
    pixel_start: u32, // window start inside the tile (see gpu_array)
    pixel_count: u32, // window length; 0 = whole tile
    _pad2: u32,
}

/// Run the full r.sun computation on the GPU.
///
/// Reads input rasters (or uses scalar fallbacks), dispatches the WGSL
/// compute shader, then writes output GeoTIFFs — matching the signature
/// of `run_raster_computation()` in lib.rs.
#[allow(clippy::too_many_arguments)]
pub fn compute_raster_gpu(
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
    pollster::block_on(compute_raster_gpu_async(
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
        day,
        step,
        solar_constant,
        out_glob,
        out_beam,
        out_diff,
        out_refl,
        out_insol,
        quiet,
    ))
}

#[allow(clippy::too_many_arguments)]
async fn compute_raster_gpu_async(
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
    // ── Solar constants (CPU, day-level) ──────────────────────────────────────
    let declination = com_declin(day);
    let g_norm_extra = com_sol_const(day, solar_constant);
    let sindecl = declination.sin() as f32;
    let cosdecl = declination.cos() as f32;

    if !quiet {
        eprintln!(
            "GPU  Day: {day}  |  Decl: {:.2}°  |  G0: {g_norm_extra:.1} W/m²",
            -declination * RAD2DEG
        );
    }

    // ── Read raster metadata ──────────────────────────────────────────────────
    let elev_ds = Dataset::open(Path::new(elev_path))?;
    let (ncols, nrows) = elev_ds.raster_size();
    let geo_transform = elev_ds.geo_transform()?;
    let projection = elev_ds.projection();
    let npixels = ncols * nrows;
    // Per-row latitude (WGS84°) — computed in Rust so the shader doesn't need
    // to know the source CRS or reproject on the fly.
    let row_lat_deg: Vec<f32> = compute_row_latitudes(&elev_ds)?
        .into_iter()
        .map(|v| v as f32)
        .collect();

    if !quiet {
        eprintln!("GPU  Raster: {ncols}×{nrows} = {npixels} pixels");
    }

    // ── Load input data into flat f32 arrays ─────────────────────────────────
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
    // Mask default = 1.0 means "compute all pixels". Any zero pixel in the mask
    // raster causes the shader to short-circuit and emit UNDEFZ for that cell.
    let mask_data = read_raster_or_constant(mask_path, ncols, nrows, 1.0)?;

    // ── Initialise wgpu ───────────────────────────────────────────────────────
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
        eprintln!("GPU  Adapter: {}", adapter.get_info().name);
    }

    // Request the adapter's *actual* per-binding storage-buffer ceiling rather
    // than wgpu's portable 128 MiB default. An 8192² f32 raster is 256 MiB per
    // buffer, which overflows the default but typically fits the device limit.
    let adapter_limits = adapter.limits();
    let max_buf_binding = adapter_limits.max_storage_buffer_binding_size;
    let max_buffer_size = adapter_limits.max_buffer_size;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("r.sun device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    // 8 read-only inputs (elev/slope/aspect/linke/albedo/mask/row_lat/horizon)
                    // + 5 read-write outputs = 13 storage buffers.
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

    // ── Shader module ─────────────────────────────────────────────────────────
    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

    let step_rad = (step * std::f64::consts::PI / 12.0) as f32;

    // ── Tile sizing ───────────────────────────────────────────────────────────
    // Even with the device's reported maximum, a single 8192² raster may still
    // exceed the per-binding limit. Process the raster in contiguous row bands
    // small enough that each per-tile buffer stays under the limit. Each pixel
    // in the shader reads only its own index, so row-banding requires no shader
    // changes — just an adjusted y_origin and nrows in the uniforms.
    let bytes_per_row = (ncols * std::mem::size_of::<f32>()) as u64;
    let binding_budget = (max_buf_binding as u64).saturating_mul(95) / 100;
    let tile_rows = ((binding_budget / bytes_per_row.max(1)) as usize)
        .max(1)
        .min(nrows);

    if !quiet {
        let full_buf_mb = (npixels * 4) / (1024 * 1024);
        let limit_mb = max_buf_binding / (1024 * 1024);
        if tile_rows < nrows {
            let n_tiles = nrows.div_ceil(tile_rows);
            eprintln!(
                "GPU  Tiling: full buffer ≈ {full_buf_mb} MB > per-binding limit \
                 {limit_mb} MB; processing in {n_tiles} row-band tile(s) \
                 of {tile_rows} rows"
            );
        }
    }

    // ── Bind group layout ─────────────────────────────────────────────────────
    // Bindings 0-10 match the shader declarations exactly.
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl"),
        entries: &[
            // 0 = uniforms
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
            // 1-5 = input storage (read-only)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 6 = mask (read-only)
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 7-11 = output storage (read-write)
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 12 = per-row latitude (read-only)
            wgpu::BindGroupLayoutEntry {
                binding: 12,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 13 = per-pixel horizon table (read-only; dummy 4-byte buf when n_az=0)
            wgpu::BindGroupLayoutEntry {
                binding: 13,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // ── Compute pipeline (shared across tiles) ────────────────────────────────
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("r.sun"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // ── Per-tile dispatch loop ────────────────────────────────────────────────
    let mut res_beam = vec![crate::UNDEFZ; npixels];
    let mut res_diff = vec![crate::UNDEFZ; npixels];
    let mut res_refl = vec![crate::UNDEFZ; npixels];
    let mut res_glob = vec![crate::UNDEFZ; npixels];
    let mut res_insol = vec![crate::UNDEFZ; npixels];

    // Shadow ray-march needs the full DEM and its global bounds — computed once.
    let dx = geo_transform[1].abs() as f32;
    let dy = geo_transform[5].abs() as f32;
    let max_z = elev_data
        .iter()
        .copied()
        .filter(|&v| v != crate::UNDEFZ)
        .fold(f32::NEG_INFINITY, f32::max);

    // Full-grid elevation is uploaded once and reused by every tile (the
    // per-tile slope/aspect/linke/albedo/mask buffers stay tile-local).
    let buf_elev_full = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("elevation_full"),
        contents: bytemuck::cast_slice(&elev_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    if !quiet {
        // Initial checkpoint so callers see the bar move even when setup,
        // dispatch, or a single-tile run dominates the wall-clock time.
        eprint!("\rProgress: 0%   ");
    }

    let mut row_start = 0usize;
    let mut tile_idx = 0usize;
    let mut chunker = crate::gpu_array::Chunker::new();
    while row_start < nrows {
        let row_end = (row_start + tile_rows).min(nrows);
        let tile_h = row_end - row_start;
        let tile_pixels = ncols * tile_h;
        let tile_byte_size = (tile_pixels * std::mem::size_of::<f32>()) as u64;
        let pixel_offset = row_start * ncols;
        let pixel_range = pixel_offset..pixel_offset + tile_pixels;
        chunker.begin_tile(crate::gpu_array::group_weights(tile_pixels, |i| {
            let j = pixel_offset + i;
            crate::gpu_array::pixel_computed(elev_data[j], slope_data[j], aspect_data[j], mask_data[j])
        }));

        // Per-tile uniforms — pixel latitude is read from the per-row buffer
        // (binding 12), sliced to the tile's row range below. The dispatch
        // window fields are filled per submission by dispatch_in_windows.
        let uniforms = Uniforms {
            sindecl,
            cosdecl,
            g_norm_extra: g_norm_extra as f32,
            step_rad,
            ncols: ncols as u32,
            nrows: tile_h as u32,
            dispatch_x_pixels: 0,
            full_nrows: nrows as u32,
            row_offset: row_start as u32,
            max_z,
            dx,
            dy,
            n_az: 0, // single-day path: always use ray-march
            pixel_start: 0,
            pixel_count: 0,
            _pad2: 0,
        };
        // Dummy 4-byte horizon buffer satisfies the shader binding when n_az=0.
        let buf_horizon_dummy = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("horizon_dummy"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Per-tile input buffers — slice each host array by the tile row range.
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

        // Per-row latitude slice for this tile.
        let buf_row_lat = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("row_lat"),
            contents: bytemuck::cast_slice(&row_lat_deg[row_start..row_end]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let make_output = |label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: tile_byte_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let buf_out_beam = make_output("out_beam");
        let buf_out_diff = make_output("out_diff");
        let buf_out_refl = make_output("out_refl");
        let buf_out_glob = make_output("out_glob");
        let buf_out_insol = make_output("out_insol");

        let make_staging = |label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: tile_byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let stage_beam = make_staging("stage_beam");
        let stage_diff = make_staging("stage_diff");
        let stage_refl = make_staging("stage_refl");
        let stage_glob = make_staging("stage_glob");
        let stage_insol = make_staging("stage_insol");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group"),
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
                    resource: buf_horizon_dummy.as_entire_binding(),
                },
            ],
        });

        let mut write = |pixel_start: u32, pixel_count: u32, dispatch_x_pixels: u32| {
            let u = Uniforms {
                dispatch_x_pixels,
                pixel_start,
                pixel_count,
                ..uniforms
            };
            queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&u));
        };
        if !quiet {
            eprintln!("GPU  Tile {tile_idx}: rows {row_start}..{row_end}");
        }
        crate::gpu_array::dispatch_in_windows(
            &device,
            &queue,
            &pipeline,
            &bind_group,
            &mut chunker,
            tile_pixels,
            &mut write,
            &|| Ok(()),
        )?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
        encoder.copy_buffer_to_buffer(&buf_out_beam, 0, &stage_beam, 0, tile_byte_size);
        encoder.copy_buffer_to_buffer(&buf_out_diff, 0, &stage_diff, 0, tile_byte_size);
        encoder.copy_buffer_to_buffer(&buf_out_refl, 0, &stage_refl, 0, tile_byte_size);
        encoder.copy_buffer_to_buffer(&buf_out_glob, 0, &stage_glob, 0, tile_byte_size);
        encoder.copy_buffer_to_buffer(&buf_out_insol, 0, &stage_insol, 0, tile_byte_size);
        queue.submit(std::iter::once(encoder.finish()));

        if !quiet {
            // Mid-tile checkpoint: dispatch is submitted, readback pending.
            // Guarantees sub-100% updates even for single-tile rasters.
            eprint!(
                "\rProgress: {:.0}%   ",
                100.0 * (row_start as f64 + tile_h as f64 * 0.5) / nrows as f64
            );
        }

        let read_staging = |buf: &wgpu::Buffer, dst: &mut [f32]| {
            let slice = buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let data = slice.get_mapped_range();
            dst.copy_from_slice(bytemuck::cast_slice(&data));
            drop(data);
            buf.unmap();
        };

        read_staging(&stage_beam, &mut res_beam[pixel_range.clone()]);
        read_staging(&stage_diff, &mut res_diff[pixel_range.clone()]);
        read_staging(&stage_refl, &mut res_refl[pixel_range.clone()]);
        read_staging(&stage_glob, &mut res_glob[pixel_range.clone()]);
        read_staging(&stage_insol, &mut res_insol[pixel_range.clone()]);

        row_start = row_end;
        tile_idx += 1;
        // Row-band progress: rows finished / total rows. Matches the CPU
        // `\rProgress: N%` format so callers parsing stderr see one scheme.
        if !quiet {
            eprint!("\rProgress: {:.0}%   ", 100.0 * row_end as f64 / nrows as f64);
        }
    }

    // ── Statistics ────────────────────────────────────────────────────────────
    if !quiet {
        let valid: Vec<f32> = res_glob.iter().copied().filter(|&v| v > 0.0).collect();
        if !valid.is_empty() {
            let mean = valid.iter().sum::<f32>() / valid.len() as f32;
            let min = valid.iter().copied().fold(f32::INFINITY, f32::min);
            let max = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "GPU  Global: min={min:.0}  mean={mean:.0}  max={max:.0} Wh/m²/day  \
                       ({} valid pixels)",
                valid.len()
            );
        }
    }

    // ── Write output GeoTIFFs ─────────────────────────────────────────────────
    let driver = gdal::DriverManager::get_driver_by_name("GTiff")?;
    if let Some(p) = out_beam {
        write_raster(
            &driver,
            p,
            ncols,
            nrows,
            &geo_transform,
            &projection,
            &res_beam,
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
            &res_diff,
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
            &res_refl,
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
            &res_glob,
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
            &res_insol,
        )?;
    }

    Ok(())
}
