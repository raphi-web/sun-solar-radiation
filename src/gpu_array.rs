//! GPU (wgpu) band compute cores on the array API — no GDAL, no file I/O.
//!
//! Mirrors `arrays.rs` (CPU cores) one-for-one: the same `BandInputs` goes
//! in, band-sized `Vec<f32>` outputs come out. The WGSL shader
//! (`shader.wgsl`) already understands row-banding (`row_offset` /
//! `full_nrows` uniforms; full-grid elevation on binding 1), so a Python-side
//! band maps directly onto one or more GPU sub-tiles.
//!
//! Sub-tiling: when a band's per-buffer byte size would exceed the adapter's
//! `max_storage_buffer_binding_size`, the band is dispatched in contiguous
//! row sub-tiles (same strategy as the path-based `gpu.rs`). Results are
//! identical either way — each pixel only reads its own index plus the
//! full-grid shadow context.
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::arrays::BandInputs;
use crate::horizon::HorizonMap;
use crate::solar::{RAD2DEG, com_declin, com_sol_const};
use crate::UNDEFZ;

/// Uniform buffer layout — must match the WGSL `Uniforms` struct exactly.
/// Total: 16 × 4 bytes = 64 bytes (multiple of 16).
#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy, Debug)]
struct Uniforms {
    sindecl: f32,
    cosdecl: f32,
    g_norm_extra: f32,
    step_rad: f32,
    ncols: u32,
    nrows: u32, // rows in THIS sub-tile
    dispatch_x_pixels: u32,
    full_nrows: u32,
    row_offset: u32, // sub-tile's first row in the FULL grid
    max_z: f32,
    dx: f32,
    dy: f32,
    n_az: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Outputs of one GPU band computation. All five components are always
/// computed (the shader writes them all); the PyO3 layer filters to the
/// requested set.
pub struct GpuDailyOut {
    pub beam: Vec<f32>,
    pub diffuse: Vec<f32>,
    pub reflected: Vec<f32>,
    pub global: Vec<f32>,
    pub insol: Vec<f32>,
}

/// Probe for a usable wgpu adapter without raising. Callers (the QGIS
/// pipeline) use this to decide GPU-vs-CPU fallback.
pub fn gpu_available() -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .is_some()
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    max_binding: u64,
}

async fn init_gpu(quiet: bool, label: &str) -> Result<GpuContext, String> {
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
    let adapter_name = adapter.get_info().name;
    if !quiet {
        eprintln!("GPU band  Adapter: {adapter_name}");
    }

    let limits = adapter.limits();
    let max_buf_binding = limits.max_storage_buffer_binding_size;
    let max_buffer_size = limits.max_buffer_size;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some(label),
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
        .await
        .map_err(|e| format!("GPU device request failed: {e}"))?;

    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

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
        label: Some("bgl"),
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
        label: Some("pipeline_layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("r.sun band"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    Ok(GpuContext {
        device,
        queue,
        pipeline,
        bgl,
        max_binding: max_buf_binding as u64,
    })
}

/// Rows per GPU sub-tile so each per-tile buffer stays under the adapter's
/// per-binding storage limit (95% budget, same as the path-based gpu.rs).
fn sub_tile_rows(ctx: &GpuContext, ncols: usize, nrows: usize) -> usize {
    let bytes_per_row = (ncols * std::mem::size_of::<f32>()) as u64;
    let budget = ctx.max_binding.saturating_mul(95) / 100;
    ((budget / bytes_per_row.max(1)) as usize)
        .max(1)
        .min(nrows)
}

/// 2D workgroup dispatch geometry for `pixels` at workgroup size 64.
fn dispatch_geometry(pixels: usize) -> (u32, u32, u32) {
    const WORKGROUP: u32 = 64;
    const MAX_DIM: u32 = 65535;
    let total = (pixels as u32).div_ceil(WORKGROUP);
    let (x, y) = if total <= MAX_DIM {
        (total, 1u32)
    } else {
        (MAX_DIM, total.div_ceil(MAX_DIM))
    };
    (x, y, x * WORKGROUP)
}

/// Compute daily irradiation for one band on the GPU.
pub fn compute_daily_band_gpu(
    inp: &BandInputs<'_>,
    day: i32,
    step: f64,
    solar_constant: f64,
    quiet: bool,
) -> Result<GpuDailyOut, String> {
    inp.validate()?;
    if !(1..=365).contains(&day) {
        return Err(format!("day must be 1-365, got {day}"));
    }
    if step <= 0.0 {
        return Err(format!("step must be > 0, got {step}"));
    }

    let declination = com_declin(day);
    let g_norm_extra = com_sol_const(day, solar_constant);
    let sindecl = declination.sin() as f32;
    let cosdecl = declination.cos() as f32;
    let step_rad = (step * std::f64::consts::PI / 12.0) as f32;
    let max_z = inp.full_max_z();

    let ncols = inp.ncols;
    let nrows = inp.nrows;
    let npixels = ncols * nrows;

    let out = pollster::block_on(async {
        let ctx = init_gpu(quiet, "r.sun daily band").await?;
        if !quiet {
            eprintln!("GPU band  day {day}  |  Decl: {:.2}°", -declination * RAD2DEG);
        }

        // Full-grid elevation — uploaded once, shared by every sub-tile
        // (binding 1; the shader indexes it via full_idx / ray-march).
        let buf_elev_full = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("elevation_full"),
                contents: bytemuck::cast_slice(inp.shadow_elev),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let mut res = GpuDailyOut {
            beam: vec![UNDEFZ; npixels],
            diffuse: vec![UNDEFZ; npixels],
            reflected: vec![UNDEFZ; npixels],
            global: vec![UNDEFZ; npixels],
            insol: vec![UNDEFZ; npixels],
        };

        let tile_rows = sub_tile_rows(&ctx, ncols, nrows);
        let dummy_horizon = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("horizon_dummy"),
                contents: bytemuck::bytes_of(&0u32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let mut start = 0usize;
        while start < nrows {
            let end = (start + tile_rows).min(nrows);
            let h = end - start;
            let pixels = ncols * h;
            let byte_size = (pixels * std::mem::size_of::<f32>()) as u64;
            let range = start * ncols..end * ncols;
            let row_range = start..end;

            let (wgx, wgy, dispatch_x_pixels) = dispatch_geometry(pixels);

            let uniforms = Uniforms {
                sindecl,
                cosdecl,
                g_norm_extra: g_norm_extra as f32,
                step_rad,
                ncols: ncols as u32,
                nrows: h as u32,
                dispatch_x_pixels,
                full_nrows: inp.full_nrows as u32,
                row_offset: (inp.row_offset + start) as u32,
                max_z: max_z as f32,
                dx: inp.dx as f32,
                dy: inp.dy as f32,
                n_az: 0, // daily: always ray-march
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let uniform_buf =
                ctx.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("uniforms"),
                        contents: bytemuck::bytes_of(&uniforms),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

            let make_input = |data: &[f32], label: &str| -> wgpu::Buffer {
                ctx.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: bytemuck::cast_slice(data),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    })
            };
            let buf_slope = make_input(&inp.slope[range.clone()], "slope");
            let buf_aspect = make_input(&inp.aspect[range.clone()], "aspect");
            let buf_linke = make_input(&inp.linke[range.clone()], "linke");
            let buf_albedo = make_input(&inp.albedo[range.clone()], "albedo");
            let buf_mask = make_input(&inp.mask[range.clone()], "mask");
            let buf_row_lat = make_input(&inp.row_lat_deg[row_range], "row_lat");

            let make_output = |label: &str| -> wgpu::Buffer {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: byte_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            };
            let buf_beam = make_output("out_beam");
            let buf_diff = make_output("out_diff");
            let buf_refl = make_output("out_refl");
            let buf_glob = make_output("out_glob");
            let buf_insol = make_output("out_insol");
            let make_staging = |label: &str| -> wgpu::Buffer {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: byte_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            let stage_beam = make_staging("stage_beam");
            let stage_diff = make_staging("stage_diff");
            let stage_refl = make_staging("stage_refl");
            let stage_glob = make_staging("stage_glob");
            let stage_insol = make_staging("stage_insol");

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bind_group"),
                layout: &ctx.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_elev_full.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_slope.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_aspect.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_linke.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_albedo.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: buf_mask.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: buf_beam.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: buf_diff.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: buf_refl.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 10, resource: buf_glob.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: buf_insol.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 12, resource: buf_row_lat.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 13, resource: dummy_horizon.as_entire_binding() },
                ],
            });

            let mut encoder =
                ctx.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("encoder"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("r.sun band pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&ctx.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(wgx, wgy, 1);
            }
            encoder.copy_buffer_to_buffer(&buf_beam, 0, &stage_beam, 0, byte_size);
            encoder.copy_buffer_to_buffer(&buf_diff, 0, &stage_diff, 0, byte_size);
            encoder.copy_buffer_to_buffer(&buf_refl, 0, &stage_refl, 0, byte_size);
            encoder.copy_buffer_to_buffer(&buf_glob, 0, &stage_glob, 0, byte_size);
            encoder.copy_buffer_to_buffer(&buf_insol, 0, &stage_insol, 0, byte_size);
            ctx.queue.submit(std::iter::once(encoder.finish()));

            let read_staging = |buf: &wgpu::Buffer, dst: &mut [f32]| {
                let slice = buf.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                ctx.device.poll(wgpu::Maintain::Wait);
                let data = slice.get_mapped_range();
                dst.copy_from_slice(bytemuck::cast_slice(&data));
                drop(data);
                buf.unmap();
            };
            read_staging(&stage_beam, &mut res.beam[range.clone()]);
            read_staging(&stage_diff, &mut res.diffuse[range.clone()]);
            read_staging(&stage_refl, &mut res.reflected[range.clone()]);
            read_staging(&stage_glob, &mut res.global[range.clone()]);
            read_staging(&stage_insol, &mut res.insol[range]);

            start = end;
        }
        Ok::<GpuDailyOut, String>(res)
    })?;

    Ok(out)
}

/// Compute annual PV potential for one band on the GPU.
///
/// `horizon` must be full-grid sized and is only honored when the band
/// covers the full grid (row_offset == 0 && nrows == full_nrows) — the
/// shader indexes it tile-locally. Otherwise the ray-march path (which
/// correctly indexes the full grid) is used, exactly like the CPU wrapper.
#[allow(clippy::too_many_arguments)]
pub fn compute_annual_band_gpu(
    inp: &BandInputs<'_>,
    days: &[i32],
    day_step: i32,
    step: f64,
    solar_constant: f64,
    panel_efficiency: f64,
    horizon: Option<&HorizonMap>,
    quiet: bool,
) -> Result<Vec<f32>, String> {
    inp.validate()?;
    if days.is_empty() {
        return Err("empty day sampling list".into());
    }
    if day_step <= 0 {
        return Err(format!("day_step must be positive, got {day_step}"));
    }

    let band_is_full = inp.row_offset == 0 && inp.nrows == inp.full_nrows;
    let n_az_u32 = if band_is_full && horizon.is_some() {
        horizon.unwrap().n_az as u32
    } else {
        0
    };

    let day_consts: Vec<(f32, f32, f32)> = days
        .iter()
        .map(|&d| {
            let decl = com_declin(d);
            (
                decl.sin() as f32,
                decl.cos() as f32,
                com_sol_const(d, solar_constant) as f32,
            )
        })
        .collect();

    let step_rad = (step * std::f64::consts::PI / 12.0) as f32;
    let max_z = inp.full_max_z();
    let day_step_f = day_step as f32;
    let kwh_factor = (panel_efficiency / 1000.0) as f32;

    let ncols = inp.ncols;
    let nrows = inp.nrows;
    let npixels = ncols * nrows;

    pollster::block_on(async {
        let ctx = init_gpu(quiet, "r.sun annual band").await?;

        let buf_elev_full = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("elevation_full"),
                contents: bytemuck::cast_slice(inp.shadow_elev),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let mut potential = vec![UNDEFZ; npixels];
        let tile_rows = sub_tile_rows(&ctx, ncols, nrows);

        let mut start = 0usize;
        while start < nrows {
            let end = (start + tile_rows).min(nrows);
            let h = end - start;
            let pixels = ncols * h;
            let byte_size = (pixels * std::mem::size_of::<f32>()) as u64;
            let range = start * ncols..end * ncols;
            let row_range = start..end;

            let (wgx, wgy, dispatch_x_pixels) = dispatch_geometry(pixels);

            let uniform_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("uniforms"),
                size: std::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let make_input = |data: &[f32], label: &str| -> wgpu::Buffer {
                ctx.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: bytemuck::cast_slice(data),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    })
            };
            let buf_slope = make_input(&inp.slope[range.clone()], "slope");
            let buf_aspect = make_input(&inp.aspect[range.clone()], "aspect");
            let buf_linke = make_input(&inp.linke[range.clone()], "linke");
            let buf_albedo = make_input(&inp.albedo[range.clone()], "albedo");
            let buf_mask = make_input(&inp.mask[range.clone()], "mask");
            let buf_row_lat = make_input(&inp.row_lat_deg[row_range], "row_lat");

            // Horizon slice for this sub-tile (only when band == full grid;
            // pixel offsets then coincide with full-grid pixel offsets).
            let buf_horizon = match horizon.filter(|_| n_az_u32 > 0) {
                Some(hm) => {
                    let n_az = hm.n_az;
                    ctx.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("horizon"),
                            contents: bytemuck::cast_slice(
                                &hm.data[start * ncols * n_az..end * ncols * n_az],
                            ),
                            usage: wgpu::BufferUsages::STORAGE,
                        })
                }
                None => ctx
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("horizon_dummy"),
                        contents: bytemuck::bytes_of(&0u32),
                        usage: wgpu::BufferUsages::STORAGE,
                    }),
            };

            let buf_glob = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("out_glob"),
                size: byte_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            // The other four outputs must be bound (the shader writes them
            // all) but are never read back on the annual path.
            let make_scratch = |label: &str| -> wgpu::Buffer {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: byte_size,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            };
            let buf_beam = make_scratch("scratch_beam");
            let buf_diff = make_scratch("scratch_diff");
            let buf_refl = make_scratch("scratch_refl");
            let buf_insol = make_scratch("scratch_insol");
            let stage_glob = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("stage_glob"),
                size: byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bind_group"),
                layout: &ctx.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf_elev_full.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buf_slope.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: buf_aspect.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: buf_linke.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_albedo.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: buf_mask.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: buf_beam.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: buf_diff.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: buf_refl.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 10, resource: buf_glob.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: buf_insol.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 12, resource: buf_row_lat.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 13, resource: buf_horizon.as_entire_binding() },
                ],
            });

            // Tile-local accumulator (Wh/m²).
            let mut annual_wh = vec![0.0f32; pixels];
            let mut day_glob = vec![0.0f32; pixels];

            for &(sindecl, cosdecl, g_norm_extra) in &day_consts {
                let uniforms = Uniforms {
                    sindecl,
                    cosdecl,
                    g_norm_extra,
                    step_rad,
                    ncols: ncols as u32,
                    nrows: h as u32,
                    dispatch_x_pixels,
                    full_nrows: inp.full_nrows as u32,
                    row_offset: (inp.row_offset + start) as u32,
                    max_z: max_z as f32,
                    dx: inp.dx as f32,
                    dy: inp.dy as f32,
                    n_az: n_az_u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                ctx.queue
                    .write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

                let mut encoder =
                    ctx.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("annual_encoder"),
                        });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("r.sun annual band pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&ctx.pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(wgx, wgy, 1);
                }
                encoder.copy_buffer_to_buffer(&buf_glob, 0, &stage_glob, 0, byte_size);
                ctx.queue.submit(std::iter::once(encoder.finish()));

                {
                    let slice = stage_glob.slice(..);
                    slice.map_async(wgpu::MapMode::Read, |_| {});
                    ctx.device.poll(wgpu::Maintain::Wait);
                    let data = slice.get_mapped_range();
                    day_glob.copy_from_slice(bytemuck::cast_slice(&data));
                    drop(data);
                    stage_glob.unmap();
                }

                for (acc, &v) in annual_wh.iter_mut().zip(day_glob.iter()) {
                    if v != UNDEFZ {
                        *acc += v * day_step_f;
                    }
                }
            }

            // Wh → kWh × efficiency; skipped pixels stay UNDEFZ.
            for (i, out) in potential[range.clone()].iter_mut().enumerate() {
                let gi = start * ncols + i;
                let m = inp.mask[gi];
                let e = inp.elevation[gi];
                if m == 0.0 || m == UNDEFZ || e == UNDEFZ || e < -1000.0
                    || inp.slope[gi] == UNDEFZ || inp.aspect[gi] == UNDEFZ
                {
                    *out = UNDEFZ;
                } else {
                    *out = annual_wh[i] * kwh_factor;
                }
            }

            start = end;
        }
        Ok::<Vec<f32>, String>(potential)
    })
}
