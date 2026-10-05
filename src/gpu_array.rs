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
//!
//! Submission budget: desktop drivers reset the GPU when one submission runs
//! longer than ~2 s (amdgpu `lockup_timeout`, Windows TDR). Each sub-tile is
//! therefore computed in pixel windows, one short submission each, sized
//! adaptively from the measured cost per pixel. Any GPU failure (lost device,
//! validation error, wgpu panic) is returned as `Err`, never aborts the host.
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};
use std::time::Instant;

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
    pixel_start: u32, // window start inside the sub-tile
    pixel_count: u32, // window length (> 0 on this path)
    _pad2: u32,
}

/// Default GPU time budget per submission [s]: a 10× margin under the ~2 s
/// driver watchdog. `SUN_GPU_MAX_SUBMIT_SECONDS` overrides it.
const DEFAULT_SUBMIT_BUDGET_S: f64 = 0.2;
/// Windows are whole 64-pixel groups (one workgroup each).
const GROUP: usize = 64;
/// First window of a tile when nothing has been measured yet.
const FIRST_GROUPS: usize = 64;
/// Max growth of the predicted cost between consecutive windows.
const MAX_GROWTH: f64 = 4.0;
/// Fraction of the budget a window is sized for.
const TARGET_FRACTION: f64 = 0.8;
/// Relative cost of a group whose pixels are all skipped (nodata/masked):
/// the shader returns after a few loads, a valid pixel ray-marches.
const SKIPPED_GROUP_WEIGHT: f32 = 0.01;

/// Submission counters of the most recent GPU band call.
#[derive(Clone, Copy, Debug)]
pub struct GpuRunStats {
    pub submissions: u64,
    pub max_submit_seconds: f64,
}

static LAST_STATS: Mutex<GpuRunStats> = Mutex::new(GpuRunStats {
    submissions: 0,
    max_submit_seconds: 0.0,
});

pub fn last_run_stats() -> GpuRunStats {
    *LAST_STATS.lock().unwrap_or_else(|e| e.into_inner())
}

fn reset_stats() {
    *LAST_STATS.lock().unwrap_or_else(|e| e.into_inner()) = GpuRunStats {
        submissions: 0,
        max_submit_seconds: 0.0,
    };
}

fn submit_budget() -> f64 {
    std::env::var("SUN_GPU_MAX_SUBMIT_SECONDS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(DEFAULT_SUBMIT_BUDGET_S)
}

/// True when the shader computes this pixel (mirrors its skip conditions).
pub(crate) fn pixel_computed(elev: f32, slope: f32, aspect: f32, mask: f32) -> bool {
    !(elev <= -1000.0
        || elev == UNDEFZ
        || mask == 0.0
        || mask == UNDEFZ
        || slope == UNDEFZ
        || aspect == UNDEFZ)
}

/// Cost weight per 64-pixel group of a tile: 1 if any pixel is computed
/// (a workgroup runs as long as its slowest pixel), else a small constant.
pub(crate) fn group_weights(pixels: usize, computed: impl Fn(usize) -> bool) -> Vec<f32> {
    (0..pixels.div_ceil(GROUP))
        .map(|g| {
            let end = ((g + 1) * GROUP).min(pixels);
            if (g * GROUP..end).any(&computed) {
                1.0
            } else {
                SKIPPED_GROUP_WEIGHT
            }
        })
        .collect()
}

/// Sizes pixel windows so each submission stays near the time budget.
///
/// Cost model per 64-pixel group: `weight × rate`. Within the first pass
/// over a tile the rate is the latest measured one. Every later pass (next
/// day, same tile) uses the per-group rates measured in the previous pass,
/// scaled by how far the current pass deviates from them, so a pass that
/// starts on expensive rows is not sized from the cheap rows the previous
/// pass ended on.
pub(crate) struct Chunker {
    budget: f64,
    weights: Vec<f32>,
    profile: Vec<f32>,
    next_profile: Vec<f32>,
    rate: f64,
    scale: f64,
    last_secs: f64,
}

impl Chunker {
    pub(crate) fn new() -> Self {
        Chunker {
            budget: submit_budget(),
            weights: Vec::new(),
            profile: Vec::new(),
            next_profile: Vec::new(),
            rate: 0.0,
            scale: 1.0,
            last_secs: 0.0,
        }
    }

    /// Start a new tile; `weights` from `group_weights`.
    pub(crate) fn begin_tile(&mut self, weights: Vec<f32>) {
        self.weights = weights;
        self.profile.clear();
        self.rate = 0.0;
    }

    fn begin_pass(&mut self, pixels: usize) {
        let groups = pixels.div_ceil(GROUP);
        if self.weights.len() != groups {
            self.begin_tile(vec![1.0; groups]);
        }
        self.next_profile = vec![0.0; groups];
        self.scale = 1.0;
        self.last_secs = 0.0;
    }

    fn end_pass(&mut self) {
        self.profile = std::mem::take(&mut self.next_profile);
    }

    fn group_cost(&self, g: usize) -> f64 {
        let w = self.weights[g] as f64;
        if self.profile.is_empty() {
            w * self.rate
        } else {
            w * self.profile[g] as f64 * self.scale
        }
    }

    /// End group (exclusive) of the next window starting at group `g0`.
    fn take(&self, g0: usize) -> usize {
        let groups = self.weights.len();
        let full = TARGET_FRACTION * self.budget;
        let target = if self.last_secs > 0.0 {
            full.min(MAX_GROWTH * self.last_secs)
        } else if !self.profile.is_empty() {
            // First window of a later pass: the profile is from another day.
            full / MAX_GROWTH
        } else {
            return (g0 + FIRST_GROUPS).min(groups);
        };
        let mut cost = 0.0;
        let mut g = g0;
        while g < groups {
            cost += self.group_cost(g);
            if cost > target && g > g0 {
                break;
            }
            g += 1;
        }
        g.max(g0 + 1)
    }

    fn record(&mut self, g0: usize, g1: usize, secs: f64) {
        {
            let mut s = LAST_STATS.lock().unwrap_or_else(|e| e.into_inner());
            s.submissions += 1;
            s.max_submit_seconds = s.max_submit_seconds.max(secs);
        }
        let secs = secs.max(1e-9);
        let weight: f64 = self.weights[g0..g1].iter().map(|&w| w as f64).sum();
        if !self.profile.is_empty() {
            let predicted: f64 = (g0..g1)
                .map(|g| self.weights[g] as f64 * self.profile[g] as f64)
                .sum();
            if predicted > 0.0 {
                self.scale = secs / predicted;
            }
        }
        self.rate = secs / weight.max(1e-9);
        for r in &mut self.next_profile[g0..g1] {
            *r = self.rate as f32;
        }
        self.last_secs = secs;
    }
}

fn gpu_failure(detail: &str) -> String {
    format!(
        "GPU computation failed: {detail}. The graphics driver may have reset \
         the GPU. Uncheck GPU acceleration to compute on the CPU."
    )
}

/// Dispatch `pixels` shader invocations in budget-sized windows, one GPU
/// submission each, so no submission runs into the driver watchdog.
/// `write_uniforms(pixel_start, pixel_count, dispatch_x_pixels)` must upload
/// the matching uniforms; `check` runs after every submission.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_in_windows(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    chunker: &mut Chunker,
    pixels: usize,
    write_uniforms: &mut dyn FnMut(u32, u32, u32),
    check: &dyn Fn() -> Result<(), String>,
) -> Result<(), String> {
    chunker.begin_pass(pixels);
    let groups = pixels.div_ceil(GROUP);
    let mut g0 = 0usize;
    while g0 < groups {
        let g1 = chunker.take(g0);
        let start = g0 * GROUP;
        let n = (g1 * GROUP).min(pixels) - start;
        let (wgx, wgy, dispatch_x_pixels) = dispatch_geometry(n);
        write_uniforms(start as u32, n as u32, dispatch_x_pixels);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("r.sun window"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("r.sun window pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(wgx, wgy, 1);
        }
        let t0 = Instant::now();
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        let secs = t0.elapsed().as_secs_f64();
        check()?;
        chunker.record(g0, g1, secs);
        g0 = g1;
    }
    chunker.end_pass();
    Ok(())
}

/// Run a GPU computation, turning wgpu panics (wgpu treats a lost device as
/// fatal) into an error. Requires `panic = "unwind"` in the release profile.
fn guarded<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown wgpu panic".into());
            Err(gpu_failure(msg.trim()))
        }
    }
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
    /// First error reported by wgpu (uncaptured error or device lost).
    error: Arc<Mutex<Option<String>>>,
}

impl GpuContext {
    fn check(&self) -> Result<(), String> {
        match self.error.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            Some(e) => Err(gpu_failure(e)),
            None => Ok(()),
        }
    }

    /// Submit one encoder and wait for it; returns the elapsed seconds.
    fn submit_and_wait(&self, encoder: wgpu::CommandEncoder) -> Result<f64, String> {
        let t0 = Instant::now();
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        let secs = t0.elapsed().as_secs_f64();
        self.check()?;
        Ok(secs)
    }

    fn read_buffer(&self, buf: &wgpu::Buffer, dst: &mut [f32]) -> Result<(), String> {
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        self.check()?;
        match rx.try_recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(gpu_failure(&format!("readback failed: {e}"))),
            Err(_) => return Err(gpu_failure("readback did not complete")),
        }
        {
            let data = slice.get_mapped_range();
            dst.copy_from_slice(bytemuck::cast_slice(&data));
        }
        buf.unmap();
        Ok(())
    }

    /// Run the shader over one sub-tile in budget-sized pixel windows.
    fn dispatch_windows(
        &self,
        chunker: &mut Chunker,
        uniform_buf: &wgpu::Buffer,
        bind_group: &wgpu::BindGroup,
        base: Uniforms,
        pixels: usize,
    ) -> Result<(), String> {
        let mut write = |pixel_start: u32, pixel_count: u32, dispatch_x_pixels: u32| {
            let u = Uniforms {
                dispatch_x_pixels,
                pixel_start,
                pixel_count,
                ..base
            };
            self.queue.write_buffer(uniform_buf, 0, bytemuck::bytes_of(&u));
        };
        dispatch_in_windows(
            &self.device,
            &self.queue,
            &self.pipeline,
            bind_group,
            chunker,
            pixels,
            &mut write,
            &|| self.check(),
        )
    }
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

    // wgpu's default handler panics on any uncaptured error; record it
    // instead and surface it as Err at the next checkpoint.
    let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    {
        let err = error.clone();
        device.on_uncaptured_error(Box::new(move |e| {
            let mut slot = err.lock().unwrap_or_else(|p| p.into_inner());
            if slot.is_none() {
                *slot = Some(format!("{e}"));
            }
        }));
    }
    {
        let err = error.clone();
        device.set_device_lost_callback(move |reason, msg| {
            if matches!(reason, wgpu::DeviceLostReason::Dropped) {
                return;
            }
            let mut slot = err.lock().unwrap_or_else(|p| p.into_inner());
            if slot.is_none() {
                *slot = Some(format!("GPU device lost ({reason:?}) {msg}"));
            }
        });
    }

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

    let ctx = GpuContext {
        device,
        queue,
        pipeline,
        bgl,
        max_binding: max_buf_binding as u64,
        error,
    };
    // Test hook: simulate a driver reset right after setup.
    if std::env::var_os("SUN_GPU_SIMULATE_LOST").is_some() {
        ctx.device.destroy();
    }
    ctx.check()?;
    Ok(ctx)
}

/// Group weights for the band pixels `[offset, offset + pixels)`.
fn tile_weights(inp: &BandInputs<'_>, offset: usize, pixels: usize) -> Vec<f32> {
    group_weights(pixels, |i| {
        let j = offset + i;
        pixel_computed(inp.elevation[j], inp.slope[j], inp.aspect[j], inp.mask[j])
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
    reset_stats();
    guarded(|| daily_band_gpu(inp, day, step, solar_constant, quiet))
}

fn daily_band_gpu(
    inp: &BandInputs<'_>,
    day: i32,
    step: f64,
    solar_constant: f64,
    quiet: bool,
) -> Result<GpuDailyOut, String> {
    let declination = com_declin(day);
    let g_norm_extra = com_sol_const(day, solar_constant);
    let sindecl = declination.sin() as f32;
    let cosdecl = declination.cos() as f32;
    let step_rad = (step * std::f64::consts::PI / 12.0) as f32;
    let max_z = inp.full_max_z();

    let ncols = inp.ncols;
    let nrows = inp.nrows;
    let npixels = ncols * nrows;

    pollster::block_on(async {
        let ctx = init_gpu(quiet, "r.sun daily band").await?;
        if !quiet {
            eprintln!("GPU band  day {day}  |  Decl: {:.2}°", -declination * RAD2DEG);
        }
        let mut chunker = Chunker::new();

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
            chunker.begin_tile(tile_weights(inp, start * ncols, pixels));
            let row_range = start..end;

            let base = Uniforms {
                sindecl,
                cosdecl,
                g_norm_extra: g_norm_extra as f32,
                step_rad,
                ncols: ncols as u32,
                nrows: h as u32,
                dispatch_x_pixels: 0,
                full_nrows: inp.full_nrows as u32,
                row_offset: (inp.row_offset + start) as u32,
                max_z: max_z as f32,
                dx: inp.dx as f32,
                dy: inp.dy as f32,
                n_az: 0, // daily: always ray-march
                pixel_start: 0,
                pixel_count: 0,
                _pad2: 0,
            };
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

            ctx.dispatch_windows(&mut chunker, &uniform_buf, &bind_group, base, pixels)?;

            let mut encoder =
                ctx.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("readback"),
                    });
            encoder.copy_buffer_to_buffer(&buf_beam, 0, &stage_beam, 0, byte_size);
            encoder.copy_buffer_to_buffer(&buf_diff, 0, &stage_diff, 0, byte_size);
            encoder.copy_buffer_to_buffer(&buf_refl, 0, &stage_refl, 0, byte_size);
            encoder.copy_buffer_to_buffer(&buf_glob, 0, &stage_glob, 0, byte_size);
            encoder.copy_buffer_to_buffer(&buf_insol, 0, &stage_insol, 0, byte_size);
            ctx.submit_and_wait(encoder)?;

            ctx.read_buffer(&stage_beam, &mut res.beam[range.clone()])?;
            ctx.read_buffer(&stage_diff, &mut res.diffuse[range.clone()])?;
            ctx.read_buffer(&stage_refl, &mut res.reflected[range.clone()])?;
            ctx.read_buffer(&stage_glob, &mut res.global[range.clone()])?;
            ctx.read_buffer(&stage_insol, &mut res.insol[range])?;

            start = end;
        }
        Ok::<GpuDailyOut, String>(res)
    })
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
    reset_stats();
    guarded(|| {
        annual_band_gpu(
            inp, days, day_step, step, solar_constant, panel_efficiency, horizon, quiet,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn annual_band_gpu(
    inp: &BandInputs<'_>,
    days: &[i32],
    day_step: i32,
    step: f64,
    solar_constant: f64,
    panel_efficiency: f64,
    horizon: Option<&HorizonMap>,
    quiet: bool,
) -> Result<Vec<f32>, String> {
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
        let mut chunker = Chunker::new();

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
            chunker.begin_tile(tile_weights(inp, start * ncols, pixels));
            let row_range = start..end;

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
                let base = Uniforms {
                    sindecl,
                    cosdecl,
                    g_norm_extra,
                    step_rad,
                    ncols: ncols as u32,
                    nrows: h as u32,
                    dispatch_x_pixels: 0,
                    full_nrows: inp.full_nrows as u32,
                    row_offset: (inp.row_offset + start) as u32,
                    max_z: max_z as f32,
                    dx: inp.dx as f32,
                    dy: inp.dy as f32,
                    n_az: n_az_u32,
                    pixel_start: 0,
                    pixel_count: 0,
                    _pad2: 0,
                };
                ctx.dispatch_windows(&mut chunker, &uniform_buf, &bind_group, base, pixels)?;

                let mut encoder =
                    ctx.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("annual_readback"),
                        });
                encoder.copy_buffer_to_buffer(&buf_glob, 0, &stage_glob, 0, byte_size);
                ctx.submit_and_wait(encoder)?;
                ctx.read_buffer(&stage_glob, &mut day_glob)?;

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
