//! Portable GPU backend (wgpu + WGSL) for the batched sphere distance query.
//!
//! Same operations and memory layout as `motion_core::cpu`; results are
//! checked against it.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use motion_core::cpu::BatchOutput;
use motion_core::layout::ModelBuffers;
use wgpu::util::DeviceExt;

const WG: u32 = 64;
/// Workgroups along x before wrapping into y (WebGPU allows 65535 per dimension).
const MAX_WG_X: u32 = 32768;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// One thread per configuration, writes per-sphere distance + gradient and the per-config minimum.
    FusedFull,
    /// One thread per configuration, writes only the per-config minimum (collision check).
    FusedMin,
    /// Three dispatches: FK -> world spheres, per-sphere distance, per-config reduction.
    Split,
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
}

impl GpuContext {
    pub fn new() -> Result<Self, String> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            );
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .map_err(|e| e.to_string())?;
            let info = adapter.get_info();
            let limits = adapter.limits();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("motion_gpu"),
                    required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                        max_buffer_size: limits.max_buffer_size,
                        max_storage_buffers_per_shader_stage: 8,
                        ..wgpu::Limits::default()
                    },
                    ..Default::default()
                })
                .await
                .map_err(|e| e.to_string())?;
            Ok(Self {
                device,
                queue,
                info,
            })
        })
    }

    pub fn max_storage_binding(&self) -> u64 {
        self.device.limits().max_storage_buffer_binding_size
    }
}

struct Pipelines {
    fused_full: wgpu::ComputePipeline,
    fused_min: wgpu::ComputePipeline,
    split_fk_spheres: wgpu::ComputePipeline,
    split_distance: wgpu::ComputePipeline,
    split_reduce: wgpu::ComputePipeline,
}

/// Model buffers on the device plus batch buffers for a fixed batch size.
pub struct GpuQuery<'a> {
    ctx: &'a GpuContext,
    pipelines: Pipelines,
    bind_group: wgpu::BindGroup,
    q: wgpu::Buffer,
    sphere_dist: wgpu::Buffer,
    sphere_grad: wgpu::Buffer,
    config_min: wgpu::Buffer,
    pub batch: usize,
    pub n_spheres: usize,
    pub dof: usize,
}

fn storage(ctx: &GpuContext, label: &str, size: u64, extra: wgpu::BufferUsages) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE | extra,
        mapped_at_creation: false,
    })
}

fn dispatch_dims(threads: u32) -> (u32, u32) {
    let groups = threads.div_ceil(WG);
    let x = groups.min(MAX_WG_X);
    (x, groups.div_ceil(x))
}

impl<'a> GpuQuery<'a> {
    pub fn new(ctx: &'a GpuContext, model: &ModelBuffers, batch: usize) -> Result<Self, String> {
        let n_spheres = model.spheres.len();
        let big = (batch * n_spheres * 16) as u64;
        if big > ctx.max_storage_binding() {
            return Err(format!(
                "batch {batch} needs {big} bytes per binding, device allows {}",
                ctx.max_storage_binding()
            ));
        }
        let device = &ctx.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("query.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("query.wgsl").into()),
        });

        let ro = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let rw = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("query"),
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
                ro(1),
                ro(2),
                ro(3),
                ro(4),
                rw(5),
                rw(6),
                rw(7),
                rw(8),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("query"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = Pipelines {
            fused_full: pipeline("fused_full"),
            fused_min: pipeline("fused_min"),
            split_fk_spheres: pipeline("split_fk_spheres"),
            split_distance: pipeline("split_distance"),
            split_reduce: pipeline("split_reduce"),
        };

        let init = |label, contents: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        let params = init(
            "params",
            bytemuck::bytes_of(&model.params(batch)),
            wgpu::BufferUsages::UNIFORM,
        );
        let links = init(
            "links",
            bytemuck::cast_slice(&model.links),
            wgpu::BufferUsages::STORAGE,
        );
        let spheres = init(
            "spheres",
            bytemuck::cast_slice(&model.spheres),
            wgpu::BufferUsages::STORAGE,
        );
        let cuboids = init(
            "cuboids",
            bytemuck::cast_slice(&model.cuboids),
            wgpu::BufferUsages::STORAGE,
        );
        let q = storage(
            ctx,
            "q",
            (model.dof * batch * 4) as u64,
            wgpu::BufferUsages::COPY_DST,
        );
        let spheres_world = storage(ctx, "spheres_world", big, wgpu::BufferUsages::empty());
        let sphere_dist = storage(
            ctx,
            "sphere_dist",
            (batch * n_spheres * 4) as u64,
            wgpu::BufferUsages::COPY_SRC,
        );
        let sphere_grad = storage(ctx, "sphere_grad", big, wgpu::BufferUsages::COPY_SRC);
        let config_min = storage(
            ctx,
            "config_min",
            (batch * 4) as u64,
            wgpu::BufferUsages::COPY_SRC,
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("query"),
            layout: &layout,
            entries: &[
                &params,
                &links,
                &spheres,
                &cuboids,
                &q,
                &spheres_world,
                &sphere_dist,
                &sphere_grad,
                &config_min,
            ]
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
        });

        Ok(Self {
            ctx,
            pipelines,
            bind_group,
            q,
            sphere_dist,
            sphere_grad,
            config_min,
            batch,
            n_spheres,
            dof: model.dof,
        })
    }

    pub fn upload_q(&self, q: &[f32]) {
        assert_eq!(q.len(), self.dof * self.batch);
        self.ctx
            .queue
            .write_buffer(&self.q, 0, bytemuck::cast_slice(q));
    }

    fn encode_pass(&self, encoder: &mut wgpu::CommandEncoder, kernel: Kernel) {
        self.encode_pass_timed(encoder, kernel, None);
    }

    fn encode_pass_timed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        kernel: Kernel,
        timestamps: Option<&wgpu::QuerySet>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: timestamps.map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        pass.set_bind_group(0, &self.bind_group, &[]);
        let per_config = dispatch_dims(self.batch as u32);
        let mut run = |p: &wgpu::ComputePipeline, (x, y): (u32, u32)| {
            pass.set_pipeline(p);
            pass.dispatch_workgroups(x, y, 1);
        };
        match kernel {
            Kernel::FusedFull => run(&self.pipelines.fused_full, per_config),
            Kernel::FusedMin => run(&self.pipelines.fused_min, per_config),
            Kernel::Split => {
                run(&self.pipelines.split_fk_spheres, per_config);
                run(
                    &self.pipelines.split_distance,
                    dispatch_dims((self.batch * self.n_spheres) as u32),
                );
                run(&self.pipelines.split_reduce, per_config);
            }
        }
    }

    fn wait(&self) {
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
    }

    /// Run the kernel `reps` times in one submission and wait; returns time per run.
    pub fn time_throughput(&self, kernel: Kernel, reps: u32) -> Duration {
        let mut encoder = self.ctx.device.create_command_encoder(&Default::default());
        for _ in 0..reps {
            self.encode_pass(&mut encoder, kernel);
        }
        let cmd = encoder.finish();
        let t = Instant::now();
        self.ctx.queue.submit([cmd]);
        self.wait();
        t.elapsed() / reps
    }

    /// One submission of one kernel run, waited for (what a solver iteration pays per launch).
    pub fn time_latency(&self, kernel: Kernel) -> Duration {
        let mut encoder = self.ctx.device.create_command_encoder(&Default::default());
        self.encode_pass(&mut encoder, kernel);
        let t = Instant::now();
        self.ctx.queue.submit([encoder.finish()]);
        self.wait();
        t.elapsed()
    }

    /// GPU-side duration of one pass (timestamp queries), if the device supports them.
    pub fn time_gpu(&self, kernel: Kernel) -> Option<Duration> {
        if !self
            .ctx
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return None;
        }
        let set = self.ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamps"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = self.ctx.device.create_command_encoder(&Default::default());
        self.encode_pass_timed(&mut encoder, kernel, Some(&set));
        encoder.resolve_query_set(&set, 0..2, &resolve, 0);
        let staging = self.copy_to_staging(&mut encoder, &resolve, 16);
        self.ctx.queue.submit([encoder.finish()]);
        let ticks = self.read_staging::<u64>(&staging);
        if ticks[0] == 0 && ticks[1] == 0 {
            // Feature advertised but no timestamps written (seen on Apple Silicon / Metal).
            return None;
        }
        let ns =
            ticks[1].saturating_sub(ticks[0]) as f64 * self.ctx.queue.get_timestamp_period() as f64;
        Some(Duration::from_nanos(ns as u64))
    }

    /// Submit an empty command buffer and wait: the floor under every launch.
    pub fn time_empty_submit(&self) -> Duration {
        let encoder = self.ctx.device.create_command_encoder(&Default::default());
        let t = Instant::now();
        self.ctx.queue.submit([encoder.finish()]);
        self.wait();
        t.elapsed()
    }

    /// Upload configurations, run, read back the per-configuration minimum: a full host round trip.
    pub fn round_trip_min(&self, q: &[f32], kernel: Kernel) -> (Vec<f32>, Duration) {
        let t = Instant::now();
        self.upload_q(q);
        let mut encoder = self.ctx.device.create_command_encoder(&Default::default());
        self.encode_pass(&mut encoder, kernel);
        let staging = self.copy_to_staging(&mut encoder, &self.config_min, (self.batch * 4) as u64);
        self.ctx.queue.submit([encoder.finish()]);
        let out = self.read_staging::<f32>(&staging);
        (out, t.elapsed())
    }

    /// Run a kernel and read back every output buffer.
    pub fn run_full(&self, q: &[f32], kernel: Kernel) -> BatchOutput {
        self.upload_q(q);
        let mut encoder = self.ctx.device.create_command_encoder(&Default::default());
        self.encode_pass(&mut encoder, kernel);
        let n = (self.batch * self.n_spheres) as u64;
        let dist = self.copy_to_staging(&mut encoder, &self.sphere_dist, n * 4);
        let grad = self.copy_to_staging(&mut encoder, &self.sphere_grad, n * 16);
        let min = self.copy_to_staging(&mut encoder, &self.config_min, (self.batch * 4) as u64);
        self.ctx.queue.submit([encoder.finish()]);
        BatchOutput {
            batch: self.batch,
            sphere_dist: self.read_staging(&dist),
            sphere_grad: self.read_staging(&grad),
            config_min: self.read_staging(&min),
        }
    }

    fn copy_to_staging(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        size: u64,
    ) -> wgpu::Buffer {
        let staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        staging
    }

    /// Asynchronous map (the only kind WebGPU has); natively we poll until the callback fires.
    fn read_staging<T: bytemuck::Pod>(&self, staging: &wgpu::Buffer) -> Vec<T> {
        let (tx, rx) = mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| tx.send(r).unwrap());
        self.wait();
        rx.recv().unwrap().expect("map staging buffer");
        let data = bytemuck::cast_slice::<u8, T>(
            &staging.slice(..).get_mapped_range().expect("mapped range")[..],
        )
        .to_vec();
        staging.unmap();
        data
    }
}
