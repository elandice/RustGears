//! wgpu (Vulkan / DX12 / Metal) implementation of the GPU kernels with pooled VRAM buffers.

use super::GpuAdapterInfo;
use std::sync::Mutex;

const MATMUL_WGSL: &str = r#"
struct Dims { m: u32, k: u32, n: u32, pad: u32 };

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> ta: array<array<f32, 16>, 16>;
var<workgroup> tb: array<array<f32, 16>, 16>;

// Tiled matrix multiplication: every 16x16 workgroup computes one output tile,
// staging tiles of A and B in fast workgroup (shared) memory.
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    var acc: f32 = 0.0;
    let tiles = (dims.k + 15u) / 16u;
    for (var t: u32 = 0u; t < tiles; t = t + 1u) {
        let ac = t * 16u + lid.x;
        if (row < dims.m && ac < dims.k) {
            ta[lid.y][lid.x] = a[row * dims.k + ac];
        } else {
            ta[lid.y][lid.x] = 0.0;
        }
        let br = t * 16u + lid.y;
        if (br < dims.k && col < dims.n) {
            tb[lid.y][lid.x] = b[br * dims.n + col];
        } else {
            tb[lid.y][lid.x] = 0.0;
        }
        workgroupBarrier();
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            acc = acc + ta[lid.y][i] * tb[i][lid.x];
        }
        workgroupBarrier();
    }
    if (row < dims.m && col < dims.n) {
        c[row * dims.n + col] = acc;
    }
}
"#;

const ELEMWISE_WGSL: &str = r#"
struct Params { op: u32, len: u32, pad0: u32, pad1: u32 };

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

fn apply(op: u32, x: f32, y: f32) -> f32 {
    switch op {
        case 0u: { return x + y; }
        case 1u: { return x - y; }
        case 2u: { return x * y; }
        case 3u: { return x / y; }
        case 10u: { return -x; }
        case 11u: { return exp(x); }
        case 12u: { return log(x); }
        case 13u: { return max(x, 0.0); }
        case 14u: { return 1.0 / (1.0 + exp(-x)); }
        case 15u: { return tanh(x); }
        case 16u: { return sqrt(x); }
        case 17u: { return 0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x * x * x))); }
        case 18u: { return abs(x); }
        default: { return x; }
    }
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 256u;
    if (idx < p.len) {
        o[idx] = apply(p.op, a[idx], b[idx]);
    }
}
"#;

/// Reusable GPU buffer pool to eliminate per-operation buffer allocation overhead.
struct BufferPool {
    a: Option<(wgpu::Buffer, u64)>,
    b: Option<(wgpu::Buffer, u64)>,
    out: Option<(wgpu::Buffer, u64)>,
    staging: Option<(wgpu::Buffer, u64)>,
    uniform: wgpu::Buffer,
}

impl BufferPool {
    fn new(device: &wgpu::Device) -> Self {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            a: None,
            b: None,
            out: None,
            staging: None,
            uniform,
        }
    }

    fn ensure_storage<'a>(
        slot: &'a mut Option<(wgpu::Buffer, u64)>,
        device: &wgpu::Device,
        min_bytes: u64,
        usage: wgpu::BufferUsages,
    ) -> &'a wgpu::Buffer {
        let needed = min_bytes.max(16);
        let recreate = match slot {
            Some((_, cap)) => *cap < needed,
            None => true,
        };
        if recreate {
            let cap = needed.next_power_of_two().max(4096);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: cap,
                usage,
                mapped_at_creation: false,
            });
            *slot = Some((buf, cap));
        }
        &slot.as_ref().unwrap().0
    }
}

pub struct GpuContext {
    pub info: GpuAdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    matmul: wgpu::ComputePipeline,
    elemwise: wgpu::ComputePipeline,
    pool: Mutex<BufferPool>,
}

/// Enumerate physical GPUs and create a device + pipelines for each.
pub fn discover() -> Vec<GpuContext> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let mut adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::PRIMARY);
    if adapters.is_empty() {
        if let Some(a) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })) {
            adapters.push(a);
        }
    }

    let mut out = Vec::new();
    for adapter in adapters {
        let info = adapter.get_info();
        if matches!(info.device_type, wgpu::DeviceType::Cpu) {
            continue;
        }
        let req = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rustgears-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ));
        let (device, queue) = match req {
            Ok(dq) => dq,
            Err(_) => continue,
        };

        let mm_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul"),
            source: wgpu::ShaderSource::Wgsl(MATMUL_WGSL.into()),
        });
        let ew_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("elemwise"),
            source: wgpu::ShaderSource::Wgsl(ELEMWISE_WGSL.into()),
        });
        let matmul = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul"),
            layout: None,
            module: &mm_mod,
            entry_point: "main",
        });
        let elemwise = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("elemwise"),
            layout: None,
            module: &ew_mod,
            entry_point: "main",
        });

        let pool = Mutex::new(BufferPool::new(&device));

        out.push(GpuContext {
            info: GpuAdapterInfo {
                name: info.name.clone(),
                api: format!("{:?}", info.backend),
                device_type: format!("{:?}", info.device_type),
            },
            device,
            queue,
            matmul,
            elemwise,
            pool,
        });
    }
    out
}

impl GpuContext {
    fn execute_pooled(
        &self,
        pipeline: &wgpu::ComputePipeline,
        a: &[f32],
        b: &[f32],
        uniform_vals: [u32; 4],
        out_len: usize,
        groups: (u32, u32),
    ) -> Vec<f32> {
        let mut pool = self.pool.lock().unwrap();
        let a_bytes: &[u8] = bytemuck::cast_slice(a);
        let b_bytes: &[u8] = bytemuck::cast_slice(b);
        let u_bytes: &[u8] = bytemuck::cast_slice(&uniform_vals);
        let out_size = (out_len * 4) as u64;

        let storage_in_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let storage_out_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let staging_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        BufferPool::ensure_storage(&mut pool.a, &self.device, a_bytes.len() as u64, storage_in_usage);
        BufferPool::ensure_storage(&mut pool.b, &self.device, b_bytes.len() as u64, storage_in_usage);
        BufferPool::ensure_storage(&mut pool.out, &self.device, out_size, storage_out_usage);
        BufferPool::ensure_storage(&mut pool.staging, &self.device, out_size, staging_usage);

        let ba = &pool.a.as_ref().unwrap().0;
        let bb = &pool.b.as_ref().unwrap().0;
        let bc = &pool.out.as_ref().unwrap().0;
        let staging = &pool.staging.as_ref().unwrap().0;
        let bu = &pool.uniform;

        self.queue.write_buffer(ba, 0, a_bytes);
        self.queue.write_buffer(bb, 0, b_bytes);
        self.queue.write_buffer(bu, 0, u_bytes);

        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ba.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bc.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bu.as_entire_binding() },
            ],
        });

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        enc.copy_buffer_to_buffer(bc, 0, staging, 0, out_size);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(0..out_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("gpu map channel").expect("gpu buffer map failed");
        let view = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();
        result
    }

    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let gx = n.div_ceil(16) as u32;
        let gy = m.div_ceil(16) as u32;
        self.execute_pooled(&self.matmul, a, b, [m as u32, k as u32, n as u32, 0], m * n, (gx, gy))
    }

    pub fn elementwise(&self, op: u32, a: &[f32], b: Option<&[f32]>) -> Vec<f32> {
        let len = a.len();
        let dummy = [0.0f32; 4];
        let b_slice = b.unwrap_or(&dummy);
        let groups = len.div_ceil(256) as u32;
        let gx = groups.min(65535);
        let gy = groups.div_ceil(gx);
        self.execute_pooled(&self.elemwise, a, b_slice, [op, len as u32, 0, 0], len, (gx, gy))
    }
}
