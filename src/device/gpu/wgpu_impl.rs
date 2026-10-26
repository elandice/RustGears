//! wgpu (Vulkan / DX12 / Metal) implementation of the GPU kernels.

use super::GpuAdapterInfo;
use wgpu::util::DeviceExt;

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

pub struct GpuContext {
    pub info: GpuAdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    matmul: wgpu::ComputePipeline,
    elemwise: wgpu::ComputePipeline,
}

/// Enumerate physical GPUs and create a device + pipelines for each.
pub fn discover() -> Vec<GpuContext> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let mut adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::PRIMARY).into_iter().collect();
    // prefer Vulkan, then DX12, then Metal
    adapters.sort_by_key(|a| match a.get_info().backend {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Dx12 => 1,
        wgpu::Backend::Metal => 2,
        _ => 3,
    });
    for adapter in adapters {
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            continue; // software rasterizers (WARP / llvmpipe) are not GPUs
        }
        if !seen.insert((info.name.clone(), info.vendor, info.device)) {
            continue; // same physical GPU exposed by another API
        }
        let limits = adapter.limits();
        let res = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rustorch-device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
            },
            None,
        ));
        let (device, queue) = match res {
            Ok(dq) => dq,
            Err(e) => {
                eprintln!("[rustorch] failed to open GPU '{}': {e}", info.name);
                continue;
            }
        };
        let make = |src: &str, label: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: "main",
            })
        };
        let matmul = make(MATMUL_WGSL, "matmul");
        let elemwise = make(ELEMWISE_WGSL, "elementwise");
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
        });
    }
    out
}

impl GpuContext {
    fn storage(&self, data: &[f32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    fn uniform(&self, v: [u32; 4]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&v),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn run(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: [&wgpu::Buffer; 4],
        out_len: usize,
        groups: (u32, u32),
    ) -> Vec<f32> {
        let size = (out_len * 4) as u64;
        let out = buffers[2];
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buffers[0].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buffers[1].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buffers[2].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buffers[3].as_entire_binding() },
            ],
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        enc.copy_buffer_to_buffer(out, 0, &staging, 0, size);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
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

    fn output(&self, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let ba = self.storage(a);
        let bb = self.storage(b);
        let bc = self.output(m * n);
        let bu = self.uniform([m as u32, k as u32, n as u32, 0]);
        let gx = ((n + 15) / 16) as u32;
        let gy = ((m + 15) / 16) as u32;
        self.run(&self.matmul, [&ba, &bb, &bc, &bu], m * n, (gx, gy))
    }

    pub fn elementwise(&self, op: u32, a: &[f32], b: Option<&[f32]>) -> Vec<f32> {
        let len = a.len();
        let ba = self.storage(a);
        let bb = match b {
            Some(b) => self.storage(b),
            None => self.storage(&[0.0; 4]),
        };
        let bo = self.output(len);
        let bu = self.uniform([op, len as u32, 0, 0]);
        let groups = ((len + 255) / 256) as u32;
        let gx = groups.min(65535);
        let gy = (groups + gx - 1) / gx;
        self.run(&self.elemwise, [&ba, &bb, &bo, &bu], len, (gx, gy))
    }
}
