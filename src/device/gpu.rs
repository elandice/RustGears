//! GPU backend.
//!
//! * With the `gpu` cargo feature (default) real GPUs are discovered through
//!   [wgpu](https://wgpu.rs) (Vulkan / DirectX 12 / Metal) and kernels run as
//!   WGSL compute shaders.
//! * Device ordinals beyond the number of physical GPUs (or every ordinal when
//!   no GPU is present) map to *virtual GPUs* that execute on the CPU. This lets
//!   multi-GPU code (e.g. [`DataParallel`](crate::parallel::DataParallel)) run
//!   unchanged on any machine.
//!
//! Small workloads are executed on the CPU because the host<->device transfer
//! would dominate; see [`set_offload_threshold`].

use super::backend::{Backend, BinaryOp, UnaryOp};
use super::cpu::CpuBackend;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "gpu")]
mod wgpu_impl;

/// Minimum `m*k*n` for a matmul to be sent to the GPU.
static MATMUL_THRESHOLD: AtomicUsize = AtomicUsize::new(48 * 48 * 48);
/// Minimum element count for an element-wise op to be sent to the GPU.
static ELEMWISE_THRESHOLD: AtomicUsize = AtomicUsize::new(1 << 20);

/// Configure when work is offloaded to a physical GPU.
///
/// * `matmul_flops` – minimum `m*k*n` for matrix multiplication.
/// * `elementwise` – minimum number of elements for element-wise kernels.
///
/// Use `set_offload_threshold(0, 0)` to force every kernel onto the GPU.
pub fn set_offload_threshold(matmul_flops: usize, elementwise: usize) {
    MATMUL_THRESHOLD.store(matmul_flops, Ordering::Relaxed);
    ELEMWISE_THRESHOLD.store(elementwise, Ordering::Relaxed);
}

/// Information about a physical GPU adapter.
#[derive(Clone, Debug)]
pub struct GpuAdapterInfo {
    pub name: String,
    pub api: String,
    pub device_type: String,
}

#[cfg(feature = "gpu")]
fn contexts() -> &'static Vec<Arc<wgpu_impl::GpuContext>> {
    static CTX: OnceLock<Vec<Arc<wgpu_impl::GpuContext>>> = OnceLock::new();
    CTX.get_or_init(|| {
        if std::env::var("RUSTGEARS_DISABLE_GPU").or_else(|_| std::env::var("RUSTORCH_DISABLE_GPU")).map(|v| v == "1").unwrap_or(false) {
            return Vec::new();
        }
        wgpu_impl::discover().into_iter().map(Arc::new).collect()
    })
}

/// Number of physical GPUs usable by the library.
pub fn physical_count() -> usize {
    #[cfg(feature = "gpu")]
    {
        contexts().len()
    }
    #[cfg(not(feature = "gpu"))]
    {
        0
    }
}

/// Information about every physical GPU.
pub fn adapters() -> Vec<GpuAdapterInfo> {
    #[cfg(feature = "gpu")]
    {
        contexts().iter().map(|c| c.info.clone()).collect()
    }
    #[cfg(not(feature = "gpu"))]
    {
        Vec::new()
    }
}

/// Backend for GPU ordinal `index`.
pub struct GpuBackend {
    index: usize,
    #[cfg(feature = "gpu")]
    ctx: Option<Arc<wgpu_impl::GpuContext>>,
    cpu: CpuBackend,
}

impl GpuBackend {
    /// `true` if this ordinal maps to real GPU hardware.
    pub fn is_physical(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.ctx.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }
}

impl Backend for GpuBackend {
    fn name(&self) -> String {
        #[cfg(feature = "gpu")]
        if let Some(ctx) = &self.ctx {
            return format!("gpu:{} {} ({})", self.index, ctx.info.name, ctx.info.api);
        }
        format!("gpu:{} (virtual, CPU-emulated)", self.index)
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        #[cfg(feature = "gpu")]
        if let Some(ctx) = &self.ctx {
            if m * k * n >= MATMUL_THRESHOLD.load(Ordering::Relaxed) && m > 0 && n > 0 && k > 0 {
                return ctx.matmul(a, b, m, k, n);
            }
        }
        self.cpu.matmul(a, b, m, k, n)
    }

    fn binary(&self, op: BinaryOp, a: &[f32], b: &[f32]) -> Vec<f32> {
        #[cfg(feature = "gpu")]
        if let Some(ctx) = &self.ctx {
            if !a.is_empty() && a.len() >= ELEMWISE_THRESHOLD.load(Ordering::Relaxed) {
                return ctx.elementwise(op.code(), a, Some(b));
            }
        }
        self.cpu.binary(op, a, b)
    }

    fn unary(&self, op: UnaryOp, a: &[f32]) -> Vec<f32> {
        #[cfg(feature = "gpu")]
        if let Some(ctx) = &self.ctx {
            if !a.is_empty() && a.len() >= ELEMWISE_THRESHOLD.load(Ordering::Relaxed) {
                return ctx.elementwise(op.code(), a, None);
            }
        }
        self.cpu.unary(op, a)
    }
}

/// Get (or lazily create) the backend for GPU ordinal `index`.
pub fn backend(index: usize) -> Arc<GpuBackend> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<GpuBackend>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap();
    map.entry(index)
        .or_insert_with(|| {
            #[cfg(feature = "gpu")]
            let ctx = contexts().get(index).cloned();
            let be = GpuBackend {
                index,
                #[cfg(feature = "gpu")]
                ctx,
                cpu: CpuBackend,
            };
            if !be.is_physical() {
                eprintln!(
                    "[rustgears] gpu:{index} is not a physical GPU ({} found) -> using a CPU-emulated virtual GPU",
                    physical_count()
                );
            }
            Arc::new(be)
        })
        .clone()
}
