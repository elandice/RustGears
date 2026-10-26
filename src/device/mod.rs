//! Device selection: decide *where* tensors are computed.
//!
//! ```no_run
//! use rustorch::prelude::*;
//! let dev = Device::best_available();      // gpu:0 if a GPU exists, otherwise cpu
//! let dev = Device::parse("gpu:1").unwrap(); // explicit selection
//! for info in rustorch::device::list_devices() { println!("{info}"); }
//! ```

pub mod backend;
pub mod cpu;
pub mod gpu;

pub use backend::{Backend, BinaryOp, UnaryOp};

use std::fmt;
use std::sync::{Arc, OnceLock};

/// A compute device used for **training and tensor math**.
///
/// NPUs are inference-only accelerators and are therefore selected through
/// [`ExecutionProvider`](crate::inference::ExecutionProvider) instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// Host CPU (multi-threaded with rayon).
    #[default]
    Cpu,
    /// GPU with the given ordinal (`gpu:0`, `gpu:1`, ...).
    Gpu(usize),
}

impl Device {
    pub fn cpu() -> Self {
        Device::Cpu
    }

    pub fn gpu(index: usize) -> Self {
        Device::Gpu(index)
    }

    /// Number of physical GPUs detected.
    pub fn gpu_count() -> usize {
        gpu::physical_count()
    }

    /// `true` if at least one physical GPU can be used.
    pub fn is_gpu_available() -> bool {
        Self::gpu_count() > 0
    }

    /// `gpu:0` when a physical GPU is present, otherwise `cpu`.
    pub fn best_available() -> Self {
        if Self::is_gpu_available() {
            Device::Gpu(0)
        } else {
            Device::Cpu
        }
    }

    /// All physical GPUs (`[gpu:0, gpu:1, ...]`).
    pub fn all_gpus() -> Vec<Device> {
        (0..Self::gpu_count()).map(Device::Gpu).collect()
    }

    /// Parse `"cpu"`, `"gpu"`, `"cuda"`, `"gpu:1"`, `"auto"`.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim().to_ascii_lowercase();
        match s.as_str() {
            "cpu" => Ok(Device::Cpu),
            "gpu" | "cuda" | "wgpu" => Ok(Device::Gpu(0)),
            "auto" | "best" => Ok(Self::best_available()),
            _ => {
                if let Some(rest) = s.strip_prefix("gpu:").or_else(|| s.strip_prefix("cuda:")) {
                    rest.parse::<usize>().map(Device::Gpu).map_err(|e| format!("invalid gpu ordinal '{rest}': {e}"))
                } else {
                    Err(format!("unknown device '{s}' (expected cpu | gpu | gpu:N | auto)"))
                }
            }
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Gpu(_))
    }

    /// The compute backend that executes kernels for this device.
    pub fn backend(&self) -> Arc<dyn Backend> {
        static CPU: OnceLock<Arc<cpu::CpuBackend>> = OnceLock::new();
        match self {
            Device::Cpu => CPU.get_or_init(|| Arc::new(cpu::CpuBackend)).clone(),
            Device::Gpu(i) => gpu::backend(*i),
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Gpu(i) => write!(f, "gpu:{i}"),
        }
    }
}

impl std::str::FromStr for Device {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Device::parse(s)
    }
}

/// Description of an available compute device.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub device: Device,
    pub name: String,
    pub api: String,
    pub kind: String,
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:<7} {:<40} [{} / {}]", self.device.to_string(), self.name, self.kind, self.api)
    }
}

/// Enumerate all training devices (CPU + physical GPUs).
pub fn list_devices() -> Vec<DeviceInfo> {
    let mut v = vec![DeviceInfo {
        device: Device::Cpu,
        name: format!("Host CPU ({} threads)", rayon::current_num_threads()),
        api: "rayon".into(),
        kind: "Cpu".into(),
    }];
    for (i, a) in gpu::adapters().into_iter().enumerate() {
        v.push(DeviceInfo { device: Device::Gpu(i), name: a.name, api: a.api, kind: a.device_type });
    }
    v
}
