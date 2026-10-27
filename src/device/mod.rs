//! 디바이스 선택 모듈: 텐서 연산이 실행될 컴퓨팅 하드웨어를 지정합니다.
//!
//! ```no_run
//! use rustgears::prelude::*;
//! let dev = Device::best_available();      // gpu:0 if a GPU exists, otherwise cpu
//! let dev = Device::parse("gpu:1").unwrap(); // explicit selection
//! for info in rustgears::device::list_devices() { println!("{info}"); }
//! ```

pub mod backend;
pub mod cpu;
pub mod gpu;

pub use backend::{Backend, BinaryOp, UnaryOp};

use std::fmt;
use std::sync::{Arc, OnceLock};

/// **학습 및 텐서 수치 연산**에 사용되는 컴퓨팅 디바이스 열거형.
///
/// NPU는 추론 전용 가속기이므로 이 열거형 대신
/// [`ExecutionProvider`](crate::inference::ExecutionProvider)를 통해 지정합니다.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// 호스트 CPU (Rayon 기반 멀티스레드 연산)
    #[default]
    Cpu,
    /// 지정된 서수(ordinal)를 갖는 GPU (`gpu:0`, `gpu:1`, ...)
    Gpu(usize),
}

impl Device {
    /// 기본 CPU 디바이스를 반환합니다.
    pub fn cpu() -> Self {
        Device::Cpu
    }

    /// 인덱스 번호에 해당하는 GPU 디바이스를 반환합니다.
    pub fn gpu(index: usize) -> Self {
        Device::Gpu(index)
    }

    /// 시스템에서 감지된 물리 GPU의 총 개수를 반환합니다.
    pub fn gpu_count() -> usize {
        gpu::physical_count()
    }

    /// 사용 가능한 물리 GPU가 1개 이상 존재하는지 확인합니다.
    pub fn is_gpu_available() -> bool {
        Self::gpu_count() > 0
    }

    /// 물리 GPU가 존재하면 `gpu:0`, 없으면 `cpu`를 반환합니다.
    pub fn best_available() -> Self {
        if Self::is_gpu_available() {
            Device::Gpu(0)
        } else {
            Device::Cpu
        }
    }

    /// 사용 가능한 모든 물리 GPU 목록(`[gpu:0, gpu:1, ...]`)을 반환합니다.
    pub fn all_gpus() -> Vec<Device> {
        (0..Self::gpu_count()).map(Device::Gpu).collect()
    }

    /// 문자열(`"cpu"`, `"gpu"`, `"cuda"`, `"gpu:1"`, `"auto"` 등)을 파싱하여 디바이스를 생성합니다.
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

    /// CPU 디바이스인지 확인합니다.
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// GPU 디바이스인지 확인합니다.
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Gpu(_))
    }

    /// 해당 디바이스의 수치 연산을 실행하는 백엔드(`Arc<dyn Backend>`)를 반환합니다.
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

/// 시스템에서 사용 가능한 연산 디바이스의 세부 정보.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    /// 디바이스 식별자
    pub device: Device,
    /// 장치 이름 (예: "NVIDIA GeForce RTX 4090", "Host CPU")
    pub name: String,
    /// 사용 중인 그래픽/연산 API (예: "Vulkan", "DirectX 12", "rayon")
    pub api: String,
    /// 디바이스 유형 (예: "DiscreteGpu", "Cpu")
    pub kind: String,
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:<7} {:<40} [{} / {}]", self.device.to_string(), self.name, self.kind, self.api)
    }
}

/// 시스템의 모든 학습용 디바이스(CPU 및 물리 GPU 목록)를 열거합니다.
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
