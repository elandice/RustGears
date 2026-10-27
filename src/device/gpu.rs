//! GPU 연산 백엔드.
//!
//! * `gpu` 카고 기능(기본 활성화)이 켜져 있으면 [wgpu](https://wgpu.rs) (Vulkan / DirectX 12 / Metal)를 통해
//!   물리 GPU를 자동 탐지하고 WGSL 컴퓨트 셰이더로 수치 연산 커널을 실행합니다.
//! * 시스템의 물리 GPU 개수를 초과하는 디바이스 서수(또는 GPU가 없는 시스템의 모든 서수)는
//!   CPU에서 동작하는 *가상 GPU*로 투명하게 폴백(fallback)됩니다. 덕분에 멀티 GPU 분산 학습 코드
//!   ([`DataParallel`](crate::parallel::DataParallel) 등)를 GPU가 없는 머신에서도 동일하게 실행할 수 있습니다.
//!
//! 작은 크기의 연산은 PCIe 버스를 통한 호스트<->디바이스 간 데이터 복사 오버헤드가 더 크므로
//! 자동으로 초고속 CPU 커널에서 실행됩니다 (자세한 설정은 [`set_offload_threshold`] 참조).

use super::backend::{Backend, BinaryOp, UnaryOp};
use super::cpu::CpuBackend;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "gpu")]
mod wgpu_impl;

/// 행렬 곱셈을 GPU로 오프로드하기 위한 최소 `m * k * n` 기준값.
/// 기본값은 약 6,400만 연산으로 설정되어 있어, 중소형 행렬은 호스트-디바이스 PCIe 왕복 지연 없이
/// AVX2+FMA Rayon CPU 커널에서 즉시 실행됩니다.
static MATMUL_THRESHOLD: AtomicUsize = AtomicUsize::new(512 * 512 * 256);
/// 요소별 연산(Element-wise)을 GPU로 오프로드하기 위한 최소 원소 개수 기준값
static ELEMWISE_THRESHOLD: AtomicUsize = AtomicUsize::new(4 << 20);

/// 연산 작업을 물리 GPU로 오프로드할 임계값을 설정합니다.
///
/// * `matmul_flops` – 행렬 곱셈 시 GPU로 보낼 최소 `m * k * n` 연산량.
/// * `elementwise` – 요소별 커널 실행 시 GPU로 보낼 최소 원소 개수.
///
/// 모든 연산을 무조건 GPU에서 실행하도록 강제하려면 `set_offload_threshold(0, 0)`을 호출하십시오.
pub fn set_offload_threshold(matmul_flops: usize, elementwise: usize) {
    MATMUL_THRESHOLD.store(matmul_flops, Ordering::Relaxed);
    ELEMWISE_THRESHOLD.store(elementwise, Ordering::Relaxed);
}

/// 물리 GPU 어댑터의 세부 정보 구조체
#[derive(Clone, Debug)]
pub struct GpuAdapterInfo {
    /// 어댑터 장치 이름
    pub name: String,
    /// 그래픽/컴퓨트 API 백엔드
    pub api: String,
    /// 장치 종류 (예: 외장 GPU, 통합 GPU)
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

/// 라이브러리에서 감지되어 사용 가능한 물리 GPU의 개수를 반환합니다.
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

/// 감지된 모든 물리 GPU의 정보를 담은 벡터를 반환합니다.
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

/// 지정된 서수(`index`)에 매핑되는 GPU 백엔드 구조체
pub struct GpuBackend {
    index: usize,
    #[cfg(feature = "gpu")]
    ctx: Option<Arc<wgpu_impl::GpuContext>>,
    cpu: CpuBackend,
}

impl GpuBackend {
    /// 이 서수가 실제 물리 GPU 하드웨어에 매핑되어 있는지 여부를 반환합니다.
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
        // 임계값 미만이거나 GPU가 없는 경우 CPU 커널로 실행
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

/// 지정된 서수(`index`)에 대한 GPU 백엔드를 가져오거나 지연 생성(캐싱)합니다.
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
