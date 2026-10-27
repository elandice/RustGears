//! 신경망 구성 요소 및 레이어 추상화 모듈 (`torch.nn`과 유사).

pub mod attention;
pub mod init;
pub mod layers;
pub mod loss;
pub mod metrics;
pub mod norm;

pub use attention::{KvCache, MultiHeadAttention, RotaryEmbedding, TransformerEncoderLayer};
pub use layers::{
    AdaptiveAvgPool2d, Conv2d, ConvTranspose2d, Dropout, Embedding, Flatten, Gelu, Identity, LeakyRelu, Linear,
    MaxPool2d, Relu, Sigmoid, Silu, Softmax, SwiGLU, Tanh,
};
pub use metrics::{confusion_matrix, precision_recall_f1, top_k_accuracy};
pub use norm::{BatchNorm1d, BatchNorm2d, LayerNorm, RMSNorm};

use crate::device::Device;
use crate::inference::LayerSpec;
use crate::tensor::Tensor;

/// 모든 신경망 레이어 및 모델이 구현해야 하는 기본 트레이트 (`torch.nn.Module`에 대응).
///
/// [`DataParallel`](crate::parallel::DataParallel)을 통해 서로 다른 디바이스에서 여러 모델 복제본이
/// 동시에 안전하게 실행될 수 있도록 모든 모듈은 `Send + Sync`를 만족해야 합니다.
pub trait Module: Send + Sync {
    /// 입력 텐서 `x`에 대한 순전파(Forward pass) 연산을 수행합니다.
    fn forward(&self, x: &Tensor) -> Tensor;

    /// 계층적 경로 이름을 포함한 학습 가능한 매개변수(Weight, Bias 등) 목록을 반환합니다.
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }

    /// 학습되지 않는 내부 상태 버퍼(예: BatchNorm의 `running_mean`, `running_var`) 목록을 반환합니다.
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }

    /// 레이어의 식별 이름을 반환합니다.
    fn name(&self) -> String {
        "Module".into()
    }

    /// 학습 모드(`true`)와 평가/추론 모드(`false`)를 전환합니다 (Dropout, BatchNorm 등의 동작 분기).
    fn set_training(&self, _training: bool) {}

    /// 레이어를 플랫폼 독립적인 추론 형식([`LayerSpec`])으로 내보냅니다 (CPU/GPU/NPU 고속 추론 세션용).
    fn export(&self) -> Result<Vec<LayerSpec>, String> {
        Err(format!("layer '{}' does not support export", self.name()))
    }

    // ---------------- 기본 제공 메서드 (Provided methods) ----------------

    /// 학습 가능한 모든 파라미터 텐서 목록을 반환합니다.
    fn parameters(&self) -> Vec<Tensor> {
        self.named_parameters().into_iter().map(|(_, t)| t).collect()
    }

    /// 모델을 학습 모드로 설정합니다.
    fn train(&self) {
        self.set_training(true)
    }

    /// 모델을 평가(추론) 모드로 설정합니다.
    fn eval(&self) {
        self.set_training(false)
    }

    /// 모든 매개변수와 버퍼를 지정된 디바이스로 이동시킵니다 (in-place).
    fn to(&self, device: Device) {
        for p in self.parameters() {
            p.set_device(device);
        }
        for (_, b) in self.named_buffers() {
            b.set_device(device);
        }
    }

    /// 모든 매개변수의 누적 기울기를 0으로 초기화합니다.
    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }

    /// 모델의 학습 가능한 총 파라미터(원소) 개수를 반환합니다.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    /// 모델 상태를 완전히 저장하거나 복원하는 데 필요한 모든 매개변수와 버퍼의 목록을 반환합니다.
    fn state_dict(&self) -> Vec<(String, Tensor)> {
        let mut v = self.named_parameters();
        v.extend(self.named_buffers());
        v
    }
}

impl<M: Module + ?Sized> Module for Box<M> {
    fn forward(&self, x: &Tensor) -> Tensor {
        (**self).forward(x)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        (**self).named_parameters()
    }
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        (**self).named_buffers()
    }
    fn name(&self) -> String {
        (**self).name()
    }
    fn set_training(&self, t: bool) {
        (**self).set_training(t)
    }
    fn export(&self) -> Result<Vec<LayerSpec>, String> {
        (**self).export()
    }
}

/// 하위 모듈의 텐서 이름 앞에 접두사(prefix)를 붙여 계층 구조를 만듭니다 (예: `"0.weight"`).
pub fn prefixed(prefix: &str, v: Vec<(String, Tensor)>) -> Vec<(String, Tensor)> {
    v.into_iter().map(|(n, t)| (format!("{prefix}.{n}"), t)).collect()
}

/// 여러 신경망 레이어를 순서대로 연결하여 실행하는 순차 컨테이너 (`nn.Sequential`과 동일).
#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// 빈 순차 컨테이너를 생성합니다.
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    /// 빌더 패턴: 레이어를 뒤에 추가합니다.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, m: M) -> Self {
        self.layers.push(Box::new(m));
        self
    }

    /// 가변 참조를 통해 새 레이어를 추가합니다.
    pub fn push<M: Module + 'static>(&mut self, m: M) {
        self.layers.push(Box::new(m));
    }

    /// 포함된 레이어 개수를 반환합니다.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// 레이어가 비어 있는지 확인합니다.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// 포함된 레이어 슬라이스를 반환합니다.
    pub fn layers(&self) -> &[Box<dyn Module>] {
        &self.layers
    }

    /// 사람이 읽기 편한 모델 구조 및 파라미터 요약 정보를 문자열로 반환합니다.
    pub fn summary(&self) -> String {
        let mut s = String::from("Sequential(\n");
        for (i, l) in self.layers.iter().enumerate() {
            let n: usize = l.parameters().iter().map(|p| p.numel()).sum();
            s += &format!("  ({i}) {:<40} params: {n}\n", l.name());
        }
        s += &format!(")\nTotal trainable parameters: {}\n", self.num_parameters());
        s
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = x.clone();
        for l in &self.layers {
            h = l.forward(&h);
        }
        h
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.layers.iter().enumerate().flat_map(|(i, l)| prefixed(&i.to_string(), l.named_parameters())).collect()
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        self.layers.iter().enumerate().flat_map(|(i, l)| prefixed(&i.to_string(), l.named_buffers())).collect()
    }

    fn name(&self) -> String {
        format!("Sequential[{}]", self.layers.len())
    }

    fn set_training(&self, t: bool) {
        for l in &self.layers {
            l.set_training(t);
        }
    }

    fn export(&self) -> Result<Vec<LayerSpec>, String> {
        let mut v = Vec::new();
        for l in &self.layers {
            v.extend(l.export()?);
        }
        Ok(v)
    }
}
