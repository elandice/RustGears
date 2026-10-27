//! # RustGears
//!
//! 순수 Rust로 작성된 PyTorch / TensorFlow 스타일의 딥러닝 프레임워크입니다.
//!
//! ## 주요 기능 및 특징
//! * **역방향 자동 미분 (Reverse-mode Autograd)** – 동적 계산 그래프 지원 (`requires_grad_`, `backward()`, `no_grad`).
//! * **디바이스 선택 (`CPU` / `GPU`)** – `rayon` 기반 멀티스레드 CPU 커널 및 `wgpu` (Vulkan / DirectX 12 / Metal) 기반 하드웨어 GPU 연산(가상 GPU 폴백 내장).
//! * **멀티 GPU 데이터 병렬화** – scatter, 동시 복제본 순전파/역전파 및 대역폭 최적화 [`ring_all_reduce`](parallel::ring_all_reduce)를 지원하는 [`DataParallel`](parallel::DataParallel).
//! * **신경망 레이어 및 옵티마이저** – `Linear`, `Conv2d`, `MaxPool2d`, `BatchNorm1d/2d`, `LayerNorm`, `Dropout`, `Embedding`, `MultiHeadAttention`, `TransformerEncoderLayer`, `Sequential`, `SGD`, `Adam`, `AdamW`, `RMSprop`, 다양한 학습률 스케줄러.
//! * **현대적인 모델 유틸리티** – Safetensors 가중치 호환, RoPE가 적용된 캐시 기반 인과적(causal) 어텐션, 제한 대기열 배치 프리페칭, `AdaptiveAvgPool2d`, `ConvTranspose2d`, `ModelEMA`, 콘솔 프로그레스 바.
//! * **NPU 추론 런타임** – 학습된 모델을 [`InferenceSession`](inference::InferenceSession)으로 내보내고, 연산자 융합(`Linear/Conv2d + BatchNorm + Activation`) 및 대칭적 **INT8 / FP16** 시스톨릭 어레이 양자화가 적용된 [`ExecutionProvider::Npu`](inference::ExecutionProvider::Npu)에서 고속 추론 가능.

pub mod autograd;
pub mod data;
pub mod device;
pub mod inference;
pub mod kernels;
pub mod nn;
pub mod optim;
pub mod parallel;
pub mod rng;
pub mod save;
pub mod tensor;

/// 편리한 일괄 임포트를 위한 프렐류드 모듈 (`use rustgears::prelude::*;`).
pub mod prelude {
    pub use crate::autograd::{no_grad, NoGradGuard};
    pub use crate::data::{DataLoader, Dataset, PrefetchIter, TensorDataset};
    pub use crate::device::{list_devices, Device};
    pub use crate::inference::{ExecutionProvider, InferenceSession, NpuPrecision, NpuProfile};
    pub use crate::nn::loss::{accuracy, bce_with_logits, cross_entropy, cross_entropy_smooth, huber_loss, l1_loss, mse_loss, nll_loss};
    pub use crate::nn::metrics::{confusion_matrix, precision_recall_f1, top_k_accuracy};
    pub use crate::nn::{
        AdaptiveAvgPool2d, BatchNorm1d, BatchNorm2d, Conv2d, ConvTranspose2d, Dropout, Embedding, Flatten, Gelu,
        Identity, KvCache, LayerNorm, LeakyRelu, Linear, MaxPool2d, Module, MultiHeadAttention, RMSNorm, Relu,
        RotaryEmbedding, Sequential, Sigmoid, Silu, Softmax, SwiGLU, Tanh, TransformerEncoderLayer,
    };
    pub use crate::optim::{
        clip_grad_norm, Adam, AdamW, CosineAnnealingLr, EarlyStopping, EarlyStoppingMode, ExponentialLr,
        GradientAccumulator, LrScheduler, ModelCheckpoint, ModelEMA, Optimizer, PlateauMode, ProgressBar,
        ReduceLROnPlateau, RmsProp, Sgd, StepLr, WarmupCosineLr,
    };
    pub use crate::parallel::{ring_all_reduce, DataParallel};
    pub use crate::rng::manual_seed;
    pub use crate::save::{
        load_inference_layers, load_safetensors, load_state_dict, read_safetensors, save_inference_model,
        save_safetensors, save_state_dict,
    };
    pub use crate::tensor::Tensor;
}
