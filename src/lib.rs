//! # RustGears
//!
//! A PyTorch / TensorFlow inspired deep-learning library written in pure Rust.
//!
//! ## Highlights
//! * **Reverse-mode Autograd** – dynamic computation graphs (`requires_grad_`, `backward()`, `no_grad`).
//! * **Device Selection (`CPU` / `GPU`)** – multi-threaded `rayon` CPU kernels and hardware GPU compute shaders via `wgpu` (Vulkan / DirectX 12 / Metal) with transparent virtual-GPU fallback.
//! * **Multi-GPU Data Parallelism** – [`DataParallel`](parallel::DataParallel) with scatter, concurrent replica forward/backward, and bandwidth-optimal [`ring_all_reduce`](parallel::ring_all_reduce).
//! * **Neural Network Layers & Optimizers** – `Linear`, `Conv2d`, `MaxPool2d`, `BatchNorm1d/2d`, `LayerNorm`, `Dropout`, `Embedding`, `MultiHeadAttention`, `TransformerEncoderLayer`, `Sequential`, `SGD`, `Adam`, `AdamW`, `RMSprop`, LR schedulers.
//! * **Modern Model Utilities** – Safetensors weight interchange, cached causal attention with RoPE, bounded batch prefetching, `AdaptiveAvgPool2d`, `ConvTranspose2d`, `ModelEMA` and terminal progress reporting.
//! * **NPU Inference Runtime** – export trained models into an [`InferenceSession`](inference::InferenceSession) and run them on [`ExecutionProvider::Npu`](inference::ExecutionProvider::Npu) with operator fusion (`Linear/Conv2d + BatchNorm + Activation`) and symmetric **INT8 / FP16** systolic-array quantisation.

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

/// Convenient glob import (`use rustgears::prelude::*;`).
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
