//! Neural network building blocks (like `torch.nn`).

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

/// Base trait for every layer / model (like `torch.nn.Module`).
///
/// Modules are `Send + Sync` so that replicas can be trained concurrently on
/// different devices by [`DataParallel`](crate::parallel::DataParallel).
pub trait Module: Send + Sync {
    /// Forward pass.
    fn forward(&self, x: &Tensor) -> Tensor;

    /// Trainable parameters with hierarchical names (`"0.weight"`, ...).
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }

    /// Non-trainable state (e.g. BatchNorm running statistics).
    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }

    /// Display name of the layer.
    fn name(&self) -> String {
        "Module".into()
    }

    /// Switch between training and evaluation behaviour (Dropout, BatchNorm).
    fn set_training(&self, _training: bool) {}

    /// Export the layer to the portable inference format (for CPU/GPU/NPU runtimes).
    fn export(&self) -> Result<Vec<LayerSpec>, String> {
        Err(format!("layer '{}' does not support export", self.name()))
    }

    // ---------------- provided methods ----------------

    fn parameters(&self) -> Vec<Tensor> {
        self.named_parameters().into_iter().map(|(_, t)| t).collect()
    }

    fn train(&self) {
        self.set_training(true)
    }

    fn eval(&self) {
        self.set_training(false)
    }

    /// Move all parameters and buffers to `device` (in place).
    fn to(&self, device: Device) {
        for p in self.parameters() {
            p.set_device(device);
        }
        for (_, b) in self.named_buffers() {
            b.set_device(device);
        }
    }

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }

    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    /// Parameters + buffers (everything needed to restore the model).
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

/// Prefix the names of a sub-module's tensors.
pub fn prefixed(prefix: &str, v: Vec<(String, Tensor)>) -> Vec<(String, Tensor)> {
    v.into_iter().map(|(n, t)| (format!("{prefix}.{n}"), t)).collect()
}

/// A chain of modules applied in order (like `nn.Sequential`).
#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    /// Builder-style append.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, m: M) -> Self {
        self.layers.push(Box::new(m));
        self
    }

    pub fn push<M: Module + 'static>(&mut self, m: M) {
        self.layers.push(Box::new(m));
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn layers(&self) -> &[Box<dyn Module>] {
        &self.layers
    }

    /// Human readable model summary (layers and parameter counts).
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
