//! Core layers: Linear, Conv2d, pooling, activations, dropout, embedding.

use super::{init, Module};
use crate::inference::{Activation, LayerSpec};
use crate::rng;
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};

// ----------------------------------------------------------------------
// Linear
// ----------------------------------------------------------------------

/// Fully connected layer `y = x W + b` (`W: [in, out]`).
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Linear {
            weight: init::kaiming_uniform(&[in_features, out_features], in_features).requires_grad_(true),
            bias: Some(init::bias_uniform(&[out_features], in_features).requires_grad_(true)),
            in_features,
            out_features,
        }
    }

    pub fn no_bias(in_features: usize, out_features: usize) -> Self {
        let mut l = Self::new(in_features, out_features);
        l.bias = None;
        l
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape().to_vec();
        assert_eq!(*shape.last().unwrap(), self.in_features, "Linear: expected last dim {}, got {:?}", self.in_features, shape);
        let flat = if shape.len() == 2 { x.clone() } else { x.reshape(&[x.numel() / self.in_features, self.in_features]) };
        let mut y = flat.matmul(&self.weight);
        if let Some(b) = &self.bias {
            y = y.add(b);
        }
        if shape.len() == 2 {
            y
        } else {
            let mut out = shape[..shape.len() - 1].to_vec();
            out.push(self.out_features);
            y.reshape(&out)
        }
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = vec![("weight".to_string(), self.weight.clone())];
        if let Some(b) = &self.bias {
            v.push(("bias".into(), b.clone()));
        }
        v
    }

    fn name(&self) -> String {
        format!("Linear({}, {})", self.in_features, self.out_features)
    }

    fn export(&self) -> Result<Vec<LayerSpec>, String> {
        Ok(vec![LayerSpec::Linear {
            in_features: self.in_features,
            out_features: self.out_features,
            weight: self.weight.to_vec(),
            bias: self.bias.as_ref().map(|b| b.to_vec()).unwrap_or_else(|| vec![0.0; self.out_features]),
        }])
    }
}

// ----------------------------------------------------------------------
// Conv2d
// ----------------------------------------------------------------------

/// 2-D convolution over `[N, C, H, W]` inputs (square kernels).
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Tensor,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Conv2d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let fan_in = in_channels * kernel_size * kernel_size;
        Conv2d {
            weight: init::kaiming_uniform(&[out_channels, in_channels, kernel_size, kernel_size], fan_in).requires_grad_(true),
            bias: init::bias_uniform(&[out_channels], fan_in).requires_grad_(true),
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
        }
    }

    pub fn stride(mut self, s: usize) -> Self {
        self.stride = s;
        self
    }

    pub fn padding(mut self, p: usize) -> Self {
        self.padding = p;
        self
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let y = x.conv2d(&self.weight, self.stride, self.padding);
        y.add(&self.bias.reshape(&[1, self.out_channels, 1, 1]))
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".into(), self.weight.clone()), ("bias".into(), self.bias.clone())]
    }

    fn name(&self) -> String {
        format!(
            "Conv2d({}, {}, k={}, s={}, p={})",
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding
        )
    }

    fn export(&self) -> Result<Vec<LayerSpec>, String> {
        Ok(vec![LayerSpec::Conv2d {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            weight: self.weight.to_vec(),
            bias: self.bias.to_vec(),
        }])
    }
}

// ----------------------------------------------------------------------
// Pooling / reshaping
// ----------------------------------------------------------------------

pub struct MaxPool2d {
    pub kernel: usize,
    pub stride: usize,
}

impl MaxPool2d {
    pub fn new(kernel: usize) -> Self {
        MaxPool2d { kernel, stride: kernel }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.max_pool2d(self.kernel, self.stride)
    }
    fn name(&self) -> String {
        format!("MaxPool2d(k={}, s={})", self.kernel, self.stride)
    }
    fn export(&self) -> Result<Vec<LayerSpec>, String> {
        Ok(vec![LayerSpec::MaxPool2d { kernel: self.kernel, stride: self.stride }])
    }
}

/// Flatten everything but the batch dimension.
pub struct Flatten;

impl Module for Flatten {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.flatten(1)
    }
    fn name(&self) -> String {
        "Flatten".into()
    }
    fn export(&self) -> Result<Vec<LayerSpec>, String> {
        Ok(vec![LayerSpec::Flatten])
    }
}

pub struct Identity;

impl Module for Identity {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.clone()
    }
    fn name(&self) -> String {
        "Identity".into()
    }
    fn export(&self) -> Result<Vec<LayerSpec>, String> {
        Ok(vec![])
    }
}

// ----------------------------------------------------------------------
// Activations
// ----------------------------------------------------------------------

macro_rules! activation {
    ($name:ident, $method:ident, $act:expr) => {
        pub struct $name;
        impl Module for $name {
            fn forward(&self, x: &Tensor) -> Tensor {
                x.$method()
            }
            fn name(&self) -> String {
                stringify!($name).into()
            }
            fn export(&self) -> Result<Vec<LayerSpec>, String> {
                Ok(vec![LayerSpec::Activation($act)])
            }
        }
    };
}

activation!(Relu, relu, Activation::Relu);
activation!(Sigmoid, sigmoid, Activation::Sigmoid);
activation!(Tanh, tanh, Activation::Tanh);
activation!(Gelu, gelu, Activation::Gelu);

/// SiLU (Sigmoid Linear Unit, also known as Swish): `y = x * sigmoid(x)`.
pub struct Silu;

impl Module for Silu {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.mul(&x.sigmoid())
    }
    fn name(&self) -> String {
        "SiLU".into()
    }
}

pub struct LeakyRelu(pub f32);

impl Module for LeakyRelu {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.leaky_relu(self.0)
    }
    fn name(&self) -> String {
        format!("LeakyReLU({})", self.0)
    }
    fn export(&self) -> Result<Vec<LayerSpec>, String> {
        Ok(vec![LayerSpec::Activation(Activation::LeakyRelu(self.0))])
    }
}

/// Softmax over the last dimension.
pub struct Softmax;

impl Module for Softmax {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.softmax(-1)
    }
    fn name(&self) -> String {
        "Softmax(dim=-1)".into()
    }
    fn export(&self) -> Result<Vec<LayerSpec>, String> {
        Ok(vec![LayerSpec::Softmax])
    }
}

// ----------------------------------------------------------------------
// Dropout
// ----------------------------------------------------------------------

/// Inverted dropout: zeroes activations with probability `p` during training.
pub struct Dropout {
    pub p: f32,
    training: AtomicBool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!((0.0..1.0).contains(&p), "dropout probability must be in [0, 1)");
        Dropout { p, training: AtomicBool::new(true) }
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Tensor {
        if !self.training.load(Ordering::Relaxed) || self.p == 0.0 {
            return x.clone();
        }
        let keep = 1.0 - self.p;
        let mask = rng::fill_with(x.numel(), |r| if r.uniform() < keep { 1.0 / keep } else { 0.0 });
        x.mul(&Tensor::from_vec(mask, x.shape()).on(x.device()))
    }
    fn name(&self) -> String {
        format!("Dropout(p={})", self.p)
    }
    fn set_training(&self, t: bool) {
        self.training.store(t, Ordering::Relaxed);
    }
    fn export(&self) -> Result<Vec<LayerSpec>, String> {
        Ok(vec![]) // identity at inference time
    }
}

// ----------------------------------------------------------------------
// Embedding
// ----------------------------------------------------------------------

/// Lookup table mapping integer ids (stored as f32) to dense vectors.
pub struct Embedding {
    pub weight: Tensor,
    pub num_embeddings: usize,
    pub dim: usize,
}

impl Embedding {
    pub fn new(num_embeddings: usize, dim: usize) -> Self {
        Embedding { weight: init::normal(&[num_embeddings, dim], 0.1).requires_grad_(true), num_embeddings, dim }
    }
}

impl Module for Embedding {
    /// `ids`: any shape of integer-valued floats -> `ids.shape + [dim]`.
    fn forward(&self, ids: &Tensor) -> Tensor {
        let idx: Vec<usize> = ids.data().iter().map(|&v| v as usize).collect();
        let (v, d) = (self.num_embeddings, self.dim);
        let w = self.weight.data();
        let mut out = Vec::with_capacity(idx.len() * d);
        for &i in &idx {
            assert!(i < v, "Embedding: index {i} out of range ({v})");
            out.extend_from_slice(&w[i * d..(i + 1) * d]);
        }
        drop(w);
        let mut shape = ids.shape().to_vec();
        shape.push(d);
        Tensor::from_op(
            out,
            shape,
            self.weight.device(),
            vec![self.weight.clone()],
            "EmbeddingBackward",
            Box::new(move |g| {
                let mut gw = vec![0.0; v * d];
                for (n, &i) in idx.iter().enumerate() {
                    for j in 0..d {
                        gw[i * d + j] += g[n * d + j];
                    }
                }
                vec![Some(gw)]
            }),
        )
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".into(), self.weight.clone())]
    }

    fn name(&self) -> String {
        format!("Embedding({}, {})", self.num_embeddings, self.dim)
    }
}

// ----------------------------------------------------------------------
// SwiGLU Feed-Forward Network
// ----------------------------------------------------------------------

/// SwiGLU Feed-Forward Network block widely adopted in modern Transformer architectures (e.g. LLaMA, PaLM):
/// `SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down`.
pub struct SwiGLU {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    pub in_features: usize,
    pub hidden_features: usize,
    pub out_features: usize,
}

impl SwiGLU {
    pub fn new(in_features: usize, hidden_features: usize, out_features: usize) -> Self {
        SwiGLU {
            gate: Linear::no_bias(in_features, hidden_features),
            up: Linear::no_bias(in_features, hidden_features),
            down: Linear::no_bias(hidden_features, out_features),
            in_features,
            hidden_features,
            out_features,
        }
    }
}

impl Module for SwiGLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        let g = self.gate.forward(x);
        let u = self.up.forward(x);
        let silu_g = g.mul(&g.sigmoid());
        let intermediate = silu_g.mul(&u);
        self.down.forward(&intermediate)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut v = super::prefixed("gate", self.gate.named_parameters());
        v.extend(super::prefixed("up", self.up.named_parameters()));
        v.extend(super::prefixed("down", self.down.named_parameters()));
        v
    }

    fn name(&self) -> String {
        format!("SwiGLU(in={}, hidden={}, out={})", self.in_features, self.hidden_features, self.out_features)
    }
}
