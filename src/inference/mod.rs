//! Portable inference format, graph optimisation, INT8 quantisation and the
//! NPU execution runtime.
//!
//! Trained [`Module`](crate::nn::Module)s can be exported into a lightweight
//! [`InferenceSession`] and executed on:
//! * [`ExecutionProvider::Npu`] – INT8/FP16 inference engine modelling an NPU
//!   systolic MAC array with operator fusion (`Linear/Conv2d + BatchNorm + Activation`),
//! * [`ExecutionProvider::Gpu`] – GPU compute backend,
//! * [`ExecutionProvider::Cpu`] – multi-threaded CPU backend.

pub mod npu;
pub mod quantize;

pub use npu::{NpuPrecision, NpuProfile, OpRecord};
pub use quantize::QuantTensor;

use crate::device::{Device, UnaryOp};
use crate::kernels;
use crate::nn::Module;
use crate::tensor::ops::softmax_raw;
use crate::tensor::Tensor;
use std::fmt;
use std::sync::Mutex;

/// Activation functions supported by the inference graph and NPU fusion unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    None,
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    LeakyRelu(f32),
}

impl Activation {
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Activation::None => x,
            Activation::Relu => UnaryOp::Relu.apply(x),
            Activation::Sigmoid => UnaryOp::Sigmoid.apply(x),
            Activation::Tanh => UnaryOp::Tanh.apply(x),
            Activation::Gelu => UnaryOp::Gelu.apply(x),
            Activation::LeakyRelu(s) => {
                if x > 0.0 {
                    x
                } else {
                    s * x
                }
            }
        }
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Activation::None => write!(f, "none"),
            Activation::Relu => write!(f, "ReLU"),
            Activation::Sigmoid => write!(f, "Sigmoid"),
            Activation::Tanh => write!(f, "Tanh"),
            Activation::Gelu => write!(f, "GELU"),
            Activation::LeakyRelu(s) => write!(f, "LeakyReLU({s})"),
        }
    }
}

/// Portable layer specification produced by [`Module::export`].
#[derive(Clone, Debug)]
pub enum LayerSpec {
    Linear {
        in_features: usize,
        out_features: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
    },
    Conv2d {
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
    },
    /// Exported form of BatchNorm: `y_c = scale_c * x_c + shift_c`.
    /// The graph optimiser folds this directly into the preceding Linear/Conv2d.
    ChannelAffine {
        scale: Vec<f32>,
        shift: Vec<f32>,
    },
    MaxPool2d {
        kernel: usize,
        stride: usize,
    },
    Flatten,
    Activation(Activation),
    Softmax,
}

/// Hardware execution provider for inference (ONNX Runtime style).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionProvider {
    /// CPU execution provider (FP32, rayon).
    Cpu,
    /// GPU execution provider (`gpu:index`, FP32).
    Gpu(usize),
    /// Neural Processing Unit (NPU) execution provider with operator fusion and
    /// INT8 / FP16 quantised MAC execution.
    Npu(NpuPrecision),
}

impl ExecutionProvider {
    /// Default NPU provider (INT8 symmetric quantisation).
    pub fn npu() -> Self {
        ExecutionProvider::Npu(NpuPrecision::Int8)
    }

    pub fn npu_int8() -> Self {
        ExecutionProvider::Npu(NpuPrecision::Int8)
    }

    pub fn npu_fp16() -> Self {
        ExecutionProvider::Npu(NpuPrecision::Fp16)
    }
}

impl fmt::Display for ExecutionProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutionProvider::Cpu => write!(f, "CPU (FP32)"),
            ExecutionProvider::Gpu(i) => write!(f, "GPU:{i} (FP32)"),
            ExecutionProvider::Npu(p) => write!(f, "NPU ({p})"),
        }
    }
}

enum CompiledProgram {
    Standard { device: Device, layers: Vec<LayerSpec> },
    Npu(npu::NpuProgram),
}

/// Compiled inference session ready to execute a trained model.
pub struct InferenceSession {
    provider: ExecutionProvider,
    raw_layers: Vec<LayerSpec>,
    program: CompiledProgram,
    last_profile: Mutex<Option<NpuProfile>>,
}

impl InferenceSession {
    /// Export `model` and compile it for `provider`.
    pub fn from_module(model: &dyn Module, provider: ExecutionProvider) -> Result<Self, String> {
        model.eval();
        let layers = model.export()?;
        Ok(Self::from_layers(layers, provider))
    }

    /// Build a session from raw [`LayerSpec`]s.
    pub fn from_layers(raw_layers: Vec<LayerSpec>, provider: ExecutionProvider) -> Self {
        let program = match provider {
            ExecutionProvider::Cpu => CompiledProgram::Standard {
                device: Device::Cpu,
                layers: fold_batch_norm(&raw_layers),
            },
            ExecutionProvider::Gpu(i) => CompiledProgram::Standard {
                device: Device::Gpu(i),
                layers: fold_batch_norm(&raw_layers),
            },
            ExecutionProvider::Npu(prec) => CompiledProgram::Npu(npu::compile(&raw_layers, prec)),
        };
        InferenceSession {
            provider,
            raw_layers,
            program,
            last_profile: Mutex::new(None),
        }
    }

    pub fn provider(&self) -> ExecutionProvider {
        self.provider
    }

    pub fn raw_layers(&self) -> &[LayerSpec] {
        &self.raw_layers
    }

    /// Profile from the most recent NPU [`run`](Self::run) call.
    pub fn last_npu_profile(&self) -> Option<NpuProfile> {
        self.last_profile.lock().unwrap().clone()
    }

    /// Summary of graph optimisations and memory compression applied by the session.
    pub fn compilation_report(&self) -> String {
        match &self.program {
            CompiledProgram::Standard { device, layers } => format!(
                "InferenceSession [provider = {}, device = {}, ops = {} (from {} raw layers)]",
                self.provider,
                device,
                layers.len(),
                self.raw_layers.len()
            ),
            CompiledProgram::Npu(prog) => prog.summary(self.raw_layers.len()),
        }
    }

    /// Execute the compiled model on `input` (no autograd graph is recorded).
    pub fn run(&self, input: &Tensor) -> Tensor {
        match &self.program {
            CompiledProgram::Standard { device, layers } => run_standard(input, *device, layers),
            CompiledProgram::Npu(prog) => {
                let (out, prof) = prog.execute(input);
                *self.last_profile.lock().unwrap() = Some(prof);
                out
            }
        }
    }
}

/// Fold `ChannelAffine` (BatchNorm) into a preceding `Linear` or `Conv2d`.
pub(crate) fn fold_batch_norm(layers: &[LayerSpec]) -> Vec<LayerSpec> {
    let mut out: Vec<LayerSpec> = Vec::with_capacity(layers.len());
    for l in layers {
        if let LayerSpec::ChannelAffine { scale, shift } = l {
            match out.last_mut() {
                Some(LayerSpec::Linear { in_features, out_features, weight, bias }) if *out_features == scale.len() => {
                    for r in 0..*in_features {
                        for c in 0..*out_features {
                            weight[r * *out_features + c] *= scale[c];
                        }
                    }
                    for c in 0..*out_features {
                        bias[c] = bias[c] * scale[c] + shift[c];
                    }
                    continue;
                }
                Some(LayerSpec::Conv2d { out_channels, in_channels, kernel, weight, bias, .. }) if *out_channels == scale.len() => {
                    let plane = *in_channels * *kernel * *kernel;
                    for o in 0..*out_channels {
                        for i in 0..plane {
                            weight[o * plane + i] *= scale[o];
                        }
                        bias[o] = bias[o] * scale[o] + shift[o];
                    }
                    continue;
                }
                _ => {}
            }
        }
        out.push(l.clone());
    }
    out
}

fn run_standard(input: &Tensor, device: Device, layers: &[LayerSpec]) -> Tensor {
    let be = device.backend();
    let mut data = input.to_vec();
    let mut shape = input.shape().to_vec();
    let mm = |a: &[f32], b: &[f32], m: usize, k: usize, n: usize| be.matmul(a, b, m, k, n);

    for l in layers {
        match l {
            LayerSpec::Linear { in_features, out_features, weight, bias } => {
                let rows = data.len() / *in_features;
                let mut y = be.matmul(&data, weight, rows, *in_features, *out_features);
                for r in 0..rows {
                    for c in 0..*out_features {
                        y[r * *out_features + c] += bias[c];
                    }
                }
                *shape.last_mut().unwrap() = *out_features;
                data = y;
            }
            LayerSpec::Conv2d { in_channels, out_channels, kernel, stride, padding, weight, bias } => {
                let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
                debug_assert_eq!(c, *in_channels);
                let (mut y, oh, ow) = kernels::conv2d_forward(&data, weight, n, c, h, w, *out_channels, *kernel, *stride, *padding, &mm);
                let spatial = oh * ow;
                for b in 0..n {
                    for (o, &bv) in bias.iter().enumerate().take(*out_channels) {
                        let base = (b * *out_channels + o) * spatial;
                        for v in &mut y[base..base + spatial] {
                            *v += bv;
                        }
                    }
                }
                shape = vec![n, *out_channels, oh, ow];
                data = y;
            }
            LayerSpec::ChannelAffine { scale, shift } => {
                let c = scale.len();
                let inner: usize = shape[2..].iter().product();
                for n in 0..shape[0] {
                    for ch in 0..c {
                        let base = (n * c + ch) * inner;
                        for v in &mut data[base..base + inner] {
                            *v = *v * scale[ch] + shift[ch];
                        }
                    }
                }
            }
            LayerSpec::MaxPool2d { kernel, stride } => {
                let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
                let oh = kernels::out_size(h, *kernel, *stride, 0);
                let ow = kernels::out_size(w, *kernel, *stride, 0);
                data = kernels::max_pool2d(&data, n, c, h, w, *kernel, *stride).0;
                shape = vec![n, c, oh, ow];
            }
            LayerSpec::Flatten => {
                let rest: usize = shape[1..].iter().product();
                shape = vec![shape[0], rest];
            }
            LayerSpec::Activation(act) => {
                for v in &mut data {
                    *v = act.apply(*v);
                }
            }
            LayerSpec::Softmax => {
                let last = *shape.last().unwrap();
                let outer = data.len() / last;
                data = softmax_raw(&data, outer, last, 1);
            }
        }
    }
    Tensor::from_vec(data, &shape).on(device)
}
