//! 이식 가능한 추론 모델 포맷, 연산 그래프 최적화, INT8 양자화 및 NPU 실행 런타임.
//!
//! 학습된 [`Module`](crate::nn::Module)을 경량화된 [`InferenceSession`]으로 내보내어
//! 다양한 하드웨어 백엔드에서 고속으로 실행할 수 있습니다:
//! * [`ExecutionProvider::Npu`] – 연산자 융합(`Linear/Conv2d + BatchNorm + Activation`)과
//!   INT8/FP16 시스톨릭 MAC 어레이를 시뮬레이션하는 NPU 추론 엔진,
//! * [`ExecutionProvider::Gpu`] – GPU(WGPU) 가속 연산 백엔드,
//! * [`ExecutionProvider::Cpu`] – Rayon 기반 멀티스레드 CPU 백엔드.

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

/// 추론 그래프 및 NPU 융합 유닛에서 지원하는 활성화 함수 열거형.
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
    /// 스칼라 값 `x`에 활성화 함수를 적용합니다.
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

/// [`Module::export`]를 통해 추출되는 이식 가능한 레이어 명세(IR).
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
    /// BatchNorm의 내보내기 형태: `y_c = scale_c * x_c + shift_c`.
    /// 그래프 최적화 단계에서 선행 Linear/Conv2d로 직접 폴딩(융합)됩니다.
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

/// 추론 실행 하드웨어 공급자 (ONNX Runtime 스타일의 Execution Provider).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionProvider {
    /// CPU 실행 공급자 (FP32, Rayon 멀티스레딩).
    Cpu,
    /// GPU 실행 공급자 (`gpu:index`, FP32).
    Gpu(usize),
    /// 연산자 융합 및 INT8 / FP16 양자화 MAC 연산을 지원하는 신경망 처리 장치(NPU) 실행 공급자.
    Npu(NpuPrecision),
}

impl ExecutionProvider {
    /// 기본 NPU 실행 공급자 생성 (INT8 대칭 양자화).
    pub fn npu() -> Self {
        ExecutionProvider::Npu(NpuPrecision::Int8)
    }

    /// INT8 양자화 NPU 공급자 생성.
    pub fn npu_int8() -> Self {
        ExecutionProvider::Npu(NpuPrecision::Int8)
    }

    /// FP16 반정밀도 NPU 공급자 생성.
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

/// 학습된 모델을 컴파일하여 최적화된 상태로 추론을 수행하는 세션.
pub struct InferenceSession {
    provider: ExecutionProvider,
    raw_layers: Vec<LayerSpec>,
    program: CompiledProgram,
    last_profile: Mutex<Option<NpuProfile>>,
}

impl InferenceSession {
    /// `model`의 가중치를 추출하고 지정된 `provider`에 맞게 컴파일합니다.
    pub fn from_module(model: &dyn Module, provider: ExecutionProvider) -> Result<Self, String> {
        model.eval();
        let layers = model.export()?;
        Ok(Self::from_layers(layers, provider))
    }

    /// 원시 [`LayerSpec`] 목록으로부터 추론 세션을 빌드합니다.
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

    /// 설정된 실행 공급자를 반환합니다.
    pub fn provider(&self) -> ExecutionProvider {
        self.provider
    }

    /// 컴파일 전 원본 레이어 목록을 반환합니다.
    pub fn raw_layers(&self) -> &[LayerSpec] {
        &self.raw_layers
    }

    /// 가장 최근 NPU [`run`](Self::run) 호출 시 기록된 프로파일 정보를 반환합니다.
    pub fn last_npu_profile(&self) -> Option<NpuProfile> {
        self.last_profile.lock().unwrap().clone()
    }

    /// 세션에 적용된 그래프 최적화 및 메모리 압축 결과 요약 보고서를 반환합니다.
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

    /// 주어진 `input` 텐서에 대해 컴파일된 모델 추론을 실행합니다 (Autograd 그래프는 기록되지 않음).
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

/// `ChannelAffine` (BatchNorm)을 선행하는 `Linear` 또는 `Conv2d`에 흡수(Fold)하는 그래프 최적화.
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

/// CPU 또는 GPU 표준 디바이스 백엔드를 사용한 추론 실행 함수.
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
