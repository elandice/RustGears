//! 핵심 신경망 레이어 모듈: Linear, Conv2d, ConvTranspose2d, 풀링, 활성화 함수, 드롭아웃, 임베딩, SwiGLU.

use super::{init, Module};
use crate::inference::{Activation, LayerSpec};
use crate::rng;
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};

// ----------------------------------------------------------------------
// 선형 계층 (Linear)
// ----------------------------------------------------------------------

/// 완전 연결 계층 (Fully Connected / Dense Layer): `y = x W + b` (`W: [in, out]`).
pub struct Linear {
    /// 가중치 행렬 (`[in_features, out_features]`)
    pub weight: Tensor,
    /// 편향 벡터 (`[out_features]`)
    pub bias: Option<Tensor>,
    /// 입력 특징 차원 수
    pub in_features: usize,
    /// 출력 특징 차원 수
    pub out_features: usize,
}

impl Linear {
    /// Kaiming 균등 초기화 가중치와 균등 초기화 편향을 가진 Linear 레이어를 생성합니다.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Linear {
            weight: init::kaiming_uniform(&[in_features, out_features], in_features).requires_grad_(true),
            bias: Some(init::bias_uniform(&[out_features], in_features).requires_grad_(true)),
            in_features,
            out_features,
        }
    }

    /// 편향(bias)이 없는 Linear 레이어를 생성합니다.
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
// 2차원 합성곱 (Conv2d)
// ----------------------------------------------------------------------

/// 4차원 이미지/특징 맵 입력(`[N, C, H, W]`)을 위한 2차원 공간 합성곱(Convolution) 레이어 (정사각형 커널).
pub struct Conv2d {
    /// 합성곱 커널 가중치 (`[out_channels, in_channels, k, k]`)
    pub weight: Tensor,
    /// 채널별 편향 (`[out_channels]`)
    pub bias: Tensor,
    /// 입력 채널 수
    pub in_channels: usize,
    /// 출력 채널 수
    pub out_channels: usize,
    /// 커널 크기 (한 변의 길이)
    pub kernel_size: usize,
    /// 스트라이드 (보폭)
    pub stride: usize,
    /// 패딩 크기
    pub padding: usize,
}

impl Conv2d {
    /// 기본 스트라이드 1, 패딩 0인 Conv2d 레이어를 생성합니다.
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

    /// 스트라이드 크기를 설정합니다 (빌더 패턴).
    pub fn stride(mut self, s: usize) -> Self {
        self.stride = s;
        self
    }

    /// 패딩 크기를 설정합니다 (빌더 패턴).
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
// 전치 합성곱 및 풀링/형상 변환
// ----------------------------------------------------------------------

/// 4차원 `[N, C, H, W]` 입력에 대한 2차원 전치 합성곱(Transposed Convolution / Deconvolution) 레이어.
/// 가중치는 PyTorch와 동일한 `[in_channels, out_channels, k, k]` 배치를 따릅니다.
pub struct ConvTranspose2d {
    pub weight: Tensor,
    pub bias: Tensor,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
}

impl ConvTranspose2d {
    /// 새 ConvTranspose2d 레이어를 생성합니다.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        assert!(in_channels > 0 && out_channels > 0 && kernel_size > 0, "ConvTranspose2d dimensions must be positive");
        let fan_in = out_channels * kernel_size * kernel_size;
        Self {
            weight: init::kaiming_uniform(&[in_channels, out_channels, kernel_size, kernel_size], fan_in).requires_grad_(true),
            bias: init::bias_uniform(&[out_channels], fan_in).requires_grad_(true),
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            output_padding: 0,
        }
    }

    /// 스트라이드를 설정합니다.
    pub fn stride(mut self, stride: usize) -> Self {
        assert!(stride > 0, "ConvTranspose2d stride must be positive");
        self.stride = stride;
        self
    }

    /// 패딩을 설정합니다.
    pub fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// 출력 패딩(output_padding)을 설정합니다.
    pub fn output_padding(mut self, output_padding: usize) -> Self {
        self.output_padding = output_padding;
        self
    }
}

impl Module for ConvTranspose2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.conv_transpose2d(&self.weight, self.stride, self.padding, self.output_padding)
            .add(&self.bias.reshape(&[1, self.out_channels, 1, 1]))
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".into(), self.weight.clone()), ("bias".into(), self.bias.clone())]
    }

    fn name(&self) -> String {
        format!("ConvTranspose2d({}, {}, k={}, s={}, p={}, op={})", self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.output_padding)
    }
}

/// 채널별로 고정된 공간 해상도 `(height, width)`로 풀링을 수행하는 적응형 2D 평균 풀링 레이어 (글로벌 `(1, 1)` 풀링 지원).
pub struct AdaptiveAvgPool2d {
    pub output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        assert!(output_size.0 > 0 && output_size.1 > 0, "AdaptiveAvgPool2d output size must be positive");
        Self { output_size }
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.adaptive_avg_pool2d(self.output_size)
    }

    fn name(&self) -> String {
        format!("AdaptiveAvgPool2d({:?})", self.output_size)
    }
}

/// 2차원 공간 최대 풀링(Max Pooling) 레이어.
pub struct MaxPool2d {
    pub kernel: usize,
    pub stride: usize,
}

impl MaxPool2d {
    /// 커널 크기와 동일한 스트라이드로 MaxPool2d 레이어를 생성합니다.
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

/// 배치 차원(0번 축)을 제외한 모든 차원을 평탄화(Flatten)하는 레이어.
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

/// 입력을 아무런 변형 없이 그대로 반환하는 항등(Identity) 레이어.
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
// 활성화 함수 (Activations)
// ----------------------------------------------------------------------

macro_rules! activation {
    ($name:ident, $method:ident, $act:expr, $doc:expr) => {
        #[doc = $doc]
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

activation!(Relu, relu, Activation::Relu, "ReLU 활성화 함수 레이어 (`max(0, x)`).");
activation!(Sigmoid, sigmoid, Activation::Sigmoid, "시그모이드 활성화 함수 레이어 (`1 / (1 + e^-x)`).");
activation!(Tanh, tanh, Activation::Tanh, "쌍곡탄젠트 활성화 함수 레이어 (`tanh(x)`).");
activation!(Gelu, gelu, Activation::Gelu, "GELU 활성화 함수 레이어.");

/// SiLU (Sigmoid Linear Unit, Swish라고도 부름): `y = x * sigmoid(x)`.
pub struct Silu;

impl Module for Silu {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.mul(&x.sigmoid())
    }
    fn name(&self) -> String {
        "SiLU".into()
    }
}

/// 음수 입력에 대해 지정된 기울기(slope)를 적용하는 Leaky ReLU 활성화 함수 레이어.
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

/// 마지막 차원에 대해 확률 분포를 계산하는 소프트맥스(Softmax) 레이어.
pub struct Softmax;

impl Module for Softmax {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.softmax(-1)
    }
    fn name(&self) -> String {
        "Softmax".into()
    }
    fn export(&self) -> Result<Vec<LayerSpec>, String> {
        Ok(vec![LayerSpec::Softmax])
    }
}

// ----------------------------------------------------------------------
// 드롭아웃 (Dropout)
// ----------------------------------------------------------------------

/// 역 드롭아웃(Inverted Dropout): 학습 도중 뉴런 활성화를 `p`의 확률로 무작위 0으로 만들고 나머지 뉴런을 `1/(1-p)`로 스케일링합니다.
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
        Ok(vec![]) // 추론 시에는 아무런 연산도 하지 않음(Identity)
    }
}

// ----------------------------------------------------------------------
// 임베딩 (Embedding)
// ----------------------------------------------------------------------

/// 정수 토큰 ID를 밀집 벡터(Dense vector)로 매핑하는 임베딩 룩업 테이블.
pub struct Embedding {
    /// 임베딩 가중치 텐서 (`[num_embeddings, dim]`)
    pub weight: Tensor,
    /// 어휘 사전 크기 (고유 토큰 개수)
    pub num_embeddings: usize,
    /// 임베딩 벡터 차원 수
    pub dim: usize,
}

impl Embedding {
    pub fn new(num_embeddings: usize, dim: usize) -> Self {
        Embedding { weight: init::normal(&[num_embeddings, dim], 0.1).requires_grad_(true), num_embeddings, dim }
    }
}

impl Module for Embedding {
    /// `ids`: 임의 형태의 정수형 텐서 -> 결과 형태: `ids.shape + [dim]`.
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
// SwiGLU 피드포워드 신경망
// ----------------------------------------------------------------------

/// LLaMA, PaLM 등 최신 대형 언어 모델(LLM)에 널리 사용되는 SwiGLU 피드포워드 신경망 블록:
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
    /// 지정된 입력, 은닉, 출력 차원으로 새 SwiGLU 블록을 생성합니다.
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
