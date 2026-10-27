//! 신경망 처리 장치(NPU) 컴파일러 및 실행 엔진.
//!
//! 실제 NPU(Apple Neural Engine, Qualcomm Hexagon, Intel AI Boost, AMD XDNA 등)는
//! 다음과 같은 세 가지 핵심 메커니즘을 통해 높은 처리량과 에너지 효율을 달성하며, 이를 모델링합니다:
//!
//! 1. **연산자 융합(Operator Fusion)** – `Linear/Conv2d + BatchNorm + Activation`을 단일 NPU
//!    매크로 명령어로 결합하여 중간 텐서가 온칩 SRAM 외부로 나가지 않도록 방지합니다.
//! 2. **저정밀도 MAC 어레이(Low-precision MAC arrays)** – 가중치와 활성화를 대칭 **INT8**
//!    (**INT32** 누적기) 또는 **FP16**으로 양자화하여 가중치 전송 대역폭을 최대 **4배** 절감합니다.
//! 3. **타일/시스톨릭 실행 프로파일링** – 모든 추론 실행 시 MAC 연산 수, 융합 연산 수,
//!    가중치 압축률 및 레이어별 지연 시간을 정밀하게 측정 및 기록합니다.

use super::quantize::{int8_gemm_fused, int8_gemm_fused_serial, round_fp16, QuantTensor};
use super::{fold_batch_norm, Activation, LayerSpec};
use crate::kernels;
use crate::tensor::ops::softmax_raw;
use crate::tensor::shape::{transpose2d, transpose2d_into};
use crate::tensor::Tensor;
use rayon::prelude::*;
use std::fmt;
use std::time::{Duration, Instant};

/// NPU MAC 어레이에서 사용하는 수치 정밀도.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NpuPrecision {
    /// 32비트 정수 누적기를 갖춘 8비트 정수 가중치 및 활성화 경로.
    Int8,
    /// 16비트 반정밀도 부동소수점 텐서 코어 경로.
    Fp16,
}

impl fmt::Display for NpuPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NpuPrecision::Int8 => write!(f, "INT8"),
            NpuPrecision::Fp16 => write!(f, "FP16"),
        }
    }
}

/// NPU 런타임에서 실행되는 융합 명령어 열거형.
#[derive(Clone, Debug)]
pub enum NpuOp {
    /// 융합된 선형 변환 (Linear + Activation).
    FusedLinear {
        in_features: usize,
        out_features: usize,
        /// 시스톨릭 내적 연산을 위해 전치된 `[out, in]` INT8 가중치.
        weight_q_t: QuantTensor,
        /// 전치된 `[out, in]` FP16 반올림 가중치.
        weight_fp16_t: Vec<f32>,
        bias: Vec<f32>,
        act: Activation,
    },
    /// 융합된 2D 합성곱 (Conv2d + Activation).
    FusedConv2d {
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        /// `[out, in*k*k]` 형태의 INT8 가중치.
        weight_q: QuantTensor,
        weight_fp16: Vec<f32>,
        bias: Vec<f32>,
        act: Activation,
    },
    /// 2D 맥스 풀링.
    MaxPool2d {
        kernel: usize,
        stride: usize,
    },
    /// 다차원 텐서 평탄화.
    Flatten,
    /// 독립 활성화 함수.
    Activation(Activation),
    /// 소프트맥스 정규화.
    Softmax,
}

impl NpuOp {
    /// 연산자의 식별 레이블 문자열을 반환합니다.
    pub fn label(&self) -> String {
        match self {
            NpuOp::FusedLinear { in_features, out_features, act, .. } => {
                if *act == Activation::None {
                    format!("NPU::Gemm({in_features}->{out_features})")
                } else {
                    format!("NPU::FusedGemmAct({in_features}->{out_features}, {act})")
                }
            }
            NpuOp::FusedConv2d { in_channels, out_channels, kernel, act, .. } => {
                if *act == Activation::None {
                    format!("NPU::Conv2d({in_channels}->{out_channels}, k={kernel})")
                } else {
                    format!("NPU::FusedConv2dAct({in_channels}->{out_channels}, k={kernel}, {act})")
                }
            }
            NpuOp::MaxPool2d { kernel, .. } => format!("NPU::MaxPool2d(k={kernel})"),
            NpuOp::Flatten => "NPU::Flatten".into(),
            NpuOp::Activation(a) => format!("NPU::Act({a})"),
            NpuOp::Softmax => "NPU::Softmax".into(),
        }
    }
}

/// NPU 명령어 1개의 실행 텔레메트리 기록.
#[derive(Clone, Debug)]
pub struct OpRecord {
    /// 연산자 레이블.
    pub op: String,
    /// 수행된 누적 곱셈(MAC) 연산 횟수.
    pub macs: u64,
    /// 실행 소요 시간.
    pub duration: Duration,
}

/// NPU 추론 실행 후 수집된 하드웨어 텔레메트리 프로파일.
#[derive(Clone, Debug)]
pub struct NpuProfile {
    /// 실행 정밀도 (INT8 또는 FP16).
    pub precision: NpuPrecision,
    /// 융합 전 원본 레이어 수.
    pub original_layers: usize,
    /// 융합 후 생성된 NPU 연산자 수.
    pub fused_ops: usize,
    /// FP32 원본 가중치 크기 (바이트).
    pub fp32_weight_bytes: usize,
    /// NPU 양자화 후 가중치 크기 (바이트).
    pub npu_weight_bytes: usize,
    /// 총 MAC 연산 수.
    pub total_macs: u64,
    /// 총 추론 소요 시간.
    pub total_time: Duration,
    /// 레이어별 세부 실행 기록.
    pub records: Vec<OpRecord>,
}

impl NpuProfile {
    /// 가중치 압축 비율 (`fp32_bytes / npu_bytes`).
    pub fn compression_ratio(&self) -> f32 {
        if self.npu_weight_bytes == 0 {
            1.0
        } else {
            self.fp32_weight_bytes as f32 / self.npu_weight_bytes as f32
        }
    }

    /// 실효 처리 성능 (GOPS, `2 * MACs / time`).
    pub fn gops(&self) -> f64 {
        let secs = self.total_time.as_secs_f64().max(1e-9);
        (self.total_macs as f64 * 2.0) / secs / 1e9
    }
}

impl fmt::Display for NpuProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "NPU Profile [{} | layers: {} -> {} fused | weights: {} B -> {} B ({:.2}x) | MACs: {} | time: {:?} ({:.2} GOPS)]",
            self.precision,
            self.original_layers,
            self.fused_ops,
            self.fp32_weight_bytes,
            self.npu_weight_bytes,
            self.compression_ratio(),
            self.total_macs,
            self.total_time,
            self.gops()
        )?;
        for r in &self.records {
            writeln!(f, "  - {:<42} {:>10} MACs   {:?}", r.op, r.macs, r.duration)?;
        }
        Ok(())
    }
}

/// NPU 컴파일 완료된 실행 프로그램.
pub struct NpuProgram {
    pub precision: NpuPrecision,
    pub original_layers: usize,
    pub ops: Vec<NpuOp>,
    pub fp32_bytes: usize,
    pub npu_bytes: usize,
}

/// 일련의 [`LayerSpec`] 레이어 명세를 융합 및 양자화된 NPU 연산 프로그램으로 컴파일합니다.
pub fn compile(raw: &[LayerSpec], precision: NpuPrecision) -> NpuProgram {
    // 1단계 최적화: BatchNorm을 선행 Linear / Conv2d에 폴딩(흡수)
    let folded = fold_batch_norm(raw);

    // 2단계 최적화: Linear/Conv2d + Activation 융합 및 가중치 양자화
    let mut ops: Vec<NpuOp> = Vec::new();
    let mut fp32_bytes = 0usize;
    let mut npu_bytes = 0usize;
    let mut i = 0;
    while i < folded.len() {
        let next_act = match folded.get(i + 1) {
            Some(LayerSpec::Activation(a)) => Some(*a),
            _ => None,
        };
        match &folded[i] {
            LayerSpec::Linear { in_features, out_features, weight, bias } => {
                let act = next_act.unwrap_or(Activation::None);
                if next_act.is_some() {
                    i += 1;
                }
                let w_t = transpose2d(weight, *in_features, *out_features);
                let w_q = QuantTensor::from_f32(&w_t, &[*out_features, *in_features]);
                let w_fp16: Vec<f32> = w_t.iter().map(|&v| round_fp16(v)).collect();
                fp32_bytes += (weight.len() + bias.len()) * 4;
                npu_bytes += match precision {
                    NpuPrecision::Int8 => w_q.byte_size() + bias.len() * 4,
                    NpuPrecision::Fp16 => weight.len() * 2 + bias.len() * 2,
                };
                ops.push(NpuOp::FusedLinear {
                    in_features: *in_features,
                    out_features: *out_features,
                    weight_q_t: w_q,
                    weight_fp16_t: w_fp16,
                    bias: bias.clone(),
                    act,
                });
            }
            LayerSpec::Conv2d { in_channels, out_channels, kernel, stride, padding, weight, bias } => {
                let act = next_act.unwrap_or(Activation::None);
                if next_act.is_some() {
                    i += 1;
                }
                let ckk = *in_channels * *kernel * *kernel;
                let w_q = QuantTensor::from_f32(weight, &[*out_channels, ckk]);
                let w_fp16: Vec<f32> = weight.iter().map(|&v| round_fp16(v)).collect();
                fp32_bytes += (weight.len() + bias.len()) * 4;
                npu_bytes += match precision {
                    NpuPrecision::Int8 => w_q.byte_size() + bias.len() * 4,
                    NpuPrecision::Fp16 => weight.len() * 2 + bias.len() * 2,
                };
                ops.push(NpuOp::FusedConv2d {
                    in_channels: *in_channels,
                    out_channels: *out_channels,
                    kernel: *kernel,
                    stride: *stride,
                    padding: *padding,
                    weight_q: w_q,
                    weight_fp16: w_fp16,
                    bias: bias.clone(),
                    act,
                });
            }
            LayerSpec::MaxPool2d { kernel, stride } => ops.push(NpuOp::MaxPool2d { kernel: *kernel, stride: *stride }),
            LayerSpec::Flatten => ops.push(NpuOp::Flatten),
            LayerSpec::Activation(a) => ops.push(NpuOp::Activation(*a)),
            LayerSpec::Softmax => ops.push(NpuOp::Softmax),
            LayerSpec::ChannelAffine { scale, shift } => {
                // 선행 Linear/Conv가 없는 독립 Affine인 경우: 대각 Linear로 표현
                let c = scale.len();
                let mut w = vec![0.0; c * c];
                for j in 0..c {
                    w[j * c + j] = scale[j];
                }
                let w_q = QuantTensor::from_f32(&w, &[c, c]);
                ops.push(NpuOp::FusedLinear {
                    in_features: c,
                    out_features: c,
                    weight_q_t: w_q,
                    weight_fp16_t: w,
                    bias: shift.clone(),
                    act: Activation::None,
                });
            }
        }
        i += 1;
    }

    NpuProgram { precision, original_layers: raw.len(), ops, fp32_bytes, npu_bytes }
}

impl NpuProgram {
    /// NPU 세션 파이프라인 및 압축 요약 문자열을 반환합니다.
    pub fn summary(&self, raw_count: usize) -> String {
        let ratio = if self.npu_bytes == 0 { 1.0 } else { self.fp32_bytes as f32 / self.npu_bytes as f32 };
        let names: Vec<String> = self.ops.iter().map(|o| o.label()).collect();
        format!(
            "InferenceSession [provider = NPU ({}), layers: {} -> {} fused ops, weights: {} B -> {} B ({:.2}x compression)]\n  Pipeline: {}",
            self.precision,
            raw_count,
            self.ops.len(),
            self.fp32_bytes,
            self.npu_bytes,
            ratio,
            names.join(" -> ")
        )
    }

    /// NPU 파이프라인을 실행하고 (결과 텐서, 실행 프로파일) 튜플을 반환합니다.
    pub fn execute(&self, input: &Tensor) -> (Tensor, NpuProfile) {
        let t_total = Instant::now();
        let mut data = input.to_vec();
        let mut shape = input.shape().to_vec();
        let mut records = Vec::with_capacity(self.ops.len());
        let mut total_macs = 0u64;

        for op in &self.ops {
            let t0 = Instant::now();
            let mut macs = 0u64;
            match op {
                NpuOp::FusedLinear { in_features, out_features, weight_q_t, weight_fp16_t, bias, act } => {
                    let rows = data.len() / *in_features;
                    macs = (rows * *in_features * *out_features) as u64;
                    data = match self.precision {
                        NpuPrecision::Int8 => {
                            let aq = QuantTensor::from_f32(&data, &[rows, *in_features]);
                            int8_gemm_fused(&aq, weight_q_t, rows, *in_features, *out_features, bias, *act)
                        }
                        NpuPrecision::Fp16 => fp16_gemm_fused(&data, weight_fp16_t, rows, *in_features, *out_features, bias, *act),
                    };
                    *shape.last_mut().unwrap() = *out_features;
                }
                NpuOp::FusedConv2d { in_channels, out_channels, kernel, stride, padding, weight_q, weight_fp16, bias, act } => {
                    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
                    debug_assert_eq!(c, *in_channels);
                    let oh = kernels::out_size(h, *kernel, *stride, *padding);
                    let ow = kernels::out_size(w, *kernel, *stride, *padding);
                    let l = oh * ow;
                    let ckk = c * *kernel * *kernel;
                    let img_stride = c * h * w;
                    let out_stride = *out_channels * l;
                    macs = (n * *out_channels * ckk * l) as u64;
                    let mut out = vec![0.0f32; n * out_stride];

                    if n >= 2 && out_stride > 0 {
                        let prec = self.precision;
                        out.par_chunks_mut(out_stride)
                            .enumerate()
                            .for_each_init(
                                || {
                                    (
                                        vec![0.0f32; ckk * l],
                                        vec![0.0f32; l * ckk],
                                        vec![0i8; l * ckk],
                                        vec![0.0f32; l * *out_channels],
                                    )
                                },
                                |(cols, cols_t, aq_buf, y_lo), (b, out_b)| {
                                    let img = &data[b * img_stride..(b + 1) * img_stride];
                                    kernels::im2col_into(img, c, h, w, *kernel, *stride, *padding, oh, ow, cols);
                                    transpose2d_into(cols, ckk, l, cols_t);
                                    match prec {
                                        NpuPrecision::Int8 => {
                                             let a_scale = QuantTensor::quantize_into(cols_t, aq_buf);
                                            int8_gemm_fused_serial(aq_buf, a_scale, weight_q, l, ckk, *out_channels, bias, *act, y_lo);
                                        }
                                        NpuPrecision::Fp16 => {
                                            fp16_gemm_fused_into(cols_t, weight_fp16, l, ckk, *out_channels, bias, *act, y_lo);
                                        }
                                    }
                                    transpose2d_into(y_lo, l, *out_channels, out_b);
                                },
                            );
                    } else {
                        let mut cols = vec![0.0f32; ckk * l];
                        let mut cols_t = vec![0.0f32; l * ckk];
                        let mut aq_buf = vec![0i8; l * ckk];
                        let mut y_lo = vec![0.0f32; l * *out_channels];
                        for b in 0..n {
                            let img = &data[b * img_stride..(b + 1) * img_stride];
                            kernels::im2col_into(img, c, h, w, *kernel, *stride, *padding, oh, ow, &mut cols);
                            transpose2d_into(&cols, ckk, l, &mut cols_t);
                            match self.precision {
                                NpuPrecision::Int8 => {
                                    let a_scale = QuantTensor::quantize_into(&cols_t, &mut aq_buf);
                                    int8_gemm_fused_serial(&aq_buf, a_scale, weight_q, l, ckk, *out_channels, bias, *act, &mut y_lo);
                                }
                                NpuPrecision::Fp16 => {
                                    fp16_gemm_fused_into(&cols_t, weight_fp16, l, ckk, *out_channels, bias, *act, &mut y_lo);
                                }
                            }
                            transpose2d_into(&y_lo, l, *out_channels, &mut out[b * out_stride..(b + 1) * out_stride]);
                        }
                    }
                    data = out;
                    shape = vec![n, *out_channels, oh, ow];
                }
                NpuOp::MaxPool2d { kernel, stride } => {
                    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
                    let oh = kernels::out_size(h, *kernel, *stride, 0);
                    let ow = kernels::out_size(w, *kernel, *stride, 0);
                    data = kernels::max_pool2d(&data, n, c, h, w, *kernel, *stride).0;
                    shape = vec![n, c, oh, ow];
                }
                NpuOp::Flatten => {
                    let rest: usize = shape[1..].iter().product();
                    shape = vec![shape[0], rest];
                }
                NpuOp::Activation(a) => {
                    for v in &mut data {
                        *v = a.apply(*v);
                    }
                }
                NpuOp::Softmax => {
                    let last = *shape.last().unwrap();
                    let outer = data.len() / last;
                    data = softmax_raw(&data, outer, last, 1);
                }
            }
            total_macs += macs;
            records.push(OpRecord { op: op.label(), macs, duration: t0.elapsed() });
        }

        let profile = NpuProfile {
            precision: self.precision,
            original_layers: self.original_layers,
            fused_ops: self.ops.len(),
            fp32_weight_bytes: self.fp32_bytes,
            npu_weight_bytes: self.npu_bytes,
            total_macs,
            total_time: t_total.elapsed(),
            records,
        };
        (Tensor::from_vec(data, &shape), profile)
    }
}

/// FP16 정밀도 융합 GEMM (출력 버퍼 신규 할당).
fn fp16_gemm_fused(a: &[f32], b_t: &[f32], m: usize, k: usize, n: usize, bias: &[f32], act: Activation) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    fp16_gemm_fused_into(a, b_t, m, k, n, bias, act, &mut out);
    out
}

/// FP16 정밀도 융합 GEMM (지정된 `out` 버퍼에 기록).
#[allow(clippy::too_many_arguments)]
fn fp16_gemm_fused_into(a: &[f32], b_t: &[f32], m: usize, k: usize, n: usize, bias: &[f32], act: Activation, out: &mut [f32]) {
    for r in 0..m {
        let arow = &a[r * k..(r + 1) * k];
        for c in 0..n {
            let brow = &b_t[c * k..(c + 1) * k];
            let mut acc = 0.0f32;
            for (&x, &w) in arow.iter().zip(brow) {
                acc += round_fp16(x) * w;
            }
            out[r * n + c] = act.apply(round_fp16(acc + bias[c]));
        }
    }
}
