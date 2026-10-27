//! Neural Processing Unit (NPU) compiler and execution engine.
//!
//! Real NPUs (Apple Neural Engine, Qualcomm Hexagon, Intel AI Boost, AMD XDNA)
//! achieve high throughput and energy efficiency through three mechanisms that
//! are explicitly modelled here:
//!
//! 1. **Operator fusion** – `Linear/Conv2d + BatchNorm + Activation` are fused
//!    into a single NPU macro-instruction so intermediate tensors never leave
//!    on-chip SRAM.
//! 2. **Low-precision MAC arrays** – weights and activations are quantised to
//!    symmetric **INT8** (with **INT32** accumulators) or **FP16**, shrinking
//!    weight bandwidth by up to **4x**.
//! 3. **Tile / systolic execution profiling** – every run records MAC count,
//!    fused op count, weight compression ratio and per-layer latency.

use super::quantize::{int8_gemm_fused, int8_gemm_fused_serial, round_fp16, QuantTensor};
use super::{fold_batch_norm, Activation, LayerSpec};
use crate::kernels;
use crate::tensor::ops::softmax_raw;
use crate::tensor::shape::{transpose2d, transpose2d_into};
use crate::tensor::Tensor;
use rayon::prelude::*;
use std::fmt;
use std::time::{Duration, Instant};

/// Numerical precision used by the NPU MAC array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NpuPrecision {
    /// 8-bit integer weights & activations with 32-bit integer accumulators.
    Int8,
    /// 16-bit floating-point tensor path.
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

/// Fused instruction executed by the NPU runtime.
#[derive(Clone, Debug)]
pub enum NpuOp {
    FusedLinear {
        in_features: usize,
        out_features: usize,
        /// Transposed `[out, in]` INT8 weights for systolic dot-products.
        weight_q_t: QuantTensor,
        /// Transposed `[out, in]` FP16-rounded weights.
        weight_fp16_t: Vec<f32>,
        bias: Vec<f32>,
        act: Activation,
    },
    FusedConv2d {
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        /// `[out, in*k*k]` INT8 weights.
        weight_q: QuantTensor,
        weight_fp16: Vec<f32>,
        bias: Vec<f32>,
        act: Activation,
    },
    MaxPool2d {
        kernel: usize,
        stride: usize,
    },
    Flatten,
    Activation(Activation),
    Softmax,
}

impl NpuOp {
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

/// Telemetry of one NPU instruction.
#[derive(Clone, Debug)]
pub struct OpRecord {
    pub op: String,
    pub macs: u64,
    pub duration: Duration,
}

/// Hardware telemetry returned after an NPU inference run.
#[derive(Clone, Debug)]
pub struct NpuProfile {
    pub precision: NpuPrecision,
    pub original_layers: usize,
    pub fused_ops: usize,
    pub fp32_weight_bytes: usize,
    pub npu_weight_bytes: usize,
    pub total_macs: u64,
    pub total_time: Duration,
    pub records: Vec<OpRecord>,
}

impl NpuProfile {
    /// Weight compression ratio (`fp32_bytes / npu_bytes`).
    pub fn compression_ratio(&self) -> f32 {
        if self.npu_weight_bytes == 0 {
            1.0
        } else {
            self.fp32_weight_bytes as f32 / self.npu_weight_bytes as f32
        }
    }

    /// Effective throughput in GOPS (`2 * MACs / time`).
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

pub struct NpuProgram {
    pub precision: NpuPrecision,
    pub original_layers: usize,
    pub ops: Vec<NpuOp>,
    pub fp32_bytes: usize,
    pub npu_bytes: usize,
}

/// Compile a sequence of [`LayerSpec`]s into fused, quantised NPU ops.
pub fn compile(raw: &[LayerSpec], precision: NpuPrecision) -> NpuProgram {
    // Pass 1: fold BatchNorm into Linear / Conv2d
    let folded = fold_batch_norm(raw);

    // Pass 2: fuse Linear/Conv2d + Activation and quantise weights
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
                // Standalone affine if no Linear/Conv preceded it: represent as diagonal Linear
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

fn fp16_gemm_fused(a: &[f32], b_t: &[f32], m: usize, k: usize, n: usize, bias: &[f32], act: Activation) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    fp16_gemm_fused_into(a, b_t, m, k, n, bias, act, &mut out);
    out
}

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
