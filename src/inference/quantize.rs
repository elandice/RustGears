//! Symmetric INT8 and simulated FP16 quantisation used by the NPU runtime.

use rayon::prelude::*;

/// Symmetrically quantised INT8 tensor (`real ≈ data[i] * scale`).
#[derive(Clone, Debug)]
pub struct QuantTensor {
    pub data: Vec<i8>,
    pub scale: f32,
    pub shape: Vec<usize>,
}

impl QuantTensor {
    /// Quantise an `f32` buffer into symmetric `i8` (`[-127, 127]`).
    pub fn from_f32(values: &[f32], shape: &[usize]) -> Self {
        let max_abs = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if max_abs < 1e-12 { 1.0 } else { max_abs / 127.0 };
        let inv = 1.0 / scale;
        let data = values.iter().map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8).collect();
        QuantTensor { data, scale, shape: shape.to_vec() }
    }

    /// Quantise `values` into an existing `i8` buffer and return `scale`.
    pub fn quantize_into(values: &[f32], out: &mut [i8]) -> f32 {
        let max_abs = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if max_abs < 1e-12 { 1.0 } else { max_abs / 127.0 };
        let inv = 1.0 / scale;
        for (dst, &v) in out.iter_mut().zip(values.iter()) {
            *dst = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
        scale
    }

    /// Dequantise back to `f32`.
    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&q| q as f32 * self.scale).collect()
    }

    /// Memory footprint in bytes (1 byte per weight + 4-byte scale).
    pub fn byte_size(&self) -> usize {
        self.data.len() + 4
    }
}

/// Round an `f32` to IEEE-754 binary16 (FP16) precision and back, modelling
/// an NPU FP16 tensor core.
#[inline]
pub fn round_fp16(x: f32) -> f32 {
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    if exp < -14 {
        return 0.0; // underflow to zero
    }
    if exp > 15 {
        return f32::from_bits(sign | 0x477f_e000); // clamp to FP16 max (65504)
    }
    // keep top 10 mantissa bits (round to nearest even)
    let round_bit = 1u32 << 12;
    let rounded = (bits + round_bit) & !0x1fff;
    f32::from_bits(rounded)
}

#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    let mut acc0: i32 = 0;
    let mut acc1: i32 = 0;
    let mut acc2: i32 = 0;
    let mut acc3: i32 = 0;
    let len = a.len();
    let mut i = 0;
    while i + 4 <= len {
        acc0 += (a[i] as i32) * (b[i] as i32);
        acc1 += (a[i + 1] as i32) * (b[i + 1] as i32);
        acc2 += (a[i + 2] as i32) * (b[i + 2] as i32);
        acc3 += (a[i + 3] as i32) * (b[i + 3] as i32);
        i += 4;
    }
    let mut total = (acc0 + acc1) + (acc2 + acc3);
    while i < len {
        total += (a[i] as i32) * (b[i] as i32);
        i += 1;
    }
    total
}

/// INT8 systolic-array GEMM:
/// `A_q[m, k] (i8) * B_q[k, n] (i8)` accumulated into `i32` registers, then
/// rescaled to `f32` (`scale_a * scale_b`) + `bias` + fused `activation`.
pub fn int8_gemm_fused(
    a: &QuantTensor,
    b_t: &QuantTensor, // transposed: [n, k] for contiguous dot products
    m: usize,
    k: usize,
    n: usize,
    bias: &[f32],
    act: super::Activation,
) -> Vec<f32> {
    let combined_scale = a.scale * b_t.scale;
    let mut out = vec![0.0f32; m * n];
    if m * k * n >= 32 * 1024 {
        out.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
            let arow = &a.data[r * k..(r + 1) * k];
            for (c, cell) in row.iter_mut().enumerate() {
                let brow = &b_t.data[c * k..(c + 1) * k];
                let acc = dot_i8(arow, brow);
                let val = (acc as f32) * combined_scale + bias[c];
                *cell = act.apply(val);
            }
        });
    } else {
        int8_gemm_fused_serial(&a.data, a.scale, b_t, m, k, n, bias, act, &mut out);
    }
    out
}

/// Single-threaded INT8 GEMM writing into `out` (used inside batch-parallel NPU Conv2d).
#[allow(clippy::too_many_arguments)]
pub fn int8_gemm_fused_serial(
    a_data: &[i8],
    a_scale: f32,
    b_t: &QuantTensor,
    m: usize,
    k: usize,
    n: usize,
    bias: &[f32],
    act: super::Activation,
    out: &mut [f32],
) {
    let combined_scale = a_scale * b_t.scale;
    for r in 0..m {
        let arow = &a_data[r * k..(r + 1) * k];
        let row = &mut out[r * n..(r + 1) * n];
        for (c, cell) in row.iter_mut().enumerate() {
            let brow = &b_t.data[c * k..(c + 1) * k];
            let acc = dot_i8(arow, brow);
            let val = (acc as f32) * combined_scale + bias[c];
            *cell = act.apply(val);
        }
    }
}
