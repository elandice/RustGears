//! NPU 런타임에서 사용하는 대칭(Symmetric) INT8 및 모의 FP16 양자화.

use rayon::prelude::*;

/// 대칭적으로 양자화된 INT8 텐서 (`실제값 ≈ data[i] * scale`).
#[derive(Clone, Debug)]
pub struct QuantTensor {
    /// 양자화된 i8 정수 데이터 (`[-127, 127]`).
    pub data: Vec<i8>,
    /// 양자화 스케일 팩터.
    pub scale: f32,
    /// 텐서 형태(Shape).
    pub shape: Vec<usize>,
}

impl QuantTensor {
    /// `f32` 버퍼를 대칭 `i8` (`[-127, 127]`) 범위로 양자화합니다.
    pub fn from_f32(values: &[f32], shape: &[usize]) -> Self {
        let max_abs = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if max_abs < 1e-12 { 1.0 } else { max_abs / 127.0 };
        let inv = 1.0 / scale;
        let data = values.iter().map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8).collect();
        QuantTensor { data, scale, shape: shape.to_vec() }
    }

    /// `values` 데이터를 기존 `i8` 버퍼에 양자화하여 기록하고 계산된 `scale`을 반환합니다.
    pub fn quantize_into(values: &[f32], out: &mut [i8]) -> f32 {
        let max_abs = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if max_abs < 1e-12 { 1.0 } else { max_abs / 127.0 };
        let inv = 1.0 / scale;
        for (dst, &v) in out.iter_mut().zip(values.iter()) {
            *dst = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
        scale
    }

    /// 양자화된 데이터를 다시 `f32` 부동소수점으로 역양자화(Dequantize)합니다.
    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&q| q as f32 * self.scale).collect()
    }

    /// 메모리 점유 바이트 크기를 반환합니다 (가중치당 1바이트 + 4바이트 스케일 계수).
    pub fn byte_size(&self) -> usize {
        self.data.len() + 4
    }
}

/// `f32` 값을 IEEE-754 binary16 (FP16) 정밀도로 반올림 변환하여 NPU FP16 텐서 코어의 수치적 거동을 모사합니다.
#[inline]
pub fn round_fp16(x: f32) -> f32 {
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    if exp < -14 {
        return 0.0; // 언더플로우 발생 시 0으로 수렴
    }
    if exp > 15 {
        return f32::from_bits(sign | 0x477f_e000); // FP16 최대값(65504)으로 클램핑
    }
    // 가수부 상위 10비트 유지 (가장 가까운 짝수로 반올림)
    let round_bit = 1u32 << 12;
    let rounded = (bits + round_bit) & !0x1fff;
    f32::from_bits(rounded)
}

/// 4-way 언롤링을 적용한 i8 벡터 간 고속 정수 내적 연산 (i32 누적).
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

/// INT8 시스톨릭 어레이 GEMM 연산:
/// `A_q[m, k] (i8) * B_q[k, n] (i8)`을 `i32` 레지스터에 누적한 뒤,
/// `f32`로 재스케일링(`scale_a * scale_b`)하고 `bias` 합산 및 융합된 `activation`을 한 번에 적용합니다.
pub fn int8_gemm_fused(
    a: &QuantTensor,
    b_t: &QuantTensor, // 전치된 행렬: 연속된 내적을 위해 [n, k] 형태
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

/// 지정된 `out` 버퍼에 결과를 기록하는 단일 스레드 INT8 GEMM (배치 병렬 NPU Conv2d 내부에서 활용).
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
