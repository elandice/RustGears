//! 자동 미분(Autograd)을 지원하는 다차원 텐서 연산자 모듈.

use super::{shape, Tensor};
use crate::autograd;
use crate::device::cpu::{cpu_matmul_into, cpu_matmul_nt_add_into, cpu_matmul_serial};
use crate::device::{BinaryOp, Device, UnaryOp};
use crate::kernels;
use rayon::prelude::*;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// 두 텐서가 동일한 디바이스(CPU/GPU)에 위치하는지 검증합니다.
fn assert_same_device(a: &Tensor, b: &Tensor, op: &str) {
    assert_eq!(
        a.device(),
        b.device(),
        "{op}: expected all tensors to be on the same device, found {} and {}",
        a.device(),
        b.device()
    );
}

/// 동일 텐서인지 확인하여 데드락 없이 두 텐서의 데이터 버퍼 잠금을 안전하게 획득합니다.
fn with_pair<R>(a: &Tensor, b: &Tensor, f: impl FnOnce(&[f32], &[f32]) -> R) -> R {
    if a.id() == b.id() {
        let d = a.data();
        f(&d, &d)
    } else {
        let da = a.data();
        let db = b.data();
        f(&da, &db)
    }
}

/// 단항 연산자별 역전파 연산자 이름을 반환합니다.
fn unary_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "NegBackward",
        UnaryOp::Exp => "ExpBackward",
        UnaryOp::Log => "LogBackward",
        UnaryOp::Relu => "ReluBackward",
        UnaryOp::Sigmoid => "SigmoidBackward",
        UnaryOp::Tanh => "TanhBackward",
        UnaryOp::Sqrt => "SqrtBackward",
        UnaryOp::Gelu => "GeluBackward",
        UnaryOp::Abs => "AbsBackward",
    }
}

/// 불필요한 `Vec<usize>` 인덱스 맵 할당을 방지하는 고속 브로드캐스팅 패턴 열거형
enum BroadcastKind {
    /// 두 텐서의 형상이 완전히 동일한 경우 (`a.shape() == b.shape()`).
    Same,
    /// `b`가 `[D]` 또는 `[1, .., 1, D]` 형태이고 `a`의 마지막 차원에 맞춰 브로드캐스팅되는 경우 (`outer = a.len() / D`).
    LastDimB { outer: usize, d: usize },
    /// `b`가 `[1, C, 1, 1]` 형태이고 `a`가 `[N, C, H, W]`인 4차원 채널 브로드캐스팅 경우.
    Channel4dB { n: usize, c: usize, spatial: usize },
    /// 사전 계산된 인덱스 맵을 사용하는 일반 NumPy 브로드캐스팅 폴백.
    General(Vec<usize>, Vec<usize>),
}

impl Tensor {
    // ------------------------------------------------------------------
    // 요소별 이항 연산 (고속 경로 브로드캐스팅 지원)
    // ------------------------------------------------------------------

    /// 브로드캐스팅과 역전파 계산 그래프 기록을 지원하는 기본 이항 연산자
    fn binary(&self, other: &Tensor, op: BinaryOp) -> Tensor {
        assert_same_device(self, other, &format!("{op:?}"));
        let a_shape = self.shape().to_vec();
        let b_shape = other.shape().to_vec();
        let out_shape = shape::broadcast_shape(&a_shape, &b_shape).unwrap_or_else(|e| panic!("{op:?}: {e}"));
        let backend = self.backend();
        let (an, bn) = (self.numel(), other.numel());

        let (data, bkind) = if a_shape == b_shape {
            (with_pair(self, other, |a, b| backend.binary(op, a, b)), BroadcastKind::Same)
        } else if out_shape == a_shape
            && bn > 0
            && an % bn == 0
            && b_shape.last() == a_shape.last()
            && b_shape[..b_shape.len() - 1].iter().all(|&s| s == 1)
        {
            let d = bn;
            let outer = an / d;
            let out = with_pair(self, other, |a, b| {
                let mut out = vec![0.0f32; an];
                for o in 0..outer {
                    let ra = &a[o * d..(o + 1) * d];
                    let ro = &mut out[o * d..(o + 1) * d];
                    for (dst, (&av, &bv)) in ro.iter_mut().zip(ra.iter().zip(b.iter())) {
                        *dst = op.apply(av, bv);
                    }
                }
                out
            });
            (out, BroadcastKind::LastDimB { outer, d })
        } else if a_shape.len() == 4
            && b_shape.len() == 4
            && b_shape[0] == 1
            && b_shape[1] == a_shape[1]
            && b_shape[2] == 1
            && b_shape[3] == 1
        {
            let (n, c, spatial) = (a_shape[0], a_shape[1], a_shape[2] * a_shape[3]);
            let out = with_pair(self, other, |a, b| {
                let mut out = vec![0.0f32; an];
                for bi in 0..n {
                    for ch in 0..c {
                        let bv = b[ch];
                        let base = (bi * c + ch) * spatial;
                        let ra = &a[base..base + spatial];
                        let ro = &mut out[base..base + spatial];
                        for (dst, &av) in ro.iter_mut().zip(ra.iter()) {
                            *dst = op.apply(av, bv);
                        }
                    }
                }
                out
            });
            (out, BroadcastKind::Channel4dB { n, c, spatial })
        } else {
            let ma = shape::broadcast_index_map(&a_shape, &out_shape);
            let mb = shape::broadcast_index_map(&b_shape, &out_shape);
            let d = with_pair(self, other, |a, b| ma.iter().zip(&mb).map(|(&i, &j)| op.apply(a[i], b[j])).collect());
            (d, BroadcastKind::General(ma, mb))
        };

        let (a, b) = (self.clone(), other.clone());
        let name = match op {
            BinaryOp::Add => "AddBackward",
            BinaryOp::Sub => "SubBackward",
            BinaryOp::Mul => "MulBackward",
            BinaryOp::Div => "DivBackward",
        };
        Tensor::from_op(
            data,
            out_shape,
            self.device(),
            vec![self.clone(), other.clone()],
            name,
            Box::new(move |g| match &bkind {
                BroadcastKind::Same => {
                    let ga = a.requires_grad().then(|| match op {
                        BinaryOp::Add | BinaryOp::Sub => g.to_vec(),
                        BinaryOp::Mul => {
                            let bd = b.data();
                            g.iter().zip(bd.iter()).map(|(&gv, &bv)| gv * bv).collect()
                        }
                        BinaryOp::Div => {
                            let bd = b.data();
                            g.iter().zip(bd.iter()).map(|(&gv, &bv)| gv / bv).collect()
                        }
                    });
                    let gb = b.requires_grad().then(|| match op {
                        BinaryOp::Add => g.to_vec(),
                        BinaryOp::Sub => g.iter().map(|&v| -v).collect(),
                        BinaryOp::Mul => {
                            let ad = a.data();
                            g.iter().zip(ad.iter()).map(|(&gv, &av)| gv * av).collect()
                        }
                        BinaryOp::Div => {
                            let ad = a.data();
                            let bd = b.data();
                            g.iter()
                                .zip(ad.iter().zip(bd.iter()))
                                .map(|(&gv, (&av, &bv))| -gv * av / (bv * bv))
                                .collect()
                        }
                    });
                    vec![ga, gb]
                }
                &BroadcastKind::LastDimB { outer, d } => {
                    let ga = a.requires_grad().then(|| match op {
                        BinaryOp::Add | BinaryOp::Sub => g.to_vec(),
                        BinaryOp::Mul => {
                            let bd = b.data();
                            let mut out = vec![0.0f32; g.len()];
                            for o in 0..outer {
                                let rg = &g[o * d..(o + 1) * d];
                                let ro = &mut out[o * d..(o + 1) * d];
                                for j in 0..d {
                                    ro[j] = rg[j] * bd[j];
                                }
                            }
                            out
                        }
                        BinaryOp::Div => {
                            let bd = b.data();
                            let mut out = vec![0.0f32; g.len()];
                            for o in 0..outer {
                                let rg = &g[o * d..(o + 1) * d];
                                let ro = &mut out[o * d..(o + 1) * d];
                                for j in 0..d {
                                    ro[j] = rg[j] / bd[j];
                                }
                            }
                            out
                        }
                    });
                    let gb = b.requires_grad().then(|| {
                        let mut out = vec![0.0f32; d];
                        match op {
                            BinaryOp::Add => {
                                for o in 0..outer {
                                    let rg = &g[o * d..(o + 1) * d];
                                    for (dst, &gv) in out.iter_mut().zip(rg.iter()) {
                                        *dst += gv;
                                    }
                                }
                            }
                            BinaryOp::Sub => {
                                for o in 0..outer {
                                    let rg = &g[o * d..(o + 1) * d];
                                    for (dst, &gv) in out.iter_mut().zip(rg.iter()) {
                                        *dst -= gv;
                                    }
                                }
                            }
                            BinaryOp::Mul => {
                                let ad = a.data();
                                for o in 0..outer {
                                    let rg = &g[o * d..(o + 1) * d];
                                    let ra = &ad[o * d..(o + 1) * d];
                                    for j in 0..d {
                                        out[j] += rg[j] * ra[j];
                                    }
                                }
                            }
                            BinaryOp::Div => {
                                let ad = a.data();
                                let bd = b.data();
                                for o in 0..outer {
                                    let rg = &g[o * d..(o + 1) * d];
                                    let ra = &ad[o * d..(o + 1) * d];
                                    for j in 0..d {
                                        out[j] -= rg[j] * ra[j] / (bd[j] * bd[j]);
                                    }
                                }
                            }
                        }
                        out
                    });
                    vec![ga, gb]
                }
                &BroadcastKind::Channel4dB { n, c, spatial } => {
                    let ga = a.requires_grad().then(|| match op {
                        BinaryOp::Add | BinaryOp::Sub => g.to_vec(),
                        BinaryOp::Mul => {
                            let bd = b.data();
                            let mut out = vec![0.0f32; g.len()];
                            for bi in 0..n {
                                for ch in 0..c {
                                    let bv = bd[ch];
                                    let base = (bi * c + ch) * spatial;
                                    for (dst, &gv) in out[base..base + spatial].iter_mut().zip(&g[base..base + spatial]) {
                                        *dst = gv * bv;
                                    }
                                }
                            }
                            out
                        }
                        BinaryOp::Div => {
                            let bd = b.data();
                            let mut out = vec![0.0f32; g.len()];
                            for bi in 0..n {
                                for ch in 0..c {
                                    let bv = bd[ch];
                                    let base = (bi * c + ch) * spatial;
                                    for (dst, &gv) in out[base..base + spatial].iter_mut().zip(&g[base..base + spatial]) {
                                        *dst = gv / bv;
                                    }
                                }
                            }
                            out
                        }
                    });
                    let gb = b.requires_grad().then(|| {
                        let mut out = vec![0.0f32; c];
                        match op {
                            BinaryOp::Add => {
                                for bi in 0..n {
                                    for ch in 0..c {
                                        let base = (bi * c + ch) * spatial;
                                        out[ch] += g[base..base + spatial].iter().sum::<f32>();
                                    }
                                }
                            }
                            BinaryOp::Sub => {
                                for bi in 0..n {
                                    for ch in 0..c {
                                        let base = (bi * c + ch) * spatial;
                                        out[ch] -= g[base..base + spatial].iter().sum::<f32>();
                                    }
                                }
                            }
                            BinaryOp::Mul => {
                                let ad = a.data();
                                for bi in 0..n {
                                    for ch in 0..c {
                                        let base = (bi * c + ch) * spatial;
                                        out[ch] += g[base..base + spatial]
                                            .iter()
                                            .zip(&ad[base..base + spatial])
                                            .map(|(&gv, &av)| gv * av)
                                            .sum::<f32>();
                                    }
                                }
                            }
                            BinaryOp::Div => {
                                let ad = a.data();
                                let bd = b.data();
                                for bi in 0..n {
                                    for ch in 0..c {
                                        let base = (bi * c + ch) * spatial;
                                        let denom = bd[ch] * bd[ch];
                                        out[ch] -= g[base..base + spatial]
                                            .iter()
                                            .zip(&ad[base..base + spatial])
                                            .map(|(&gv, &av)| gv * av / denom)
                                            .sum::<f32>();
                                    }
                                }
                            }
                        }
                        out
                    });
                    vec![ga, gb]
                }
                BroadcastKind::General(ma, mb) => {
                    let n = g.len();
                    let need_vals = matches!(op, BinaryOp::Mul | BinaryOp::Div);
                    let ad = if need_vals { a.to_vec() } else { Vec::new() };
                    let bd = if need_vals { b.to_vec() } else { Vec::new() };
                    let ga = a.requires_grad().then(|| -> Vec<f32> {
                        match op {
                            BinaryOp::Add | BinaryOp::Sub => g.to_vec(),
                            BinaryOp::Mul => (0..n).map(|i| g[i] * bd[mb[i]]).collect(),
                            BinaryOp::Div => (0..n).map(|i| g[i] / bd[mb[i]]).collect(),
                        }
                    });
                    let gb = b.requires_grad().then(|| -> Vec<f32> {
                        match op {
                            BinaryOp::Add => g.to_vec(),
                            BinaryOp::Sub => g.iter().map(|v| -v).collect(),
                            BinaryOp::Mul => (0..n).map(|i| g[i] * ad[ma[i]]).collect(),
                            BinaryOp::Div => (0..n)
                                .map(|i| {
                                    let bv = bd[mb[i]];
                                    -g[i] * ad[ma[i]] / (bv * bv)
                                })
                                .collect(),
                        }
                    });
                    vec![
                        ga.map(|v| shape::reduce_by_map(&v, ma, an)),
                        gb.map(|v| shape::reduce_by_map(&v, mb, bn)),
                    ]
                }
            }),
        )
    }

    /// 두 텐서의 요소별 덧셈 (`+`, 브로드캐스팅 지원)
    pub fn add(&self, other: &Tensor) -> Tensor {
        self.binary(other, BinaryOp::Add)
    }
    /// 두 텐서의 요소별 뺄셈 (`-`, 브로드캐스팅 지원)
    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.binary(other, BinaryOp::Sub)
    }
    /// 두 텐서의 요소별 곱셈 (`*`, 아다마르 곱, 브로드캐스팅 지원)
    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.binary(other, BinaryOp::Mul)
    }
    /// 두 텐서의 요소별 나눗셈 (`/`, 브로드캐스팅 지원)
    pub fn div(&self, other: &Tensor) -> Tensor {
        self.binary(other, BinaryOp::Div)
    }

    // ------------------------------------------------------------------
    // 스칼라 연산 (Scalar ops)
    // ------------------------------------------------------------------

    /// 함수 `f`와 도함수 `df`를 적용하여 텐서의 각 원소를 변환하는 헬퍼 함수
    fn scalar_map(&self, name: &'static str, f: impl Fn(f32) -> f32, df: impl Fn(f32) -> f32 + Send + Sync + 'static) -> Tensor {
        let data: Vec<f32> = self.data().iter().map(|&x| f(x)).collect();
        let x = self.clone();
        Tensor::from_op(
            data,
            self.shape().to_vec(),
            self.device(),
            vec![self.clone()],
            name,
            Box::new(move |g| {
                let xd = x.data();
                vec![Some(g.iter().zip(xd.iter()).map(|(gv, &xv)| gv * df(xv)).collect())]
            }),
        )
    }

    /// 모든 원소에 스칼라 `s`를 더합니다.
    pub fn add_scalar(&self, s: f32) -> Tensor {
        self.scalar_map("AddScalarBackward", move |x| x + s, |_| 1.0)
    }

    /// 모든 원소에 스칼라 `s`를 곱합니다.
    pub fn mul_scalar(&self, s: f32) -> Tensor {
        self.scalar_map("MulScalarBackward", move |x| x * s, move |_| s)
    }

    /// 모든 원소를 스칼라 `s`로 나눕니다.
    pub fn div_scalar(&self, s: f32) -> Tensor {
        self.mul_scalar(1.0 / s)
    }

    /// 모든 원소에 스칼라 `p` 거듭제곱을 취합니다 (`x^p`).
    pub fn pow_scalar(&self, p: f32) -> Tensor {
        self.scalar_map("PowBackward", move |x| x.powf(p), move |x| p * x.powf(p - 1.0))
    }

    /// 모든 원소를 제곱합니다 (`x^2`).
    pub fn square(&self) -> Tensor {
        self.scalar_map("SquareBackward", |x| x * x, |x| 2.0 * x)
    }

    /// Leaky ReLU 활성화 함수를 적용합니다 (`x > 0 ? x : slope * x`).
    pub fn leaky_relu(&self, slope: f32) -> Tensor {
        self.scalar_map(
            "LeakyReluBackward",
            move |x| if x > 0.0 { x } else { slope * x },
            move |x| if x > 0.0 { 1.0 } else { slope },
        )
    }

    /// 모든 원소 값을 `[lo, hi]` 범위 내로 제한(클램핑)합니다.
    pub fn clamp(&self, lo: f32, hi: f32) -> Tensor {
        self.scalar_map("ClampBackward", move |x| x.clamp(lo, hi), move |x| if x >= lo && x <= hi { 1.0 } else { 0.0 })
    }

    // ------------------------------------------------------------------
    // 단항 연산 (디바이스 백엔드로 디스패치)
    // ------------------------------------------------------------------

    /// 단항 연산자를 백엔드 커널로 실행하고 미분 노드를 등록합니다.

    fn unary(&self, op: UnaryOp) -> Tensor {
        let y = self.backend().unary(op, &self.data());
        let needs_grad = autograd::is_grad_enabled() && self.requires_grad();
        let needs_y = matches!(op, UnaryOp::Exp | UnaryOp::Sigmoid | UnaryOp::Tanh | UnaryOp::Sqrt);
        let yc = if needs_grad && needs_y { y.clone() } else { Vec::new() };
        let x = self.clone();
        Tensor::from_op(
            y,
            self.shape().to_vec(),
            self.device(),
            vec![self.clone()],
            unary_name(op),
            Box::new(move |g| {
                let xd = x.data();
                let grad = if matches!(op, UnaryOp::Relu) {
                    if g.len() >= 32 * 1024 {
                        g.par_iter()
                            .zip(xd.par_iter())
                            .map(|(&gv, &xv)| if xv > 0.0 { gv } else { 0.0 })
                            .collect()
                    } else {
                        g.iter()
                            .zip(xd.iter())
                            .map(|(&gv, &xv)| if xv > 0.0 { gv } else { 0.0 })
                            .collect()
                    }
                } else if needs_y {
                    g.iter().enumerate().map(|(i, &gv)| gv * op.derivative(xd[i], yc[i])).collect()
                } else {
                    g.iter().enumerate().map(|(i, &gv)| gv * op.derivative(xd[i], 0.0)).collect()
                };
                vec![Some(grad)]
            }),
        )
    }

    /// 부호 반전 (`-x`)
    pub fn neg(&self) -> Tensor {
        self.unary(UnaryOp::Neg)
    }
    /// 밑이 e인 지수 함수 (`e^x`)
    pub fn exp(&self) -> Tensor {
        self.unary(UnaryOp::Exp)
    }
    /// 자연로그 (`ln(x)`)
    pub fn log(&self) -> Tensor {
        self.unary(UnaryOp::Log)
    }
    /// 정류 선형 유닛 활성화 함수 (ReLU, `max(0, x)`)
    pub fn relu(&self) -> Tensor {
        self.unary(UnaryOp::Relu)
    }
    /// 로지스틱 시그모이드 활성화 함수 (`1 / (1 + e^-x)`)
    pub fn sigmoid(&self) -> Tensor {
        self.unary(UnaryOp::Sigmoid)
    }
    /// 하이퍼볼릭 탄젠트 활성화 함수 (`tanh(x)`)
    pub fn tanh(&self) -> Tensor {
        self.unary(UnaryOp::Tanh)
    }
    /// 제곱근 (`sqrt(x)`)
    pub fn sqrt(&self) -> Tensor {
        self.unary(UnaryOp::Sqrt)
    }
    /// 가우스 오차 선형 유닛 활성화 함수 (GELU)
    pub fn gelu(&self) -> Tensor {
        self.unary(UnaryOp::Gelu)
    }
    /// 절댓값 (`|x|`)
    pub fn abs(&self) -> Tensor {
        self.unary(UnaryOp::Abs)
    }

    // ------------------------------------------------------------------
    // 선형대수 연산 (Linear algebra)
    // ------------------------------------------------------------------

    /// 행렬 곱셈 연산:
    /// `[m, k] @ [k, n]`, `[b, m, k] @ [k, n]`, `[b, m, k] @ [b, k, n]`, `[k] @ [k, n]`을 지원합니다.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        match (self.ndim(), other.ndim()) {
            (2, 2) => self.mm(other),
            (1, 2) => {
                let k = self.shape()[0];
                let n = other.shape()[1];
                self.reshape(&[1, k]).mm(other).reshape(&[n])
            }
            (3, 2) => {
                let (b, m, k) = (self.shape()[0], self.shape()[1], self.shape()[2]);
                let n = other.shape()[1];
                self.reshape(&[b * m, k]).mm(other).reshape(&[b, m, n])
            }
            (3, 3) => self.bmm(other),
            (a, b) => panic!("matmul: unsupported ranks {a} and {b} (shapes {:?}, {:?})", self.shape(), other.shape()),
        }
    }

    fn mm(&self, other: &Tensor) -> Tensor {
        assert_same_device(self, other, "matmul");
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (k2, n) = (other.shape()[0], other.shape()[1]);
        assert_eq!(k, k2, "matmul: shape mismatch {:?} @ {:?}", self.shape(), other.shape());
        let be = self.backend();
        let out = with_pair(self, other, |a, b| be.matmul(a, b, m, k, n));
        let (a, b) = (self.clone(), other.clone());
        Tensor::from_op(
            out,
            vec![m, n],
            self.device(),
            vec![self.clone(), other.clone()],
            "MmBackward",
            Box::new(move |g| {
                let ga = a.requires_grad().then(|| {
                    let bt = shape::transpose2d(&b.data(), k, n);
                    be.matmul(g, &bt, m, n, k)
                });
                let gb = b.requires_grad().then(|| {
                    let at = shape::transpose2d(&a.data(), m, k);
                    be.matmul(&at, g, k, m, n)
                });
                vec![ga, gb]
            }),
        )
    }

    /// 배치 행렬 곱셈: `[b, m, k] @ [b, k, n] -> [b, m, n]`
    pub fn bmm(&self, other: &Tensor) -> Tensor {
        assert_same_device(self, other, "bmm");
        assert!(self.ndim() == 3 && other.ndim() == 3, "bmm expects 3-D tensors");
        let (bs, m, k) = (self.shape()[0], self.shape()[1], self.shape()[2]);
        let (bs2, k2, n) = (other.shape()[0], other.shape()[1], other.shape()[2]);
        assert!(bs == bs2 && k == k2, "bmm: shape mismatch {:?} @ {:?}", self.shape(), other.shape());
        let be = self.backend();
        let out = with_pair(self, other, |a, b| {
            let mut out = Vec::with_capacity(bs * m * n);
            for i in 0..bs {
                out.extend(be.matmul(&a[i * m * k..(i + 1) * m * k], &b[i * k * n..(i + 1) * k * n], m, k, n));
            }
            out
        });
        let (a, b) = (self.clone(), other.clone());
        Tensor::from_op(
            out,
            vec![bs, m, n],
            self.device(),
            vec![self.clone(), other.clone()],
            "BmmBackward",
            Box::new(move |g| {
                let ad = a.data();
                let bd = b.data();
                let mut ga = a.requires_grad().then(|| Vec::with_capacity(bs * m * k));
                let mut gb = b.requires_grad().then(|| Vec::with_capacity(bs * k * n));
                for i in 0..bs {
                    let gi = &g[i * m * n..(i + 1) * m * n];
                    if let Some(ga) = ga.as_mut() {
                        let bt = shape::transpose2d(&bd[i * k * n..(i + 1) * k * n], k, n);
                        ga.extend(be.matmul(gi, &bt, m, n, k));
                    }
                    if let Some(gb) = gb.as_mut() {
                        let at = shape::transpose2d(&ad[i * m * k..(i + 1) * m * k], m, k);
                        gb.extend(be.matmul(&at, gi, k, m, n));
                    }
                }
                vec![ga, gb]
            }),
        )
    }

    // ------------------------------------------------------------------
    // 형상 조작 (Shape manipulation)
    // ------------------------------------------------------------------

    /// 텐서의 형태(Shape)를 변경합니다 (원소의 총 개수는 일치해야 함).
    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        let n: usize = new_shape.iter().product();
        assert_eq!(n, self.numel(), "reshape: cannot reshape {:?} into {:?}", self.shape(), new_shape);
        Tensor::from_op(
            self.to_vec(),
            new_shape.to_vec(),
            self.device(),
            vec![self.clone()],
            "ReshapeBackward",
            Box::new(|g| vec![Some(g.to_vec())]),
        )
    }

    /// `start_dim`부터 마지막 차원까지를 단일 차원으로 평탄화합니다.
    pub fn flatten(&self, start_dim: usize) -> Tensor {
        let mut s: Vec<usize> = self.shape()[..start_dim].to_vec();
        s.push(self.shape()[start_dim..].iter().product());
        self.reshape(&s)
    }

    /// 지정된 `dim` 위치에 크기 1인 새로운 차원을 삽입합니다.
    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        let mut s = self.shape().to_vec();
        s.insert(dim, 1);
        self.reshape(&s)
    }

    /// 주어진 순열(`perm`)에 따라 차원 축들을 재배치합니다.
    pub fn permute(&self, perm: &[usize]) -> Tensor {
        let (data, out_shape) = shape::permute(&self.data(), self.shape(), perm);
        let in_shape = self.shape().to_vec();
        let mut inv = vec![0; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inv[p] = i;
        }
        let os = out_shape.clone();
        Tensor::from_op(
            data,
            out_shape,
            self.device(),
            vec![self.clone()],
            "PermuteBackward",
            Box::new(move |g| {
                let (gi, s) = shape::permute(g, &os, &inv);
                debug_assert_eq!(s, in_shape);
                vec![Some(gi)]
            }),
        )
    }

    /// 두 차원 `d0`과 `d1`의 축을 맞바꿉니다.
    pub fn transpose(&self, d0: usize, d1: usize) -> Tensor {
        let mut perm: Vec<usize> = (0..self.ndim()).collect();
        perm.swap(d0, d1);
        self.permute(&perm)
    }

    /// 2차원 텐서의 행과 열을 전치(Transpose)합니다.
    pub fn t(&self) -> Tensor {
        assert_eq!(self.ndim(), 2, "t() expects a 2-D tensor");
        self.transpose(0, 1)
    }

    /// 텐서를 다른 디바이스(CPU/GPU)로 복사합니다 (미분 이력 유지).
    pub fn to(&self, device: Device) -> Tensor {
        if device == self.device() {
            return self.clone();
        }
        Tensor::from_op(
            self.to_vec(),
            self.shape().to_vec(),
            device,
            vec![self.clone()],
            "ToDeviceBackward",
            Box::new(|g| vec![Some(g.to_vec())]),
        )
    }

    // ------------------------------------------------------------------
    // 집계 및 축소 연산 (Reductions)
    // ------------------------------------------------------------------

    /// 텐서의 모든 원소를 합산하여 스칼라 텐서로 반환합니다.
    pub fn sum(&self) -> Tensor {
        let s: f32 = self.data().iter().sum();
        let n = self.numel();
        Tensor::from_op(vec![s], vec![], self.device(), vec![self.clone()], "SumBackward", Box::new(move |g| vec![Some(vec![g[0]; n])]))
    }

    /// 텐서의 모든 원소에 대한 평균값을 계산하여 스칼라 텐서로 반환합니다.
    pub fn mean(&self) -> Tensor {
        let n = self.numel().max(1) as f32;
        self.sum().mul_scalar(1.0 / n)
    }

    /// 지정한 차원(`dim`)을 따라 합계를 계산합니다 (`keepdim`이 참이면 축소된 차원을 크기 1로 유지).
    pub fn sum_dim(&self, dim: isize, keepdim: bool) -> Tensor {
        let d = self.dim_index(dim);
        let (outer, size, inner) = shape::split_at_dim(self.shape(), d);
        let x = self.data();
        let mut out = vec![0.0; outer * inner];
        for o in 0..outer {
            for j in 0..size {
                let base = (o * size + j) * inner;
                for i in 0..inner {
                    out[o * inner + i] += x[base + i];
                }
            }
        }
        drop(x);
        let mut s = self.shape().to_vec();
        if keepdim {
            s[d] = 1;
        } else {
            s.remove(d);
        }
        Tensor::from_op(
            out,
            s,
            self.device(),
            vec![self.clone()],
            "SumDimBackward",
            Box::new(move |g| {
                let mut gx = vec![0.0; outer * size * inner];
                for o in 0..outer {
                    for j in 0..size {
                        let base = (o * size + j) * inner;
                        gx[base..base + inner].copy_from_slice(&g[o * inner..(o + 1) * inner]);
                    }
                }
                vec![Some(gx)]
            }),
        )
    }

    /// 지정한 차원(`dim`)을 따라 평균을 계산합니다.
    pub fn mean_dim(&self, dim: isize, keepdim: bool) -> Tensor {
        let n = self.size(dim) as f32;
        self.sum_dim(dim, keepdim).mul_scalar(1.0 / n)
    }

    /// 지정한 차원(`dim`)을 따라 편향 분산(Biased variance)을 계산합니다.
    pub fn var_dim(&self, dim: isize, keepdim: bool) -> Tensor {
        let mu = self.mean_dim(dim, true);
        let c = self.sub(&mu);
        c.square().mean_dim(dim, keepdim)
    }

    /// 지정한 축(`dim`)을 따라 수치적으로 안정적인 소프트맥스(Softmax)를 계산합니다.
    pub fn softmax(&self, dim: isize) -> Tensor {
        let d = self.dim_index(dim);
        let (outer, size, inner) = shape::split_at_dim(self.shape(), d);
        let y = softmax_raw(&self.data(), outer, size, inner);
        let yc = y.clone();
        Tensor::from_op(
            y,
            self.shape().to_vec(),
            self.device(),
            vec![self.clone()],
            "SoftmaxBackward",
            Box::new(move |g| {
                let mut gx = vec![0.0; g.len()];
                for o in 0..outer {
                    for i in 0..inner {
                        let idx = |j: usize| (o * size + j) * inner + i;
                        let dot: f32 = (0..size).map(|j| g[idx(j)] * yc[idx(j)]).sum();
                        for j in 0..size {
                            gx[idx(j)] = yc[idx(j)] * (g[idx(j)] - dot);
                        }
                    }
                }
                vec![Some(gx)]
            }),
        )
    }

    /// 지정한 축(`dim`)을 따라 수치적으로 안정적인 로그 소프트맥스(Log-Softmax)를 계산합니다.
    pub fn log_softmax(&self, dim: isize) -> Tensor {
        let d = self.dim_index(dim);
        let (outer, size, inner) = shape::split_at_dim(self.shape(), d);
        let sm = softmax_raw(&self.data(), outer, size, inner);
        let y: Vec<f32> = sm.iter().map(|v| v.max(1e-30).ln()).collect();
        Tensor::from_op(
            y,
            self.shape().to_vec(),
            self.device(),
            vec![self.clone()],
            "LogSoftmaxBackward",
            Box::new(move |g| {
                let mut gx = vec![0.0; g.len()];
                for o in 0..outer {
                    for i in 0..inner {
                        let idx = |j: usize| (o * size + j) * inner + i;
                        let gs: f32 = (0..size).map(|j| g[idx(j)]).sum();
                        for j in 0..size {
                            gx[idx(j)] = g[idx(j)] - sm[idx(j)] * gs;
                        }
                    }
                }
                vec![Some(gx)]
            }),
        )
    }

    // ------------------------------------------------------------------
    // 합성곱 및 풀링 연산 (Convolution / Pooling)
    // ------------------------------------------------------------------

    /// 2차원 합성곱 연산: `self`: `[N, C, H, W]`, `weight`: `[O, C, k, k]` -> `[N, O, OH, OW]`
    pub fn conv2d(&self, weight: &Tensor, stride: usize, padding: usize) -> Tensor {
        assert_same_device(self, weight, "conv2d");
        assert_eq!(self.ndim(), 4, "conv2d input must be [N,C,H,W], got {:?}", self.shape());
        assert_eq!(weight.ndim(), 4, "conv2d weight must be [O,C,k,k]");
        let (n, c, h, w) = (self.shape()[0], self.shape()[1], self.shape()[2], self.shape()[3]);
        let (o, wc, k, k2) = (weight.shape()[0], weight.shape()[1], weight.shape()[2], weight.shape()[3]);
        assert!(wc == c && k == k2, "conv2d: weight {:?} incompatible with input {:?}", weight.shape(), self.shape());
        let be = self.backend();
        let mm = |a: &[f32], b: &[f32], m: usize, kk: usize, nn: usize| be.matmul(a, b, m, kk, nn);
        let (out, oh, ow) = with_pair(self, weight, |x, wt| kernels::conv2d_forward(x, wt, n, c, h, w, o, k, stride, padding, &mm));
        let (x, wt) = (self.clone(), weight.clone());
        Tensor::from_op(
            out,
            vec![n, o, oh, ow],
            self.device(),
            vec![self.clone(), weight.clone()],
            "Conv2dBackward",
            Box::new(move |g| {
                let l = oh * ow;
                let ckk = c * k * k;
                let img_stride = c * h * w;
                let out_stride = o * l;
                let need_gw = wt.requires_grad();
                let need_gx = x.requires_grad();
                let xd = x.data();
                let wd = wt.data();
                let wt_t = if need_gx { shape::transpose2d(&wd, o, ckk) } else { Vec::new() };

                if n >= 2 && (need_gw || need_gx) {
                    let mut gx = need_gx.then(|| vec![0.0f32; n * img_stride]);
                    let gw = if let Some(gx_slice) = gx.as_mut() {
                        let gw_reduced = gx_slice
                            .par_chunks_mut(img_stride)
                            .enumerate()
                            .fold(
                                || {
                                    (
                                        vec![0.0f32; if need_gw { o * ckk } else { 0 }],
                                        vec![0.0f32; if need_gw { ckk * l } else { 0 }],
                                        vec![0.0f32; ckk * l],
                                    )
                                },
                                |(mut gw_acc, mut cols, mut dcols), (b, gx_b)| {
                                    let gb = &g[b * out_stride..(b + 1) * out_stride];
                                    if need_gw {
                                        let img = &xd[b * img_stride..(b + 1) * img_stride];
                                        kernels::im2col_into(img, c, h, w, k, stride, padding, oh, ow, &mut cols);
                                        cpu_matmul_nt_add_into(gb, &cols, &mut gw_acc, o, ckk, l);
                                    }
                                    cpu_matmul_into(&wt_t, gb, &mut dcols, ckk, o, l);
                                    kernels::col2im(&dcols, c, h, w, k, stride, padding, gx_b);
                                    (gw_acc, cols, dcols)
                                },
                            )
                            .map(|(gw_acc, _, _)| gw_acc)
                            .reduce(
                                || vec![0.0f32; if need_gw { o * ckk } else { 0 }],
                                |mut a, b| {
                                    for (av, bv) in a.iter_mut().zip(b) {
                                        *av += bv;
                                    }
                                    a
                                },
                            );
                        need_gw.then_some(gw_reduced)
                    } else if need_gw {
                        let gw_reduced = (0..n)
                            .into_par_iter()
                            .fold(
                                || (vec![0.0f32; o * ckk], vec![0.0f32; ckk * l]),
                                |(mut gw_acc, mut cols), b| {
                                    let gb = &g[b * out_stride..(b + 1) * out_stride];
                                    let img = &xd[b * img_stride..(b + 1) * img_stride];
                                    kernels::im2col_into(img, c, h, w, k, stride, padding, oh, ow, &mut cols);
                                    cpu_matmul_nt_add_into(gb, &cols, &mut gw_acc, o, ckk, l);
                                    (gw_acc, cols)
                                },
                            )
                            .map(|(gw_acc, _)| gw_acc)
                            .reduce(
                                || vec![0.0f32; o * ckk],
                                |mut a, b| {
                                    for (av, bv) in a.iter_mut().zip(b) {
                                        *av += bv;
                                    }
                                    a
                                },
                            );
                        Some(gw_reduced)
                    } else {
                        None
                    };
                    vec![gx, gw]
                } else {
                    let mut gw = vec![0.0f32; o * ckk];
                    let mut gx = need_gx.then(|| vec![0.0f32; n * img_stride]);
                    let mut cols = vec![0.0f32; ckk * l];
                    for b in 0..n {
                        let gb = &g[b * out_stride..(b + 1) * out_stride];
                        if need_gw {
                            kernels::im2col_into(&xd[b * img_stride..(b + 1) * img_stride], c, h, w, k, stride, padding, oh, ow, &mut cols);
                            cpu_matmul_nt_add_into(gb, &cols, &mut gw, o, ckk, l);
                        }
                        if let Some(gx) = gx.as_mut() {
                            let dcols = cpu_matmul_serial(&wt_t, gb, ckk, o, l);
                            kernels::col2im(&dcols, c, h, w, k, stride, padding, &mut gx[b * img_stride..(b + 1) * img_stride]);
                        }
                    }
                    vec![gx, need_gw.then_some(gw)]
                }
            }),
        )
    }

    /// `[N, C, H, W]` 입력과 `[C, O, k, k]` 가중치에 대한 2차원 전치 합성곱 (Transposed Convolution).
    /// 정사각형 커널, 팽창 계수(dilation) 1 및 단일 그룹을 지원합니다.
    pub fn conv_transpose2d(&self, weight: &Tensor, stride: usize, padding: usize, output_padding: usize) -> Tensor {
        assert_same_device(self, weight, "conv_transpose2d");
        assert_eq!(self.ndim(), 4, "conv_transpose2d input must be [N,C,H,W]");
        assert_eq!(weight.ndim(), 4, "conv_transpose2d weight must be [C,O,k,k]");
        let (n, c, h, w) = (self.shape()[0], self.shape()[1], self.shape()[2], self.shape()[3]);
        let (wc, o, k, k2) = (weight.shape()[0], weight.shape()[1], weight.shape()[2], weight.shape()[3]);
        assert!(c > 0 && o > 0 && h > 0 && w > 0 && k > 0 && c == wc && k == k2, "conv_transpose2d: incompatible input/weight shapes");
        assert!(stride > 0 && output_padding < stride, "conv_transpose2d requires stride > 0 and output_padding < stride");
        let output_size = |s: usize| {
            (s - 1).checked_mul(stride)
                .and_then(|v| v.checked_add(k))
                .and_then(|v| v.checked_add(output_padding))
                .and_then(|v| padding.checked_mul(2).and_then(|p| v.checked_sub(p)))
                .filter(|&v| v > 0)
                .expect("conv_transpose2d: invalid or overflowing output size")
        };
        let (oh, ow) = (output_size(h), output_size(w));
        let (l, okk) = (h * w, o * k * k);
        let be = self.backend();
        let out = with_pair(self, weight, |x, wt| {
            let wt_t = shape::transpose2d(wt, c, okk);
            let mut out = vec![0.0; n * o * oh * ow];
            for b in 0..n {
                let cols = be.matmul(&wt_t, &x[b * c * l..(b + 1) * c * l], okk, c, l);
                kernels::col2im(&cols, o, oh, ow, k, stride, padding, &mut out[b * o * oh * ow..(b + 1) * o * oh * ow]);
            }
            out
        });
        let (x, wt) = (self.clone(), weight.clone());
        Tensor::from_op(out, vec![n, o, oh, ow], self.device(), vec![self.clone(), weight.clone()], "ConvTranspose2dBackward", Box::new(move |g| {
            let mut gx = x.requires_grad().then(|| vec![0.0; x.numel()]);
            let mut gw = wt.requires_grad().then(|| vec![0.0; wt.numel()]);
            for b in 0..n {
                let cols = kernels::im2col(&g[b * o * oh * ow..(b + 1) * o * oh * ow], o, oh, ow, k, stride, padding);
                if let Some(gx) = gx.as_mut() {
                    let gb = be.matmul(&wt.data(), &cols, c, okk, l);
                    gx[b * c * l..(b + 1) * c * l].copy_from_slice(&gb);
                }
                if let Some(gw) = gw.as_mut() {
                    let cols_t = shape::transpose2d(&cols, okk, l);
                    let gb = be.matmul(&x.data()[b * c * l..(b + 1) * c * l], &cols_t, c, l, okk);
                    gw.iter_mut().zip(gb).for_each(|(a, v)| *a += v);
                }
            }
            vec![gx, gw]
        }))
    }

    /// `[N, C, H, W]` 입력에 대해 고정된 `(height, width)` 크기로 2D 적응형 평균 풀링을 수행합니다.
    /// 나누어떨어지지 않는 입력 크기는 바닥(floor)/올림(ceil)을 적용한 오버랩 영역을 사용합니다.
    pub fn adaptive_avg_pool2d(&self, output_size: (usize, usize)) -> Tensor {
        assert_eq!(self.ndim(), 4, "adaptive_avg_pool2d input must be [N,C,H,W]");
        let (n, c, h, w) = (self.shape()[0], self.shape()[1], self.shape()[2], self.shape()[3]);
        let (oh, ow) = output_size;
        assert!(h > 0 && w > 0 && oh > 0 && ow > 0, "adaptive_avg_pool2d spatial sizes must be positive");
        let regions: Vec<_> = (0..oh).flat_map(|y| (0..ow).map(move |x| {
            (y * h / oh, ((y + 1) * h).div_ceil(oh), x * w / ow, ((x + 1) * w).div_ceil(ow))
        })).collect();
        let mut out = Vec::with_capacity(n * c * oh * ow);
        let d = self.data();
        for plane in 0..n * c {
            for &(y0, y1, x0, x1) in &regions {
                let mut sum = 0.0;
                for y in y0..y1 {
                    for x in x0..x1 {
                        sum += d[(plane * h + y) * w + x];
                    }
                }
                out.push(sum / ((y1 - y0) * (x1 - x0)) as f32);
            }
        }
        let numel = self.numel();
        Tensor::from_op(out, vec![n, c, oh, ow], self.device(), vec![self.clone()], "AdaptiveAvgPool2dBackward", Box::new(move |g| {
            let mut gx = vec![0.0; numel];
            for plane in 0..n * c {
                for (i, &(y0, y1, x0, x1)) in regions.iter().enumerate() {
                    let value = g[plane * oh * ow + i] / ((y1 - y0) * (x1 - x0)) as f32;
                    for y in y0..y1 {
                        for x in x0..x1 {
                            gx[(plane * h + y) * w + x] += value;
                        }
                    }
                }
            }
            vec![Some(gx)]
        }))
    }

    /// `[N, C, H, W]` 입력에 대한 2차원 최대 풀링 (Max Pooling).
    pub fn max_pool2d(&self, kernel: usize, stride: usize) -> Tensor {
        assert_eq!(self.ndim(), 4, "max_pool2d input must be [N,C,H,W]");
        let (n, c, h, w) = (self.shape()[0], self.shape()[1], self.shape()[2], self.shape()[3]);
        let (out, arg) = kernels::max_pool2d(&self.data(), n, c, h, w, kernel, stride);
        let oh = kernels::out_size(h, kernel, stride, 0);
        let ow = kernels::out_size(w, kernel, stride, 0);
        let numel = self.numel();
        Tensor::from_op(
            out,
            vec![n, c, oh, ow],
            self.device(),
            vec![self.clone()],
            "MaxPool2dBackward",
            Box::new(move |g| {
                let mut gx = vec![0.0; numel];
                for (gv, &i) in g.iter().zip(&arg) {
                    gx[i] += gv;
                }
                vec![Some(gx)]
            }),
        )
    }
}

/// 최댓값을 감산하여 오버플로를 방지하는 수치적으로 안정적인 소프트맥스 원시 계산 함수
pub(crate) fn softmax_raw(x: &[f32], outer: usize, size: usize, inner: usize) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    for o in 0..outer {
        for i in 0..inner {
            let idx = |j: usize| (o * size + j) * inner + i;
            let mx = (0..size).map(|j| x[idx(j)]).fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0.0;
            for j in 0..size {
                let e = (x[idx(j)] - mx).exp();
                y[idx(j)] = e;
                s += e;
            }
            for j in 0..size {
                y[idx(j)] /= s;
            }
        }
    }
    y
}

// ----------------------------------------------------------------------
// 연산자 오버로딩: &a + &b, a * 2.0, -a 등
// ----------------------------------------------------------------------

macro_rules! impl_binop {
    ($trait:ident, $method:ident, $func:ident, $scalar:expr) => {
        impl $trait<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::$func(self, rhs)
            }
        }
        impl $trait<Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                Tensor::$func(&self, &rhs)
            }
        }
        impl $trait<&Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::$func(&self, rhs)
            }
        }
        impl $trait<Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                Tensor::$func(self, &rhs)
            }
        }
        impl $trait<f32> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f32) -> Tensor {
                let f: fn(&Tensor, f32) -> Tensor = $scalar;
                f(self, rhs)
            }
        }
        impl $trait<f32> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f32) -> Tensor {
                let f: fn(&Tensor, f32) -> Tensor = $scalar;
                f(&self, rhs)
            }
        }
    };
}

impl_binop!(Add, add, add, |t, s| t.add_scalar(s));
impl_binop!(Sub, sub, sub, |t, s| t.add_scalar(-s));
impl_binop!(Mul, mul, mul, |t, s| t.mul_scalar(s));
impl_binop!(Div, div, div, |t, s| t.mul_scalar(1.0 / s));

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(&self)
    }
}
