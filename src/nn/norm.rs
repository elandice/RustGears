//! 신경망 정규화(Normalization) 레이어 모듈.

use super::Module;
use crate::autograd;
use crate::inference::LayerSpec;
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};

/// 채널 차원(dim 1)을 기준으로 정규화를 수행하는 배치 정규화(Batch Normalization) 레이어.
///
/// * `BatchNorm1d` – 2차원 입력 `[N, C]` 지원
/// * `BatchNorm2d` – 4차원 공간 입력 `[N, C, H, W]` 지원 (N, H, W 축에 걸쳐 통계량 산출)
pub struct BatchNorm {
    /// 채널별 스케일(Scale) 학습 파라미터 ($\gamma$)
    pub gamma: Tensor,
    /// 채널별 시프트(Shift/Bias) 학습 파라미터 ($\beta$)
    pub beta: Tensor,
    /// 추론에 사용되는 채널별 누적 이동 평균
    pub running_mean: Tensor,
    /// 추론에 사용되는 채널별 누적 이동 분산
    pub running_var: Tensor,
    /// 입력 특징(채널) 개수
    pub num_features: usize,
    /// 이동 통계량 갱신 모멘텀 (기본값: 0.1)
    pub momentum: f32,
    /// 분모 0 방지용 소수 epsilon (기본값: 1e-5)
    pub eps: f32,
    /// 4차원 공간(spatial: `[N, C, H, W]`) 데이터 여부
    spatial: bool,
    /// 학습/평가 모드 플래그
    training: AtomicBool,
}

pub type BatchNorm1d = BatchNorm;

/// 4차원 이미지/특징 맵 입력(`[N, C, H, W]`)을 위한 생성자 헬퍼 구조체
pub struct BatchNorm2d;

impl BatchNorm2d {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(num_features: usize) -> BatchNorm {
        BatchNorm::with_spatial(num_features, true)
    }
}

impl BatchNorm {
    /// 2차원 입력(`[N, C]`)을 위한 1D 배치 정규화 레이어를 생성합니다.
    pub fn new(num_features: usize) -> Self {
        Self::with_spatial(num_features, false)
    }

    /// 내부 생성자: 공간 차원 여부를 지정하여 초기화합니다.
    fn with_spatial(num_features: usize, spatial: bool) -> Self {
        assert!(num_features > 0, "BatchNorm feature count must be positive");
        BatchNorm {
            gamma: Tensor::ones(&[num_features]).requires_grad_(true),
            beta: Tensor::zeros(&[num_features]).requires_grad_(true),
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            num_features,
            momentum: 0.1,
            eps: 1e-5,
            spatial,
            training: AtomicBool::new(true),
        }
    }
}

impl Module for BatchNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        let c = self.num_features;
        assert_eq!(x.ndim(), if self.spatial { 4 } else { 2 }, "BatchNorm input rank mismatch");
        assert_eq!(x.shape()[1], c, "BatchNorm channel count mismatch");

        let n = x.shape()[0];
        let s = if self.spatial { x.shape()[2] * x.shape()[3] } else { 1 };
        let m_count = n * s;
        let total = n * c * s;
        let is_train = self.training.load(Ordering::Relaxed);
        let record_grad = autograd::is_grad_enabled()
            && (x.requires_grad() || self.gamma.requires_grad() || self.beta.requires_grad());

        if is_train {
            assert!(m_count > 0, "BatchNorm training statistics must not be empty");
            let xd = x.data();
            let gd = self.gamma.data();
            let bd = self.beta.data();

            let inv_m = 1.0 / (m_count as f32);
            let mut mean = vec![0.0f32; c];
            let mut var = vec![0.0f32; c];
            let mut inv_std = vec![0.0f32; c];

            // 연속된 공간 슬라이스에 대해 채널별 평균 및 편향 분산 직접 계산
            for bi in 0..n {
                for ch in 0..c {
                    let base = (bi * c + ch) * s;
                    let slice = &xd[base..base + s];
                    let mut sum = 0.0f32;
                    for &v in slice {
                        sum += v;
                    }
                    mean[ch] += sum;
                }
            }
            for ch in 0..c {
                mean[ch] *= inv_m;
            }

            for bi in 0..n {
                for ch in 0..c {
                    let mu = mean[ch];
                    let base = (bi * c + ch) * s;
                    let slice = &xd[base..base + s];
                    let mut vsum = 0.0f32;
                    for &v in slice {
                        let d = v - mu;
                        vsum += d * d;
                    }
                    var[ch] += vsum;
                }
            }
            for ch in 0..c {
                var[ch] *= inv_m;
                inv_std[ch] = 1.0 / (var[ch] + self.eps).sqrt();
            }

            // 누적 이동 통계량 갱신 (Bessel 보정 적용)
            {
                let mom = self.momentum;
                let count_f = m_count as f32;
                let unbias = if count_f > 1.0 { count_f / (count_f - 1.0) } else { 1.0 };
                let mut rm = self.running_mean.data_mut();
                let mut rv = self.running_var.data_mut();
                for ch in 0..c {
                    rm[ch] = (1.0 - mom) * rm[ch] + mom * mean[ch];
                    rv[ch] = (1.0 - mom) * rv[ch] + mom * var[ch] * unbias;
                }
            }

            let mut y = vec![0.0f32; total];
            let mut xhat = if record_grad { vec![0.0f32; total] } else { Vec::new() };

            for bi in 0..n {
                for ch in 0..c {
                    let mu = mean[ch];
                    let istd = inv_std[ch];
                    let gam = gd[ch];
                    let bet = bd[ch];
                    let base = (bi * c + ch) * s;
                    let xs = &xd[base..base + s];
                    let ys = &mut y[base..base + s];
                    if record_grad {
                        let xhs = &mut xhat[base..base + s];
                        for i in 0..s {
                            let xh = (xs[i] - mu) * istd;
                            xhs[i] = xh;
                            ys[i] = xh * gam + bet;
                        }
                    } else {
                        let scale = gam * istd;
                        let shift = bet - mu * scale;
                        for i in 0..s {
                            ys[i] = xs[i] * scale + shift;
                        }
                    }
                }
            }
            drop(xd);
            drop(gd);
            drop(bd);

            let (xt, gt, bt) = (x.clone(), self.gamma.clone(), self.beta.clone());
            Tensor::from_op(
                y,
                x.shape().to_vec(),
                x.device(),
                vec![x.clone(), self.gamma.clone(), self.beta.clone()],
                "BatchNormBackward",
                Box::new(move |dy| {
                    let mut dgamma = vec![0.0f32; c];
                    let mut dbeta = vec![0.0f32; c];
                    for bi in 0..n {
                        for ch in 0..c {
                            let base = (bi * c + ch) * s;
                            let dys = &dy[base..base + s];
                            let xhs = &xhat[base..base + s];
                            let mut sb = 0.0f32;
                            let mut sg = 0.0f32;
                            for i in 0..s {
                                sb += dys[i];
                                sg += dys[i] * xhs[i];
                            }
                            dbeta[ch] += sb;
                            dgamma[ch] += sg;
                        }
                    }

                    let dx = xt.requires_grad().then(|| {
                        let gd = gt.data();
                        let mf = m_count as f32;
                        let mut dx = vec![0.0f32; total];
                        for bi in 0..n {
                            for ch in 0..c {
                                let factor = gd[ch] * inv_std[ch] / mf;
                                let db = dbeta[ch];
                                let dg = dgamma[ch];
                                let base = (bi * c + ch) * s;
                                let dys = &dy[base..base + s];
                                let xhs = &xhat[base..base + s];
                                let dxs = &mut dx[base..base + s];
                                for i in 0..s {
                                    dxs[i] = factor * (mf * dys[i] - db - xhs[i] * dg);
                                }
                            }
                        }
                        dx
                    });

                    vec![
                        dx,
                        gt.requires_grad().then_some(dgamma),
                        bt.requires_grad().then_some(dbeta),
                    ]
                }),
            )
        } else {
            // 평가(추론) 모드: 고정된 이동 통계량 사용
            let xd = x.data();
            let gd = self.gamma.data();
            let bd = self.beta.data();
            let rm = self.running_mean.data();
            let rv = self.running_var.data();

            let mut inv_std = vec![0.0f32; c];
            let mut scale = vec![0.0f32; c];
            let mut shift = vec![0.0f32; c];
            for ch in 0..c {
                inv_std[ch] = 1.0 / (rv[ch] + self.eps).sqrt();
                scale[ch] = gd[ch] * inv_std[ch];
                shift[ch] = bd[ch] - rm[ch] * scale[ch];
            }

            let mut y = vec![0.0f32; total];
            for bi in 0..n {
                for ch in 0..c {
                    let sc = scale[ch];
                    let sh = shift[ch];
                    let base = (bi * c + ch) * s;
                    let xs = &xd[base..base + s];
                    let ys = &mut y[base..base + s];
                    for i in 0..s {
                        ys[i] = xs[i] * sc + sh;
                    }
                }
            }
            let rm_vec = if record_grad { rm.clone() } else { Vec::new() };
            drop(xd);
            drop(gd);
            drop(bd);
            drop(rm);
            drop(rv);

            let (xt, gt, bt) = (x.clone(), self.gamma.clone(), self.beta.clone());
            Tensor::from_op(
                y,
                x.shape().to_vec(),
                x.device(),
                vec![x.clone(), self.gamma.clone(), self.beta.clone()],
                "BatchNormEvalBackward",
                Box::new(move |dy| {
                    let dx = xt.requires_grad().then(|| {
                        let mut dx = vec![0.0f32; total];
                        for bi in 0..n {
                            for ch in 0..c {
                                let sc = scale[ch];
                                let base = (bi * c + ch) * s;
                                for (dst, &g) in dx[base..base + s].iter_mut().zip(&dy[base..base + s]) {
                                    *dst = g * sc;
                                }
                            }
                        }
                        dx
                    });
                    let dgamma = gt.requires_grad().then(|| {
                        let xd = xt.data();
                        let mut dg = vec![0.0f32; c];
                        for bi in 0..n {
                            for ch in 0..c {
                                let base = (bi * c + ch) * s;
                                let mu = rm_vec[ch];
                                let istd = inv_std[ch];
                                let mut sum = 0.0f32;
                                for i in 0..s {
                                    sum += dy[base + i] * (xd[base + i] - mu) * istd;
                                }
                                dg[ch] += sum;
                            }
                        }
                        dg
                    });
                    let dbeta = bt.requires_grad().then(|| {
                        let mut db = vec![0.0f32; c];
                        for bi in 0..n {
                            for ch in 0..c {
                                let base = (bi * c + ch) * s;
                                db[ch] += dy[base..base + s].iter().sum::<f32>();
                            }
                        }
                        db
                    });
                    vec![dx, dgamma, dbeta]
                }),
            )
        }
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".into(), self.gamma.clone()), ("bias".into(), self.beta.clone())]
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        vec![("running_mean".into(), self.running_mean.clone()), ("running_var".into(), self.running_var.clone())]
    }

    fn name(&self) -> String {
        format!("BatchNorm{}({})", if self.spatial { "2d" } else { "1d" }, self.num_features)
    }

    fn set_training(&self, t: bool) {
        self.training.store(t, Ordering::Relaxed);
    }

    /// 채널별 아핀 변환으로 내보냅니다 (추론 최적화 시 이전 Linear/Conv 레이어와 융합됨).
    fn export(&self) -> Result<Vec<LayerSpec>, String> {
        let g = self.gamma.to_vec();
        let b = self.beta.to_vec();
        let rm = self.running_mean.to_vec();
        let rv = self.running_var.to_vec();
        let scale: Vec<f32> = (0..self.num_features).map(|i| g[i] / (rv[i] + self.eps).sqrt()).collect();
        let shift: Vec<f32> = (0..self.num_features).map(|i| b[i] - rm[i] * scale[i]).collect();
        Ok(vec![LayerSpec::ChannelAffine { scale, shift }])
    }
}

/// 마지막 차원을 따라 평균과 분산을 정규화하는 레이어 정규화 (Layer Normalization).
pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub dim: usize,
    pub eps: f32,
}

impl LayerNorm {
    /// 지정된 특징 차원 크기로 새 LayerNorm을 생성합니다.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "LayerNorm dimension must be positive");
        LayerNorm {
            gamma: Tensor::ones(&[dim]).requires_grad_(true),
            beta: Tensor::zeros(&[dim]).requires_grad_(true),
            dim,
            eps: 1e-5,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.shape().last(), Some(&self.dim), "LayerNorm last dimension mismatch");
        let mean = x.mean_dim(-1, true);
        let var = x.var_dim(-1, true);
        x.sub(&mean).div(&var.add_scalar(self.eps).sqrt()).mul(&self.gamma).add(&self.beta)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".into(), self.gamma.clone()), ("bias".into(), self.beta.clone())]
    }

    fn name(&self) -> String {
        format!("LayerNorm({})", self.dim)
    }
}

/// 평균 제곱근 레이어 정규화 (Root Mean Square Layer Normalization: RMSNorm).
///
/// LLaMA, Gemma, Mistral 등 최신 대형 언어 모델(LLM)에서 연산 효율성과 학습 안정성을 위해 널리 사용됩니다:
/// `y = (x / RMS(x)) * weight`, 여기서 `RMS(x) = sqrt(mean(x^2) + eps)`.
pub struct RMSNorm {
    pub weight: Tensor,
    pub dim: usize,
    pub eps: f32,
}

impl RMSNorm {
    /// 지정된 특징 차원 크기로 새 RMSNorm을 생성합니다.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "RMSNorm dimension must be positive");
        RMSNorm {
            weight: Tensor::ones(&[dim]).requires_grad_(true),
            dim,
            eps: 1e-6,
        }
    }

    /// 분모 안정화 상수 epsilon을 설정합니다.
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.shape().last(), Some(&self.dim), "RMSNorm last dimension mismatch");
        let rms = x.square().mean_dim(-1, true).add_scalar(self.eps).sqrt();
        x.div(&rms).mul(&self.weight)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".into(), self.weight.clone())]
    }

    fn name(&self) -> String {
        format!("RMSNorm({})", self.dim)
    }
}
