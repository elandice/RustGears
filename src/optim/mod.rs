//! 신경망 최적화 알고리즘 모듈 (`torch.optim`과 유사).
//!
//! SGD(모멘텀, Nesterov 지원), Adam, AdamW(분리 가중치 감쇠), RMSprop 옵티마이저와
//! 학습률 스케줄러, 콜백, 기울기 클리핑(`clip_grad_norm`)을 제공합니다.

pub mod callback;
pub mod ema;
pub mod progress;
pub mod scheduler;

pub use callback::{EarlyStopping, EarlyStoppingMode, GradientAccumulator, ModelCheckpoint};
pub use ema::ModelEMA;
pub use progress::ProgressBar;
pub use scheduler::{
    CosineAnnealingLr, ExponentialLr, LrScheduler, PlateauMode, ReduceLROnPlateau, StepLr, WarmupCosineLr,
};

use crate::tensor::Tensor;
use rayon::prelude::*;

/// 모든 옵티마이저의 기본 인터페이스 트레이트.
pub trait Optimizer {
    /// 계산된 기울기를 바탕으로 가중치를 1회 갱신합니다.
    fn step(&mut self);
    /// 이 옵티마이저가 관리하는 파라미터 슬라이스를 반환합니다.
    fn params(&self) -> &[Tensor];
    /// 현재 학습률을 반환합니다.
    fn lr(&self) -> f32;
    /// 학습률을 설정합니다.
    fn set_lr(&mut self, lr: f32);

    /// 관리 중인 모든 파라미터의 기울기를 0으로 초기화합니다.
    fn zero_grad(&self) {
        for p in self.params() {
            p.zero_grad();
        }
    }
}

/// Rayon 멀티스레드 병렬 처리를 통해 각 파라미터를 동시 갱신하는 헬퍼 함수
fn for_each_param(params: &[Tensor], f: impl Fn(usize, &mut [f32], &[f32]) + Sync) {
    params.par_iter().enumerate().for_each(|(i, p)| {
        if let Some(g) = p.grad_vec() {
            let mut d = p.data_mut();
            f(i, &mut d, &g);
        }
    });
}

// ----------------------------------------------------------------------
// 확률적 경사 하강법 (SGD)
// ----------------------------------------------------------------------

/// 확률적 경사 하강법 (Stochastic Gradient Descent) 옵티마이저.
/// 모멘텀(Momentum), 네스테로프 가속 경사(Nesterov), L2 가중치 감쇠(Weight Decay)를 지원합니다.
pub struct Sgd {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    nesterov: bool,
    velocity: Vec<std::sync::Mutex<Vec<f32>>>,
}

impl Sgd {
    /// 파라미터 목록과 학습률로 새 SGD 옵티마이저를 생성합니다.
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let velocity = params.iter().map(|p| std::sync::Mutex::new(vec![0.0; p.numel()])).collect();
        Sgd { params, lr, momentum: 0.0, weight_decay: 0.0, nesterov: false, velocity }
    }

    /// 모멘텀 계수를 설정합니다 (기본: 0.0).
    pub fn momentum(mut self, m: f32) -> Self {
        self.momentum = m;
        self
    }

    /// L2 가중치 감쇠 계수를 설정합니다 (기본: 0.0).
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Nesterov 가속 모멘텀 사용 여부를 설정합니다 (기본: false).
    pub fn nesterov(mut self, n: bool) -> Self {
        self.nesterov = n;
        self
    }
}

impl Optimizer for Sgd {
    fn step(&mut self) {
        let (lr, mom, wd, nest) = (self.lr, self.momentum, self.weight_decay, self.nesterov);
        let vel = &self.velocity;
        for_each_param(&self.params, |i, w, g| {
            let mut v = vel[i].lock().unwrap();
            for j in 0..w.len() {
                let mut gj = g[j] + wd * w[j];
                if mom != 0.0 {
                    v[j] = mom * v[j] + gj;
                    gj = if nest { gj + mom * v[j] } else { v[j] };
                }
                w[j] -= lr * gj;
            }
        });
    }
    fn params(&self) -> &[Tensor] {
        &self.params
    }
    fn lr(&self) -> f32 {
        self.lr
    }
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

// ----------------------------------------------------------------------
// Adam / AdamW
// ----------------------------------------------------------------------

/// Adam 옵티마이저 (Kingma & Ba, 2015). `decoupled = true`인 경우 AdamW로 동작합니다.
pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    decoupled: bool,
    t: i32,
    state: Vec<std::sync::Mutex<(Vec<f32>, Vec<f32>)>>,
}

impl Adam {
    /// 파라미터 목록과 학습률로 새 Adam 옵티마이저를 생성합니다.
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let state = params.iter().map(|p| std::sync::Mutex::new((vec![0.0; p.numel()], vec![0.0; p.numel()]))).collect();
        Adam { params, lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0, decoupled: false, t: 0, state }
    }

    /// 모멘텀 계수 beta1, beta2를 설정합니다 (기본값: 0.9, 0.999).
    pub fn betas(mut self, b1: f32, b2: f32) -> Self {
        self.beta1 = b1;
        self.beta2 = b2;
        self
    }

    /// 수치적 안정성을 위한 epsilon 값을 설정합니다 (기본값: 1e-8).
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// 가중치 감쇠 계수를 설정합니다.
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

/// AdamW 옵티마이저: L2 정규화 대신 분리된 가중치 감쇠(Decoupled Weight Decay)를 적용한 Adam (Loshchilov & Hutter, 2019).
pub struct AdamW;

impl AdamW {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(params: Vec<Tensor>, lr: f32, weight_decay: f32) -> Adam {
        let mut a = Adam::new(params, lr).weight_decay(weight_decay);
        a.decoupled = true;
        a
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.t += 1;
        let (lr, b1, b2, eps, wd, dec) = (self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, self.decoupled);
        let bc1 = 1.0 - b1.powi(self.t);
        let bc2 = 1.0 - b2.powi(self.t);
        let state = &self.state;
        for_each_param(&self.params, |i, w, g| {
            let mut st = state[i].lock().unwrap();
            let (m, v) = &mut *st;
            for j in 0..w.len() {
                let mut gj = g[j];
                if dec {
                    w[j] -= lr * wd * w[j];
                } else {
                    gj += wd * w[j];
                }
                m[j] = b1 * m[j] + (1.0 - b1) * gj;
                v[j] = b2 * v[j] + (1.0 - b2) * gj * gj;
                let mh = m[j] / bc1;
                let vh = v[j] / bc2;
                w[j] -= lr * mh / (vh.sqrt() + eps);
            }
        });
    }
    fn params(&self) -> &[Tensor] {
        &self.params
    }
    fn lr(&self) -> f32 {
        self.lr
    }
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

// ----------------------------------------------------------------------
// RMSprop
// ----------------------------------------------------------------------

/// RMSprop 옵티마이저 (이동 평균 제곱 기울기를 활용한 적응형 학습률).
pub struct RmsProp {
    params: Vec<Tensor>,
    lr: f32,
    alpha: f32,
    eps: f32,
    sq: Vec<std::sync::Mutex<Vec<f32>>>,
}

impl RmsProp {
    /// 파라미터 목록과 학습률로 새 RmsProp 옵티마이저를 생성합니다.
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let sq = params.iter().map(|p| std::sync::Mutex::new(vec![0.0; p.numel()])).collect();
        RmsProp { params, lr, alpha: 0.99, eps: 1e-8, sq }
    }

    /// 이동 평균 평활 계수 alpha를 설정합니다 (기본값: 0.99).
    pub fn alpha(mut self, a: f32) -> Self {
        self.alpha = a;
        self
    }
}

impl Optimizer for RmsProp {
    fn step(&mut self) {
        let (lr, a, eps) = (self.lr, self.alpha, self.eps);
        let sq = &self.sq;
        for_each_param(&self.params, |i, w, g| {
            let mut s = sq[i].lock().unwrap();
            for j in 0..w.len() {
                s[j] = a * s[j] + (1.0 - a) * g[j] * g[j];
                w[j] -= lr * g[j] / (s[j].sqrt() + eps);
            }
        });
    }
    fn params(&self) -> &[Tensor] {
        &self.params
    }
    fn lr(&self) -> f32 {
        self.lr
    }
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

// ----------------------------------------------------------------------
// 최적화 유틸리티
// ----------------------------------------------------------------------

/// 모든 파라미터의 전역 L2 그래디언트 노름(Norm)을 `max_norm` 이하로 클리핑(제한)합니다.
/// 클리핑 수행 전의 원래 노름 값을 반환합니다.
pub fn clip_grad_norm(params: &[Tensor], max_norm: f32) -> f32 {
    let total: f32 = params
        .iter()
        .filter_map(|p| p.grad_vec())
        .map(|g| g.iter().map(|v| v * v).sum::<f32>())
        .sum::<f32>()
        .sqrt();
    if total > max_norm {
        let s = max_norm / (total + 1e-6);
        for p in params {
            p.with_grad_mut(|g| g.iter_mut().for_each(|v| *v *= s));
        }
    }
    total
}
