//! Optimisation algorithms (like `torch.optim`).

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

/// Common optimizer interface.
pub trait Optimizer {
    /// Apply one update using the accumulated gradients.
    fn step(&mut self);
    /// Parameters managed by this optimizer.
    fn params(&self) -> &[Tensor];
    fn lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);

    /// Clear gradients of all managed parameters.
    fn zero_grad(&self) {
        for p in self.params() {
            p.zero_grad();
        }
    }
}

/// Update each parameter (in parallel) with `f(param_index, data, grad)`.
fn for_each_param(params: &[Tensor], f: impl Fn(usize, &mut [f32], &[f32]) + Sync) {
    params.par_iter().enumerate().for_each(|(i, p)| {
        if let Some(g) = p.grad_vec() {
            let mut d = p.data_mut();
            f(i, &mut d, &g);
        }
    });
}

// ----------------------------------------------------------------------
// SGD
// ----------------------------------------------------------------------

/// Stochastic gradient descent with optional momentum, Nesterov and L2 decay.
pub struct Sgd {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    nesterov: bool,
    velocity: Vec<std::sync::Mutex<Vec<f32>>>,
}

impl Sgd {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let velocity = params.iter().map(|p| std::sync::Mutex::new(vec![0.0; p.numel()])).collect();
        Sgd { params, lr, momentum: 0.0, weight_decay: 0.0, nesterov: false, velocity }
    }
    pub fn momentum(mut self, m: f32) -> Self {
        self.momentum = m;
        self
    }
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
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

/// Adam (Kingma & Ba, 2015). With `decoupled = true` this is AdamW.
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
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let state = params.iter().map(|p| std::sync::Mutex::new((vec![0.0; p.numel()], vec![0.0; p.numel()]))).collect();
        Adam { params, lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0, decoupled: false, t: 0, state }
    }
    pub fn betas(mut self, b1: f32, b2: f32) -> Self {
        self.beta1 = b1;
        self.beta2 = b2;
        self
    }
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

/// AdamW: Adam with decoupled weight decay (Loshchilov & Hutter, 2019).
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

pub struct RmsProp {
    params: Vec<Tensor>,
    lr: f32,
    alpha: f32,
    eps: f32,
    sq: Vec<std::sync::Mutex<Vec<f32>>>,
}

impl RmsProp {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let sq = params.iter().map(|p| std::sync::Mutex::new(vec![0.0; p.numel()])).collect();
        RmsProp { params, lr, alpha: 0.99, eps: 1e-8, sq }
    }
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
// utilities
// ----------------------------------------------------------------------

/// Clip the global L2 norm of the gradients to `max_norm`. Returns the norm
/// before clipping.
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
