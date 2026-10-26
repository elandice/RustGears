//! Learning-rate schedulers.

use super::Optimizer;

/// Call [`LrScheduler::step`] once per epoch (or per iteration) after
/// `optimizer.step()`.
pub trait LrScheduler {
    fn step(&mut self, opt: &mut dyn Optimizer);
    fn current_lr(&self) -> f32;
}

/// Multiply the LR by `gamma` every `step_size` steps.
pub struct StepLr {
    base: f32,
    step_size: usize,
    gamma: f32,
    t: usize,
    lr: f32,
}

impl StepLr {
    pub fn new(base_lr: f32, step_size: usize, gamma: f32) -> Self {
        StepLr { base: base_lr, step_size, gamma, t: 0, lr: base_lr }
    }
}

impl LrScheduler for StepLr {
    fn step(&mut self, opt: &mut dyn Optimizer) {
        self.t += 1;
        self.lr = self.base * self.gamma.powi((self.t / self.step_size) as i32);
        opt.set_lr(self.lr);
    }
    fn current_lr(&self) -> f32 {
        self.lr
    }
}

/// `lr = base * gamma^t`.
pub struct ExponentialLr {
    gamma: f32,
    lr: f32,
}

impl ExponentialLr {
    pub fn new(base_lr: f32, gamma: f32) -> Self {
        ExponentialLr { gamma, lr: base_lr }
    }
}

impl LrScheduler for ExponentialLr {
    fn step(&mut self, opt: &mut dyn Optimizer) {
        self.lr *= self.gamma;
        opt.set_lr(self.lr);
    }
    fn current_lr(&self) -> f32 {
        self.lr
    }
}

/// Cosine annealing from `base_lr` to `eta_min` over `t_max` steps.
pub struct CosineAnnealingLr {
    base: f32,
    eta_min: f32,
    t_max: usize,
    t: usize,
    lr: f32,
}

impl CosineAnnealingLr {
    pub fn new(base_lr: f32, t_max: usize, eta_min: f32) -> Self {
        CosineAnnealingLr { base: base_lr, eta_min, t_max, t: 0, lr: base_lr }
    }
}

impl LrScheduler for CosineAnnealingLr {
    fn step(&mut self, opt: &mut dyn Optimizer) {
        self.t = (self.t + 1).min(self.t_max);
        let c = (std::f32::consts::PI * self.t as f32 / self.t_max as f32).cos();
        self.lr = self.eta_min + 0.5 * (self.base - self.eta_min) * (1.0 + c);
        opt.set_lr(self.lr);
    }
    fn current_lr(&self) -> f32 {
        self.lr
    }
}

/// Linear warm-up for `warmup` steps followed by cosine decay (Transformer recipe).
pub struct WarmupCosineLr {
    base: f32,
    warmup: usize,
    total: usize,
    t: usize,
    lr: f32,
}

impl WarmupCosineLr {
    pub fn new(base_lr: f32, warmup: usize, total: usize) -> Self {
        WarmupCosineLr { base: base_lr, warmup, total, t: 0, lr: base_lr / warmup.max(1) as f32 }
    }
}

impl LrScheduler for WarmupCosineLr {
    fn step(&mut self, opt: &mut dyn Optimizer) {
        self.t += 1;
        self.lr = if self.t < self.warmup {
            self.base * (self.t + 1) as f32 / self.warmup as f32
        } else {
            let p = (self.t - self.warmup) as f32 / (self.total - self.warmup).max(1) as f32;
            0.5 * self.base * (1.0 + (std::f32::consts::PI * p.min(1.0)).cos())
        };
        opt.set_lr(self.lr);
    }
    fn current_lr(&self) -> f32 {
        self.lr
    }
}

/// Optimization target direction for `ReduceLROnPlateau`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlateauMode {
    Min,
    Max,
}

/// Reduce learning rate when a monitored metric has stopped improving.
pub struct ReduceLROnPlateau {
    mode: PlateauMode,
    factor: f32,
    patience: usize,
    threshold: f32,
    min_lr: f32,
    cooldown: usize,
    cooldown_counter: usize,
    best: Option<f32>,
    num_bad_epochs: usize,
    lr: f32,
}

impl ReduceLROnPlateau {
    pub fn new(base_lr: f32) -> Self {
        ReduceLROnPlateau {
            mode: PlateauMode::Min,
            factor: 0.1,
            patience: 10,
            threshold: 1e-4,
            min_lr: 0.0,
            cooldown: 0,
            cooldown_counter: 0,
            best: None,
            num_bad_epochs: 0,
            lr: base_lr,
        }
    }

    pub fn mode(mut self, mode: PlateauMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn factor(mut self, factor: f32) -> Self {
        assert!(factor > 0.0 && factor < 1.0, "factor must be in (0.0, 1.0)");
        self.factor = factor;
        self
    }

    pub fn patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.abs();
        self
    }

    pub fn min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }

    pub fn cooldown(mut self, cooldown: usize) -> Self {
        self.cooldown = cooldown;
        self
    }

    pub fn current_lr(&self) -> f32 {
        self.lr
    }

    pub fn num_bad_epochs(&self) -> usize {
        self.num_bad_epochs
    }

    pub fn best(&self) -> Option<f32> {
        self.best
    }

    /// Step the scheduler with the latest evaluated metric.
    pub fn step(&mut self, opt: &mut dyn Optimizer, metric: f32) {
        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            self.num_bad_epochs = 0;
            return;
        }

        let is_improved = match self.best {
            None => true,
            Some(best) => match self.mode {
                PlateauMode::Min => metric < (best - self.threshold),
                PlateauMode::Max => metric > (best + self.threshold),
            },
        };

        if is_improved {
            self.best = Some(metric);
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
            if self.num_bad_epochs >= self.patience {
                let new_lr = (self.lr * self.factor).max(self.min_lr);
                if new_lr < self.lr {
                    self.lr = new_lr;
                    opt.set_lr(self.lr);
                    self.cooldown_counter = self.cooldown;
                    self.num_bad_epochs = 0;
                }
            }
        }
    }
}
