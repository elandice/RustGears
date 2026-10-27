//! 학습률 스케줄러(Learning-Rate Schedulers) 구현 모듈.

use super::Optimizer;

/// 학습률 스케줄러 트레이트.
/// `optimizer.step()` 실행 후 에포크 또는 반복 단위마다 [`step`](Self::step)을 호출합니다.
pub trait LrScheduler {
    /// 옵티마이저의 학습률을 갱신합니다.
    fn step(&mut self, opt: &mut dyn Optimizer);
    /// 현재 스케줄러의 학습률을 반환합니다.
    fn current_lr(&self) -> f32;
}

/// 일정한 스텝 간격(`step_size`)마다 학습률에 감쇠 계수 `gamma`를 곱하는 단계별 스케줄러.
pub struct StepLr {
    base: f32,
    step_size: usize,
    gamma: f32,
    t: usize,
    lr: f32,
}

impl StepLr {
    /// 기본 학습률, 스텝 주기, 감쇠율을 지정하여 StepLr을 생성합니다.
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

/// 매 스텝마다 지수적으로 학습률을 감소시키는 지수 감쇠 스케줄러 (`lr = base * gamma^t`).
pub struct ExponentialLr {
    gamma: f32,
    lr: f32,
}

impl ExponentialLr {
    /// 기본 학습률과 감쇠 계수 `gamma`로 ExponentialLr을 생성합니다.
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

/// `t_max` 스텝에 걸쳐 코사인 함수 곡선을 따라 `base_lr`에서 `eta_min`까지 학습률을 점진적으로 낮추는 코사인 어닐링 스케줄러.
pub struct CosineAnnealingLr {
    base: f32,
    eta_min: f32,
    t_max: usize,
    t: usize,
    lr: f32,
}

impl CosineAnnealingLr {
    /// 기본 학습률, 최대 스텝 수 `t_max`, 최저 학습률 `eta_min`을 지정하여 생성합니다.
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

/// 초기 `warmup` 스텝 동안 학습률을 0에서 선형으로 끌어올린 후 코사인 감쇠를 적용하는 웜업 코사인 스케줄러 (트랜스포머 표준).
pub struct WarmupCosineLr {
    base: f32,
    warmup: usize,
    total: usize,
    t: usize,
    lr: f32,
}

impl WarmupCosineLr {
    /// 기본 학습률, 웜업 스텝 수, 전체 총 스텝 수를 지정하여 생성합니다.
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

/// `ReduceLROnPlateau`의 모니터링 지표 목표 방향 열거형
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlateauMode {
    /// 지표 값이 작아질수록 개선으로 판단 (예: 검증 손실)
    Min,
    /// 지표 값이 커질수록 개선으로 판단 (예: 정확도, F1 점수)
    Max,
}

/// 모니터링 중인 평가 지표가 정체(Plateau) 상태에 빠졌을 때 학습률을 감소시키는 스케줄러.
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
    /// 기본 학습률로 ReduceLROnPlateau 인스턴스를 생성합니다.
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

    /// 모니터링 모드(`Min` 또는 `Max`)를 설정합니다.
    pub fn mode(mut self, mode: PlateauMode) -> Self {
        self.mode = mode;
        self
    }

    /// 학습률 감소 배율(`factor`)을 설정합니다 (`new_lr = lr * factor`).
    pub fn factor(mut self, factor: f32) -> Self {
        assert!(factor > 0.0 && factor < 1.0, "factor must be in (0.0, 1.0)");
        self.factor = factor;
        self
    }

    /// 개선이 없을 때 학습률 감소 전 대기할 에포크 수(`patience`)를 설정합니다.
    pub fn patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// 개선으로 인정할 최소 변화량(`threshold`)을 설정합니다.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.abs();
        self
    }

    /// 감소 가능한 최소 학습률 하한선(`min_lr`)을 설정합니다.
    pub fn min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// 학습률 감소 후 정상 동작 재개 전 대기할 쿨다운 에포크 수를 설정합니다.
    pub fn cooldown(mut self, cooldown: usize) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// 현재 학습률을 반환합니다.
    pub fn current_lr(&self) -> f32 {
        self.lr
    }

    /// 개선되지 않은 연속 에포크 수를 반환합니다.
    pub fn num_bad_epochs(&self) -> usize {
        self.num_bad_epochs
    }

    /// 지금까지 관측된 최고 지표 점수를 반환합니다.
    pub fn best(&self) -> Option<f32> {
        self.best
    }

    /// 새로운 평가 지표 값을 입력받아 스케줄러 상태를 갱신하고 필요 시 옵티마이저 학습률을 감소시킵니다.
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
