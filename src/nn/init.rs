//! 신경망 가중치 매개변수 초기화 기법 모듈.

use crate::tensor::Tensor;

/// Kaiming / He 균등 초기화 (ReLU 계열 신경망에 최적화된 기본값).
pub fn kaiming_uniform(shape: &[usize], fan_in: usize) -> Tensor {
    let bound = (6.0 / fan_in.max(1) as f32).sqrt();
    Tensor::uniform(shape, -bound, bound)
}

/// Xavier / Glorot 균등 초기화 (tanh / 시그모이드 신경망에 적합한 기본값).
pub fn xavier_uniform(shape: &[usize], fan_in: usize, fan_out: usize) -> Tensor {
    let bound = (6.0 / (fan_in + fan_out).max(1) as f32).sqrt();
    Tensor::uniform(shape, -bound, bound)
}

/// 평균 0, 표준편차 `std`를 따르는 정규 분포 초기화 `N(0, std^2)`.
pub fn normal(shape: &[usize], std: f32) -> Tensor {
    Tensor::randn(shape).detach_scaled(std)
}

/// 편향(Bias) 균등 초기화 `U(-1/sqrt(fan_in), 1/sqrt(fan_in))`.
pub fn bias_uniform(shape: &[usize], fan_in: usize) -> Tensor {
    let bound = 1.0 / (fan_in.max(1) as f32).sqrt();
    Tensor::uniform(shape, -bound, bound)
}

impl Tensor {
    /// 초기화 시 그래프 기록 없이 스칼라 값을 곱하는 헬퍼 메서드 (`self * s`).
    pub(crate) fn detach_scaled(&self, s: f32) -> Tensor {
        Tensor::from_vec(self.data().iter().map(|v| v * s).collect(), self.shape())
    }
}
