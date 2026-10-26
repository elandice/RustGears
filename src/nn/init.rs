//! Parameter initialisation schemes.

use crate::tensor::Tensor;

/// Kaiming / He uniform initialisation (good default for ReLU networks).
pub fn kaiming_uniform(shape: &[usize], fan_in: usize) -> Tensor {
    let bound = (6.0 / fan_in.max(1) as f32).sqrt();
    Tensor::uniform(shape, -bound, bound)
}

/// Xavier / Glorot uniform initialisation (good default for tanh / sigmoid).
pub fn xavier_uniform(shape: &[usize], fan_in: usize, fan_out: usize) -> Tensor {
    let bound = (6.0 / (fan_in + fan_out).max(1) as f32).sqrt();
    Tensor::uniform(shape, -bound, bound)
}

/// Normal initialisation `N(0, std^2)`.
pub fn normal(shape: &[usize], std: f32) -> Tensor {
    Tensor::randn(shape).detach_scaled(std)
}

/// Bias initialisation `U(-1/sqrt(fan_in), 1/sqrt(fan_in))`.
pub fn bias_uniform(shape: &[usize], fan_in: usize) -> Tensor {
    let bound = 1.0 / (fan_in.max(1) as f32).sqrt();
    Tensor::uniform(shape, -bound, bound)
}

impl Tensor {
    /// Non-differentiable `self * s` used during initialisation.
    pub(crate) fn detach_scaled(&self, s: f32) -> Tensor {
        Tensor::from_vec(self.data().iter().map(|v| v * s).collect(), self.shape())
    }
}
