//! Normalisation layers.

use super::Module;
use crate::autograd::no_grad;
use crate::inference::LayerSpec;
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};

/// Batch normalisation over the channel dimension (dim 1).
///
/// * `BatchNorm1d` – input `[N, C]`
/// * `BatchNorm2d` – input `[N, C, H, W]` (statistics over N, H, W)
pub struct BatchNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub num_features: usize,
    pub momentum: f32,
    pub eps: f32,
    spatial: bool,
    training: AtomicBool,
}

pub type BatchNorm1d = BatchNorm;

/// Alias constructor for the spatial (4-D) variant.
pub struct BatchNorm2d;

impl BatchNorm2d {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(num_features: usize) -> BatchNorm {
        BatchNorm::with_spatial(num_features, true)
    }
}

impl BatchNorm {
    pub fn new(num_features: usize) -> Self {
        Self::with_spatial(num_features, false)
    }

    fn with_spatial(num_features: usize, spatial: bool) -> Self {
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
        // bring channels to the last axis: [M, C]
        let (x2, back) = if self.spatial {
            assert_eq!(x.ndim(), 4, "BatchNorm2d expects [N,C,H,W]");
            let (n, h, w) = (x.shape()[0], x.shape()[2], x.shape()[3]);
            (x.permute(&[0, 2, 3, 1]).reshape(&[n * h * w, c]), Some((n, h, w)))
        } else {
            assert_eq!(x.ndim(), 2, "BatchNorm1d expects [N,C]");
            (x.clone(), None)
        };

        let xhat = if self.training.load(Ordering::Relaxed) {
            let mean = x2.mean_dim(0, true);
            let var = x2.var_dim(0, true);
            // update running statistics (not part of the graph)
            no_grad(|| {
                let m = self.momentum;
                let count = x2.shape()[0] as f32;
                let unbias = if count > 1.0 { count / (count - 1.0) } else { 1.0 };
                let (mv, vv) = (mean.to_vec(), var.to_vec());
                let mut rm = self.running_mean.data_mut();
                let mut rv = self.running_var.data_mut();
                for i in 0..c {
                    rm[i] = (1.0 - m) * rm[i] + m * mv[i];
                    rv[i] = (1.0 - m) * rv[i] + m * vv[i] * unbias;
                }
            });
            x2.sub(&mean).div(&var.add_scalar(self.eps).sqrt())
        } else {
            let rm = self.running_mean.clone();
            let rstd = Tensor::from_vec(self.running_var.data().iter().map(|v| (v + self.eps).sqrt()).collect(), &[c])
                .on(x.device());
            x2.sub(&rm).div(&rstd)
        };
        let y = xhat.mul(&self.gamma).add(&self.beta);

        match back {
            Some((n, h, w)) => y.reshape(&[n, h, w, c]).permute(&[0, 3, 1, 2]),
            None => y,
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

    /// Exported as a per-channel affine transform (folded into the previous
    /// Linear/Conv by the inference optimiser).
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

/// Layer normalisation over the last dimension.
pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub dim: usize,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
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

/// Root Mean Square Layer Normalization (RMSNorm).
///
/// Widely used in modern LLMs (e.g. LLaMA, Gemma, Mistral) for faster computation and training stability:
/// `y = (x / RMS(x)) * weight`, where `RMS(x) = sqrt(mean(x^2) + eps)`.
pub struct RMSNorm {
    pub weight: Tensor,
    pub dim: usize,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize) -> Self {
        RMSNorm {
            weight: Tensor::ones(&[dim]).requires_grad_(true),
            dim,
            eps: 1e-6,
        }
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
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
