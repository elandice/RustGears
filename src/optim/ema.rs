//! Exponential moving averages for evaluating a smoothed model.

use crate::nn::Module;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Detached averages of parameters and floating-point buffers, including
/// BatchNorm statistics. Call `update` after each optimizer step and `copy_to`
/// on a separate evaluation model to keep training weights intact.
pub struct ModelEMA {
    decay: f32,
    updates: usize,
    state: Vec<(String, Tensor)>,
}

impl ModelEMA {
    pub fn new(model: &dyn Module, decay: f32) -> Self {
        assert!(decay.is_finite() && (0.0..1.0).contains(&decay), "EMA decay must be in [0, 1)");
        let state = model.state_dict().into_iter().map(|(name, tensor)| (name, tensor.detach())).collect();
        let ema = Self { decay, updates: 0, state };
        ema.matching_state(model).expect("ModelEMA requires unique state names");
        ema
    }

    /// `average = decay * average + (1 - decay) * current`.
    /// Name and shape mismatches return an error before any average changes.
    pub fn update(&mut self, model: &dyn Module) -> Result<(), String> {
        let current = self.matching_state(model)?;
        for ((_, average), tensor) in self.state.iter().zip(current) {
            let data = tensor.to_vec();
            for (a, value) in average.data_mut().iter_mut().zip(data) {
                *a = self.decay * *a + (1.0 - self.decay) * value;
            }
        }
        self.updates += 1;
        Ok(())
    }

    /// Copy averages into a compatible model, preserving its device and gradients.
    /// A mismatched model is left unchanged.
    pub fn copy_to(&self, model: &dyn Module) -> Result<(), String> {
        let target = self.matching_state(model)?;
        for ((_, average), tensor) in self.state.iter().zip(target) {
            tensor.copy_from_slice(&average.to_vec());
        }
        Ok(())
    }

    pub fn num_updates(&self) -> usize {
        self.updates
    }

    /// Independent snapshots with no autograd history.
    pub fn state_dict(&self) -> Vec<(String, Tensor)> {
        self.state.iter().map(|(name, tensor)| (name.clone(), tensor.detach())).collect()
    }

    fn matching_state(&self, model: &dyn Module) -> Result<Vec<Tensor>, String> {
        let state = model.state_dict();
        let count = state.len();
        let mut by_name: HashMap<_, _> = state.into_iter().collect();
        if count != by_name.len() || count != self.state.len() {
            return Err("ModelEMA: state count mismatch or duplicate names".into());
        }
        self.state
            .iter()
            .map(|(name, average)| {
                let tensor = by_name.remove(name).ok_or_else(|| format!("ModelEMA: missing state '{name}'"))?;
                if tensor.shape() != average.shape() {
                    return Err(format!("ModelEMA: shape mismatch for '{name}'"));
                }
                Ok(tensor)
            })
            .collect()
    }
}
