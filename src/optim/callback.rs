//! Training loop callbacks: EarlyStopping, ModelCheckpoint, and GradientAccumulator.

use crate::nn::Module;
use crate::save::save_state_dict;
use crate::tensor::Tensor;
use std::io;
use std::path::{Path, PathBuf};

/// Optimization target direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyStoppingMode {
    /// Metric should be minimized (e.g. loss).
    Min,
    /// Metric should be maximized (e.g. accuracy, F1 score).
    Max,
}

/// Early stopping monitor to stop training when a monitored metric stops improving.
///
/// Optionally saves and restores the best model parameters.
pub struct EarlyStopping {
    mode: EarlyStoppingMode,
    patience: usize,
    min_delta: f32,
    restore_best_weights: bool,
    counter: usize,
    best_score: Option<f32>,
    early_stop: bool,
    best_weights: Option<Vec<(String, Vec<f32>)>>,
}

impl EarlyStopping {
    /// Create a new `EarlyStopping` instance with given patience.
    /// Default mode is `Min` with `min_delta = 0.0` and `restore_best_weights = true`.
    pub fn new(patience: usize) -> Self {
        EarlyStopping {
            mode: EarlyStoppingMode::Min,
            patience,
            min_delta: 0.0,
            restore_best_weights: true,
            counter: 0,
            best_score: None,
            early_stop: false,
            best_weights: None,
        }
    }

    /// Set optimization mode (`Min` or `Max`).
    pub fn mode(mut self, mode: EarlyStoppingMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set minimum change to qualify as an improvement.
    pub fn min_delta(mut self, delta: f32) -> Self {
        self.min_delta = delta.abs();
        self
    }

    /// Whether to restore model weights from the epoch with the best value.
    pub fn restore_best_weights(mut self, restore: bool) -> Self {
        self.restore_best_weights = restore;
        self
    }

    /// Check if the metric improved. Returns `true` if an improvement occurred.
    ///
    /// When `restore_best_weights` is active, snapshot of parameters is updated on improvement,
    /// and automatically restored if early stopping triggers.
    pub fn step(&mut self, score: f32, model: &dyn Module) -> bool {
        let is_improved = match self.best_score {
            None => true,
            Some(best) => match self.mode {
                EarlyStoppingMode::Min => score < (best - self.min_delta),
                EarlyStoppingMode::Max => score > (best + self.min_delta),
            },
        };

        if is_improved {
            self.best_score = Some(score);
            self.counter = 0;
            if self.restore_best_weights {
                let snapshot = model
                    .state_dict()
                    .into_iter()
                    .map(|(name, tensor)| (name, tensor.to_vec()))
                    .collect();
                self.best_weights = Some(snapshot);
            }
            true
        } else {
            self.counter += 1;
            if self.counter >= self.patience {
                self.early_stop = true;
                if self.restore_best_weights {
                    self.restore(model);
                }
            }
            false
        }
    }

    /// Restores the best weights to the model if recorded. Returns true if restored.
    pub fn restore(&self, model: &dyn Module) -> bool {
        if let Some(ref snapshot) = self.best_weights {
            for (name, tensor) in model.state_dict() {
                if let Some((_, data)) = snapshot.iter().find(|(n, _)| n == &name) {
                    tensor.copy_from_slice(data);
                }
            }
            true
        } else {
            false
        }
    }

    /// Whether early stopping has been triggered.
    pub fn should_stop(&self) -> bool {
        self.early_stop
    }

    /// Current count of consecutive epochs without improvement.
    pub fn counter(&self) -> usize {
        self.counter
    }

    /// Patience threshold.
    pub fn patience(&self) -> usize {
        self.patience
    }

    /// Best score observed so far.
    pub fn best_score(&self) -> Option<f32> {
        self.best_score
    }

    /// Reset early stopping state.
    pub fn reset(&mut self) {
        self.counter = 0;
        self.best_score = None;
        self.early_stop = false;
        self.best_weights = None;
    }
}

/// Callback that saves model checkpoints to disk when a monitored metric improves.
pub struct ModelCheckpoint {
    filepath: PathBuf,
    mode: EarlyStoppingMode,
    save_best_only: bool,
    best_score: Option<f32>,
}

impl ModelCheckpoint {
    /// Create a new checkpoint callback. Default is `save_best_only = true` and `mode = Min`.
    pub fn new(filepath: impl Into<PathBuf>) -> Self {
        ModelCheckpoint {
            filepath: filepath.into(),
            mode: EarlyStoppingMode::Min,
            save_best_only: true,
            best_score: None,
        }
    }

    pub fn mode(mut self, mode: EarlyStoppingMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn save_best_only(mut self, best_only: bool) -> Self {
        self.save_best_only = best_only;
        self
    }

    /// Record a new score and save the checkpoint if conditions are met.
    /// Returns `Ok(true)` if checkpoint was saved to disk.
    pub fn step(&mut self, score: f32, model: &dyn Module) -> io::Result<bool> {
        let is_improved = match self.best_score {
            None => true,
            Some(best) => match self.mode {
                EarlyStoppingMode::Min => score < best,
                EarlyStoppingMode::Max => score > best,
            },
        };

        if is_improved {
            self.best_score = Some(score);
        }

        if !self.save_best_only || is_improved {
            save_state_dict(model, &self.filepath)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn best_score(&self) -> Option<f32> {
        self.best_score
    }

    pub fn filepath(&self) -> &Path {
        &self.filepath
    }
}

/// Helper for gradient accumulation over multiple mini-batches.
///
/// Useful for simulating larger batch sizes when hardware memory is constrained.
pub struct GradientAccumulator {
    accumulation_steps: usize,
    current_step: usize,
}

impl GradientAccumulator {
    /// Create a new gradient accumulator that triggers updates every `accumulation_steps`.
    pub fn new(accumulation_steps: usize) -> Self {
        assert!(accumulation_steps > 0, "accumulation_steps must be > 0");
        GradientAccumulator {
            accumulation_steps,
            current_step: 0,
        }
    }

    /// Scales the loss by `1.0 / accumulation_steps` before backward pass.
    pub fn scale_loss(&self, loss: &Tensor) -> Tensor {
        loss.mul_scalar(1.0 / self.accumulation_steps as f32)
    }

    /// Advance one step and return true if optimizer step and zero_grad should be performed.
    pub fn step(&mut self) -> bool {
        self.current_step += 1;
        if self.current_step >= self.accumulation_steps {
            self.current_step = 0;
            true
        } else {
            false
        }
    }

    /// Reset step counter.
    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    /// Configured accumulation steps.
    pub fn accumulation_steps(&self) -> usize {
        self.accumulation_steps
    }
}
