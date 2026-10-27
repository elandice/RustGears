//! 학습 루프 콜백 모듈: 조기 종료(EarlyStopping), 체크포인트 저장(ModelCheckpoint), 기울기 누적(GradientAccumulator).

use crate::nn::Module;
use crate::save::save_state_dict;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

/// 평가 지표의 최적화 목표 방향 열거형
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyStoppingMode {
    /// 지표 값이 작아져야 개선됨 (예: 손실값)
    Min,
    /// 지표 값이 커져야 개선됨 (예: 정확도, F1 점수)
    Max,
}

/// 모니터링 지표가 일정 기간(patience) 동안 개선되지 않으면 학습을 조기 중단하는 조기 종료 콜백.
///
/// 설정에 따라 가장 우수한 성적을 낸 시점의 모델 파라미터를 메모리에 보관하고 자동 복원합니다.
pub struct EarlyStopping {
    mode: EarlyStoppingMode,
    patience: usize,
    min_delta: f32,
    restore_best_weights: bool,
    counter: usize,
    best_score: Option<f32>,
    early_stop: bool,
    best_weights: Option<Vec<(String, Tensor)>>,
}

impl EarlyStopping {
    /// 지정된 대기 에포크 수(`patience`)로 새 EarlyStopping 인스턴스를 생성합니다.
    /// 기본 모드는 `Min`, `min_delta = 0.0`, `restore_best_weights = true`입니다.
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

    /// 최적화 모드(`Min` 또는 `Max`)를 설정합니다.
    pub fn mode(mut self, mode: EarlyStoppingMode) -> Self {
        self.mode = mode;
        self
    }

    /// 개선으로 판정할 최소 변화량(`min_delta`)을 설정합니다.
    pub fn min_delta(mut self, delta: f32) -> Self {
        assert!(delta.is_finite(), "min_delta must be finite");
        self.min_delta = delta.abs();
        self
    }

    /// 조기 종료 시 최고 성능 에포크의 모델 가중치로 복원할지 여부를 설정합니다.
    pub fn restore_best_weights(mut self, restore: bool) -> Self {
        self.restore_best_weights = restore;
        self
    }

    /// 현재 에포크의 평가 점수를 확인합니다. 개선이 이루어졌다면 `true`를 반환합니다.
    ///
    /// `restore_best_weights`가 켜져 있으면 개선 시 스냅샷이 갱신되며,
    /// 조기 중단이 발동할 때 자동으로 최고 가중치로 복원됩니다.
    pub fn step(&mut self, score: f32, model: &dyn Module) -> bool {
        let is_improved = score.is_finite() && match self.best_score {
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
                    .map(|(name, tensor)| (name, tensor.detach()))
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

    /// 보관된 최고 가중치와 버퍼 스냅샷을 모델로 복원합니다.
    pub fn restore(&self, model: &dyn Module) -> bool {
        let Some(snapshot) = &self.best_weights else { return false };
        let state = model.state_dict();
        if state.len() != snapshot.len() {
            return false;
        }
        let mut by_name: HashMap<_, _> = state.into_iter().collect();
        if by_name.len() != snapshot.len() {
            return false;
        }
        let mut compatible = Vec::with_capacity(snapshot.len());
        for (name, saved) in snapshot {
            let Some(tensor) = by_name.remove(name) else { return false };
            if tensor.shape() != saved.shape() {
                return false;
            }
            compatible.push((tensor, saved));
        }
        for (tensor, saved) in compatible {
            tensor.copy_from_slice(&saved.data());
        }
        true
    }

    /// 조기 종료 조건이 충족되어 학습을 중단해야 하는지 여부를 반환합니다.
    pub fn should_stop(&self) -> bool {
        self.early_stop
    }

    /// 지표 개선이 없었던 연속 에포크 카운트를 반환합니다.
    pub fn counter(&self) -> usize {
        self.counter
    }

    /// 설정된 인내 한도(patience)를 반환합니다.
    pub fn patience(&self) -> usize {
        self.patience
    }

    /// 지금까지 기록된 최고 점수를 반환합니다.
    pub fn best_score(&self) -> Option<f32> {
        self.best_score
    }

    /// 조기 종료 상태를 초기화합니다.
    pub fn reset(&mut self) {
        self.counter = 0;
        self.best_score = None;
        self.early_stop = false;
        self.best_weights = None;
    }
}

/// 모니터링 중인 평가 지표가 개선될 때마다 모델 가중치를 디스크 파일로 저장하는 체크포인트 콜백.
pub struct ModelCheckpoint {
    filepath: PathBuf,
    mode: EarlyStoppingMode,
    save_best_only: bool,
    best_score: Option<f32>,
}

impl ModelCheckpoint {
    /// 지정된 파일 경로로 새 ModelCheckpoint 인스턴스를 생성합니다 (기본: `save_best_only = true`, `mode = Min`).
    pub fn new(filepath: impl Into<PathBuf>) -> Self {
        ModelCheckpoint {
            filepath: filepath.into(),
            mode: EarlyStoppingMode::Min,
            save_best_only: true,
            best_score: None,
        }
    }

    /// 최적화 모드(`Min` 또는 `Max`)를 설정합니다.
    pub fn mode(mut self, mode: EarlyStoppingMode) -> Self {
        self.mode = mode;
        self
    }

    /// 최고 성능을 기록할 때만 저장할지 여부를 설정합니다.
    pub fn save_best_only(mut self, best_only: bool) -> Self {
        self.save_best_only = best_only;
        self
    }

    /// 평가 점수를 기록하고 조건 충족 시 체크포인트 파일을 저장합니다. 파일이 저장되면 `Ok(true)`를 반환합니다.
    pub fn step(&mut self, score: f32, model: &dyn Module) -> io::Result<bool> {
        if !score.is_finite() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "checkpoint score must be finite"));
        }
        let is_improved = match self.best_score {
            None => true,
            Some(best) => match self.mode {
                EarlyStoppingMode::Min => score < best,
                EarlyStoppingMode::Max => score > best,
            },
        };

        if !self.save_best_only || is_improved {
            save_state_dict(model, &self.filepath)?;
            if is_improved {
                self.best_score = Some(score);
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// 지금까지 기록된 최고 점수를 반환합니다.
    pub fn best_score(&self) -> Option<f32> {
        self.best_score
    }

    /// 저장 경로를 반환합니다.
    pub fn filepath(&self) -> &Path {
        &self.filepath
    }
}

/// 여러 미니배치에 걸쳐 기울기를 누적하여 대용량 배치 크기를 시뮬레이션하는 유틸리티.
/// GPU/CPU 메모리 한계로 인해 큰 배치를 직접 처리할 수 없을 때 유용합니다.
pub struct GradientAccumulator {
    accumulation_steps: usize,
    current_step: usize,
}

impl GradientAccumulator {
    /// 지정된 누적 스텝 수로 새 기울기 누적기를 생성합니다.
    pub fn new(accumulation_steps: usize) -> Self {
        assert!(accumulation_steps > 0, "accumulation_steps must be > 0");
        GradientAccumulator {
            accumulation_steps,
            current_step: 0,
        }
    }

    /// 역전파 전에 손실(Loss) 텐서를 `1.0 / accumulation_steps`로 스케일링합니다.
    pub fn scale_loss(&self, loss: &Tensor) -> Tensor {
        loss.mul_scalar(1.0 / self.accumulation_steps as f32)
    }

    /// 한 스텝을 진행하고, 옵티마이저의 `step()` 및 `zero_grad()`를 실행해야 할 시점이면 `true`를 반환합니다.
    pub fn step(&mut self) -> bool {
        self.current_step += 1;
        if self.current_step >= self.accumulation_steps {
            self.current_step = 0;
            true
        } else {
            false
        }
    }

    /// 누적 스텝 카운터를 초기화합니다.
    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    /// 설정된 누적 스텝 수를 반환합니다.
    pub fn accumulation_steps(&self) -> usize {
        self.accumulation_steps
    }
}
