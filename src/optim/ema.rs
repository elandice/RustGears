//! 모델 평가를 위한 가중치 지수 이동 평균(Exponential Moving Average: EMA) 모듈.

use crate::nn::Module;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// 학습 가중치와 분리된 모델 파라미터 및 부동소수점 버퍼(BatchNorm 통계량 포함)의 지수 이동 평균.
/// 각 옵티마이저 step() 호출 직후 `update`를 호출하고, 평가 모델에 `copy_to`를 수행하여 가중치를 적용합니다.
pub struct ModelEMA {
    decay: f32,
    updates: usize,
    state: Vec<(String, Tensor)>,
}

impl ModelEMA {
    /// 대상 모델과 감쇠율(`decay`: `[0, 1)`)로 새 ModelEMA 인스턴스를 생성합니다.
    pub fn new(model: &dyn Module, decay: f32) -> Self {
        assert!(decay.is_finite() && (0.0..1.0).contains(&decay), "EMA decay must be in [0, 1)");
        let state = model.state_dict().into_iter().map(|(name, tensor)| (name, tensor.detach())).collect();
        let ema = Self { decay, updates: 0, state };
        ema.matching_state(model).expect("ModelEMA requires unique state names");
        ema
    }

    /// 현재 모델 가중치를 기반으로 지수 이동 평균을 갱신합니다:
    /// `average = decay * average + (1 - decay) * current`.
    pub fn update(&mut self, model: &dyn Module) -> Result<(), String> {
        let current = self.matching_state(model)?;
        let decay = self.decay;
        let one_minus_decay = 1.0 - decay;
        for ((_, average), tensor) in self.state.iter().zip(current) {
            if average.id() == tensor.id() {
                continue;
            }
            let src = tensor.data();
            let mut dst = average.data_mut();
            for (a, &value) in dst.iter_mut().zip(src.iter()) {
                *a = decay * *a + one_minus_decay * value;
            }
        }
        self.updates += 1;
        Ok(())
    }

    /// 계산된 EMA 가중치를 호환되는 평가 모델로 복사합니다 (디바이스 및 기울기 설정 보존).
    pub fn copy_to(&self, model: &dyn Module) -> Result<(), String> {
        let target = self.matching_state(model)?;
        for ((_, average), tensor) in self.state.iter().zip(target) {
            if average.id() == tensor.id() {
                continue;
            }
            let src = average.data();
            let mut dst = tensor.data_mut();
            dst.copy_from_slice(&src);
        }
        Ok(())
    }

    /// 지금까지 수행된 누적 갱신 횟수를 반환합니다.
    pub fn num_updates(&self) -> usize {
        self.updates
    }

    /// 계산 그래프 이력이 없는 독립적인 EMA 스냅샷 state_dict를 반환합니다.
    pub fn state_dict(&self) -> Vec<(String, Tensor)> {
        self.state.iter().map(|(name, tensor)| (name.clone(), tensor.detach())).collect()
    }

    /// 모델의 state_dict와 EMA 상태의 이름 및 형상이 일치하는지 검증합니다.
    fn matching_state(&self, model: &dyn Module) -> Result<Vec<Tensor>, String> {
        let state = model.state_dict();
        if state.len() == self.state.len()
            && state
                .iter()
                .zip(&self.state)
                .all(|((n1, t1), (n2, t2))| n1 == n2 && t1.shape() == t2.shape())
        {
            return Ok(state.into_iter().map(|(_, t)| t).collect());
        }

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
