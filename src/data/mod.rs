//! 데이터셋 및 미니배치 데이터 로딩 모듈 (`torch.utils.data`와 유사).

pub mod datasets;

use crate::device::Device;
use crate::rng::Rng;
use crate::tensor::Tensor;
use rayon::prelude::*;
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread::{Scope, ScopedJoinHandle};

/// `(input, target)` 샘플 쌍을 인덱스로 접근할 수 있는 맵 스타일(Map-style) 데이터셋 트레이트.
pub trait Dataset: Send + Sync {
    /// 데이터셋의 전체 샘플 개수를 반환합니다.
    fn len(&self) -> usize;
    /// `i`번째 샘플의 평탄화된 입력 및 타깃 벡터를 반환합니다.
    fn get(&self, i: usize) -> (Vec<f32>, Vec<f32>);
    /// 단일 입력 샘플의 형상(배치 차원 제외)을 반환합니다.
    fn input_shape(&self) -> Vec<usize>;
    /// 단일 타깃 샘플의 형상(분류 클래스 라벨의 경우 `[]`)을 반환합니다.
    fn target_shape(&self) -> Vec<usize>;

    /// 지정된 `indices` 순서대로 입력과 타깃을 일괄(bulk) 조회합니다 (선택적 최적화 구현).
    /// `None`을 반환하면 Rayon을 통한 샘플별 병렬 읽기를 수행합니다.
    fn get_batch(&self, _indices: &[usize]) -> Option<(Vec<f32>, Vec<f32>)> {
        None
    }

    /// 데이터셋이 비어 있는지 확인합니다.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// 인메모리 텐서(0번 축 = 샘플 축)를 기반으로 하는 기본 데이터셋 구조체.
pub struct TensorDataset {
    x: Vec<f32>,
    y: Vec<f32>,
    x_shape: Vec<usize>,
    y_shape: Vec<usize>,
    n: usize,
}

impl TensorDataset {
    /// 입력 텐서 `x`와 정답 텐서 `y`로부터 새 `TensorDataset`을 생성합니다.
    pub fn new(x: &Tensor, y: &Tensor) -> Self {
        assert!(x.ndim() > 0 && y.ndim() > 0, "TensorDataset: tensors must have a sample dimension");
        assert_eq!(x.shape()[0], y.shape()[0], "TensorDataset: x and y must have the same number of samples");
        TensorDataset {
            x: x.to_vec(),
            y: y.to_vec(),
            x_shape: x.shape()[1..].to_vec(),
            y_shape: y.shape()[1..].to_vec(),
            n: x.shape()[0],
        }
    }

    /// 지정된 테스트 비율(`test_fraction`)과 시드로 데이터셋을 (학습용, 테스트용)으로 분할합니다.
    pub fn train_test_split(&self, test_fraction: f32, seed: u64) -> (TensorDataset, TensorDataset) {
        assert!((0.0..=1.0).contains(&test_fraction), "test_fraction must be in [0, 1]");
        let mut idx: Vec<usize> = (0..self.n).collect();
        Rng::new(seed).shuffle(&mut idx);
        let n_test = (self.n as f32 * test_fraction).round() as usize;
        let (test, train) = idx.split_at(n_test);
        (self.subset(train), self.subset(test))
    }

    /// 지정된 인덱스들에 해당하는 샘플들만 추출하여 부분 데이터셋(Subset)을 생성합니다.
    pub fn subset(&self, idx: &[usize]) -> TensorDataset {
        let xs: usize = self.x_shape.iter().product();
        let ys: usize = self.y_shape.iter().product();
        let mut x = Vec::with_capacity(idx.len() * xs);
        let mut y = Vec::with_capacity(idx.len() * ys);
        for &i in idx {
            x.extend_from_slice(&self.x[i * xs..(i + 1) * xs]);
            y.extend_from_slice(&self.y[i * ys..(i + 1) * ys]);
        }
        TensorDataset { x, y, x_shape: self.x_shape.clone(), y_shape: self.y_shape.clone(), n: idx.len() }
    }

    /// 전체 데이터셋을 `(x, y)` 형태의 텐서 쌍으로 변환하여 반환합니다.
    pub fn tensors(&self) -> (Tensor, Tensor) {
        let mut xs = vec![self.n];
        xs.extend(&self.x_shape);
        let mut ys = vec![self.n];
        ys.extend(&self.y_shape);
        (Tensor::from_vec(self.x.clone(), &xs), Tensor::from_vec(self.y.clone(), &ys))
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.n
    }
    fn get(&self, i: usize) -> (Vec<f32>, Vec<f32>) {
        let xs: usize = self.x_shape.iter().product();
        let ys: usize = self.y_shape.iter().product();
        (self.x[i * xs..(i + 1) * xs].to_vec(), self.y[i * ys..(i + 1) * ys].to_vec())
    }
    fn input_shape(&self) -> Vec<usize> {
        self.x_shape.clone()
    }
    fn target_shape(&self) -> Vec<usize> {
        self.y_shape.clone()
    }

    fn get_batch(&self, indices: &[usize]) -> Option<(Vec<f32>, Vec<f32>)> {
        assert!(indices.iter().all(|&i| i < self.n), "TensorDataset: sample index out of range");
        if indices.is_empty() {
            return Some((Vec::new(), Vec::new()));
        }
        let xs: usize = self.x_shape.iter().product();
        let ys: usize = self.y_shape.iter().product();
        // 인덱스가 연속된 경우 슬라이스 복사로 고속화
        if indices.windows(2).all(|pair| pair[1] == pair[0] + 1) {
            let start = indices[0];
            let end = start + indices.len();
            return Some((self.x[start * xs..end * xs].to_vec(), self.y[start * ys..end * ys].to_vec()));
        }
        let mut x = Vec::with_capacity(indices.len() * xs);
        let mut y = Vec::with_capacity(indices.len() * ys);
        for &i in indices {
            x.extend_from_slice(&self.x[i * xs..(i + 1) * xs]);
            y.extend_from_slice(&self.y[i * ys..(i + 1) * ys]);
        }
        Some((x, y))
    }
}

/// 데이터셋을 (선택적으로 셔플된) 미니배치 단위로 순회하는 데이터 로더.
/// 고속 일괄 조회가 지원되면 일괄 읽기를 사용하고, 그렇지 않으면 Rayon 스레드 풀에서 샘플을 병렬 수집합니다.
pub struct DataLoader<'a> {
    dataset: &'a dyn Dataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    device: Device,
    rng: Rng,
}

impl<'a> DataLoader<'a> {
    /// 지정된 데이터셋과 미니배치 크기로 새 DataLoader를 생성합니다.
    pub fn new(dataset: &'a dyn Dataset, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");
        DataLoader { dataset, batch_size, shuffle: false, drop_last: false, device: Device::Cpu, rng: Rng::new(42) }
    }

    /// 에포크마다 데이터를 무작위로 섞을지 여부를 설정합니다.
    pub fn shuffle(mut self, s: bool) -> Self {
        self.shuffle = s;
        self
    }

    /// 마지막 불완전한 미니배치를 버릴지 여부를 설정합니다.
    pub fn drop_last(mut self, d: bool) -> Self {
        self.drop_last = d;
        self
    }

    /// 셔플 난수 생성기 시드를 설정합니다.
    pub fn seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    /// 생성된 미니배치 텐서가 상주할 대상 디바이스를 설정합니다.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// 1 에포크당 생성되는 총 미니배치 개수를 반환합니다.
    pub fn num_batches(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
        }
    }

    /// 1 에포크 분량의 `(inputs, targets)` 미니배치 반복자를 반환합니다.
    pub fn iter(&mut self) -> impl Iterator<Item = (Tensor, Tensor)> + '_ {
        self.batches(true)
    }

    /// 배치 생성 내부 구현 함수
    fn batches(&mut self, parallel_samples: bool) -> impl Iterator<Item = (Tensor, Tensor)> + '_ {
        let mut order: Vec<usize> = (0..self.dataset.len()).collect();
        if self.shuffle {
            self.rng.shuffle(&mut order);
        }
        let bs = self.batch_size;
        let drop_last = self.drop_last;
        let ds = self.dataset;
        let device = self.device;
        let xs = ds.input_shape();
        let ys = ds.target_shape();
        let n = order.len();
        (0..n).step_by(bs).filter(move |&start| !drop_last || n - start >= bs).map(move |start| {
            let idx = &order[start..start + bs.min(n - start)];
            let xn: usize = xs.iter().product();
            let yn: usize = ys.iter().product();
            let (x, y) = if let Some(batch) = ds.get_batch(idx) {
                batch
            } else {
                let samples: Vec<(Vec<f32>, Vec<f32>)> = if parallel_samples {
                    idx.par_iter().map(|&i| ds.get(i)).collect()
                } else {
                    idx.iter().map(|&i| ds.get(i)).collect()
                };
                for (a, b) in &samples {
                    assert_eq!(a.len(), xn, "Dataset::get returned the wrong input size");
                    assert_eq!(b.len(), yn, "Dataset::get returned the wrong target size");
                }
                let mut x = Vec::with_capacity(idx.len() * xn);
                let mut y = Vec::with_capacity(idx.len() * yn);
                for (a, b) in samples {
                    x.extend(a);
                    y.extend(b);
                }
                (x, y)
            };
            assert_eq!(x.len(), idx.len() * xn, "Dataset::get_batch returned the wrong input batch size");
            assert_eq!(y.len(), idx.len() * yn, "Dataset::get_batch returned the wrong target batch size");
            let mut xshape = vec![idx.len()];
            xshape.extend(&xs);
            let mut yshape = vec![idx.len()];
            yshape.extend(&ys);
            (Tensor::from_vec(x, &xshape).on(device), Tensor::from_vec(y, &yshape).on(device))
        })
    }

    /// 백그라운드 스레드에서 최대 `capacity`개의 미니배치를 사전 적재(Prefetching)하는 반복자를 반환합니다.
    /// 배치는 원래 순서를 유지하며 생성됩니다.
    /// 빌린 데이터셋의 유효성을 보장하기 위해 `std::thread::scope` 내부에서 호출하십시오.
    pub fn iter_prefetch<'scope, 'env>(
        &'env mut self,
        scope: &'scope Scope<'scope, 'env>,
        capacity: usize,
    ) -> PrefetchIter<'scope>
    where
        'a: 'env,
    {
        assert!(capacity > 0, "prefetch capacity must be > 0");
        // Rayon 스레드 풀 데드락을 방지하기 위해 이미 Rayon 스레드인 경우 동기 읽기 사용
        let batches = self.batches(rayon::current_thread_index().is_none());
        let (sender, receiver) = sync_channel(capacity);
        let worker = scope.spawn(move || {
            for batch in batches {
                if sender.send(batch).is_err() {
                    break;
                }
            }
        });
        PrefetchIter { receiver: Some(receiver), worker: Some(worker) }
    }
}

/// [`DataLoader::iter_prefetch`]가 반환하는 순서 보장형, 큐 크기 제한형 사전 적재(Prefetch) 반복자.
pub struct PrefetchIter<'scope> {
    receiver: Option<Receiver<(Tensor, Tensor)>>,
    worker: Option<ScopedJoinHandle<'scope, ()>>,
}

impl Iterator for PrefetchIter<'_> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        match self.receiver.as_ref()?.recv() {
            Ok(batch) => Some(batch),
            Err(_) => {
                self.receiver.take();
                if let Some(worker) = self.worker.take() {
                    if let Err(panic) = worker.join() {
                        std::panic::resume_unwind(panic);
                    }
                }
                None
            }
        }
    }
}

impl Drop for PrefetchIter<'_> {
    fn drop(&mut self) {
        self.receiver.take(); // 조인 전 대기열에 막힌 생산자 스레드의 차단을 해제
        if let Some(worker) = self.worker.take() {
            if let Err(panic) = worker.join() {
                if !std::thread::panicking() {
                    std::panic::resume_unwind(panic);
                }
            }
        }
    }
}
