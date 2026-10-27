//! 동기식 데이터 병렬 처리 (DistributedDataParallel 스타일).

use super::allreduce::{ring_all_reduce, AllReduceStats};
use crate::device::Device;
use crate::nn::Module;
use crate::optim::Optimizer;
use crate::tensor::Tensor;
use std::time::{Duration, Instant};

/// 학습 스텝 중 각 모델 레플리카(디바이스)별 통계 정보.
#[derive(Clone, Debug)]
pub struct ReplicaStats {
    /// 해당 레플리카가 위치한 디바이스.
    pub device: Device,
    /// 처리한 샘플 수.
    pub samples: usize,
    /// 계산된 손실값(Loss).
    pub loss: f32,
    /// 순전파 및 역전파 연산 소요 시간.
    pub compute_time: Duration,
}

/// [`DataParallel::train_step`]의 수행 결과 보고서.
#[derive(Clone, Debug)]
pub struct StepReport {
    /// 전체 배치에 걸친 샘플 가중 평균 손실값.
    pub loss: f32,
    /// 각 레플리카별 실행 통계.
    pub replicas: Vec<ReplicaStats>,
    /// 올리듀스(All-Reduce) 통신 통계.
    pub allreduce: AllReduceStats,
    /// 올리듀스 통신에 소요된 시간.
    pub allreduce_time: Duration,
}

/// 모델을 여러 디바이스에 복제하여 데이터 병렬 방식으로 분산 학습을 수행합니다:
///
/// 1. **scatter**: 미니배치 데이터를 디바이스별 샤드로 분할합니다.
/// 2. 활성화된 각 레플리카에서 **순전파와 역전파를 병렬로 동시 실행**합니다 (디바이스가 1개인 경우 인라인 실행).
/// 3. **ring all-reduce**를 통해 모든 레플리카의 기울기를 평균 합산 축약합니다.
/// 4. 옵티마이저가 마스터 레플리카의 매개변수를 갱신하면, 갱신된 파라미터를 모든 레플리카로 **브로드캐스트**하여 동기화합니다.
pub struct DataParallel<M: Module> {
    replicas: Vec<M>,
    devices: Vec<Device>,
}

impl<M: Module> DataParallel<M> {
    /// `factory` 클로저를 사용하여 디바이스당 하나의 레플리카를 인스턴스화하고,
    /// 가중치를 0번(마스터) 레플리카와 동기화합니다.
    pub fn new(devices: Vec<Device>, factory: impl Fn() -> M) -> Self {
        assert!(!devices.is_empty(), "DataParallel needs at least one device");
        let replicas: Vec<M> = devices
            .iter()
            .map(|&d| {
                let m = factory();
                m.to(d);
                m
            })
            .collect();
        let dp = DataParallel { replicas, devices };
        dp.broadcast_parameters();
        dp
    }

    /// 마스터 레플리카(디바이스 0)의 참조를 반환합니다. 모델 평가나 내보내기 시 사용합니다.
    pub fn module(&self) -> &M {
        &self.replicas[0]
    }

    /// 할당된 디바이스 슬라이스를 반환합니다.
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    /// 마스터 레플리카의 학습 가능한 매개변수 목록을 반환합니다 (옵티마이저 전달용).
    pub fn parameters(&self) -> Vec<Tensor> {
        self.replicas[0].parameters()
    }

    /// 모든 레플리카를 학습 모드로 전환합니다.
    pub fn train(&self) {
        self.replicas.iter().for_each(|r| r.train());
    }

    /// 모든 레플리카를 평가(추론) 모드로 전환합니다.
    pub fn eval(&self) {
        self.replicas.iter().for_each(|r| r.eval());
    }

    /// 마스터 레플리카의 파라미터 및 버퍼 데이터를 다른 모든 레플리카로 복사합니다.
    pub fn broadcast_parameters(&self) {
        let master = self.replicas[0].state_dict();
        for rep in &self.replicas[1..] {
            for ((_, src), (_, dst)) in master.iter().zip(rep.state_dict()) {
                dst.copy_from_slice(&src.data());
            }
        }
    }

    /// 마스터 레플리카에서 순전파 연산을 수행합니다.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.replicas[0].forward(&x.to(self.devices[0]))
    }

    /// 분할(Scatter) → 병렬 순전파/역전파 → 그래디언트 Ring All-Reduce를 순차적으로 실행합니다.
    ///
    /// 호출 완료 후 모든 레플리카의 매개변수는 배치 전체에 대해 *평균화된* 기울기를 갖게 됩니다.
    pub fn train_step<L>(&self, x: &Tensor, y: &Tensor, loss_fn: L) -> StepReport
    where
        L: Fn(&Tensor, &Tensor) -> Tensor + Sync,
    {
        assert!(x.ndim() > 0 && y.ndim() > 0, "DataParallel inputs must have a batch dimension");
        let n = x.shape()[0];
        assert!(n > 0, "DataParallel batch must not be empty");
        assert_eq!(y.shape()[0], n, "DataParallel input and target batch sizes must match");
        let k = self.devices.len();

        let compute = |rep: &M, dev: Device, start: usize, len: usize| {
            let t0 = Instant::now();
            rep.zero_grad();
            let xs = x.narrow_rows(start, len).on(dev);
            let ys = y.narrow_rows(start, len).on(dev);
            let out = rep.forward(&xs);
            let loss = loss_fn(&out, &ys);
            loss.backward();
            ReplicaStats { device: dev, samples: len, loss: loss.item(), compute_time: t0.elapsed() }
        };

        // 단일 디바이스 fast-path: 호출 스레드의 그래디언트 활성화 상태를 확인하여 바로 실행
        if k == 1 && crate::autograd::is_grad_enabled() {
            let replica = compute(&self.replicas[0], self.devices[0], 0, n);
            let t0 = Instant::now();
            let params = self.replicas[0].parameters();
            let elements = params.iter().map(Tensor::numel).sum();
            for p in params {
                let mut present = false;
                p.with_grad_mut(|_| present = true);
                if !present {
                    p.set_grad(Some(vec![0.0; p.numel()]));
                }
            }
            return StepReport {
                loss: replica.loss,
                replicas: vec![replica],
                allreduce: AllReduceStats { workers: 1, elements, steps: 0, bytes_sent: 0 },
                allreduce_time: t0.elapsed(),
            };
        }

        // 샤드 크기 계산 (최대한 균등하게 분할)
        let (base, extra) = (n / k, n % k);
        let shards: Vec<(usize, usize)> =
            (0..k).map(|r| (r * base + r.min(extra), base + usize::from(r < extra))).collect();

        let compute = &compute;
        let replicas: Vec<ReplicaStats> = std::thread::scope(|s| {
            let handles: Vec<_> = self
                .replicas
                .iter()
                .zip(&self.devices)
                .zip(&shards)
                .map(|((rep, &dev), &(start, len))| {
                    if len == 0 {
                        rep.zero_grad();
                        None
                    } else {
                        Some(s.spawn(move || compute(rep, dev, start, len)))
                    }
                })
                .collect();
            handles
                .into_iter()
                .zip(&self.devices)
                .map(|(handle, &device)| {
                    handle.map_or_else(
                        || ReplicaStats { device, samples: 0, loss: 0.0, compute_time: Duration::ZERO },
                        |handle| handle.join().expect("replica thread panicked"),
                    )
                })
                .collect()
        });

        // 각 레플리카의 기울기를 1차원 버킷으로 평탄화(Flatten)하고 배치 샘플 비율에 따라 사전 스케일링
        let t0 = Instant::now();
        let params: Vec<Vec<Tensor>> = self.replicas.iter().map(|r| r.parameters()).collect();
        let mut buckets: Vec<Vec<f32>> = params
            .iter()
            .zip(&replicas)
            .map(|(ps, st)| {
                let w = st.samples as f32 / n as f32;
                let mut b = vec![0.0; ps.iter().map(Tensor::numel).sum()];
                let mut offset = 0;
                for p in ps {
                    let end = offset + p.numel();
                    p.with_grad_mut(|g| {
                        for (dst, &src) in b[offset..end].iter_mut().zip(g.iter()) {
                            *dst = src * w;
                        }
                    });
                    offset = end;
                }
                b
            })
            .collect();
        let stats = ring_all_reduce(&mut buckets);
        // 평균화된 기울기를 모든 레플리카 매개변수에 다시 기록
        for (ps, bucket) in params.iter().zip(&buckets) {
            let mut off = 0;
            for p in ps {
                let len = p.numel();
                p.set_grad(Some(bucket[off..off + len].to_vec()));
                off += len;
            }
        }
        let allreduce_time = t0.elapsed();

        let loss = replicas.iter().map(|r| r.loss * r.samples as f32).sum::<f32>() / n as f32;
        StepReport { loss, replicas, allreduce: stats, allreduce_time }
    }

    /// 완전한 1스텝 실행: `train_step` + `optimizer.step()` + 파라미터 브로드캐스트.
    /// 옵티마이저는 반드시 [`DataParallel::parameters`]로 초기화되어야 합니다.
    pub fn fit_batch<L>(&self, x: &Tensor, y: &Tensor, loss_fn: L, opt: &mut dyn Optimizer) -> StepReport
    where
        L: Fn(&Tensor, &Tensor) -> Tensor + Sync,
    {
        let report = self.train_step(x, y, loss_fn);
        opt.step();
        self.broadcast_parameters();
        report
    }
}
