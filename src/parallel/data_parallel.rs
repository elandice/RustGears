//! Synchronous data parallelism (DistributedDataParallel style).

use super::allreduce::{ring_all_reduce, AllReduceStats};
use crate::device::Device;
use crate::nn::Module;
use crate::optim::Optimizer;
use crate::tensor::Tensor;
use std::time::{Duration, Instant};

/// Per-replica statistics of a training step.
#[derive(Clone, Debug)]
pub struct ReplicaStats {
    pub device: Device,
    pub samples: usize,
    pub loss: f32,
    pub compute_time: Duration,
}

/// Result of [`DataParallel::train_step`].
#[derive(Clone, Debug)]
pub struct StepReport {
    /// Sample-weighted mean loss over the whole batch.
    pub loss: f32,
    pub replicas: Vec<ReplicaStats>,
    pub allreduce: AllReduceStats,
    pub allreduce_time: Duration,
}

/// Replicates a model on several devices and trains it data-parallel:
///
/// 1. **scatter** the mini-batch into one shard per device,
/// 2. run **forward + backward concurrently** on active replicas (one device runs inline),
/// 3. average the gradients with a **ring all-reduce**,
/// 4. the optimizer updates the master replica, then parameters are
///    **broadcast** back to all replicas.
pub struct DataParallel<M: Module> {
    replicas: Vec<M>,
    devices: Vec<Device>,
}

impl<M: Module> DataParallel<M> {
    /// Build one replica per device with `factory` and synchronise their
    /// weights with replica 0.
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

    /// The master replica (device 0) – use it for evaluation / export.
    pub fn module(&self) -> &M {
        &self.replicas[0]
    }

    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    /// Parameters of the master replica (pass these to the optimizer).
    pub fn parameters(&self) -> Vec<Tensor> {
        self.replicas[0].parameters()
    }

    pub fn train(&self) {
        self.replicas.iter().for_each(|r| r.train());
    }

    pub fn eval(&self) {
        self.replicas.iter().for_each(|r| r.eval());
    }

    /// Copy master parameters and buffers to every other replica.
    pub fn broadcast_parameters(&self) {
        let master = self.replicas[0].state_dict();
        for rep in &self.replicas[1..] {
            for ((_, src), (_, dst)) in master.iter().zip(rep.state_dict()) {
                dst.copy_from_slice(&src.data());
            }
        }
    }

    /// Forward pass on the master replica.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.replicas[0].forward(&x.to(self.devices[0]))
    }

    /// Scatter → parallel forward/backward → ring all-reduce of gradients.
    ///
    /// After this call every replica holds the *averaged* gradient.
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

        // Worker threads have gradient recording enabled independently of the caller.
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

        // shard sizes (as even as possible)
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

        // flatten each replica's gradients into one bucket, pre-scaled by its share of the batch
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
        // write the averaged gradients back into every replica
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

    /// Full step: `train_step` + `optimizer.step()` + parameter broadcast.
    /// The optimizer must have been created from [`DataParallel::parameters`].
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
