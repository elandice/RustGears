//! Datasets and mini-batch loading (like `torch.utils.data`).

pub mod datasets;

use crate::device::Device;
use crate::rng::Rng;
use crate::tensor::Tensor;
use rayon::prelude::*;
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread::{Scope, ScopedJoinHandle};

/// A map-style dataset of `(input, target)` samples.
pub trait Dataset: Send + Sync {
    fn len(&self) -> usize;
    /// Return the flattened input and target of sample `i`.
    fn get(&self, i: usize) -> (Vec<f32>, Vec<f32>);
    /// Shape of one input sample (without batch dim).
    fn input_shape(&self) -> Vec<usize>;
    /// Shape of one target sample (`[]` for class indices).
    fn target_shape(&self) -> Vec<usize>;

    /// Optional bulk read of flattened inputs and targets in `indices` order.
    /// Return `None` to use parallel per-sample reads. Implementations must
    /// preserve each sample's declared shape; the loader checks batch lengths.
    fn get_batch(&self, _indices: &[usize]) -> Option<(Vec<f32>, Vec<f32>)> {
        None
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Dataset backed by two in-memory tensors (first dim = sample index).
pub struct TensorDataset {
    x: Vec<f32>,
    y: Vec<f32>,
    x_shape: Vec<usize>,
    y_shape: Vec<usize>,
    n: usize,
}

impl TensorDataset {
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

    /// Split into (train, test) with the given test fraction (shuffled with `seed`).
    pub fn train_test_split(&self, test_fraction: f32, seed: u64) -> (TensorDataset, TensorDataset) {
        assert!((0.0..=1.0).contains(&test_fraction), "test_fraction must be in [0, 1]");
        let mut idx: Vec<usize> = (0..self.n).collect();
        Rng::new(seed).shuffle(&mut idx);
        let n_test = (self.n as f32 * test_fraction).round() as usize;
        let (test, train) = idx.split_at(n_test);
        (self.subset(train), self.subset(test))
    }

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

    /// Whole dataset as `(x, y)` tensors.
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

/// Iterates a dataset in (optionally shuffled) mini-batches. Uses bulk reads
/// when available, otherwise gathers samples on the Rayon thread pool.
pub struct DataLoader<'a> {
    dataset: &'a dyn Dataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    device: Device,
    rng: Rng,
}

impl<'a> DataLoader<'a> {
    pub fn new(dataset: &'a dyn Dataset, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");
        DataLoader { dataset, batch_size, shuffle: false, drop_last: false, device: Device::Cpu, rng: Rng::new(42) }
    }

    pub fn shuffle(mut self, s: bool) -> Self {
        self.shuffle = s;
        self
    }

    pub fn drop_last(mut self, d: bool) -> Self {
        self.drop_last = d;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    /// Place produced batches on `device`.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn num_batches(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
        }
    }

    /// One epoch of `(inputs, targets)` batches.
    pub fn iter(&mut self) -> impl Iterator<Item = (Tensor, Tensor)> + '_ {
        self.batches(true)
    }

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

    /// Prepare up to `capacity` batches ahead on a background thread. Sample
    /// reads use Rayon unless called inside a Rayon worker, avoiding pool deadlocks.
    /// Batches retain their original order.
    /// Use inside `std::thread::scope` so borrowed datasets stay valid.
    /// Dropping the iterator cancels pending sends and joins the producer.
    pub fn iter_prefetch<'scope, 'env>(
        &'env mut self,
        scope: &'scope Scope<'scope, 'env>,
        capacity: usize,
    ) -> PrefetchIter<'scope>
    where
        'a: 'env,
    {
        assert!(capacity > 0, "prefetch capacity must be > 0");
        // A blocked Rayon consumer must not occupy the pool needed by the producer.
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

/// Ordered, bounded batch prefetching returned by [`DataLoader::iter_prefetch`].
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
        self.receiver.take(); // Unblock a producer waiting on a full queue before joining.
        if let Some(worker) = self.worker.take() {
            if let Err(panic) = worker.join() {
                if !std::thread::panicking() {
                    std::panic::resume_unwind(panic);
                }
            }
        }
    }
}
