//! Datasets and mini-batch loading (like `torch.utils.data`).

pub mod datasets;

use crate::device::Device;
use crate::rng::Rng;
use crate::tensor::Tensor;
use rayon::prelude::*;

/// A map-style dataset of `(input, target)` samples.
pub trait Dataset: Send + Sync {
    fn len(&self) -> usize;
    /// Return the flattened input and target of sample `i`.
    fn get(&self, i: usize) -> (Vec<f32>, Vec<f32>);
    /// Shape of one input sample (without batch dim).
    fn input_shape(&self) -> Vec<usize>;
    /// Shape of one target sample (`[]` for class indices).
    fn target_shape(&self) -> Vec<usize>;

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
}

/// Iterates a dataset in (optionally shuffled) mini-batches. Samples of a
/// batch are gathered in parallel on the rayon thread pool.
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
        let chunks: Vec<Vec<usize>> = order.chunks(bs).filter(|c| !drop_last || c.len() == bs).map(|c| c.to_vec()).collect();
        chunks.into_iter().map(move |idx| {
            let samples: Vec<(Vec<f32>, Vec<f32>)> = idx.par_iter().map(|&i| ds.get(i)).collect();
            let mut x = Vec::new();
            let mut y = Vec::new();
            for (a, b) in samples {
                x.extend(a);
                y.extend(b);
            }
            let mut xshape = vec![idx.len()];
            xshape.extend(&xs);
            let mut yshape = vec![idx.len()];
            yshape.extend(&ys);
            (Tensor::from_vec(x, &xshape).on(device), Tensor::from_vec(y, &yshape).on(device))
        })
    }
}
