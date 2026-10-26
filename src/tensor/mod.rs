//! N-dimensional tensor with automatic differentiation.

pub(crate) mod ops;
pub mod shape;

use crate::autograd::{self, BackwardFn, Node};
use crate::device::{Backend, Device};
use crate::rng;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

pub(crate) struct TensorInner {
    id: usize,
    data: RwLock<Vec<f32>>,
    shape: Vec<usize>,
    device: RwLock<Device>,
    requires_grad: AtomicBool,
    grad: Mutex<Option<Vec<f32>>>,
    node: Option<Node>,
}

/// A reference-counted, thread-safe, n-dimensional `f32` tensor.
///
/// Cloning a `Tensor` is cheap (it clones an `Arc`); both handles refer to the
/// same storage, exactly like Python references to a `torch.Tensor`.
#[derive(Clone)]
pub struct Tensor(Arc<TensorInner>);

impl Tensor {
    // ------------------------------------------------------------------
    // construction
    // ------------------------------------------------------------------

    fn build(data: Vec<f32>, shape: Vec<usize>, device: Device, requires_grad: bool, node: Option<Node>) -> Tensor {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel, "data length {} does not match shape {:?}", data.len(), shape);
        Tensor(Arc::new(TensorInner {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            data: RwLock::new(data),
            shape,
            device: RwLock::new(device),
            requires_grad: AtomicBool::new(requires_grad),
            grad: Mutex::new(None),
            node,
        }))
    }

    /// Create a tensor from raw data and a shape (on the CPU).
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Self::build(data, shape.to_vec(), Device::Cpu, false, None)
    }

    /// Create a tensor from a slice.
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Tensor {
        Self::from_vec(data.to_vec(), shape)
    }

    /// Create a 2-D tensor from nested rows.
    pub fn from_rows(rows: &[Vec<f32>]) -> Tensor {
        let r = rows.len();
        let c = rows.first().map(|x| x.len()).unwrap_or(0);
        let data: Vec<f32> = rows
            .iter()
            .flat_map(|row| {
                assert_eq!(row.len(), c, "ragged rows");
                row.iter().copied()
            })
            .collect();
        Self::from_vec(data, &[r, c])
    }

    /// 0-dimensional (scalar) tensor.
    pub fn scalar(v: f32) -> Tensor {
        Self::from_vec(vec![v], &[])
    }

    pub fn full(shape: &[usize], value: f32) -> Tensor {
        Self::from_vec(vec![value; shape.iter().product()], shape)
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Self::full(shape, 0.0)
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Self::full(shape, 1.0)
    }

    pub fn zeros_like(t: &Tensor) -> Tensor {
        Self::zeros(t.shape()).on(t.device())
    }

    pub fn ones_like(t: &Tensor) -> Tensor {
        Self::ones(t.shape()).on(t.device())
    }

    /// Samples from `N(0, 1)`.
    pub fn randn(shape: &[usize]) -> Tensor {
        let n = shape.iter().product();
        Self::from_vec(rng::fill_with(n, |r| r.normal()), shape)
    }

    /// Samples from `U[0, 1)`.
    pub fn rand(shape: &[usize]) -> Tensor {
        let n = shape.iter().product();
        Self::from_vec(rng::fill_with(n, |r| r.uniform()), shape)
    }

    /// Samples from `U[lo, hi)`.
    pub fn uniform(shape: &[usize], lo: f32, hi: f32) -> Tensor {
        let n = shape.iter().product();
        Self::from_vec(rng::fill_with(n, |r| r.uniform_range(lo, hi)), shape)
    }

    /// `[0, 1, ..., n-1]`.
    pub fn arange(n: usize) -> Tensor {
        Self::from_vec((0..n).map(|i| i as f32).collect(), &[n])
    }

    /// Identity matrix.
    pub fn eye(n: usize) -> Tensor {
        let mut d = vec![0.0; n * n];
        for i in 0..n {
            d[i * n + i] = 1.0;
        }
        Self::from_vec(d, &[n, n])
    }

    /// Builder: mark as a trainable leaf.
    pub fn requires_grad_(self, flag: bool) -> Tensor {
        assert!(self.0.node.is_none() || !flag, "requires_grad_ can only be set on leaf tensors");
        self.0.requires_grad.store(flag, Ordering::Relaxed);
        self
    }

    /// Builder: place this tensor on `device` (in place, no copy).
    pub fn on(self, device: Device) -> Tensor {
        self.set_device(device);
        self
    }

    /// Internal constructor used by differentiable ops.
    pub(crate) fn from_op(
        data: Vec<f32>,
        shape: Vec<usize>,
        device: Device,
        inputs: Vec<Tensor>,
        name: &'static str,
        backward: BackwardFn,
    ) -> Tensor {
        let needs = autograd::is_grad_enabled() && inputs.iter().any(|t| t.requires_grad());
        if needs {
            let node = Node { name, inputs, backward };
            Self::build(data, shape, device, true, Some(node))
        } else {
            Self::build(data, shape, device, false, None)
        }
    }

    // ------------------------------------------------------------------
    // accessors
    // ------------------------------------------------------------------

    pub fn id(&self) -> usize {
        self.0.id
    }

    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    /// Size of dimension `d` (negative values count from the end).
    pub fn size(&self, d: isize) -> usize {
        self.0.shape[self.dim_index(d)]
    }

    pub fn ndim(&self) -> usize {
        self.0.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.0.shape.iter().product()
    }

    pub fn device(&self) -> Device {
        *self.0.device.read().unwrap()
    }

    /// Move this tensor (and its future computations) to `device` in place.
    /// Used by `Module::to` – like `param.data = param.data.to(device)`.
    pub fn set_device(&self, device: Device) {
        *self.0.device.write().unwrap() = device;
    }

    pub(crate) fn backend(&self) -> Arc<dyn Backend> {
        self.device().backend()
    }

    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad.load(Ordering::Relaxed)
    }

    pub fn is_leaf(&self) -> bool {
        self.0.node.is_none()
    }

    pub(crate) fn node(&self) -> Option<&Node> {
        self.0.node.as_ref()
    }

    /// Name of the op that produced this tensor (`None` for leaves).
    pub fn grad_fn(&self) -> Option<&'static str> {
        self.0.node.as_ref().map(|n| n.name)
    }

    /// Read access to the underlying storage.
    pub fn data(&self) -> RwLockReadGuard<'_, Vec<f32>> {
        self.0.data.read().unwrap()
    }

    /// Write access to the underlying storage (used by optimizers).
    pub fn data_mut(&self) -> RwLockWriteGuard<'_, Vec<f32>> {
        self.0.data.write().unwrap()
    }

    /// Copy the data out as a `Vec`.
    pub fn to_vec(&self) -> Vec<f32> {
        self.data().clone()
    }

    /// Value of a single-element tensor.
    pub fn item(&self) -> f32 {
        assert_eq!(self.numel(), 1, "item() requires a single-element tensor, got {:?}", self.shape());
        self.data()[0]
    }

    /// Element at a multi-dimensional index.
    pub fn at(&self, idx: &[usize]) -> f32 {
        let strides = shape::strides(self.shape());
        let off: usize = idx.iter().zip(&strides).map(|(i, s)| i * s).sum();
        self.data()[off]
    }

    /// Overwrite the data (shape must be unchanged).
    pub fn copy_from_slice(&self, src: &[f32]) {
        let mut d = self.data_mut();
        assert_eq!(d.len(), src.len(), "copy_from_slice: size mismatch");
        d.copy_from_slice(src);
    }

    // ------------------------------------------------------------------
    // gradients
    // ------------------------------------------------------------------

    /// Accumulated gradient as a tensor.
    pub fn grad(&self) -> Option<Tensor> {
        self.0.grad.lock().unwrap().as_ref().map(|g| Tensor::from_vec(g.clone(), self.shape()).on(self.device()))
    }

    /// Accumulated gradient as a raw vector.
    pub fn grad_vec(&self) -> Option<Vec<f32>> {
        self.0.grad.lock().unwrap().clone()
    }

    /// Replace the gradient (used by all-reduce / gradient clipping).
    pub fn set_grad(&self, g: Option<Vec<f32>>) {
        if let Some(g) = &g {
            assert_eq!(g.len(), self.numel(), "set_grad: size mismatch");
        }
        *self.0.grad.lock().unwrap() = g;
    }

    pub(crate) fn accumulate_grad(&self, g: &[f32]) {
        let mut slot = self.0.grad.lock().unwrap();
        match slot.as_mut() {
            Some(acc) => acc.iter_mut().zip(g).for_each(|(a, b)| *a += b),
            None => *slot = Some(g.to_vec()),
        }
    }

    /// Mutate the gradient in place, if any.
    pub fn with_grad_mut(&self, f: impl FnOnce(&mut Vec<f32>)) {
        if let Some(g) = self.0.grad.lock().unwrap().as_mut() {
            f(g)
        }
    }

    pub fn zero_grad(&self) {
        *self.0.grad.lock().unwrap() = None;
    }

    /// Back-propagate from this (scalar) tensor.
    pub fn backward(&self) {
        assert_eq!(self.numel(), 1, "backward() without arguments requires a scalar output (got {:?})", self.shape());
        assert!(
            self.requires_grad(),
            "backward(): tensor does not require grad (no trainable inputs or no_grad active)"
        );
        autograd::run_backward(self, vec![1.0]);
    }

    /// Back-propagate with an explicit output gradient.
    pub fn backward_with(&self, grad: &Tensor) {
        assert_eq!(grad.shape(), self.shape(), "backward_with: gradient shape mismatch");
        autograd::run_backward(self, grad.to_vec());
    }

    /// New tensor sharing no graph history (copy of the data).
    pub fn detach(&self) -> Tensor {
        Tensor::from_vec(self.to_vec(), self.shape()).on(self.device())
    }

    pub(crate) fn dim_index(&self, d: isize) -> usize {
        let n = self.ndim() as isize;
        let i = if d < 0 { n + d } else { d };
        assert!(i >= 0 && i < n, "dimension {d} out of range for tensor with {n} dims");
        i as usize
    }

    /// Select rows (first dimension) – non differentiable, used by data loading.
    pub fn index_rows(&self, rows: &[usize]) -> Tensor {
        let row = self.numel() / self.shape()[0].max(1);
        let d = self.data();
        let mut out = Vec::with_capacity(rows.len() * row);
        for &r in rows {
            out.extend_from_slice(&d[r * row..(r + 1) * row]);
        }
        let mut shape = self.shape().to_vec();
        shape[0] = rows.len();
        Tensor::from_vec(out, &shape).on(self.device())
    }

    /// Contiguous slice `[start, start+len)` along dim 0 – non differentiable.
    pub fn narrow_rows(&self, start: usize, len: usize) -> Tensor {
        assert!(self.ndim() > 0, "narrow_rows requires a batch dimension");
        let rows = self.shape()[0];
        assert!(start <= rows && len <= rows - start, "narrow_rows range exceeds the batch dimension");
        let row = self.numel() / rows.max(1);
        let data = self.data()[start * row..(start + len) * row].to_vec();
        let mut shape = self.shape().to_vec();
        shape[0] = len;
        Self::build(data, shape, self.device(), false, None)
    }

    /// Index of the maximum along the last dimension.
    pub fn argmax(&self) -> Vec<usize> {
        let c = *self.shape().last().expect("argmax on scalar");
        self.data()
            .chunks(c)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) })
                    .0
            })
            .collect()
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let d = self.data();
        let shown: Vec<String> = d.iter().take(8).map(|v| format!("{v:.4}")).collect();
        let more = if d.len() > 8 { ", ..." } else { "" };
        write!(f, "tensor([{}{}], shape={:?}, device={}", shown.join(", "), more, self.shape(), self.device())?;
        if let Some(n) = self.grad_fn() {
            write!(f, ", grad_fn={n}")?;
        } else if self.requires_grad() {
            write!(f, ", requires_grad=true")?;
        }
        write!(f, ")")
    }
}
