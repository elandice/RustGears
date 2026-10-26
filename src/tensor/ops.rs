//! Differentiable tensor operations.

use super::{shape, Tensor};
use crate::device::{BinaryOp, UnaryOp};
use crate::kernels;
use std::ops::{Add, Div, Mul, Neg, Sub};

fn assert_same_device(a: &Tensor, b: &Tensor, op: &str) {
    assert_eq!(
        a.device(),
        b.device(),
        "{op}: expected all tensors to be on the same device, found {} and {}",
        a.device(),
        b.device()
    );
}

/// Read two tensors' storage without taking the same lock twice.
fn with_pair<R>(a: &Tensor, b: &Tensor, f: impl FnOnce(&[f32], &[f32]) -> R) -> R {
    if a.id() == b.id() {
        let d = a.data();
        f(&d, &d)
    } else {
        let da = a.data();
        let db = b.data();
        f(&da, &db)
    }
}

fn unary_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "NegBackward",
        UnaryOp::Exp => "ExpBackward",
        UnaryOp::Log => "LogBackward",
        UnaryOp::Relu => "ReluBackward",
        UnaryOp::Sigmoid => "SigmoidBackward",
        UnaryOp::Tanh => "TanhBackward",
        UnaryOp::Sqrt => "SqrtBackward",
        UnaryOp::Gelu => "GeluBackward",
        UnaryOp::Abs => "AbsBackward",
    }
}

impl Tensor {
    // ------------------------------------------------------------------
    // element-wise binary (with broadcasting)
    // ------------------------------------------------------------------

    fn binary(&self, other: &Tensor, op: BinaryOp) -> Tensor {
        assert_same_device(self, other, &format!("{op:?}"));
        let a_shape = self.shape().to_vec();
        let b_shape = other.shape().to_vec();
        let out_shape = shape::broadcast_shape(&a_shape, &b_shape).unwrap_or_else(|e| panic!("{op:?}: {e}"));
        let backend = self.backend();
        let (data, maps) = if a_shape == b_shape {
            (with_pair(self, other, |a, b| backend.binary(op, a, b)), None)
        } else {
            let ma = shape::broadcast_index_map(&a_shape, &out_shape);
            let mb = shape::broadcast_index_map(&b_shape, &out_shape);
            let d = with_pair(self, other, |a, b| ma.iter().zip(&mb).map(|(&i, &j)| op.apply(a[i], b[j])).collect());
            (d, Some((ma, mb)))
        };
        let (a, b) = (self.clone(), other.clone());
        let (an, bn) = (self.numel(), other.numel());
        let name = match op {
            BinaryOp::Add => "AddBackward",
            BinaryOp::Sub => "SubBackward",
            BinaryOp::Mul => "MulBackward",
            BinaryOp::Div => "DivBackward",
        };
        Tensor::from_op(
            data,
            out_shape,
            self.device(),
            vec![self.clone(), other.clone()],
            name,
            Box::new(move |g| {
                let n = g.len();
                let ai = |i: usize| maps.as_ref().map_or(i, |m| m.0[i]);
                let bi = |i: usize| maps.as_ref().map_or(i, |m| m.1[i]);
                let need_vals = matches!(op, BinaryOp::Mul | BinaryOp::Div);
                let ad = if need_vals { a.to_vec() } else { Vec::new() };
                let bd = if need_vals { b.to_vec() } else { Vec::new() };
                let ga = a.requires_grad().then(|| -> Vec<f32> {
                    match op {
                        BinaryOp::Add | BinaryOp::Sub => g.to_vec(),
                        BinaryOp::Mul => (0..n).map(|i| g[i] * bd[bi(i)]).collect(),
                        BinaryOp::Div => (0..n).map(|i| g[i] / bd[bi(i)]).collect(),
                    }
                });
                let gb = b.requires_grad().then(|| -> Vec<f32> {
                    match op {
                        BinaryOp::Add => g.to_vec(),
                        BinaryOp::Sub => g.iter().map(|v| -v).collect(),
                        BinaryOp::Mul => (0..n).map(|i| g[i] * ad[ai(i)]).collect(),
                        BinaryOp::Div => (0..n)
                            .map(|i| {
                                let bv = bd[bi(i)];
                                -g[i] * ad[ai(i)] / (bv * bv)
                            })
                            .collect(),
                    }
                });
                match &maps {
                    None => vec![ga, gb],
                    Some((ma, mb)) => vec![
                        ga.map(|v| shape::reduce_by_map(&v, ma, an)),
                        gb.map(|v| shape::reduce_by_map(&v, mb, bn)),
                    ],
                }
            }),
        )
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        self.binary(other, BinaryOp::Add)
    }
    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.binary(other, BinaryOp::Sub)
    }
    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.binary(other, BinaryOp::Mul)
    }
    pub fn div(&self, other: &Tensor) -> Tensor {
        self.binary(other, BinaryOp::Div)
    }

    // ------------------------------------------------------------------
    // scalar ops
    // ------------------------------------------------------------------

    fn scalar_map(&self, name: &'static str, f: impl Fn(f32) -> f32, df: impl Fn(f32) -> f32 + Send + Sync + 'static) -> Tensor {
        let data: Vec<f32> = self.data().iter().map(|&x| f(x)).collect();
        let x = self.clone();
        Tensor::from_op(
            data,
            self.shape().to_vec(),
            self.device(),
            vec![self.clone()],
            name,
            Box::new(move |g| {
                let xd = x.data();
                vec![Some(g.iter().zip(xd.iter()).map(|(gv, &xv)| gv * df(xv)).collect())]
            }),
        )
    }

    pub fn add_scalar(&self, s: f32) -> Tensor {
        self.scalar_map("AddScalarBackward", move |x| x + s, |_| 1.0)
    }

    pub fn mul_scalar(&self, s: f32) -> Tensor {
        self.scalar_map("MulScalarBackward", move |x| x * s, move |_| s)
    }

    pub fn div_scalar(&self, s: f32) -> Tensor {
        self.mul_scalar(1.0 / s)
    }

    pub fn pow_scalar(&self, p: f32) -> Tensor {
        self.scalar_map("PowBackward", move |x| x.powf(p), move |x| p * x.powf(p - 1.0))
    }

    pub fn square(&self) -> Tensor {
        self.scalar_map("SquareBackward", |x| x * x, |x| 2.0 * x)
    }

    pub fn leaky_relu(&self, slope: f32) -> Tensor {
        self.scalar_map(
            "LeakyReluBackward",
            move |x| if x > 0.0 { x } else { slope * x },
            move |x| if x > 0.0 { 1.0 } else { slope },
        )
    }

    pub fn clamp(&self, lo: f32, hi: f32) -> Tensor {
        self.scalar_map("ClampBackward", move |x| x.clamp(lo, hi), move |x| if x >= lo && x <= hi { 1.0 } else { 0.0 })
    }

    // ------------------------------------------------------------------
    // unary (dispatched to the device backend)
    // ------------------------------------------------------------------

    fn unary(&self, op: UnaryOp) -> Tensor {
        let y = self.backend().unary(op, &self.data());
        let x = self.clone();
        let yc = y.clone();
        Tensor::from_op(
            y,
            self.shape().to_vec(),
            self.device(),
            vec![self.clone()],
            unary_name(op),
            Box::new(move |g| {
                let xd = x.data();
                vec![Some(g.iter().enumerate().map(|(i, gv)| gv * op.derivative(xd[i], yc[i])).collect())]
            }),
        )
    }

    pub fn neg(&self) -> Tensor {
        self.unary(UnaryOp::Neg)
    }
    pub fn exp(&self) -> Tensor {
        self.unary(UnaryOp::Exp)
    }
    pub fn log(&self) -> Tensor {
        self.unary(UnaryOp::Log)
    }
    pub fn relu(&self) -> Tensor {
        self.unary(UnaryOp::Relu)
    }
    pub fn sigmoid(&self) -> Tensor {
        self.unary(UnaryOp::Sigmoid)
    }
    pub fn tanh(&self) -> Tensor {
        self.unary(UnaryOp::Tanh)
    }
    pub fn sqrt(&self) -> Tensor {
        self.unary(UnaryOp::Sqrt)
    }
    pub fn gelu(&self) -> Tensor {
        self.unary(UnaryOp::Gelu)
    }
    pub fn abs(&self) -> Tensor {
        self.unary(UnaryOp::Abs)
    }

    // ------------------------------------------------------------------
    // linear algebra
    // ------------------------------------------------------------------

    /// Matrix product supporting `[m,k]@[k,n]`, `[b,m,k]@[k,n]`,
    /// `[b,m,k]@[b,k,n]` and `[k]@[k,n]`.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        match (self.ndim(), other.ndim()) {
            (2, 2) => self.mm(other),
            (1, 2) => {
                let k = self.shape()[0];
                let n = other.shape()[1];
                self.reshape(&[1, k]).mm(other).reshape(&[n])
            }
            (3, 2) => {
                let (b, m, k) = (self.shape()[0], self.shape()[1], self.shape()[2]);
                let n = other.shape()[1];
                self.reshape(&[b * m, k]).mm(other).reshape(&[b, m, n])
            }
            (3, 3) => self.bmm(other),
            (a, b) => panic!("matmul: unsupported ranks {a} and {b} (shapes {:?}, {:?})", self.shape(), other.shape()),
        }
    }

    fn mm(&self, other: &Tensor) -> Tensor {
        assert_same_device(self, other, "matmul");
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (k2, n) = (other.shape()[0], other.shape()[1]);
        assert_eq!(k, k2, "matmul: shape mismatch {:?} @ {:?}", self.shape(), other.shape());
        let be = self.backend();
        let out = with_pair(self, other, |a, b| be.matmul(a, b, m, k, n));
        let (a, b) = (self.clone(), other.clone());
        Tensor::from_op(
            out,
            vec![m, n],
            self.device(),
            vec![self.clone(), other.clone()],
            "MmBackward",
            Box::new(move |g| {
                let ga = a.requires_grad().then(|| {
                    let bt = shape::transpose2d(&b.data(), k, n);
                    be.matmul(g, &bt, m, n, k)
                });
                let gb = b.requires_grad().then(|| {
                    let at = shape::transpose2d(&a.data(), m, k);
                    be.matmul(&at, g, k, m, n)
                });
                vec![ga, gb]
            }),
        )
    }

    /// Batched matrix multiply `[b,m,k] @ [b,k,n] -> [b,m,n]`.
    pub fn bmm(&self, other: &Tensor) -> Tensor {
        assert_same_device(self, other, "bmm");
        assert!(self.ndim() == 3 && other.ndim() == 3, "bmm expects 3-D tensors");
        let (bs, m, k) = (self.shape()[0], self.shape()[1], self.shape()[2]);
        let (bs2, k2, n) = (other.shape()[0], other.shape()[1], other.shape()[2]);
        assert!(bs == bs2 && k == k2, "bmm: shape mismatch {:?} @ {:?}", self.shape(), other.shape());
        let be = self.backend();
        let out = with_pair(self, other, |a, b| {
            let mut out = Vec::with_capacity(bs * m * n);
            for i in 0..bs {
                out.extend(be.matmul(&a[i * m * k..(i + 1) * m * k], &b[i * k * n..(i + 1) * k * n], m, k, n));
            }
            out
        });
        let (a, b) = (self.clone(), other.clone());
        Tensor::from_op(
            out,
            vec![bs, m, n],
            self.device(),
            vec![self.clone(), other.clone()],
            "BmmBackward",
            Box::new(move |g| {
                let ad = a.to_vec();
                let bd = b.to_vec();
                let mut ga = a.requires_grad().then(|| Vec::with_capacity(bs * m * k));
                let mut gb = b.requires_grad().then(|| Vec::with_capacity(bs * k * n));
                for i in 0..bs {
                    let gi = &g[i * m * n..(i + 1) * m * n];
                    if let Some(ga) = ga.as_mut() {
                        let bt = shape::transpose2d(&bd[i * k * n..(i + 1) * k * n], k, n);
                        ga.extend(be.matmul(gi, &bt, m, n, k));
                    }
                    if let Some(gb) = gb.as_mut() {
                        let at = shape::transpose2d(&ad[i * m * k..(i + 1) * m * k], m, k);
                        gb.extend(be.matmul(&at, gi, k, m, n));
                    }
                }
                vec![ga, gb]
            }),
        )
    }

    // ------------------------------------------------------------------
    // shape manipulation
    // ------------------------------------------------------------------

    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        let n: usize = new_shape.iter().product();
        assert_eq!(n, self.numel(), "reshape: cannot reshape {:?} into {:?}", self.shape(), new_shape);
        Tensor::from_op(
            self.to_vec(),
            new_shape.to_vec(),
            self.device(),
            vec![self.clone()],
            "ReshapeBackward",
            Box::new(|g| vec![Some(g.to_vec())]),
        )
    }

    /// Flatten all dimensions from `start_dim` onward.
    pub fn flatten(&self, start_dim: usize) -> Tensor {
        let mut s: Vec<usize> = self.shape()[..start_dim].to_vec();
        s.push(self.shape()[start_dim..].iter().product());
        self.reshape(&s)
    }

    /// Insert a dimension of size 1.
    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        let mut s = self.shape().to_vec();
        s.insert(dim, 1);
        self.reshape(&s)
    }

    pub fn permute(&self, perm: &[usize]) -> Tensor {
        let (data, out_shape) = shape::permute(&self.data(), self.shape(), perm);
        let in_shape = self.shape().to_vec();
        let mut inv = vec![0; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inv[p] = i;
        }
        let os = out_shape.clone();
        Tensor::from_op(
            data,
            out_shape,
            self.device(),
            vec![self.clone()],
            "PermuteBackward",
            Box::new(move |g| {
                let (gi, s) = shape::permute(g, &os, &inv);
                debug_assert_eq!(s, in_shape);
                vec![Some(gi)]
            }),
        )
    }

    pub fn transpose(&self, d0: usize, d1: usize) -> Tensor {
        let mut perm: Vec<usize> = (0..self.ndim()).collect();
        perm.swap(d0, d1);
        self.permute(&perm)
    }

    /// Transpose of a 2-D tensor.
    pub fn t(&self) -> Tensor {
        assert_eq!(self.ndim(), 2, "t() expects a 2-D tensor");
        self.transpose(0, 1)
    }

    /// Copy to another device (differentiable).
    pub fn to(&self, device: Device) -> Tensor {
        if device == self.device() {
            return self.clone();
        }
        Tensor::from_op(
            self.to_vec(),
            self.shape().to_vec(),
            device,
            vec![self.clone()],
            "ToDeviceBackward",
            Box::new(|g| vec![Some(g.to_vec())]),
        )
    }

    // ------------------------------------------------------------------
    // reductions
    // ------------------------------------------------------------------

    /// Sum of all elements (scalar).
    pub fn sum(&self) -> Tensor {
        let s: f32 = self.data().iter().sum();
        let n = self.numel();
        Tensor::from_op(vec![s], vec![], self.device(), vec![self.clone()], "SumBackward", Box::new(move |g| vec![Some(vec![g[0]; n])]))
    }

    /// Mean of all elements (scalar).
    pub fn mean(&self) -> Tensor {
        let n = self.numel().max(1) as f32;
        self.sum().mul_scalar(1.0 / n)
    }

    /// Sum along `dim` (negative indexes from the end).
    pub fn sum_dim(&self, dim: isize, keepdim: bool) -> Tensor {
        let d = self.dim_index(dim);
        let (outer, size, inner) = shape::split_at_dim(self.shape(), d);
        let x = self.data();
        let mut out = vec![0.0; outer * inner];
        for o in 0..outer {
            for j in 0..size {
                let base = (o * size + j) * inner;
                for i in 0..inner {
                    out[o * inner + i] += x[base + i];
                }
            }
        }
        drop(x);
        let mut s = self.shape().to_vec();
        if keepdim {
            s[d] = 1;
        } else {
            s.remove(d);
        }
        Tensor::from_op(
            out,
            s,
            self.device(),
            vec![self.clone()],
            "SumDimBackward",
            Box::new(move |g| {
                let mut gx = vec![0.0; outer * size * inner];
                for o in 0..outer {
                    for j in 0..size {
                        let base = (o * size + j) * inner;
                        gx[base..base + inner].copy_from_slice(&g[o * inner..(o + 1) * inner]);
                    }
                }
                vec![Some(gx)]
            }),
        )
    }

    pub fn mean_dim(&self, dim: isize, keepdim: bool) -> Tensor {
        let n = self.size(dim) as f32;
        self.sum_dim(dim, keepdim).mul_scalar(1.0 / n)
    }

    /// Biased variance along `dim`.
    pub fn var_dim(&self, dim: isize, keepdim: bool) -> Tensor {
        let mu = self.mean_dim(dim, true);
        let c = self.sub(&mu);
        c.square().mean_dim(dim, keepdim)
    }

    /// Numerically stable softmax along `dim`.
    pub fn softmax(&self, dim: isize) -> Tensor {
        let d = self.dim_index(dim);
        let (outer, size, inner) = shape::split_at_dim(self.shape(), d);
        let y = softmax_raw(&self.data(), outer, size, inner);
        let yc = y.clone();
        Tensor::from_op(
            y,
            self.shape().to_vec(),
            self.device(),
            vec![self.clone()],
            "SoftmaxBackward",
            Box::new(move |g| {
                let mut gx = vec![0.0; g.len()];
                for o in 0..outer {
                    for i in 0..inner {
                        let idx = |j: usize| (o * size + j) * inner + i;
                        let dot: f32 = (0..size).map(|j| g[idx(j)] * yc[idx(j)]).sum();
                        for j in 0..size {
                            gx[idx(j)] = yc[idx(j)] * (g[idx(j)] - dot);
                        }
                    }
                }
                vec![Some(gx)]
            }),
        )
    }

    /// Numerically stable log-softmax along `dim`.
    pub fn log_softmax(&self, dim: isize) -> Tensor {
        let d = self.dim_index(dim);
        let (outer, size, inner) = shape::split_at_dim(self.shape(), d);
        let sm = softmax_raw(&self.data(), outer, size, inner);
        let y: Vec<f32> = sm.iter().map(|v| v.max(1e-30).ln()).collect();
        Tensor::from_op(
            y,
            self.shape().to_vec(),
            self.device(),
            vec![self.clone()],
            "LogSoftmaxBackward",
            Box::new(move |g| {
                let mut gx = vec![0.0; g.len()];
                for o in 0..outer {
                    for i in 0..inner {
                        let idx = |j: usize| (o * size + j) * inner + i;
                        let gs: f32 = (0..size).map(|j| g[idx(j)]).sum();
                        for j in 0..size {
                            gx[idx(j)] = g[idx(j)] - sm[idx(j)] * gs;
                        }
                    }
                }
                vec![Some(gx)]
            }),
        )
    }

    // ------------------------------------------------------------------
    // convolution / pooling
    // ------------------------------------------------------------------

    /// 2-D convolution. `self`: `[N,C,H,W]`, `weight`: `[O,C,k,k]` -> `[N,O,OH,OW]`.
    pub fn conv2d(&self, weight: &Tensor, stride: usize, padding: usize) -> Tensor {
        assert_same_device(self, weight, "conv2d");
        assert_eq!(self.ndim(), 4, "conv2d input must be [N,C,H,W], got {:?}", self.shape());
        assert_eq!(weight.ndim(), 4, "conv2d weight must be [O,C,k,k]");
        let (n, c, h, w) = (self.shape()[0], self.shape()[1], self.shape()[2], self.shape()[3]);
        let (o, wc, k, k2) = (weight.shape()[0], weight.shape()[1], weight.shape()[2], weight.shape()[3]);
        assert!(wc == c && k == k2, "conv2d: weight {:?} incompatible with input {:?}", weight.shape(), self.shape());
        let be = self.backend();
        let mm = |a: &[f32], b: &[f32], m: usize, kk: usize, nn: usize| be.matmul(a, b, m, kk, nn);
        let (out, oh, ow) = with_pair(self, weight, |x, wt| kernels::conv2d_forward(x, wt, n, c, h, w, o, k, stride, padding, &mm));
        let (x, wt) = (self.clone(), weight.clone());
        let be2 = be.clone();
        Tensor::from_op(
            out,
            vec![n, o, oh, ow],
            self.device(),
            vec![self.clone(), weight.clone()],
            "Conv2dBackward",
            Box::new(move |g| {
                let l = oh * ow;
                let ckk = c * k * k;
                let xd = x.to_vec();
                let wd = wt.to_vec();
                let wt_t = shape::transpose2d(&wd, o, ckk);
                let mut gw = vec![0.0f32; o * ckk];
                let mut gx = x.requires_grad().then(|| vec![0.0f32; n * c * h * w]);
                for b in 0..n {
                    let gb = &g[b * o * l..(b + 1) * o * l];
                    if wt.requires_grad() {
                        let cols = kernels::im2col(&xd[b * c * h * w..(b + 1) * c * h * w], c, h, w, k, stride, padding);
                        let cols_t = shape::transpose2d(&cols, ckk, l);
                        let gwb = be2.matmul(gb, &cols_t, o, l, ckk);
                        gw.iter_mut().zip(gwb).for_each(|(a, v)| *a += v);
                    }
                    if let Some(gx) = gx.as_mut() {
                        let dcols = be2.matmul(&wt_t, gb, ckk, o, l);
                        kernels::col2im(&dcols, c, h, w, k, stride, padding, &mut gx[b * c * h * w..(b + 1) * c * h * w]);
                    }
                }
                vec![gx, wt.requires_grad().then_some(gw)]
            }),
        )
    }

    /// Transposed convolution on `[N,C,H,W]` with weights `[C,O,k,k]`.
    /// Square kernels, unit dilation and one group are supported.
    pub fn conv_transpose2d(&self, weight: &Tensor, stride: usize, padding: usize, output_padding: usize) -> Tensor {
        assert_same_device(self, weight, "conv_transpose2d");
        assert_eq!(self.ndim(), 4, "conv_transpose2d input must be [N,C,H,W]");
        assert_eq!(weight.ndim(), 4, "conv_transpose2d weight must be [C,O,k,k]");
        let (n, c, h, w) = (self.shape()[0], self.shape()[1], self.shape()[2], self.shape()[3]);
        let (wc, o, k, k2) = (weight.shape()[0], weight.shape()[1], weight.shape()[2], weight.shape()[3]);
        assert!(c > 0 && o > 0 && h > 0 && w > 0 && k > 0 && c == wc && k == k2, "conv_transpose2d: incompatible input/weight shapes");
        assert!(stride > 0 && output_padding < stride, "conv_transpose2d requires stride > 0 and output_padding < stride");
        let output_size = |s: usize| {
            (s - 1).checked_mul(stride)
                .and_then(|v| v.checked_add(k))
                .and_then(|v| v.checked_add(output_padding))
                .and_then(|v| padding.checked_mul(2).and_then(|p| v.checked_sub(p)))
                .filter(|&v| v > 0)
                .expect("conv_transpose2d: invalid or overflowing output size")
        };
        let (oh, ow) = (output_size(h), output_size(w));
        let (l, okk) = (h * w, o * k * k);
        let be = self.backend();
        let out = with_pair(self, weight, |x, wt| {
            let wt_t = shape::transpose2d(wt, c, okk);
            let mut out = vec![0.0; n * o * oh * ow];
            for b in 0..n {
                let cols = be.matmul(&wt_t, &x[b * c * l..(b + 1) * c * l], okk, c, l);
                kernels::col2im(&cols, o, oh, ow, k, stride, padding, &mut out[b * o * oh * ow..(b + 1) * o * oh * ow]);
            }
            out
        });
        let (x, wt) = (self.clone(), weight.clone());
        Tensor::from_op(out, vec![n, o, oh, ow], self.device(), vec![self.clone(), weight.clone()], "ConvTranspose2dBackward", Box::new(move |g| {
            let mut gx = x.requires_grad().then(|| vec![0.0; x.numel()]);
            let mut gw = wt.requires_grad().then(|| vec![0.0; wt.numel()]);
            for b in 0..n {
                let cols = kernels::im2col(&g[b * o * oh * ow..(b + 1) * o * oh * ow], o, oh, ow, k, stride, padding);
                if let Some(gx) = gx.as_mut() {
                    let gb = be.matmul(&wt.data(), &cols, c, okk, l);
                    gx[b * c * l..(b + 1) * c * l].copy_from_slice(&gb);
                }
                if let Some(gw) = gw.as_mut() {
                    let cols_t = shape::transpose2d(&cols, okk, l);
                    let gb = be.matmul(&x.data()[b * c * l..(b + 1) * c * l], &cols_t, c, l, okk);
                    gw.iter_mut().zip(gb).for_each(|(a, v)| *a += v);
                }
            }
            vec![gx, gw]
        }))
    }

    /// Average pooling on `[N,C,H,W]` to a fixed `(height, width)`.
    /// Non-divisible input sizes use overlapping floor/ceil pooling regions.
    pub fn adaptive_avg_pool2d(&self, output_size: (usize, usize)) -> Tensor {
        assert_eq!(self.ndim(), 4, "adaptive_avg_pool2d input must be [N,C,H,W]");
        let (n, c, h, w) = (self.shape()[0], self.shape()[1], self.shape()[2], self.shape()[3]);
        let (oh, ow) = output_size;
        assert!(h > 0 && w > 0 && oh > 0 && ow > 0, "adaptive_avg_pool2d spatial sizes must be positive");
        let regions: Vec<_> = (0..oh).flat_map(|y| (0..ow).map(move |x| {
            (y * h / oh, ((y + 1) * h).div_ceil(oh), x * w / ow, ((x + 1) * w).div_ceil(ow))
        })).collect();
        let mut out = Vec::with_capacity(n * c * oh * ow);
        let d = self.data();
        for plane in 0..n * c {
            for &(y0, y1, x0, x1) in &regions {
                let mut sum = 0.0;
                for y in y0..y1 {
                    for x in x0..x1 {
                        sum += d[(plane * h + y) * w + x];
                    }
                }
                out.push(sum / ((y1 - y0) * (x1 - x0)) as f32);
            }
        }
        let numel = self.numel();
        Tensor::from_op(out, vec![n, c, oh, ow], self.device(), vec![self.clone()], "AdaptiveAvgPool2dBackward", Box::new(move |g| {
            let mut gx = vec![0.0; numel];
            for plane in 0..n * c {
                for (i, &(y0, y1, x0, x1)) in regions.iter().enumerate() {
                    let value = g[plane * oh * ow + i] / ((y1 - y0) * (x1 - x0)) as f32;
                    for y in y0..y1 {
                        for x in x0..x1 {
                            gx[(plane * h + y) * w + x] += value;
                        }
                    }
                }
            }
            vec![Some(gx)]
        }))
    }

    /// 2-D max pooling on `[N,C,H,W]`.
    pub fn max_pool2d(&self, kernel: usize, stride: usize) -> Tensor {
        assert_eq!(self.ndim(), 4, "max_pool2d input must be [N,C,H,W]");
        let (n, c, h, w) = (self.shape()[0], self.shape()[1], self.shape()[2], self.shape()[3]);
        let (out, arg) = kernels::max_pool2d(&self.data(), n, c, h, w, kernel, stride);
        let oh = kernels::out_size(h, kernel, stride, 0);
        let ow = kernels::out_size(w, kernel, stride, 0);
        let numel = self.numel();
        Tensor::from_op(
            out,
            vec![n, c, oh, ow],
            self.device(),
            vec![self.clone()],
            "MaxPool2dBackward",
            Box::new(move |g| {
                let mut gx = vec![0.0; numel];
                for (gv, &i) in g.iter().zip(&arg) {
                    gx[i] += gv;
                }
                vec![Some(gx)]
            }),
        )
    }
}

use crate::device::Device;

pub(crate) fn softmax_raw(x: &[f32], outer: usize, size: usize, inner: usize) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    for o in 0..outer {
        for i in 0..inner {
            let idx = |j: usize| (o * size + j) * inner + i;
            let mx = (0..size).map(|j| x[idx(j)]).fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0.0;
            for j in 0..size {
                let e = (x[idx(j)] - mx).exp();
                y[idx(j)] = e;
                s += e;
            }
            for j in 0..size {
                y[idx(j)] /= s;
            }
        }
    }
    y
}

// ----------------------------------------------------------------------
// operator overloading:  &a + &b, a * 2.0, -a, ...
// ----------------------------------------------------------------------

macro_rules! impl_binop {
    ($trait:ident, $method:ident, $func:ident, $scalar:expr) => {
        impl $trait<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::$func(self, rhs)
            }
        }
        impl $trait<Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                Tensor::$func(&self, &rhs)
            }
        }
        impl $trait<&Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::$func(&self, rhs)
            }
        }
        impl $trait<Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                Tensor::$func(self, &rhs)
            }
        }
        impl $trait<f32> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f32) -> Tensor {
                let f: fn(&Tensor, f32) -> Tensor = $scalar;
                f(self, rhs)
            }
        }
        impl $trait<f32> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f32) -> Tensor {
                let f: fn(&Tensor, f32) -> Tensor = $scalar;
                f(&self, rhs)
            }
        }
    };
}

impl_binop!(Add, add, add, |t, s| t.add_scalar(s));
impl_binop!(Sub, sub, sub, |t, s| t.add_scalar(-s));
impl_binop!(Mul, mul, mul, |t, s| t.mul_scalar(s));
impl_binop!(Div, div, div, |t, s| t.mul_scalar(1.0 / s));

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self)
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(&self)
    }
}
