//! Raw (non-autograd) numeric kernels shared by training ops and the inference
//! runtime: im2col / col2im convolution helpers and max pooling.

/// Output spatial size of a convolution / pooling window.
pub fn out_size(input: usize, kernel: usize, stride: usize, pad: usize) -> usize {
    assert!(input + 2 * pad >= kernel, "kernel {kernel} larger than padded input {}", input + 2 * pad);
    (input + 2 * pad - kernel) / stride + 1
}

/// Unfold one image `[C,H,W]` into columns `[C*k*k, OH*OW]`.
pub fn im2col(x: &[f32], c: usize, h: usize, w: usize, k: usize, stride: usize, pad: usize) -> Vec<f32> {
    let oh = out_size(h, k, stride, pad);
    let ow = out_size(w, k, stride, pad);
    let l = oh * ow;
    let mut cols = vec![0.0f32; c * k * k * l];
    for ch in 0..c {
        for ki in 0..k {
            for kj in 0..k {
                let row = (ch * k + ki) * k + kj;
                let base = row * l;
                for oy in 0..oh {
                    let iy = (oy * stride + ki) as isize - pad as isize;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for ox in 0..ow {
                        let ix = (ox * stride + kj) as isize - pad as isize;
                        if ix < 0 || ix >= w as isize {
                            continue;
                        }
                        cols[base + oy * ow + ox] = x[(ch * h + iy as usize) * w + ix as usize];
                    }
                }
            }
        }
    }
    cols
}

/// Fold columns `[C*k*k, OH*OW]` back into an image `[C,H,W]` (accumulating).
#[allow(clippy::too_many_arguments)]
pub fn col2im(cols: &[f32], c: usize, h: usize, w: usize, k: usize, stride: usize, pad: usize, out: &mut [f32]) {
    let oh = out_size(h, k, stride, pad);
    let ow = out_size(w, k, stride, pad);
    let l = oh * ow;
    for ch in 0..c {
        for ki in 0..k {
            for kj in 0..k {
                let row = (ch * k + ki) * k + kj;
                let base = row * l;
                for oy in 0..oh {
                    let iy = (oy * stride + ki) as isize - pad as isize;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for ox in 0..ow {
                        let ix = (ox * stride + kj) as isize - pad as isize;
                        if ix < 0 || ix >= w as isize {
                            continue;
                        }
                        out[(ch * h + iy as usize) * w + ix as usize] += cols[base + oy * ow + ox];
                    }
                }
            }
        }
    }
}

/// Max pooling over `[N,C,H,W]`. Returns `(output, argmax_indices)` where each
/// index points into the flattened input.
pub fn max_pool2d(x: &[f32], n: usize, c: usize, h: usize, w: usize, k: usize, stride: usize) -> (Vec<f32>, Vec<usize>) {
    let oh = out_size(h, k, stride, 0);
    let ow = out_size(w, k, stride, 0);
    let mut out = Vec::with_capacity(n * c * oh * ow);
    let mut arg = Vec::with_capacity(n * c * oh * ow);
    for plane in 0..n * c {
        let base = plane * h * w;
        for oy in 0..oh {
            for ox in 0..ow {
                let mut best = f32::NEG_INFINITY;
                let mut bi = 0;
                for ki in 0..k {
                    for kj in 0..k {
                        let idx = base + (oy * stride + ki) * w + ox * stride + kj;
                        if x[idx] > best {
                            best = x[idx];
                            bi = idx;
                        }
                    }
                }
                out.push(best);
                arg.push(bi);
            }
        }
    }
    (out, arg)
}

/// Convolution forward for a batch using a matmul function
/// (`W[O, C*k*k] @ cols[C*k*k, L]`). Returns `[N, O, OH, OW]` data.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_forward(
    x: &[f32],
    weight: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    o: usize,
    k: usize,
    stride: usize,
    pad: usize,
    matmul: &dyn Fn(&[f32], &[f32], usize, usize, usize) -> Vec<f32>,
) -> (Vec<f32>, usize, usize) {
    let oh = out_size(h, k, stride, pad);
    let ow = out_size(w, k, stride, pad);
    let l = oh * ow;
    let ckk = c * k * k;
    let mut out = Vec::with_capacity(n * o * l);
    for b in 0..n {
        let img = &x[b * c * h * w..(b + 1) * c * h * w];
        let cols = im2col(img, c, h, w, k, stride, pad);
        out.extend(matmul(weight, &cols, o, ckk, l));
    }
    (out, oh, ow)
}
