//! Raw (non-autograd) numeric kernels shared by training ops and the inference
//! runtime: im2col / col2im convolution helpers and max pooling.

use crate::device::cpu::cpu_matmul_into;
use rayon::prelude::*;

/// Output spatial size of a convolution / pooling window.
pub fn out_size(input: usize, kernel: usize, stride: usize, pad: usize) -> usize {
    assert!(kernel > 0 && stride > 0, "kernel and stride must be positive");
    let padded = pad.checked_mul(2).and_then(|p| input.checked_add(p)).expect("padded input size overflow");
    assert!(padded >= kernel, "kernel {kernel} larger than padded input {padded}");
    (padded - kernel) / stride + 1
}

/// Unfold one image `[C,H,W]` into columns `[C*k*k, OH*OW]`.
pub fn im2col(x: &[f32], c: usize, h: usize, w: usize, k: usize, stride: usize, pad: usize) -> Vec<f32> {
    let oh = out_size(h, k, stride, pad);
    let ow = out_size(w, k, stride, pad);
    let mut cols = vec![0.0f32; c * k * k * oh * ow];
    im2col_into(x, c, h, w, k, stride, pad, oh, ow, &mut cols);
    cols
}

/// Unfold one image `[C,H,W]` into a pre-allocated buffer `cols` of length `C*k*k*OH*OW`.
#[allow(clippy::too_many_arguments)]
pub fn im2col_into(
    x: &[f32],
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    stride: usize,
    pad: usize,
    oh: usize,
    ow: usize,
    cols: &mut [f32],
) {
    let l = oh * ow;
    if pad > 0 {
        cols.fill(0.0);
    }
    if stride == 1 {
        for ch in 0..c {
            let ch_off = ch * h * w;
            for ki in 0..k {
                let dy = ki as isize - pad as isize;
                let oy_start = (-dy).max(0) as usize;
                let oy_end = (h as isize - dy).min(oh as isize).max(0) as usize;
                for kj in 0..k {
                    let dx = kj as isize - pad as isize;
                    let ox_start = (-dx).max(0) as usize;
                    let ox_end = (w as isize - dx).min(ow as isize).max(0) as usize;
                    if ox_start >= ox_end {
                        continue;
                    }
                    let ix_start = (ox_start as isize + dx) as usize;
                    let copy_len = ox_end - ox_start;
                    let row = (ch * k + ki) * k + kj;
                    let base = row * l;
                    for oy in oy_start..oy_end {
                        let iy = (oy as isize + dy) as usize;
                        let dst_off = base + oy * ow + ox_start;
                        let src_off = ch_off + iy * w + ix_start;
                        cols[dst_off..dst_off + copy_len].copy_from_slice(&x[src_off..src_off + copy_len]);
                    }
                }
            }
        }
    } else {
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
    }
}

/// Fold columns `[C*k*k, OH*OW]` back into an image `[C,H,W]` (accumulating).
#[allow(clippy::too_many_arguments)]
pub fn col2im(cols: &[f32], c: usize, h: usize, w: usize, k: usize, stride: usize, pad: usize, out: &mut [f32]) {
    let oh = out_size(h, k, stride, pad);
    let ow = out_size(w, k, stride, pad);
    let l = oh * ow;
    if stride == 1 {
        for ch in 0..c {
            let ch_off = ch * h * w;
            for ki in 0..k {
                let dy = ki as isize - pad as isize;
                let oy_start = (-dy).max(0) as usize;
                let oy_end = (h as isize - dy).min(oh as isize).max(0) as usize;
                for kj in 0..k {
                    let dx = kj as isize - pad as isize;
                    let ox_start = (-dx).max(0) as usize;
                    let ox_end = (w as isize - dx).min(ow as isize).max(0) as usize;
                    if ox_start >= ox_end {
                        continue;
                    }
                    let ix_start = (ox_start as isize + dx) as usize;
                    let copy_len = ox_end - ox_start;
                    let row = (ch * k + ki) * k + kj;
                    let base = row * l;
                    for oy in oy_start..oy_end {
                        let iy = (oy as isize + dy) as usize;
                        let src_off = base + oy * ow + ox_start;
                        let dst_off = ch_off + iy * w + ix_start;
                        let dst = &mut out[dst_off..dst_off + copy_len];
                        let src = &cols[src_off..src_off + copy_len];
                        for (d, &s) in dst.iter_mut().zip(src) {
                            *d += s;
                        }
                    }
                }
            }
        }
    } else {
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
}

/// Max pooling over `[N,C,H,W]`. Returns `(output, argmax_indices)` where each
/// index points into the flattened input.
pub fn max_pool2d(x: &[f32], n: usize, c: usize, h: usize, w: usize, k: usize, stride: usize) -> (Vec<f32>, Vec<usize>) {
    let oh = out_size(h, k, stride, 0);
    let ow = out_size(w, k, stride, 0);
    let l = oh * ow;
    let total = n * c * l;
    let mut out = vec![0.0f32; total];
    let mut arg = vec![0usize; total];

    let pool_plane = |(plane, (out_plane, arg_plane)): (usize, (&mut [f32], &mut [usize]))| {
        let base = plane * h * w;
        for oy in 0..oh {
            let iy0 = oy * stride;
            for ox in 0..ow {
                let ix0 = ox * stride;
                let mut best = f32::NEG_INFINITY;
                let mut bi = base + iy0 * w + ix0;
                for ki in 0..k {
                    let row_base = base + (iy0 + ki) * w + ix0;
                    for kj in 0..k {
                        let idx = row_base + kj;
                        let v = x[idx];
                        if v > best || v.is_nan() {
                            best = v;
                            bi = idx;
                        }
                    }
                }
                let out_idx = oy * ow + ox;
                out_plane[out_idx] = best;
                arg_plane[out_idx] = bi;
            }
        }
    };

    if total >= 8192 && l > 0 {
        out.par_chunks_mut(l)
            .zip(arg.par_chunks_mut(l))
            .enumerate()
            .for_each(pool_plane);
    } else if l > 0 {
        out.chunks_mut(l)
            .zip(arg.chunks_mut(l))
            .enumerate()
            .for_each(pool_plane);
    }
    (out, arg)
}

/// Convolution forward for a batch (`W[O, C*k*k] @ cols[C*k*k, L]`).
/// Parallelizes across batch samples using thread-local `im2col` buffers and AVX2 GEMM.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
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
    let img_stride = c * h * w;
    let out_stride = o * l;

    if n >= 2 && n * o * ckk * l >= 16384 && out_stride > 0 {
        let mut out = vec![0.0f32; n * out_stride];
        out.par_chunks_mut(out_stride)
            .enumerate()
            .for_each_init(
                || vec![0.0f32; ckk * l],
                |cols, (b, out_b)| {
                    let img = &x[b * img_stride..(b + 1) * img_stride];
                    im2col_into(img, c, h, w, k, stride, pad, oh, ow, cols);
                    cpu_matmul_into(weight, cols, out_b, o, ckk, l);
                },
            );
        (out, oh, ow)
    } else {
        let mut out = Vec::with_capacity(n * out_stride);
        let mut cols = vec![0.0f32; ckk * l];
        for b in 0..n {
            let img = &x[b * img_stride..(b + 1) * img_stride];
            im2col_into(img, c, h, w, k, stride, pad, oh, ow, &mut cols);
            out.extend(matmul(weight, &cols, o, ckk, l));
        }
        (out, oh, ow)
    }
}
