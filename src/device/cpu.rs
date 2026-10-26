//! Multi-threaded CPU backend (rayon work-stealing thread pool).

use super::backend::{Backend, BinaryOp, UnaryOp};
use rayon::prelude::*;

/// Below this many elements the work is executed on the calling thread.
const PAR_THRESHOLD: usize = 16 * 1024;

#[derive(Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn name(&self) -> String {
        format!("cpu (rayon x{})", rayon::current_num_threads())
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        cpu_matmul(a, b, m, k, n)
    }

    fn binary(&self, op: BinaryOp, a: &[f32], b: &[f32]) -> Vec<f32> {
        debug_assert_eq!(a.len(), b.len());
        if a.len() >= PAR_THRESHOLD {
            a.par_iter().zip(b.par_iter()).map(|(&x, &y)| op.apply(x, y)).collect()
        } else {
            a.iter().zip(b.iter()).map(|(&x, &y)| op.apply(x, y)).collect()
        }
    }

    fn unary(&self, op: UnaryOp, a: &[f32]) -> Vec<f32> {
        if a.len() >= PAR_THRESHOLD {
            a.par_iter().map(|&x| op.apply(x)).collect()
        } else {
            a.iter().map(|&x| op.apply(x)).collect()
        }
    }
}

/// Cache friendly (i-k-j) matmul, parallelised over output rows.
pub fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matmul: lhs size mismatch");
    assert_eq!(b.len(), k * n, "matmul: rhs size mismatch");
    let mut c = vec![0.0f32; m * n];
    if m == 0 || n == 0 {
        return c;
    }
    let row = |(i, crow): (usize, &mut [f32])| {
        let arow = &a[i * k..(i + 1) * k];
        for (p, &av) in arow.iter().enumerate() {
            if av == 0.0 {
                continue;
            }
            let brow = &b[p * n..(p + 1) * n];
            for (cv, &bv) in crow.iter_mut().zip(brow) {
                *cv += av * bv;
            }
        }
    };
    if m * k * n >= 32 * 1024 {
        c.par_chunks_mut(n).enumerate().for_each(row);
    } else {
        c.chunks_mut(n).enumerate().for_each(row);
    }
    c
}
