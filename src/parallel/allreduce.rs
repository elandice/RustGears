//! Ring all-reduce.
//!
//! `N` workers are arranged in a ring and every buffer is split into `N`
//! chunks. The algorithm runs in two phases of `N-1` steps each:
//!
//! 1. **reduce-scatter** – each worker sends one chunk to its right neighbour,
//!    which adds it to its own copy. After `N-1` steps worker `r` owns the fully
//!    reduced chunk `(r+1) mod N`.
//! 2. **all-gather** – the reduced chunks circulate around the ring until every
//!    worker has all of them.
//!
//! Each worker transmits `2 (N-1)/N` times the buffer size, independent of `N`
//! — which is why it scales far better than a central parameter server.

use rayon::prelude::*;

/// Communication statistics of one all-reduce.
#[derive(Clone, Debug, Default)]
pub struct AllReduceStats {
    pub workers: usize,
    pub elements: usize,
    pub steps: usize,
    /// Total bytes sent by all workers.
    pub bytes_sent: usize,
}

fn chunk_range(len: usize, n: usize, c: usize) -> std::ops::Range<usize> {
    (c * len / n)..((c + 1) * len / n)
}

/// Sum `buffers` element-wise in place; afterwards every buffer contains the
/// total. All buffers must have equal length.
pub fn ring_all_reduce(buffers: &mut [Vec<f32>]) -> AllReduceStats {
    let n = buffers.len();
    let len = buffers.first().map(|b| b.len()).unwrap_or(0);
    assert!(buffers.iter().all(|b| b.len() == len), "ring_all_reduce: buffers must have equal length");
    let mut stats = AllReduceStats { workers: n, elements: len, steps: 0, bytes_sent: 0 };
    if n <= 1 {
        return stats;
    }

    // Phase 1: reduce-scatter
    for step in 0..n - 1 {
        // all workers send simultaneously: rank r sends chunk (r - step) to r+1
        let msgs: Vec<(usize, usize, Vec<f32>)> = (0..n)
            .into_par_iter()
            .map(|r| {
                let c = (r + n - step) % n;
                (((r + 1) % n), c, buffers[r][chunk_range(len, n, c)].to_vec())
            })
            .collect();
        for (dst, c, data) in msgs {
            stats.bytes_sent += data.len() * 4;
            let range = chunk_range(len, n, c);
            for (a, b) in buffers[dst][range].iter_mut().zip(data) {
                *a += b;
            }
        }
        stats.steps += 1;
    }

    // Phase 2: all-gather. Rank r now owns reduced chunk (r+1) mod n.
    for step in 0..n - 1 {
        let msgs: Vec<(usize, usize, Vec<f32>)> = (0..n)
            .into_par_iter()
            .map(|r| {
                let c = (r + 1 + n - step) % n;
                (((r + 1) % n), c, buffers[r][chunk_range(len, n, c)].to_vec())
            })
            .collect();
        for (dst, c, data) in msgs {
            stats.bytes_sent += data.len() * 4;
            let range = chunk_range(len, n, c);
            buffers[dst][range].copy_from_slice(&data);
        }
        stats.steps += 1;
    }
    stats
}
