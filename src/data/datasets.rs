//! Synthetic datasets for experiments and examples.

use super::TensorDataset;
use crate::rng::Rng;
use crate::tensor::Tensor;

/// Interleaved spirals: `classes` arms with `n_per_class` 2-D points each.
pub fn spirals(n_per_class: usize, classes: usize, noise: f32, seed: u64) -> TensorDataset {
    let mut rng = Rng::new(seed);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for c in 0..classes {
        for i in 0..n_per_class {
            let r = i as f32 / n_per_class as f32;
            let t = c as f32 * 4.0 + r * 4.0 + rng.normal() * noise;
            x.push(r * t.sin());
            x.push(r * t.cos());
            y.push(c as f32);
        }
    }
    let n = n_per_class * classes;
    TensorDataset::new(&Tensor::from_vec(x, &[n, 2]), &Tensor::from_vec(y, &[n]))
}

/// Gaussian blobs in `dim` dimensions.
pub fn blobs(n_per_class: usize, classes: usize, dim: usize, spread: f32, seed: u64) -> TensorDataset {
    let mut rng = Rng::new(seed);
    let centers: Vec<Vec<f32>> = (0..classes).map(|_| (0..dim).map(|_| rng.uniform_range(-5.0, 5.0)).collect()).collect();
    let mut x = Vec::new();
    let mut y = Vec::new();
    for (c, center) in centers.iter().enumerate() {
        for _ in 0..n_per_class {
            for &m in center {
                x.push(m + rng.normal() * spread);
            }
            y.push(c as f32);
        }
    }
    let n = n_per_class * classes;
    TensorDataset::new(&Tensor::from_vec(x, &[n, dim]), &Tensor::from_vec(y, &[n]))
}

/// Noisy regression data `y = sin(x) + 0.1 x^2 + noise`, `x in [-3, 3]`.
pub fn sine_regression(n: usize, noise: f32, seed: u64) -> TensorDataset {
    let mut rng = Rng::new(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let v = rng.uniform_range(-3.0, 3.0);
        x.push(v);
        y.push(v.sin() + 0.1 * v * v + rng.normal() * noise);
    }
    TensorDataset::new(&Tensor::from_vec(x, &[n, 1]), &Tensor::from_vec(y, &[n, 1]))
}

/// Tiny image dataset `[N, 1, size, size]` with 4 classes of strokes:
/// 0 = horizontal, 1 = vertical, 2 = diagonal "\", 3 = anti-diagonal "/".
pub fn stroke_images(n: usize, size: usize, noise: f32, seed: u64) -> TensorDataset {
    let mut rng = Rng::new(seed);
    let mut x = Vec::with_capacity(n * size * size);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let class = rng.below(4);
        let mut img = vec![0.0f32; size * size];
        let off = rng.below(size) as isize;
        let thick = 1 + rng.below(2) as isize;
        for i in 0..size as isize {
            for t in 0..thick {
                let (r, c) = match class {
                    0 => (off + t, i),
                    1 => (i, off + t),
                    2 => (i, i + off - size as isize / 2 + t),
                    _ => (i, size as isize - 1 - i + off - size as isize / 2 + t),
                };
                if r >= 0 && c >= 0 && (r as usize) < size && (c as usize) < size {
                    img[r as usize * size + c as usize] = 1.0;
                }
            }
        }
        for p in img.iter_mut() {
            *p += rng.normal() * noise;
        }
        x.extend(img);
        y.push(class as f32);
    }
    TensorDataset::new(&Tensor::from_vec(x, &[n, 1, size, size]), &Tensor::from_vec(y, &[n]))
}

/// Sequence task for Transformers: input is a random token sequence, target is
/// the same sequence sorted ascending (`[N, T]` -> `[N, T]`).
pub fn sort_sequences(n: usize, seq_len: usize, vocab: usize, seed: u64) -> TensorDataset {
    let mut rng = Rng::new(seed);
    let mut x = Vec::with_capacity(n * seq_len);
    let mut y = Vec::with_capacity(n * seq_len);
    for _ in 0..n {
        let s: Vec<usize> = (0..seq_len).map(|_| rng.below(vocab)).collect();
        let mut sorted = s.clone();
        sorted.sort();
        x.extend(s.iter().map(|&v| v as f32));
        y.extend(sorted.iter().map(|&v| v as f32));
    }
    TensorDataset::new(&Tensor::from_vec(x, &[n, seq_len]), &Tensor::from_vec(y, &[n, seq_len]))
}
