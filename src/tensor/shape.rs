//! Shape utilities: strides, broadcasting and permutation.

/// Row-major (C-contiguous) strides.
pub fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// NumPy-style broadcast of two shapes.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let n = a.len().max(b.len());
    let mut out = vec![0; n];
    for i in 0..n {
        let da = if i + a.len() >= n { a[i + a.len() - n] } else { 1 };
        let db = if i + b.len() >= n { b[i + b.len() - n] } else { 1 };
        out[i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return Err(format!("shapes {a:?} and {b:?} are not broadcastable"));
        };
    }
    Ok(out)
}

/// For every element of `out_shape`, the linear index into a tensor of
/// `src_shape` that broadcasts to it.
pub fn broadcast_index_map(src: &[usize], out: &[usize]) -> Vec<usize> {
    let n = out.len();
    let off = n - src.len();
    let sstr = strides(src);
    // effective stride per output dim (0 where broadcast)
    let eff: Vec<usize> = (0..n)
        .map(|i| if i < off || src[i - off] == 1 { 0 } else { sstr[i - off] })
        .collect();
    let total: usize = out.iter().product();
    let mut map = Vec::with_capacity(total);
    let mut idx = vec![0usize; n];
    let mut cur = 0usize;
    for _ in 0..total {
        map.push(cur);
        // increment multi-index
        for d in (0..n).rev() {
            idx[d] += 1;
            cur += eff[d];
            if idx[d] < out[d] {
                break;
            }
            cur -= eff[d] * idx[d];
            idx[d] = 0;
        }
    }
    map
}

/// Sum a gradient of shape `out` down to `src` using a broadcast index map.
pub fn reduce_by_map(grad: &[f32], map: &[usize], src_numel: usize) -> Vec<f32> {
    let mut g = vec![0.0; src_numel];
    for (v, &i) in grad.iter().zip(map) {
        g[i] += v;
    }
    g
}

/// Permute dimensions of a contiguous buffer.
pub fn permute(data: &[f32], shape: &[usize], perm: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let n = shape.len();
    assert_eq!(perm.len(), n, "permute: wrong number of dims");
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let in_str = strides(shape);
    let eff: Vec<usize> = perm.iter().map(|&p| in_str[p]).collect();
    let total = data.len();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; n];
    let mut cur = 0usize;
    for _ in 0..total {
        out.push(data[cur]);
        for d in (0..n).rev() {
            idx[d] += 1;
            cur += eff[d];
            if idx[d] < out_shape[d] {
                break;
            }
            cur -= eff[d] * idx[d];
            idx[d] = 0;
        }
    }
    (out, out_shape)
}

/// Transpose a row-major `rows x cols` matrix into a new vector.
pub fn transpose2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    transpose2d_into(data, rows, cols, &mut out);
    out
}

/// Cache-tiled transpose of a row-major `rows x cols` matrix into `out`.
pub fn transpose2d_into(data: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    const TILE: usize = 32;
    let mut r0 = 0;
    while r0 < rows {
        let r1 = (r0 + TILE).min(rows);
        let mut c0 = 0;
        while c0 < cols {
            let c1 = (c0 + TILE).min(cols);
            for r in r0..r1 {
                let row_off = r * cols;
                for c in c0..c1 {
                    out[c * rows + r] = data[row_off + c];
                }
            }
            c0 = c1;
        }
        r0 = r1;
    }
}

/// Split a shape around `dim` into (outer, dim_size, inner).
pub fn split_at_dim(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    (outer, shape[dim], inner)
}
