//! 텐서 형태(Shape) 유틸리티: 스트라이드(stride), 브로드캐스팅(broadcasting), 차원 순열 치환(permutation).

/// 행 우선(Row-major, C-contiguous) 방식의 차원별 스트라이드(stride)를 계산합니다.
pub fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// NumPy 스타일 브로드캐스팅 규칙에 따라 두 텐서 형상(`a`, `b`)을 브로드캐스팅한 결과 형상을 계산합니다.
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

/// `src_shape`를 `out_shape`로 브로드캐스트할 때,
/// 출력의 각 원소가 읽어야 할 원본 데이터의 1차원 선형 인덱스 맵을 생성합니다.
pub fn broadcast_index_map(src: &[usize], out: &[usize]) -> Vec<usize> {
    let n = out.len();
    let off = n - src.len();
    let sstr = strides(src);
    // 브로드캐스트된 차원은 스트라이드가 0이 됨
    let eff: Vec<usize> = (0..n)
        .map(|i| if i < off || src[i - off] == 1 { 0 } else { sstr[i - off] })
        .collect();
    let total: usize = out.iter().product();
    let mut map = Vec::with_capacity(total);
    let mut idx = vec![0usize; n];
    let mut cur = 0usize;
    for _ in 0..total {
        map.push(cur);
        // 다차원 인덱스 1 증가
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

/// 브로드캐스트 인덱스 맵을 역으로 순회하여 출력 기울기를 원래 입력 크기(`src_numel`)로 축소 합산(reduce sum)합니다.
pub fn reduce_by_map(grad: &[f32], map: &[usize], src_numel: usize) -> Vec<f32> {
    let mut g = vec![0.0; src_numel];
    for (v, &i) in grad.iter().zip(map) {
        g[i] += v;
    }
    g
}

/// 연속된 메모리 버퍼의 차원 축 순서를 주어진 순열(`perm`)에 따라 재배치합니다.
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

/// 행 우선(Row-major) `rows x cols` 행렬을 전치(Transpose)하여 새로운 벡터로 반환합니다.
pub fn transpose2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    transpose2d_into(data, rows, cols, &mut out);
    out
}

/// 캐시 지역성(Tile: 32x32)을 고려하여 `rows x cols` 행렬을 `out` 버퍼에 고속으로 전치합니다.
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

/// 형상을 특정 차원(`dim`)을 기준으로 `(outer, dim_size, inner)` 3개 크기로 분할합니다.
pub fn split_at_dim(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    (outer, shape[dim], inner)
}
