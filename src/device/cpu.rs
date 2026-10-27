//! 멀티스레드 CPU 연산 백엔드 (Rayon 작업 훔치기(work-stealing) 스레드 풀 + AVX2/FMA SIMD 마이크로커널).

use super::backend::{Backend, BinaryOp, UnaryOp};
use rayon::prelude::*;

/// 병렬 오버헤드를 피하기 위해 이 개수 미만의 요소는 호출 스레드에서 동기 실행합니다.
const PAR_THRESHOLD: usize = 32 * 1024;
/// 행렬 곱셈 시 Rayon 멀티스레드 태스크를 생성하기 위한 최소 `m * k * n` 연산량 기준
const MATMUL_PAR_THRESHOLD: usize = 128 * 1024;

/// Rayon 및 AVX2 SIMD 기반 CPU 연산 백엔드
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

/// 캐시 블록화 및 AVX2+FMA 레지스터 블록화가 적용된 CPU 행렬 곱셈 (`C[m,n] = A[m,k] @ B[k,n]`).
pub fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matmul: lhs size mismatch");
    assert_eq!(b.len(), k * n, "matmul: rhs size mismatch");
    let mut c = vec![0.0f32; m * n];
    if m == 0 || n == 0 || k == 0 {
        return c;
    }
    if m * k * n >= MATMUL_PAR_THRESHOLD && m >= 8 {
        // 각 워커가 완전한 4x16 레지스터 블록을 실행할 수 있도록 4개 행 단위로 분할 처리
        let rows_per_chunk = 4;
        let chunk_elems = rows_per_chunk * n;
        c.par_chunks_mut(chunk_elems).enumerate().for_each(|(chunk_idx, c_chunk)| {
            let row_start = chunk_idx * rows_per_chunk;
            let num_rows = c_chunk.len() / n;
            let a_chunk = &a[row_start * k..(row_start + num_rows) * k];
            matmul_kernel(a_chunk, b, c_chunk, num_rows, k, n);
        });
    } else {
        matmul_kernel(a, b, &mut c, m, k, n);
    }
    c
}

/// 단일 스레드로 새 버퍼에 행렬 곱셈을 수행합니다 (바깥쪽 병렬 루프 내부에서 사용).
#[inline]
pub fn cpu_matmul_serial(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    if m > 0 && n > 0 && k > 0 {
        matmul_kernel(a, b, &mut c, m, k, n);
    }
    c
}

/// 사전에 할당된 `c` 슬라이스에 직접 행렬 곱 결과를 쓰는 단일 스레드 함수 (먼저 0으로 채움).
#[inline]
pub fn cpu_matmul_into(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    c.fill(0.0);
    if m > 0 && n > 0 && k > 0 {
        matmul_kernel(a, b, c, m, k, n);
    }
}

/// 기존 `c` 슬라이스에 행렬 곱셈 결과를 누적(`C += A @ B`)하는 단일 스레드 함수.
#[inline]
pub fn cpu_matmul_add_into(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if m > 0 && n > 0 && k > 0 {
        matmul_kernel(a, b, c, m, k, n);
    }
}

/// `B`가 `[n, k]` 형태로 저장되어 있을 때 전치하지 않고 직접 누적하는 단일 스레드 행렬 곱 (`C[m, n] += A[m, k] @ B[n, k]^T`).
#[inline]
pub fn cpu_matmul_nt_add_into(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                matmul_nt_avx2(a, b, c, m, n, k);
            }
            return;
        }
    }
    // 스칼라 폴백 경로
    for i in 0..m {
        let arow = &a[i * k..(i + 1) * k];
        let crow = &mut c[i * n..(i + 1) * n];
        for j in 0..n {
            let brow = &b[j * k..(j + 1) * k];
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += arow[p] * brow[p];
            }
            crow[j] += sum;
        }
    }
}

/// AVX2 + FMA 명령어를 사용한 `matmul_nt` SIMD 마이크로커널
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_nt_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    for i in 0..m {
        let ai = a_ptr.add(i * k);
        for j in 0..n {
            let bj = b_ptr.add(j * k);
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut p = 0;
            while p + 16 <= k {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ai.add(p)), _mm256_loadu_ps(bj.add(p)), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(ai.add(p + 8)), _mm256_loadu_ps(bj.add(p + 8)), acc1);
                p += 16;
            }
            if p + 8 <= k {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ai.add(p)), _mm256_loadu_ps(bj.add(p)), acc0);
                p += 8;
            }
            let sum256 = _mm256_add_ps(acc0, acc1);
            let hi128 = _mm256_extractf128_ps(sum256, 1);
            let lo128 = _mm256_castps256_ps128(sum256);
            let sum128 = _mm_add_ps(lo128, hi128);
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            let mut total = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
            while p < k {
                total += *ai.add(p) * *bj.add(p);
                p += 1;
            }
            *c_ptr.add(i * n + j) += total;
        }
    }
}

/// 런타임 하드웨어 기능을 확인하여 최적화된 행렬 곱 커널을 호출합니다.
#[inline]
fn matmul_kernel(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2 및 fma 지원 여부를 런타임에 확인 완료
            unsafe {
                matmul_avx2(a, b, c, m, k, n);
            }
            return;
        }
    }
    matmul_scalar(a, b, c, m, k, n);
}

/// 캐시 타일링(`KC = 128`)과 4행 언롤링이 적용된 스칼라 행렬 곱셈 폴백 함수
fn matmul_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    const KC: usize = 128;
    let mut p0 = 0;
    while p0 < k {
        let p1 = (p0 + KC).min(k);
        let mut i = 0;
        while i + 4 <= m {
            let (c0, rest) = c[i * n..(i + 4) * n].split_at_mut(n);
            let (c1, rest) = rest.split_at_mut(n);
            let (c2, c3) = rest.split_at_mut(n);
            for p in p0..p1 {
                let a0 = a[i * k + p];
                let a1 = a[(i + 1) * k + p];
                let a2 = a[(i + 2) * k + p];
                let a3 = a[(i + 3) * k + p];
                let brow = &b[p * n..(p + 1) * n];
                for j in 0..n {
                    let bv = brow[j];
                    c0[j] += a0 * bv;
                    c1[j] += a1 * bv;
                    c2[j] += a2 * bv;
                    c3[j] += a3 * bv;
                }
            }
            i += 4;
        }
        while i < m {
            let crow = &mut c[i * n..(i + 1) * n];
            let arow = &a[i * k..(i + 1) * k];
            for p in p0..p1 {
                let av = arow[p];
                if av == 0.0 {
                    continue;
                }
                let brow = &b[p * n..(p + 1) * n];
                for (cv, &bv) in crow.iter_mut().zip(brow) {
                    *cv += av * bv;
                }
            }
            i += 1;
        }
        p0 = p1;
    }
}

/// 4x16 YMM 레지스터 블록 및 캐시 블록화(`KC = 256`)가 적용된 AVX2+FMA 마이크로커널
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const KC: usize = 256;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    let mut p0 = 0;
    while p0 < k {
        let p1 = (p0 + KC).min(k);
        let mut i = 0;

        // 4행 레지스터 블록 마이크로커널
        while i + 4 <= m {
            let mut j = 0;
            // 4 x 16 레지스터 블록 (8개의 YMM 누적 레지스터)
            while j + 16 <= n {
                let c0_ptr = c_ptr.add(i * n + j);
                let c1_ptr = c_ptr.add((i + 1) * n + j);
                let c2_ptr = c_ptr.add((i + 2) * n + j);
                let c3_ptr = c_ptr.add((i + 3) * n + j);

                let mut c00 = _mm256_loadu_ps(c0_ptr);
                let mut c01 = _mm256_loadu_ps(c0_ptr.add(8));
                let mut c10 = _mm256_loadu_ps(c1_ptr);
                let mut c11 = _mm256_loadu_ps(c1_ptr.add(8));
                let mut c20 = _mm256_loadu_ps(c2_ptr);
                let mut c21 = _mm256_loadu_ps(c2_ptr.add(8));
                let mut c30 = _mm256_loadu_ps(c3_ptr);
                let mut c31 = _mm256_loadu_ps(c3_ptr.add(8));

                for p in p0..p1 {
                    let bp = b_ptr.add(p * n + j);
                    let b0 = _mm256_loadu_ps(bp);
                    let b1 = _mm256_loadu_ps(bp.add(8));

                    let a0 = _mm256_set1_ps(*a_ptr.add(i * k + p));
                    c00 = _mm256_fmadd_ps(a0, b0, c00);
                    c01 = _mm256_fmadd_ps(a0, b1, c01);

                    let a1 = _mm256_set1_ps(*a_ptr.add((i + 1) * k + p));
                    c10 = _mm256_fmadd_ps(a1, b0, c10);
                    c11 = _mm256_fmadd_ps(a1, b1, c11);

                    let a2 = _mm256_set1_ps(*a_ptr.add((i + 2) * k + p));
                    c20 = _mm256_fmadd_ps(a2, b0, c20);
                    c21 = _mm256_fmadd_ps(a2, b1, c21);

                    let a3 = _mm256_set1_ps(*a_ptr.add((i + 3) * k + p));
                    c30 = _mm256_fmadd_ps(a3, b0, c30);
                    c31 = _mm256_fmadd_ps(a3, b1, c31);
                }

                _mm256_storeu_ps(c0_ptr, c00);
                _mm256_storeu_ps(c0_ptr.add(8), c01);
                _mm256_storeu_ps(c1_ptr, c10);
                _mm256_storeu_ps(c1_ptr.add(8), c11);
                _mm256_storeu_ps(c2_ptr, c20);
                _mm256_storeu_ps(c2_ptr.add(8), c21);
                _mm256_storeu_ps(c3_ptr, c30);
                _mm256_storeu_ps(c3_ptr.add(8), c31);

                j += 16;
            }

            // 4 x 8 나머지 블록
            if j + 8 <= n {
                let c0_ptr = c_ptr.add(i * n + j);
                let c1_ptr = c_ptr.add((i + 1) * n + j);
                let c2_ptr = c_ptr.add((i + 2) * n + j);
                let c3_ptr = c_ptr.add((i + 3) * n + j);

                let mut c00 = _mm256_loadu_ps(c0_ptr);
                let mut c10 = _mm256_loadu_ps(c1_ptr);
                let mut c20 = _mm256_loadu_ps(c2_ptr);
                let mut c30 = _mm256_loadu_ps(c3_ptr);

                for p in p0..p1 {
                    let b0 = _mm256_loadu_ps(b_ptr.add(p * n + j));
                    let a0 = _mm256_set1_ps(*a_ptr.add(i * k + p));
                    let a1 = _mm256_set1_ps(*a_ptr.add((i + 1) * k + p));
                    let a2 = _mm256_set1_ps(*a_ptr.add((i + 2) * k + p));
                    let a3 = _mm256_set1_ps(*a_ptr.add((i + 3) * k + p));

                    c00 = _mm256_fmadd_ps(a0, b0, c00);
                    c10 = _mm256_fmadd_ps(a1, b0, c10);
                    c20 = _mm256_fmadd_ps(a2, b0, c20);
                    c30 = _mm256_fmadd_ps(a3, b0, c30);
                }

                _mm256_storeu_ps(c0_ptr, c00);
                _mm256_storeu_ps(c1_ptr, c10);
                _mm256_storeu_ps(c2_ptr, c20);
                _mm256_storeu_ps(c3_ptr, c30);
                j += 8;
            }

            // 4개 행에 대한 스칼라 열 꼬리(tail) 처리
            if j < n {
                for p in p0..p1 {
                    let a0 = *a_ptr.add(i * k + p);
                    let a1 = *a_ptr.add((i + 1) * k + p);
                    let a2 = *a_ptr.add((i + 2) * k + p);
                    let a3 = *a_ptr.add((i + 3) * k + p);
                    for col in j..n {
                        let bv = *b_ptr.add(p * n + col);
                        *c_ptr.add(i * n + col) += a0 * bv;
                        *c_ptr.add((i + 1) * n + col) += a1 * bv;
                        *c_ptr.add((i + 2) * n + col) += a2 * bv;
                        *c_ptr.add((i + 3) * n + col) += a3 * bv;
                    }
                }
            }

            i += 4;
        }

        // 남은 행 처리 (1..3개 행)
        while i < m {
            let mut j = 0;
            while j + 16 <= n {
                let cp = c_ptr.add(i * n + j);
                let mut c0 = _mm256_loadu_ps(cp);
                let mut c1 = _mm256_loadu_ps(cp.add(8));
                for p in p0..p1 {
                    let av = _mm256_set1_ps(*a_ptr.add(i * k + p));
                    let bp = b_ptr.add(p * n + j);
                    c0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp), c0);
                    c1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp.add(8)), c1);
                }
                _mm256_storeu_ps(cp, c0);
                _mm256_storeu_ps(cp.add(8), c1);
                j += 16;
            }
            if j + 8 <= n {
                let cp = c_ptr.add(i * n + j);
                let mut c0 = _mm256_loadu_ps(cp);
                for p in p0..p1 {
                    let av = _mm256_set1_ps(*a_ptr.add(i * k + p));
                    c0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b_ptr.add(p * n + j)), c0);
                }
                _mm256_storeu_ps(cp, c0);
                j += 8;
            }
            if j < n {
                for p in p0..p1 {
                    let av = *a_ptr.add(i * k + p);
                    for col in j..n {
                        *c_ptr.add(i * n + col) += av * *b_ptr.add(p * n + col);
                    }
                }
            }
            i += 1;
        }

        p0 = p1;
    }
}
