//! 링 올리듀스(Ring All-Reduce) 분산 통신 알고리즘.
//!
//! `N`개의 워커가 링(원형) 형태로 배치되며, 각 버퍼는 `N`개의 청크(조각)로 균등하게 분할됩니다.
//! 알고리즘은 각각 `N-1` 스텝으로 구성된 두 가지 단계로 동작합니다:
//!
//! 1. **reduce-scatter** – 각 워커가 자신의 청크 중 하나를 오른쪽 이웃 워커로 전송하고,
//!    수신한 워커는 자신의 청크에 이를 누적 합산합니다. `N-1` 스텝 후 워커 `r`은
//!    완전히 합산 축약된 청크 `(r+1) mod N`을 보유하게 됩니다.
//! 2. **all-gather** – 축약이 완료된 청크들을 링을 따라 순환 전달하여,
//!    모든 워커가 전체 청크의 최종 합산본을 공유하도록 만듭니다.
//!
//! 각 워커의 총 전송량은 버퍼 크기의 `2 (N-1)/N` 배로, 워커 수 `N`이 늘어나도
//! 거의 일정하게 유지되므로 중앙 집중식 매개변수 서버(Parameter Server) 방식보다
//! 대규모 분산 환경에서 훨씬 뛰어난 확장성을 보입니다.

use rayon::prelude::*;

/// 1회 all-reduce 통신에 대한 통계 정보.
#[derive(Clone, Debug, Default)]
pub struct AllReduceStats {
    /// 통신에 참여한 워커(디바이스) 수.
    pub workers: usize,
    /// 축약된 버퍼의 원소 개수.
    pub elements: usize,
    /// 통신 스텝 수.
    pub steps: usize,
    /// 모든 워커가 전송한 총 바이트 수.
    pub bytes_sent: usize,
}

/// 전체 길이 `len`을 `n`개로 분할할 때 `c`번째 청크의 슬라이스 인덱스 범위 반환.
fn chunk_range(len: usize, n: usize, c: usize) -> std::ops::Range<usize> {
    (c * len / n)..((c + 1) * len / n)
}

/// 주어진 버퍼 슬라이스들의 요소를 in-place로 원소별 합산합니다.
///
/// 연산이 완료되면 모든 버퍼가 전체 요소의 합산본을 동일하게 공유하게 됩니다.
/// 모든 버퍼는 동일한 길이를 가져야 합니다.
pub fn ring_all_reduce(buffers: &mut [Vec<f32>]) -> AllReduceStats {
    let n = buffers.len();
    let len = buffers.first().map(|b| b.len()).unwrap_or(0);
    assert!(buffers.iter().all(|b| b.len() == len), "ring_all_reduce: buffers must have equal length");
    let mut stats = AllReduceStats { workers: n, elements: len, steps: 0, bytes_sent: 0 };
    if n <= 1 {
        return stats;
    }

    // 1단계: reduce-scatter (청크별 분산 합산)
    for step in 0..n - 1 {
        // 모든 워커가 동시에 전송: 랭크 r은 청크 (r - step)을 r+1에게 전송
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

    // 2단계: all-gather (축약 완료된 청크를 전파하여 버퍼 복원). 랭크 r은 이제 축약된 청크 (r+1) mod n을 보유함.
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
