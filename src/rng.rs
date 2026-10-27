//! 결정론적 의사 난수 생성기 (SplitMix64 알고리즘).
//!
//! 전역 난수 생성기는 매개변수 가중치 초기화, 드롭아웃 마스크 생성 등에 사용됩니다.
//! 실험의 재현성을 보장하려면 [`manual_seed`]를 호출하여 난수 시드를 고정할 수 있습니다.

use std::sync::Mutex;

/// 작고 빠르며 재현 가능한 의사 난수 생성기 (SplitMix64).
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// 지정된 시드로 새 난수 생성기 인스턴스를 초기화합니다.
    pub fn new(seed: u64) -> Self {
        Rng { state: seed ^ 0x9E37_79B9_7F4A_7C15 }
    }

    /// 다음 64비트 부호 없는 정수 난수를 반환합니다 (SplitMix64 상태 전이).
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `[0, 1)` 범위의 균등 분포 난수(`f32`)를 생성합니다.
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// `[lo, hi)` 범위의 균등 분포 난수(`f32`)를 생성합니다.
    pub fn uniform_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }

    /// Box-Muller 변환을 사용하여 표준 정규 분포(평균 0, 표준편차 1) 난수를 생성합니다.
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7); // 로그 계산 시 0 방지
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// `[0, n)` 범위의 부호 없는 정수 난수를 균등 분포로 생성합니다.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Fisher-Yates 알고리즘을 사용하여 주어진 슬라이스의 요소를 무작위로 섞습니다.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

/// 전역 난수 생성기를 관리하는 뮤텍스 싱글톤
static GLOBAL: Mutex<Option<Rng>> = Mutex::new(None);

/// 전역 난수 생성기에 접근하여 주어진 클로저를 실행하는 헬퍼 함수
fn with_global<R>(f: impl FnOnce(&mut Rng) -> R) -> R {
    let mut g = GLOBAL.lock().unwrap();
    let rng = g.get_or_insert_with(|| Rng::new(0x5EED));
    f(rng)
}

/// 전역 난수 생성기의 시드를 설정합니다 (`torch.manual_seed`와 동일한 역할).
pub fn manual_seed(seed: u64) {
    *GLOBAL.lock().unwrap() = Some(Rng::new(seed));
}

/// 전역 난수 생성기를 사용하여 `[0, 1)` 범위의 균등 분포 난수를 반환합니다.
pub fn uniform() -> f32 {
    with_global(|r| r.uniform())
}

/// 전역 난수 생성기를 사용하여 `[lo, hi)` 범위의 균등 분포 난수를 반환합니다.
pub fn uniform_range(lo: f32, hi: f32) -> f32 {
    with_global(|r| r.uniform_range(lo, hi))
}

/// 전역 난수 생성기를 사용하여 표준 정규 분포(평균 0, 표준편차 1) 난수를 반환합니다.
pub fn normal() -> f32 {
    with_global(|r| r.normal())
}

/// 전역 난수 생성기를 사용하여 다음 64비트 정수 난수를 반환합니다.
pub fn next_u64() -> u64 {
    with_global(|r| r.next_u64())
}

/// 전역 난수 생성기를 사용하여 클로저 `f`가 생성한 `n`개의 샘플로 구성된 벡터를 반환합니다.
pub fn fill_with(n: usize, mut f: impl FnMut(&mut Rng) -> f32) -> Vec<f32> {
    with_global(|r| (0..n).map(|_| f(r)).collect())
}
