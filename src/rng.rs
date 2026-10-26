//! Deterministic pseudo random number generation (SplitMix64).
//!
//! A global generator is used for parameter initialisation, dropout masks,
//! etc. Call [`manual_seed`] to make experiments reproducible.

use std::sync::Mutex;

/// Small, fast, reproducible PRNG (SplitMix64).
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed ^ 0x9E37_79B9_7F4A_7C15 }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform sample in `[0, 1)`.
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform sample in `[lo, hi)`.
    pub fn uniform_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }

    /// Standard normal sample (Box-Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Uniform integer in `[0, n)`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

static GLOBAL: Mutex<Option<Rng>> = Mutex::new(None);

fn with_global<R>(f: impl FnOnce(&mut Rng) -> R) -> R {
    let mut g = GLOBAL.lock().unwrap();
    let rng = g.get_or_insert_with(|| Rng::new(0x5EED));
    f(rng)
}

/// Seed the global generator (like `torch.manual_seed`).
pub fn manual_seed(seed: u64) {
    *GLOBAL.lock().unwrap() = Some(Rng::new(seed));
}

pub fn uniform() -> f32 {
    with_global(|r| r.uniform())
}

pub fn uniform_range(lo: f32, hi: f32) -> f32 {
    with_global(|r| r.uniform_range(lo, hi))
}

pub fn normal() -> f32 {
    with_global(|r| r.normal())
}

pub fn next_u64() -> u64 {
    with_global(|r| r.next_u64())
}

/// Fill a vector with `n` samples produced by `f` using the global generator.
pub fn fill_with(n: usize, mut f: impl FnMut(&mut Rng) -> f32) -> Vec<f32> {
    with_global(|r| (0..n).map(|_| f(r)).collect())
}
