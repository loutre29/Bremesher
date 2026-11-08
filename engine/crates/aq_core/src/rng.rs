//! Deterministic RNG (C15). The only RNG of the engine: `SplitMix64` (Steele, Lea, Flood 2014).
//!
//! The seed drives exactly three things (C15): the dual tree root tie-break, the loop seed
//! order and the h0 jitter of the ladder. Nothing else is random.

/// SplitMix64 generator. Same seed, same sequence, on every platform.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [lo, hi).
    pub fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Uniform integer in [0, n) (0 when n == 0), unbiased multiply-shift with rejection.
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let threshold = n.wrapping_neg() % n;
        loop {
            let m = (self.next_u64() as u128) * (n as u128);
            if (m as u64) >= threshold {
                return (m >> 64) as u64;
            }
        }
    }

    /// Deterministic Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        let n = v.len();
        for i in (1..n).rev() {
            let j = self.below(i as u64 + 1) as usize;
            v.swap(i, j);
        }
    }
}
