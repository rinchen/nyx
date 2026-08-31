//! Sparse / stride context model.
//!
//! Instead of the dense last-`order` bytes, this model hashes a sparse set of positions
//! relative to the current byte (offsets -1, -4, and the most recent byte `p`). Sparse
//! contexts capture long-range regularities that dense order-N models miss at modest
//! memory cost.

use super::BitModel;
use super::ByteAssembler;

const INIT: u32 = 32;
const MAX_PROB: u16 = 4095;
const MIN_PROB: u16 = 1;

/// Sparse-context bit model keyed on **byte** history (offsets -1, -4, and the most
/// recent completed byte).
pub struct Sparse {
    asm: ByteAssembler,
    tables: std::collections::HashMap<u64, [u32; 2]>,
}

impl Sparse {
    /// Create a sparse model (history of the last 4 bytes is kept for offset addressing).
    #[must_use]
    pub fn new() -> Self {
        Self {
            asm: ByteAssembler::new(8),
            tables: std::collections::HashMap::new(),
        }
    }

    fn key(&self) -> u64 {
        let n = self.asm.bytes_len();
        let last = self.asm.last_byte();
        let m1 = *self.asm.last(2).first().unwrap_or(&0);
        let m4 = *self.asm.last(5).first().unwrap_or(&0);
        // Include n in the key so partial history (fewer than 4 bytes seen) maps to
        // distinct, initially-uniform contexts rather than colliding with full history.
        (n << 40) | (u64::from(last) << 16) | (u64::from(m4) << 8) | u64::from(m1)
    }

    fn entry(&self) -> [u32; 2] {
        self.tables.get(&self.key()).copied().unwrap_or([INIT, INIT])
    }
}

impl BitModel for Sparse {
    fn predict(&self) -> u16 {
        let [c0, c1] = self.entry();
        let tot = f64::from(c0 + c1);
        (f64::from(c1) / tot * f64::from(MAX_PROB))
            .clamp(f64::from(MIN_PROB), f64::from(MAX_PROB)) as u16
    }

    fn update(&mut self, bit: bool) {
        // Advance byte context when a full byte completes; the key then reflects it.
        let _ = self.asm.push_bit(bit);
        let k = self.key();
        let e = self.tables.entry(k).or_insert([INIT, INIT]);
        e[usize::from(bit)] += 1;
    }

    fn reset(&mut self) {
        self.asm.reset();
        self.tables.clear();
    }
}

impl Default for Sparse {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapts_toward_one_after_ones() {
        let mut m = Sparse::new();
        for _ in 0..12 {
            m.update(true);
        }
        assert!(m.predict() > 2048);
    }

    #[test]
    fn reset_clears_context() {
        let mut m = Sparse::new();
        m.update(true);
        m.reset();
        assert!((i32::from(m.predict()) - 2048).abs() <= 1);
    }
}
