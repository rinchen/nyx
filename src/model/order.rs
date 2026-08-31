//! Order-N adaptive bit model with per-context bit frequency counts.
//!
//! Context is the last `order` bytes (or fewer at the start of a block). Each context
//! holds a `[count0, count1]` tally; the prediction is `count1 / (count0+count1)`,
//! quantized to 12-bit and clamped away from the extremes so the entropy coder never
//! sees a probability of exactly 0 or 1.

use super::BitModel;
use super::ByteAssembler;

const INIT: u32 = 32; // initial count per side so a fresh context is ~50/50
const MAX_PROB: u16 = 4095;
const MIN_PROB: u16 = 1;

/// Order-N context model over **bytes**.
///
/// `update` receives one coded bit at a time; a [`ByteAssembler`] rebuilds whole
/// bytes from those bits, and the model's context is the last `order` *completed
/// bytes* (not bits). Each byte-context holds a `[count0, count1]` tally for the
/// current bit position; the prediction is `count1 / (count0+count1)`, quantized to
/// 12-bit and clamped away from the extremes.
pub struct OrderN {
    order: usize,
    asm: ByteAssembler,
    // Keyed by (byte-context, bit-position-in-byte). bit_pos in 0..8.
    tables: std::collections::HashMap<(u64, u8), [u32; 2]>,
}

impl OrderN {
    /// Create an order-`order` model. `order == 0` is a flat (order-0) model.
    #[must_use]
    pub fn new(order: usize) -> Self {
        Self {
            order,
            asm: ByteAssembler::new(order.max(1)),
            tables: std::collections::HashMap::new(),
        }
    }

    fn key(&self) -> (u64, u8) {
        let bytes = self.asm.last(self.order);
        let mut k = 0u64;
        for &b in bytes {
            k = (k << 8) | u64::from(b);
        }
        // The bit position within the current byte matters: the same preceding
        // bytes predict different bits depending on which bit of the byte we're on.
        let bit_pos = self.asm.nbits();
        (k, bit_pos)
    }

    fn entry(&self) -> [u32; 2] {
        self.tables.get(&self.key()).copied().unwrap_or([INIT, INIT])
    }
}

impl BitModel for OrderN {
    fn predict(&self) -> u16 {
        let [c0, c1] = self.entry();
        let tot = f64::from(c0 + c1);
        (f64::from(c1) / tot * f64::from(MAX_PROB))
            .clamp(f64::from(MIN_PROB), f64::from(MAX_PROB)) as u16
    }

    fn update(&mut self, bit: bool) {
        // Record the tally under the bit position we're currently coding (before the
        // byte assembler advances), so the 8th bit of a byte is filed at bit_pos 7, not 0.
        let bit_pos = self.asm.nbits();
        self.asm.push_bit(bit);
        let mut k = 0u64;
        for &b in self.asm.last(self.order) {
            k = (k << 8) | u64::from(b);
        }
        let key = (k, bit_pos);
        let e = self.tables.entry(key).or_insert([INIT, INIT]);
        e[bit as usize] += 1;
    }

    fn reset(&mut self) {
        self.asm.reset();
        self.tables.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed a whole byte (MSB→LSB) through a `BitModel`, matching the codec's loop.
    fn feed_byte<M: BitModel>(m: &mut M, byte: u8) {
        for bit_idx in (0..8).rev() {
            let bit = (byte >> bit_idx) & 1 == 1;
            m.update(bit);
        }
    }

    #[test]
    fn predicts_uniform_when_fresh() {
        let m = OrderN::new(2);
        assert!((i32::from(m.predict()) - 2048).abs() <= 1, "fresh model ~50/50");
    }

    #[test]
    fn adapts_toward_one_after_ones() {
        let mut m = OrderN::new(1);
        // Feed many 0xFF bytes; an order-1 byte model should learn the repetition
        // and predict >50% for the next 1-bit.
        for _ in 0..12 {
            feed_byte(&mut m, 0xFF);
        }
        assert!(m.predict() > 2048, "after many 0xFF bytes, expect >50%");
    }

    #[test]
    fn reset_clears_context() {
        let mut m = OrderN::new(1);
        feed_byte(&mut m, 0xFF);
        m.reset();
        assert!((i32::from(m.predict()) - 2048).abs() <= 1);
    }
}
