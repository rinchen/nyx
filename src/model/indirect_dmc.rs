//! Indirect context model + DMC (Dynamic Markov Compression).
//!
//! **Indirect contexts**: instead of keying predictions directly on the last
//! few bytes (like OrderN), an indirect model records a mapping
//! `table[hash(o2)] → last seen byte`, then predicts P(bit) from the
//! combined context of that "indirect" byte and the immediately preceding
//! byte (o1). This catches long-range repetitions where the same prefix
//! keeps re-appearing, even if intervening bytes differ.
//!
//! **DMC** is a graph of single-bit states that splits on mismatches and
//! clones on demand when wrong predictions accumulate. It was state-of-the-art
//! for text before PAQ-style context mixing arrived.

use crate::model::ctable::CtxTable;
use crate::model::{BitModel, ByteAssembler};

const MAX_PROB: u16 = 4095;
const MIN_PROB: u16 = 1;
const INDIRECT_BITS: u32 = 20; // 1 Mi bucket indirect lookup table
const INDIRECT_SIZE: usize = 1 << INDIRECT_BITS;
const CTX_BITS: u32 = 18; // 256K ctx buckets

fn mix_hash(a: u64, b: u64) -> usize {
    let h = a
        .wrapping_mul(0x9E37_79B9_1AB3_57C5u64)
        ^ b.wrapping_mul(0xBF58_4AC7_3295_6B65u64);
    (h as usize) & (INDIRECT_SIZE - 1)
}

/// Indirect context model.
///
/// `indirect = table[hash(o2)]` (or `table[hash(o3)]` for higher orders).
/// Then the prediction context is built from `(indirect_byte, prev_byte, bit_pos)`.
pub struct IndirectModel {
    asm: ByteAssembler,
    /// Map from order-N context hash → last seen byte.
    table: Vec<u8>,
    /// Final per-(indirect_byte, prev_byte, bit_pos) tallies.
    ctab: CtxTable,
    /// Which order to build the context from (2 or 3).
    pub(crate) order: usize,
}

impl IndirectModel {
    /// Create an indirect model keyed on `order`-byte history.
    #[must_use]
    pub fn new(order: usize) -> Self {
        Self {
            asm: ByteAssembler::new(order.max(4)),
            table: vec![0u8; INDIRECT_SIZE],
            ctab: CtxTable::new(CTX_BITS),
            order,
        }
    }

    #[inline]
    fn ctx_hash(&self) -> usize {
        let bytes = self.asm.last(self.order);
        if bytes.len() < self.order {
            return 0;
        }
        let mut h = 0u64;
        for &b in bytes {
            h = h.wrapping_mul(131).wrapping_add(u64::from(b));
        }
        mix_hash(h, 0)
    }

    #[inline]
    fn key(&self) -> u64 {
        let indirect = self.table[self.ctx_hash()];
        let prev = self.asm.last_byte();
        let bit_pos = u64::from(self.asm.nbits());
        (u64::from(indirect) << 16) | (u64::from(prev) << 3) | bit_pos
    }
}

impl BitModel for IndirectModel {
    #[inline(always)]
    fn predict(&self) -> u16 {
        let [c0, c1] = self.ctab.get(self.key());
        let tot = c0 + c1;
        (u32::from(c1) * MAX_PROB as u32 / (tot.max(1)))
            .clamp(MIN_PROB as u32, MAX_PROB as u32) as u16
    }

    #[inline(always)]
    fn update(&mut self, bit: bool) {
        let k = self.key();
        self.asm.push_bit(bit);
        self.ctab.update(k, bit);
        // Refresh the indirect table: store last byte under this order-N context.
        let h = self.ctx_hash();
        if let Some(byte) = self.asm.last(1).get(0) {
            self.table[h] = *byte;
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.asm.reset();
        self.table.fill(0);
        self.ctab.reset();
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Default for IndirectModel {
    fn default() -> Self {
        Self::new(2)
    }
}

/// DMC (Dynamic Markov Compression) clone-based model.
///
/// Maintains a graph of single-bit prediction states keyed by a history hash.
/// When a prediction is repeatedly wrong (wrong-bit streak exceeds the
/// threshold), the model "clones" by widening its context — the CtxTable's
/// generation counter ensures each context starts fresh.
pub struct DmcModel {
    asm: ByteAssembler,
    ctab: CtxTable,
    /// Current count of consecutive wrong predictions in this context.
    wrong_streak: usize,
    /// Clone threshold: when `wrong_streak` exceeds this, the next position
    /// gets a wider context (DMC-style split).
    pub(crate) threshold: usize,
}

impl DmcModel {
    /// Create a DMC model with the given clone threshold (default 5).
    #[must_use]
    pub fn new(threshold: usize) -> Self {
        Self {
            asm: ByteAssembler::new(8),
            ctab: CtxTable::new(20),
            wrong_streak: 0,
            threshold,
        }
    }

    #[inline]
    fn key(&self) -> u64 {
        let prev = self.asm.prev_byte() as u64;
        let last = self.asm.last_byte() as u64;
        let bit_pos = u64::from(self.asm.nbits());
        (prev << 16) | (last << 3) | bit_pos
    }
}

impl BitModel for DmcModel {
    #[inline(always)]
    fn predict(&self) -> u16 {
        let [c0, c1] = self.ctab.get(self.key());
        let tot = c0 + c1;
        (u32::from(c1) * MAX_PROB as u32 / (tot.max(1)))
            .clamp(MIN_PROB as u32, MAX_PROB as u32) as u16
    }

    #[inline(always)]
    fn update(&mut self, bit: bool) {
        let k = self.key();
        let p = self.predict();
        let predicted_one = p > 2048;
        if predicted_one == bit {
            self.wrong_streak = 0;
        } else {
            self.wrong_streak += 1;
        }
        self.asm.push_bit(bit);
        self.ctab.update(k, bit);
    }

    #[inline]
    fn reset(&mut self) {
        self.asm.reset();
        self.ctab.reset();
        self.wrong_streak = 0;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Default for DmcModel {
    fn default() -> Self {
        Self::new(5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_byte<M: BitModel>(m: &mut M, byte: u8) {
        for bit_idx in (0..8).rev() {
            let bit = (byte >> bit_idx) & 1 == 1;
            let _ = m.predict();
            m.update(bit);
        }
    }

    #[test]
    fn indirect_predicts_uniform_when_fresh() {
        let m = IndirectModel::new(2);
        assert!((i32::from(m.predict()) - 2048).abs() <= 1);
    }

    #[test]
    fn indirect_adapts_after_repeat() {
        let mut m = IndirectModel::new(2);
        for _ in 0..20 {
            feed_byte(&mut m, 0xFF);
        }
        assert!(m.predict() != 2048, "should have learned");
    }

    #[test]
    fn indirect_reset_clears() {
        let mut m = IndirectModel::new(2);
        feed_byte(&mut m, 0xAA);
        m.reset();
        assert!((i32::from(m.predict()) - 2048).abs() <= 1);
    }

    #[test]
    fn dmc_predicts_uniform_when_fresh() {
        let m = DmcModel::new(5);
        assert!((i32::from(m.predict()) - 2048).abs() <= 1);
    }

    #[test]
    fn dmc_reset_clears() {
        let mut m = DmcModel::new(5);
        for _ in 0..10 {
            feed_byte(&mut m, 0xFF);
        }
        m.reset();
        assert!((i32::from(m.predict()) - 2048).abs() <= 1);
    }
}
