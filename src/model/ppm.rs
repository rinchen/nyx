//! PPM-style (Prediction by Partial Matching) bit model.
//!
//! Internally stacks order-0 through order-`max_order` context tables. For each bit,
//! `predict()` walks from the highest order down to order-0 and returns the prediction
//! from the first order whose context total count meets `escape_threshold`. If no order
//! has enough data, it falls back to the flat prior (2048). This is the classic PPM escape
//! rule: use the longest reliable context, otherwise escape to a shorter one.
//!
//! `update()` records the bit under every order's context (conservative — the mixer can
//! learn to trust the shorter-order predictions when the long-order context is noisy).

use super::BitModel;
use super::ByteAssembler;
use super::ctable::CtxTable;

const MAX_PROB: u16 = 4095;
const MIN_PROB: u16 = 1;
const CTX_BITS: u32 = 21;

/// PPM-style model with `max_order` stacked context tables and an escape threshold.
pub struct PpmModel {
    orders: Vec<CtxTable>,
    asm: ByteAssembler,
    max_order: usize,
    escape_threshold: u32,
}

impl PpmModel {
    /// Create a PPM model. `max_order` must be ≥ 0. `escape_threshold` is the minimum
    /// total count (`c0 + c1`) a context must have before it is trusted; below that the
    /// model escapes to the next lower order.
    #[must_use]
    pub fn new(max_order: usize, escape_threshold: u32) -> Self {
        let orders = (0..=max_order)
            .map(|_| CtxTable::new(CTX_BITS))
            .collect();
        Self {
            orders,
            asm: ByteAssembler::new(max_order.max(1)),
            max_order,
            escape_threshold,
        }
    }

    /// Context key for a given order, from the current assembler state (pre-push).
    #[inline]
    fn ctx(&self, order: usize) -> u64 {
        let bytes = self.asm.last(order);
        let mut k = 0u64;
        for &b in bytes {
            k = (k << 8) | u64::from(b);
        }
        (k << 3) | u64::from(self.asm.nbits())
    }
}

impl BitModel for PpmModel {
    fn predict(&self) -> u16 {
        // Walk from highest order down; use the first context that has enough data.
        for order in (0..=self.max_order).rev() {
            let [c0, c1] = self.orders[order].get(self.ctx(order));
            let tot = c0 + c1;
            if tot >= self.escape_threshold {
                let p = (c1 as f64 / tot as f64 * f64::from(MAX_PROB))
                    .clamp(f64::from(MIN_PROB), f64::from(MAX_PROB));
                return p as u16;
            }
        }
        // All orders escaped — flat prior.
        2048
    }

    fn update(&mut self, bit: bool) {
        // Conservative: update every order's context (the mixer will learn to weight
        // noisy high-order predictions down if they're not reliable).
        for order in 0..=self.max_order {
            self.orders[order].update(self.ctx(order), bit);
        }
        self.asm.push_bit(bit);
    }

    fn reset(&mut self) {
        self.asm.reset();
        for t in &self.orders {
            t.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppm_escapes_to_lower_order_on_sparse_data() {
        // Feed many different random bytes so order-3+ contexts are almost never hit.
        // With escape_threshold=4, predict() should return ~2048 (prior) because no
        // context accumulates enough count.
        let mut ppm = PpmModel::new(5, 4);
        let mut rng = 0x9E37_79B9u32;
        for _ in 0..1000 {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let byte = rng as u8;
            for bit_idx in (0..8).rev() {
                let bit = (byte >> bit_idx) & 1 == 1;
                ppm.update(bit);
            }
        }
        // After random data, most predictions should be near the prior (escape).
        let mut sum = 0u64;
        let mut n = 0u32;
        for _ in 0..256 {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let byte = rng as u8;
            for bit_idx in (0..8).rev() {
                let bit = (byte >> bit_idx) & 1 == 1;
                sum += u64::from(ppm.predict());
                ppm.update(bit);
                n += 1;
            }
        }
        let mean = sum / u64::from(n);
        // Should be near 2048 (within ±256) because escape dominates.
        assert!(
            (mean.cast_signed() - 2048).abs() <= 256,
            "ppm should escape on sparse data (mean prob = {mean}, expected ~2048)"
        );
    }

    #[test]
    fn ppm_uses_higher_order_on_repetitive_data() {
        // Feed many repeats of the same byte; order-2 context of 0xFF should accumulate
        // count quickly and predict >2048 for the next 1-bit.
        let mut ppm = PpmModel::new(5, 4);
        for _ in 0..20 {
            for bit_idx in (0..8).rev() {
                let bit = (0xFF >> bit_idx) & 1 == 1;
                ppm.update(bit);
            }
        }
        // Now feed another 0xFF; the first bit should be predicted above prior.
        let p = ppm.predict();
        assert!(
            p > 2048,
            "ppm should learn the repetition (predicted {p}, expected >2048)"
        );
    }

    #[test]
    fn ppm_reset_clears_context() {
        let mut ppm = PpmModel::new(2, 4);
        for bit_idx in (0..8).rev() {
            ppm.update((0xFF >> bit_idx) & 1 == 1);
        }
        ppm.reset();
        // After reset the tables are re-seeded to INIT=32 each; 32/64*4095 truncates to
        // 2047, so we allow ±1.
        assert!(
            (i32::from(ppm.predict()) - 2048).abs() <= 1,
            "after reset, ppm should return prior"
        );
    }
}
