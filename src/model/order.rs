//! Order-N adaptive bit model with per-context bit frequency counts.
//!
//! Context is the last `order` bytes (or fewer at the start of a block). Each context
//! holds a `[count0, count1]` tally; the prediction is `count1 / (count0+count1)`,
//! quantized to 12-bit and clamped away from the extremes so the entropy coder never
//! sees a probability of exactly 0 or 1.

use super::BitModel;

const INIT: u32 = 32; // initial count per side so a fresh context is ~50/50
const MAX_PROB: u16 = 4095;
const MIN_PROB: u16 = 1;

/// Order-N context model over bytes.
pub struct OrderN {
    order: usize,
    ctx: Vec<u8>,
    tables: std::collections::HashMap<u64, [u32; 2]>,
}

impl OrderN {
    /// Create an order-`order` model. `order == 0` is a flat (order-0) model.
    #[must_use]
    pub fn new(order: usize) -> Self {
        Self {
            order,
            ctx: Vec::with_capacity(order),
            tables: std::collections::HashMap::new(),
        }
    }

    fn key(&self) -> u64 {
        let mut k = 0u64;
        for &b in &self.ctx {
            k = (k << 8) | u64::from(b);
        }
        k
    }

    fn entry(&self) -> [u32; 2] {
        self.tables
            .get(&self.key())
            .copied()
            .unwrap_or([INIT, INIT])
    }
}

impl BitModel for OrderN {
    fn predict(&self) -> u16 {
        let [c0, c1] = self.entry();
        let tot = f64::from(c0 + c1);
        let p = (f64::from(c1) / tot * f64::from(MAX_PROB))
            .clamp(f64::from(MIN_PROB), f64::from(MAX_PROB));
        p as u16
    }

    fn update(&mut self, bit: bool) {
        let k = self.key();
        let e = self.tables.entry(k).or_insert([INIT, INIT]);
        e[bit as usize] += 1;
        let b = bit as u8;
        if self.order > 0 {
            if self.ctx.len() == self.order {
                self.ctx.remove(0);
            }
            self.ctx.push(b);
        }
    }

    fn reset(&mut self) {
        self.ctx.clear();
        self.tables.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicts_uniform_when_fresh() {
        let m = OrderN::new(2);
        assert!((i32::from(m.predict()) - 2048).abs() <= 1, "fresh model ~50/50");
    }

    #[test]
    fn adapts_toward_one_after_ones() {
        let mut m = OrderN::new(1);
        for _ in 0..12 {
            assert!(m.predict() >= 1);
            m.update(true);
        }
        assert!(m.predict() > 2048, "after many 1s, expect >50%");
    }

    #[test]
    fn reset_clears_context() {
        let mut m = OrderN::new(1);
        m.update(true);
        m.reset();
        assert!((i32::from(m.predict()) - 2048).abs() <= 1);
    }
}
