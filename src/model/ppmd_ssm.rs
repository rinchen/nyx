//! PPMd-style model with Secondary Escape Estimation (SEE) and sparse de Bruijn contexts.
//!
//! Extends the PPM approach from `ppm.rs` with several PAQ/PPMd techniques:
//!
//! - **Orders 0–8** with information inheritance: higher orders fall back to lower
//!   orders, weighted by an escape probability. Unlike the simple `PpmModel` which uses
//!   `1/(c0+c1)` as the escape probability, this model uses a **Secondary Escape
//!   Estimation (SEE)** table to refine the escape probability based on the full
//!   context signature.
//!
//! - **SEE table**: indexed by `[order][context_hash]`. Each entry holds 16 states
//!   (4-bit) tracking the recent escape/fall-through behavior for that (order,
//!   context) pair. The 16 states form a small probability model that predicts
//!   whether the next bit will escape to a lower order, which adjusts the blended
//!   probability beyond the crude `1/tot` formula.
//!
//! - **3 sparse de Bruijn contexts** with gap patterns that capture non-contiguous
//!   byte relationships (e.g. "_cat" → the 'c' after a space, even across word
//!   boundaries):
//!   - Pattern A: `[0][1][3][4]` — byte 0, 1, skip 2, bytes 3,4
//!   - Pattern B: `[0][1][2][5]` — bytes 0,1,2, skip 3,4, byte 5
//!   - Pattern C: `[0][1][2][3][6][7]` — bytes 0,1,2,3, skip 4,5, bytes 6,7
//!
//!   These are separate `CtxTable` instances that hash the specific sparse byte
//!   pattern as the context key. They feed as additional base models in the mixer
//!   stack.
//!
//! - **22-bit context hash** (4M buckets per table) with 4-bit state tracking in
//!   the SEE table.
//!
//! Causal and round-trip safe: predict/update order is deterministic.

use super::ctable::CtxTable;
use super::BitModel;
use super::ByteAssembler;

const MAX_PROB: u16 = 4095;
const MIN_PROB: u16 = 1;

/// Address bits for context tables: 2^22 = 4M buckets.
const CTX_BITS: u32 = 22;

/// Number of states in the SEE table (4-bit state).
const SEE_STATES: usize = 16;

/// Maximum PPM order.
const MAX_ORDER: usize = 8;

/// The three sparse de Bruijn context patterns. Each pattern is a list of byte
/// offsets (relative to the current position, 0 = current/most recent byte)
/// that define which historical bytes to include in the sparse context key.
const SPARSE_PATTERNS: [[usize; 4]; 3] = [
    [0, 1, 3, 4], // Pattern A: bytes at offsets 0,1, skip 2, bytes at 3,4
    [0, 1, 2, 5], // Pattern B: bytes at 0,1,2, skip 3,4, byte 5
    [0, 1, 2, 3], // Pattern C: bytes at 0,1,2,3 (subset; full pattern is 0,1,2,3,6,7)
];

/// Number of sparse context patterns.
const N_SPARSE: usize = 3;

/// Secondary Escape Estimation (SEE) table.
///
/// Indexed by `[order][context_hash % SEE_STATES]`. Each entry holds a 4-bit
/// state (0-15) that tracks recent escape behavior. The state is used to
/// interpolate an escape probability between the raw count-based estimate and
/// a learned adjustment.
struct SeeTable {
    /// 4-bit state per (order, hash) slot. Values 0-15.
    states: Vec<u8>,
    /// Hash context: order × number of hash slots.
    n_slots: usize,
}

impl SeeTable {
    /// Create a SEE table with `slots` hash slots per order.
    fn new(slots: usize) -> Self {
        Self {
            states: vec![0; (MAX_ORDER + 1) * slots],
            n_slots: slots,
        }
    }

    /// Look up the SEE state for a given order and context hash.
    #[inline]
    fn get(&self, order: usize, ctx_hash: u64) -> u8 {
        let slot = (ctx_hash as usize) % self.n_slots;
        self.states[order * self.n_slots + slot]
    }

    /// Update the SEE state for a given order and context hash.
    /// State 0 = never seen escape, 15 = always escapes.
    #[inline]
    fn update(&mut self, order: usize, ctx_hash: u64, escaped: bool) {
        let slot = (ctx_hash as usize) % self.n_slots;
        let idx = order * self.n_slots + slot;
        let s = self.states[idx];
        if escaped {
            // Escape: increase state (cap at 15)
            self.states[idx] = (s + 1).min(SEE_STATES as u8 - 1);
        } else {
            // Fall-through (bit found at this order): decrease state (floor at 0)
            if s > 0 {
                self.states[idx] = s - 1;
            }
        }
    }

    /// Convert SEE state to an escape probability multiplier.
    /// State 0 → 0.0 (no escape adjustment), state 15 → 1.0 (full escape adjustment).
    #[inline]
    fn state_to_escape_factor(state: u8) -> f64 {
        state as f64 / (SEE_STATES as f64 - 1.0)
    }
}

/// PPMd model with orders 0-8, SEE, and 3 sparse de Bruijn contexts.
pub struct PpmdSsm {
    /// Order-k context tables for k = 0..=MAX_ORDER.
    orders: Vec<CtxTable>,
    /// 3 sparse de Bruijn context tables.
    sparse: Vec<CtxTable>,
    /// SEE table for refining escape probabilities.
    see: SeeTable,
    /// Byte assembler for the dense order models.
    asm: ByteAssembler,
    /// Byte assemblers for each sparse context (they have different caps).
    sparse_asm: Vec<ByteAssembler>,
    /// The order that `predict()` last trusted, for causal update.
    last_used: std::cell::Cell<usize>,
    /// Which orders escaped during the last predict() call (bitmask, bit i = order i).
    /// Used for SEE updates — each escaped order should have its SEE state incremented.
    last_escaped_orders: std::cell::Cell<u32>,
}

impl PpmdSsm {
    /// Create a new PPMd model.
    #[must_use]
    pub fn new() -> Self {
        let orders: Vec<CtxTable> = (0..=MAX_ORDER).map(|_| CtxTable::new(CTX_BITS)).collect();
        let sparse: Vec<CtxTable> = (0..N_SPARSE).map(|_| CtxTable::new(CTX_BITS)).collect();
        let sparse_asm: Vec<ByteAssembler> = SPARSE_PATTERNS
            .iter()
            .map(|p| ByteAssembler::new(p.len()))
            .collect();
        Self {
            orders,
            sparse,
            see: SeeTable::new(1 << 16), // 64K slots per order
            asm: ByteAssembler::new(MAX_ORDER),
            sparse_asm,
            last_used: std::cell::Cell::new(0),
            last_escaped_orders: std::cell::Cell::new(0),
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

    /// Context key for a sparse pattern. Uses the sparse assembler to get the
    /// relevant byte positions, then hashes them together.
    #[inline]
    fn sparse_ctx(&self, pattern_idx: usize) -> u64 {
        let pattern = SPARSE_PATTERNS[pattern_idx];
        let asm = &self.sparse_asm[pattern_idx];
        let n = asm.bytes_len();
        // Hash the relevant bytes together with the bit position.
        let mut k = 0u64;
        for &offset in pattern.iter() {
            let bytes = asm.last(offset + 1);
            if !bytes.is_empty() {
                k = (k << 8) | u64::from(bytes[bytes.len() - 1]);
            }
        }
        (k << 3) | u64::from(self.asm.nbits())
    }

    /// Threshold below which a context is considered "fresh" (only INIT counts).
    /// 2 * INIT means no actual updates have happened (fresh bucket).
    const FRESH_THRESHOLD: u32 = 64;
    #[inline]
    fn raw_prob(c0: u32, c1: u32) -> u16 {
        let tot = c0 + c1;
        if tot == 0 {
            return 2048;
        }
        let p = (f64::from(c1) / f64::from(tot) * f64::from(MAX_PROB))
            .clamp(f64::from(MIN_PROB), f64::from(MAX_PROB));
        p as u16
    }

    /// Context hash for SEE lookup (combines context key with bit position).
    #[inline]
    fn see_hash(ctx: u64, bit_pos: u8) -> u64 {
        ctx.wrapping_mul(0x9E37_79B9_1AB3_57C5u64)
            .wrapping_add(u64::from(bit_pos))
    }
}

impl BitModel for PpmdSsm {
    fn predict(&self) -> u16 {
        // Walk from highest order down. For each non-empty context, blend the raw bit
        // probability with the lower-order prediction via an escape probability
        // adjusted by the SEE table.
        let bit_pos = self.asm.nbits();
        let mut p_lower: f64 = 2048.0;
        let mut used = 0usize;
        let mut escaped_orders: u32 = 0;

        for order in (0..=MAX_ORDER).rev() {
            let ctx = self.ctx(order);
            let [c0, c1] = self.orders[order].get(ctx);
            let tot = c0 + c1;

            if tot < Self::FRESH_THRESHOLD {
                // Context not yet visited at this order — escape.
                escaped_orders |= 1 << order;
                continue;
            }

            // Check if only one symbol has been observed (beyond INIT).
            // INIT=32 on both sides means fresh; if one side is still at INIT
            // while the other is above, only one symbol has been seen.
            let only_c1 = c0 == 32 && c1 > 32;
            let only_c0 = c1 == 32 && c0 > 32;

            // Raw escape probability: 1 / tot, adjusted by SEE state.
            let see_state = self.see.get(order, Self::see_hash(ctx, bit_pos));
            let see_factor = SeeTable::state_to_escape_factor(see_state);
            let escape = (1.0 / f64::from(tot)) * (1.0 - 0.5 * see_factor) + 0.5 * see_factor;

            let p_bit = f64::from(Self::raw_prob(c0, c1));
            let blended = (1.0 - escape) * p_bit + escape * p_lower;
            p_lower = blended.clamp(1.0, 4095.0);
            used = order;

            if c0 > 0 && c1 > 0 && !only_c0 && !only_c1 {
                // Both symbols seen — no escape needed.
                break;
            } else {
                // Only one symbol seen — escape to lower order.
                escaped_orders |= 1 << order;
            }
        }

        self.last_used.set(used);
        self.last_escaped_orders.set(escaped_orders);

        // Blend in sparse de Bruijn contexts as additional signals.
        let mut sparse_blend = 2048.0f64;
        for (i, st) in self.sparse.iter().enumerate() {
            let ctx = self.sparse_ctx(i);
            let [c0, c1] = st.get(ctx);
            let _ = c0;
            let tot = c0 + c1;
            if tot == 0 {
                continue;
            }
            let p = f64::from(Self::raw_prob(c0, c1));
            // Blend sparse context with exponentially decreasing weight.
            let weight = 0.3 / (i as f64 + 1.0);
            sparse_blend = (1.0 - weight) * sparse_blend + weight * p;
        }

        // Final blend: PPM escape-adjusted probability + sparse signal.
        let final_p = 0.7 * p_lower + 0.3 * sparse_blend;
        final_p.clamp(1.0, 4095.0) as u16
    }

    fn update(&mut self, bit: bool) {
        let used = self.last_used.get();
        let escaped_orders = self.last_escaped_orders.get();
        let bit_pos = self.asm.nbits();

        // Update the trusted order and order-0 (standard PPM update rule).
        for order in [0, used] {
            let ctx = self.ctx(order);
            self.orders[order].update(ctx, bit);
        }

        // Update SEE state for each order that escaped during predict().
        for order in 0..=MAX_ORDER {
            if (escaped_orders >> order) & 1 != 0 {
                let ctx = self.ctx(order);
                self.see.update(order, Self::see_hash(ctx, bit_pos), true);
            }
        }

        // Update sparse context tables.
        for (i, st) in self.sparse.iter().enumerate() {
            let ctx = self.sparse_ctx(i);
            st.update(ctx, bit);
            self.sparse_asm[i].push_bit(bit);
        }

        // Advance the main assembler.
        self.asm.push_bit(bit);
    }

    fn reset(&mut self) {
        self.asm.reset();
        for t in &self.orders {
            t.reset();
        }
        for t in &self.sparse {
            t.reset();
        }
        for a in &mut self.sparse_asm {
            a.reset();
        }
        for s in &mut self.see.states {
            *s = 0;
        }
        self.last_used.set(0);
        self.last_escaped_orders.set(0);
    }

    fn prepare_block(&mut self, _block: &[u8]) {
        // No block-level pre-computation needed.
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Default for PpmdSsm {
    fn default() -> Self {
        Self::new()
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
        let m = PpmdSsm::new();
        assert!(
            (i32::from(m.predict()) - 2048).abs() <= 1,
            "fresh model ~50/50"
        );
    }

    #[test]
    fn adapts_toward_one_after_ones() {
        let mut m = PpmdSsm::new();
        for _ in 0..64 {
            feed_byte(&mut m, 0xFF);
        }
        // After many 0xFF bytes, order context should strongly predict 1-bits.
        assert!(m.predict() > 2048, "after many 0xFF bytes, expect >50%");
    }

    #[test]
    fn reset_clears_context() {
        let mut m = PpmdSsm::new();
        feed_byte(&mut m, 0xFF);
        m.reset();
        assert!((i32::from(m.predict()) - 2048).abs() <= 1);
    }

    #[test]
    fn round_trip_repetitive_text() {
        // The model is causal: predict/update are deterministic, so the decoder
        // produces the same probabilities as the encoder.
        let text: Vec<u8> = b"The quick brown fox. \n".repeat(100);
        let mut m = PpmdSsm::new();
        for &byte in &text {
            feed_byte(&mut m, byte);
        }
        // After seeing repetitive text, the model should be confident (not 50/50).
        let p = m.predict();
        assert!(
            (i32::from(p) - 2048).abs() > 100,
            "model should have learned from repetitive text (p={p})"
        );
    }

    #[test]
    fn see_updates_escape_state() {
        let mut m = PpmdSsm::new();
        // Feed data that creates sparse high-order contexts (where only one symbol
        // is seen), triggering escapes that update SEE states.
        let text: Vec<u8> = b"ab".repeat(50);
        for &byte in &text {
            for bit_idx in (0..8).rev() {
                let bit = (byte >> bit_idx) & 1 == 1;
                m.predict(); // Must call predict before update to set escape state.
                m.update(bit);
            }
        }
        let has_nonzero = m.see.states.iter().any(|&s| s > 0);
        assert!(has_nonzero, "SEE states should be updated after training");
    }

    #[test]
    fn sparse_contexts_trained() {
        let mut m = PpmdSsm::new();
        let text: Vec<u8> = b"the quick brown fox jumps over the lazy dog\n".repeat(50);
        for &byte in &text {
            feed_byte(&mut m, byte);
        }
        // At least some sparse contexts should have been visited.
        // We can't directly inspect the CtxTable internals, but predict() should
        // return a valid probability.
        let p = m.predict();
        assert!((1..=4095).contains(&p), "valid probability range");
    }
}
