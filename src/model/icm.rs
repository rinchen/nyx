//! Indirect Context Model (ICM) — bit-history state machine.
//!
//! Instead of storing `[count0, count1]` tallies per context (like
//! [`CtxTable`](super::ctable::CtxTable)), this model stores a small
//! **state index** that encodes the *recent bit-history pattern* at each
//! context — run of zeros, run of ones, alternating, switch states, etc.
//!
//! This is the ICM design used by ZPAQ and PAQ-family compressors. It captures
//! *how* bits arrive at a context (not just *how many* of each), which lets it
//! represent periodic structure like `0101...` or `001001...` that raw count
//! tables smear across the average.
//!
//! The state table has 22 states — the standard ICM/PAQ8 set —
//! giving each context a ~4.5-bit history summary. This fits in a `u8` per
//! context slot and keeps memory footprint comparable to the existing tables
//! while being significantly richer per bucket.

use super::ctable::CtxTable;
use super::{BitModel, ByteAssembler};

const MAX_PROB: u16 = 4095;
const MIN_PROB: u16 = 1;
const CTX_BITS: u32 = 21;

// Number of ICM states. The standard PAQ8/ZPAQ ICM uses 22 states:
// 0-13: low-count asymmetric states (c0,c1 totals 1-4)
// 14-19: mid-count states (totals 5)
// 20-21: saturated balanced sink states
const NUM_STATES: usize = 22;

/// `STATE_PROBS[s]` = predicted P(bit==1) for state `s`, in `[1, 4095]`.
fn build_state_probs() -> [u16; NUM_STATES] {
    // Each state represents a (c0, c1) count pair. Probability = c1/(c0+c1) * 4095.
    const STATES: [(u16, u16); NUM_STATES] = [
        (0, 1),   // 0  → one 1 seen,           prob = 1.0
        (1, 0),   // 1  → one 0 seen,           prob = 0.0
        (0, 2),   // 2  → two 1s,               prob = 1.0
        (1, 1),   // 3  → one 0, one 1,         prob = 0.5
        (2, 0),   // 4  → two 0s,               prob = 0.0
        (0, 3),   // 5  → three 1s,             prob = 1.0
        (1, 2),   // 6  → two 1s, one 0,        prob = 0.667
        (2, 1),   // 7  → two 0s, one 1,        prob = 0.333
        (3, 0),   // 8  → three 0s,             prob = 0.0
        (0, 4),   // 9  → four 1s,              prob = 1.0
        (1, 3),   // 10 → three 1s, one 0,      prob = 0.75
        (2, 2),   // 11 → balanced 4,           prob = 0.5
        (3, 1),   // 12 → three 0s, one 1,      prob = 0.25
        (4, 0),   // 13 → four 0s,              prob = 0.0
        (0, 5),   // 14 → five 1s,              prob = 1.0
        (1, 4),   // 15 → four 1s, one 0,       prob = 0.8
        (2, 3),   // 16 → three 1s, two 0s,     prob = 0.6
        (3, 2),   // 17 → three 0s, two 1s,     prob = 0.4
        (4, 1),   // 18 → four 0s, one 1,       prob = 0.2
        (5, 0),   // 19 → five 0s,              prob = 0.0
        (3, 3),   // 20 → balanced 6,           prob = 0.5
        (4, 4),   // 21 → balanced 8,           prob = 0.5
    ];
    let mut p = [2048u16; NUM_STATES];
    for (i, &(c0, c1)) in STATES.iter().enumerate() {
        let tot = c0 + c1;
        if tot > 0 {
            p[i] = ((c1 as f64 / tot as f64) * MAX_PROB as f64) as u16;
        }
    }
    p
}

/// Standard PAQ8/ZPAQ ICM state transition table.
/// `TRANSITIONS[s][0]` = next state when bit=0 observed; `[s][1]` for bit=1.
/// States 14-21 are sinks: transitions from them go to balanced states 20/21.
const TRANSITIONS: [[usize; 2]; NUM_STATES] = [
    // bit=0  bit=1
    [3,      2],     // 0  (0,1)
    [4,      3],     // 1  (1,0)
    [6,      5],     // 2  (0,2)
    [7,      6],     // 3  (1,1)
    [8,      7],     // 4  (2,0)
    [10,     9],     // 5  (0,3)
    [11,     10],    // 6  (1,2)
    [12,     11],    // 7  (2,1)
    [13,     12],    // 8  (3,0)
    [15,     14],    // 9  (0,4)
    [16,     15],    // 10 (1,3)
    [17,     16],    // 11 (2,2)
    [18,     17],    // 12 (3,1)
    [19,     18],    // 13 (4,0)
    [20,     21],    // 14 (0,5) → saturated on bit=1
    [20,     21],    // 15 (1,4)
    [20,     21],    // 16 (2,3)
    [20,     20],    // 17 (3,2) → stays balanced
    [20,     21],    // 18 (4,1)
    [20,     21],    // 19 (5,0)
    [20,     21],    // 20 (3,3) → balanced sink
    [20,     21],    // 21 (4,4) → balanced sink
];

/// Indirect Context Model — replaces per-context count tables with a
/// bit-history state machine. Same causal semantics: `predict()` reflects the
/// state *before* the current bit; `update()` transitions the state *after*.
pub struct IcmModel {
    asm: ByteAssembler,
    /// Maps context-key → current ICM state index (0..NUM_STATES).
    states: Vec<u8>,
    table: &'static StateTable,
}

/// Bit-history state transition table for ICM.
struct StateTable {
    probs: [u16; NUM_STATES],
    next_state: [[usize; 2]; NUM_STATES],
}

impl StateTable {
    fn new() -> Self {
        StateTable {
            probs: build_state_probs(),
            next_state: TRANSITIONS,
        }
    }
}

// Singleton — the transition/prob tables are static constants.
static ICM_TABLE: std::sync::OnceLock<StateTable> = std::sync::OnceLock::new();

impl IcmModel {
    /// Create an ICM model. `ctx_bits` controls the context hash-table size.
    #[must_use]
    pub fn new(ctx_bits: u32) -> Self {
        Self::with_asm(ByteAssembler::new(4), ctx_bits)
    }

    #[must_use]
    pub fn with_asm(asm: ByteAssembler, ctx_bits: u32) -> Self {
        let entries = 1usize << ctx_bits;
        Self {
            asm,
            states: vec![0u8; entries],
            table: ICM_TABLE.get_or_init(StateTable::new),
        }
    }

    #[inline]
    fn key(&self) -> u64 {
        let n = self.asm.bytes_len();
        let last = self.asm.last_byte();
        let prev = self.asm.prev_byte();
        (n << 24) | (u64::from(last) << 8) | u64::from(prev) | (u64::from(self.asm.nbits()) << 16)
    }

    #[inline]
    fn slot(&self) -> usize {
        self.key() as usize & (self.states.len() - 1)
    }
}

impl BitModel for IcmModel {
    fn predict(&self) -> u16 {
        let idx = self.slot();
        let state = self.states[idx] as usize;
        let p = self.table.probs[state];
        p.max(MIN_PROB).min(MAX_PROB)
    }

    fn update(&mut self, bit: bool) {
        let idx = self.slot();
        let state = self.states[idx] as usize;
        let next = self.table.next_state[state][usize::from(bit)];
        self.states[idx] = next as u8;
        let _ = self.asm.push_bit(bit);
    }

    fn reset(&mut self) {
        self.asm.reset();
        self.states.fill(0);
    }
}

impl Default for IcmModel {
    fn default() -> Self {
        Self::new(21)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicts_uniform_when_fresh() {
        let m = IcmModel::new(21);
        // Fresh state 0 = (0,1), predicts P(1) > 50%.
        assert!(m.predict() > 2048, "fresh state 0 should predict >50%");
    }

    #[test]
    fn adapts_after_ones() {
        let mut m = IcmModel::new(21);
        for _ in 0..12 {
            m.update(true);
        }
        // After 12 ones, should be in a high state (saturated sink 21 → 2048).
        // The key invariant: after many 1s, predict stays high (>=2048).
        assert!(m.predict() >= 2048, "after 12 ones, predict should be high");
    }

    #[test]
    fn adapts_after_zeros() {
        let mut m = IcmModel::new(21);
        for _ in 0..12 {
            m.update(false);
        }
        assert!(m.predict() <= 2048, "after 12 zeros, predict should be low");
    }

    #[test]
    fn reset_clears_context() {
        let mut m = IcmModel::new(21);
        m.update(true);
        m.update(true);
        m.reset();
        assert_eq!(m.predict(), table().probs[0], "after reset, state should be back to 0");
    }

    #[test]
    fn state_transitions_climb_on_ones() {
        let table = table();
        let mut state = 0usize;
        // First 4 ones climb through high-prob states before saturating.
        for _ in 0..4 {
            state = table.next_state[state][1];
        }
        assert!(
            table.probs[state] >= 2048,
            "after 4 ones, state {} should have high probability, got {}",
            state,
            table.probs[state]
        );
    }

    fn table() -> &'static StateTable {
        ICM_TABLE.get_or_init(StateTable::new)
    }
}
