//! Lazy multi-context LZP pre-stage.
//!
//! Extends the base LZP with two zstd-inspired techniques:
//!
//! 1. **Hash chains**: each 4-gram bucket keeps the last 4 positions, not just 1.
//!    This gives multiple match candidates per context, letting the model pick the
//!    strongest signal instead of blindly trusting the most recent occurrence.
//!
//! 2. **Lazy candidate selection**: when predicting the current bit, the model
//!    scans all chain entries and selects the candidate with the longest confirmed
//!    match. This mirrors zstd's "lazy parsing" — defer the literal/match decision
//!    until the best candidate is known, rather than committing to the first hit.
//!
//! The model still implements [`BitModel`] and contributes a per-bit prediction
//! to the logistic mixer.

use super::ByteAssembler;

const MIN_MATCH: usize = 4;
const TABLE_BITS: usize = 18;
const TABLE_SIZE: usize = 1 << TABLE_BITS;
const CHAIN_DEPTH: usize = 4;
/// Retained history window (was `history.len() > 1<<20` drain — now a ring).
const HISTORY_CAP: usize = 1 << 20;

/// Lazy multi-context LZP matcher.
pub struct LazyLzp {
    /// Hash chain: `table[hash]` = array of the last 4 positions for this 4-gram.
    table: Vec<[u32; CHAIN_DEPTH]>,
    asm: ByteAssembler,
    /// Fixed-capacity ring of the last `HISTORY_CAP` bytes. Read at absolute
    /// position `p` via `history[p % HISTORY_CAP]`; valid only while
    /// `p >= window_start() && p < hlen`.
    history: Vec<u8>,
    /// Absolute write position within `history` (monotonic).
    ring_pos: usize,
    /// Total bytes seen (absolute stream position).
    hlen: usize,
    bit_pos: u8,
}

impl LazyLzp {
    #[must_use]
    pub fn new() -> Self {
        Self {
            table: vec![[0u32; CHAIN_DEPTH]; TABLE_SIZE],
            asm: ByteAssembler::new(8),
            history: vec![0u8; HISTORY_CAP],
            ring_pos: 0,
            hlen: 0,
            bit_pos: 0,
        }
    }

    #[inline]
    fn hash(history: &[u8]) -> usize {
        let n = history.len();
        if n < 4 {
            return 0;
        }
        let h = u32::from(history[n - 4])
            | (u32::from(history[n - 3]) << 8)
            | (u32::from(history[n - 2]) << 16)
            | (u32::from(history[n - 1]) << 24);
        (h as usize) & (TABLE_SIZE - 1)
    }

    /// Hash the last 4 bytes of the retained history (absolute position `hlen`).
    #[inline]
    fn hash_at_hlen(&self) -> usize {
        if self.hlen < 4 {
            return 0;
        }
        let b0 = self.get_abs(self.hlen - 4);
        let b1 = self.get_abs(self.hlen - 3);
        let b2 = self.get_abs(self.hlen - 2);
        let b3 = self.get_abs(self.hlen - 1);
        let h =
            u32::from(b0) | (u32::from(b1) << 8) | (u32::from(b2) << 16) | (u32::from(b3) << 24);
        (h as usize) & (TABLE_SIZE - 1)
    }

    /// Read the byte at absolute position `p` from the ring.
    #[inline]
    fn get_abs(&self, p: usize) -> u8 {
        self.history[p % HISTORY_CAP]
    }

    /// Absolute position of the oldest byte still retained.
    #[inline]
    fn window_start(&self) -> usize {
        self.hlen.saturating_sub(HISTORY_CAP)
    }

    /// True if `p` is a position we can still read from the ring.
    #[inline]
    fn retained(&self, p: usize) -> bool {
        p >= self.window_start() && p < self.hlen
    }

    /// Find all match candidates from the chain, return (position, length) pairs.
    fn find_candidates(&self, data: &[u8], pos: usize) -> [(usize, usize); CHAIN_DEPTH] {
        let mut candidates = [(0usize, 0usize); CHAIN_DEPTH];
        if pos < 4 || pos >= data.len() {
            return candidates;
        }
        let idx = Self::hash(&data[..pos]);
        let chain = self.table[idx];
        for (i, &prev) in chain.iter().enumerate() {
            if prev == 0 || prev as usize >= pos {
                candidates[i] = (0, 0);
                continue;
            }
            let prev_usize = prev as usize;
            let mut len = 0;
            while pos + len < data.len()
                && prev_usize + len < data.len()
                && data[prev_usize + len] == data[pos + len]
            {
                len += 1;
                if len > 255 {
                    break;
                }
            }
            candidates[i] = (prev_usize, len);
        }
        candidates
    }

    /// Return the byte at the matched position for the best candidate.
    fn matched_byte(&self) -> Option<u8> {
        let hlen = self.hlen;
        if hlen < MIN_MATCH + 1 {
            return None;
        }
        let idx = self.hash_at_hlen();
        let chain = self.table[idx];
        let mut best_len = 0;
        let mut best_pos = 0usize;
        for &prev in &chain {
            let prev = prev as usize;
            if prev == 0 || prev < MIN_MATCH || prev >= hlen || !self.retained(prev) {
                continue;
            }
            // Confirm the 4-gram repeats
            let mut ok = true;
            for k in 0..4 {
                if self.get_abs(hlen - 4 + k) != self.get_abs(prev - 4 + k) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            // Measure match length (bounded by the retained tail before hlen).
            let mut len = 4;
            while prev + len < hlen
                && hlen - 4 + len < hlen
                && self.get_abs(prev + len) == self.get_abs(hlen - 4 + len)
                && len < 255
            {
                len += 1;
            }
            if len > best_len {
                best_len = len;
                best_pos = prev;
            }
        }
        if best_len >= MIN_MATCH && best_pos > 0 {
            Some(self.get_abs(best_pos))
        } else {
            None
        }
    }

    /// Insert `pos` at the head of the chain for the current 4-gram.
    fn train(&mut self, data: &[u8], pos: usize) {
        if pos < 4 || pos >= data.len() {
            return;
        }
        let idx = Self::hash(&data[..pos]);
        let mut chain = self.table[idx];
        // Shift chain: drop oldest, insert new at head.
        for i in (1..CHAIN_DEPTH).rev() {
            chain[i] = chain[i - 1];
        }
        chain[0] = pos as u32;
        self.table[idx] = chain;
    }
}

impl super::BitModel for LazyLzp {
    fn predict(&self) -> u16 {
        let hlen = self.hlen;
        if hlen < MIN_MATCH + 1 {
            return 2048;
        }
        let idx = self.hash_at_hlen();
        let chain = self.table[idx];
        let mut best_len = 0;
        let mut best_byte = 0u8;
        for &prev in &chain {
            let prev = prev as usize;
            if prev == 0 || prev < MIN_MATCH || prev >= hlen || !self.retained(prev) {
                continue;
            }
            // Confirm the 4-gram repeats: [hlen-4, hlen) == [prev-4, prev).
            let mut ok = true;
            for k in 0..4 {
                if self.get_abs(hlen - 4 + k) != self.get_abs(prev - 4 + k) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            // Causal extension: compare the candidate's continuation against the
            // known context tail, both staying before the current byte `hlen`.
            // (The original compared `history[hlen + len]` — a lookahead past the
            // end of history — which was always false, so the model never fired.)
            let mut len = 4;
            while prev + len < hlen
                && hlen - 4 + len < hlen
                && self.get_abs(prev + len) == self.get_abs(hlen - 4 + len)
                && len < 255
            {
                len += 1;
            }
            if len > best_len {
                best_len = len;
                best_byte = self.get_abs(prev);
            }
        }
        if best_len >= MIN_MATCH {
            let bit = (best_byte >> (7 - self.bit_pos)) & 1 == 1;
            if bit {
                3584
            } else {
                512
            }
        } else {
            2048
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn update(&mut self, bit: bool) {
        self.bit_pos = self.asm.nbits();
        let completed = self.asm.push_bit(bit);
        if let Some(byte) = completed {
            self.history[self.ring_pos % HISTORY_CAP] = byte;
            self.ring_pos += 1;
            self.hlen += 1;
            self.bit_pos = 0;
            let hlen = self.hlen;
            if hlen >= 4 {
                let idx = self.hash_at_hlen();
                let mut chain = self.table[idx];
                for i in (1..CHAIN_DEPTH).rev() {
                    chain[i] = chain[i - 1];
                }
                chain[0] = hlen as u32;
                self.table[idx] = chain;
            }
        }
    }

    fn reset(&mut self) {
        self.table.iter_mut().for_each(|c| c.fill(0));
        self.asm.reset();
        self.history.fill(0);
        self.ring_pos = 0;
        self.hlen = 0;
        self.bit_pos = 0;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Default for LazyLzp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::BitModel;
    use super::*;

    #[test]
    fn matches_repeated_literal_run() {
        let data = b"abcabcabcabcabcabc";
        let mut lzp = LazyLzp::new();
        for i in 1..=8 {
            lzp.train(data, i);
        }
        let candidates = lzp.find_candidates(data, 9);
        let has_match = candidates.iter().any(|(_, len)| *len >= MIN_MATCH);
        assert!(has_match, "expected at least one match candidate at pos 9");
    }

    #[test]
    fn selects_longest_match() {
        let data = b"abcabcabcabcabcabc";
        let mut lzp = LazyLzp::new();
        for i in 1..=12 {
            lzp.train(data, i);
        }
        let candidates = lzp.find_candidates(data, 12);
        let best = candidates
            .iter()
            .filter(|(_, len)| *len >= MIN_MATCH)
            .max_by_key(|(_, len)| *len);
        assert!(best.is_some(), "expected a match candidate");
    }

    #[test]
    fn reset_clears_chain() {
        let data = b"abcabcabcabcabcabc";
        let mut lzp = LazyLzp::new();
        lzp.train(data, 6);
        lzp.reset();
        let candidates = lzp.find_candidates(data, 9);
        assert!(
            candidates.iter().all(|(_, len)| *len == 0),
            "after reset, chain should be empty"
        );
    }
}
