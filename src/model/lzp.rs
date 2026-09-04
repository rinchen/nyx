//! LZP pre-stage with hash chains and cross-block persistence.
//!
//! Long-range repetition needs a real LDM-style window. zstd-1 uses up to 8MB;
//! we start with 4MB. To make that useful, the hash table becomes 8-entry chains
//! instead of a single last-position-per-bucket, so we can walk candidates and
//! pick the longest match.
//!
//! Cross-block persistence:
//! - The hash table and position chain survive across blocks within a stream.
//! - Mixer weights decay by `*0.995` per block instead of reset, so learning
//!   transfers across block boundaries.
//! - On decompress, the decoder reconstructs the same state from the decoded
//!   stream because the state update is deterministic from byte history.
//!
//! Block size classifier-aware:
//! - Text blocks: 1–4 MB (longer window for repeated phrases, quotes, boilerplate).
//! - Binary/Exec/Random: 64 KiB.

use super::BitModel;
use super::ByteAssembler;

const MIN_MATCH: usize = 4;
const TABLE_BITS: u32 = 19; // 512K entries (down from 1M to reduce memory)
const TABLE_SIZE: usize = 1 << TABLE_BITS;
const MAX_CHAIN: usize = 8; // walk at most 8 candidates per lookup
const DEFAULT_WINDOW: usize = 4 * 1024 * 1024; // 4MB sliding window
const DEFAULT_HISTORY_CAP: usize = 4 * 1024 * 1024; // 4MiB history ring (matches window)
const NEXT_SIZE: usize = 1 << 21; // 2M entry chain-link table

/// LZP matcher with hash chains and cross-block persistence.
///
/// State is intentionally reconstructible: both encoder and decoder build
/// identical hash tables by processing the same decoded prefix, so round-trip
/// correctness holds without serializing the table.
pub struct Lzp {
    /// Hash chain heads: `heads[hash]` = newest position for this 4-gram.
    heads: Vec<u32>,
    /// Hash chain links: indexed by `position % NEXT_SIZE`.
    next: Vec<u32>,
    /// Current absolute position in the decoded stream.
    current_pos: usize,
    /// Sliding window: only keep positions within `current_pos - window`.
    window: usize,
    /// Assembled byte stream so we can key the hash on whole bytes, not bits.
    asm: ByteAssembler,
    /// Ring of recent bytes for match confirmation.
    history: Vec<u8>,
    /// Bit index within the current (in-progress) byte.
    bit_pos: u8,
}

impl Lzp {
    #[must_use]
    pub fn new() -> Self {
        Self {
            heads: vec![0u32; TABLE_SIZE],
            next: vec![0u32; NEXT_SIZE],
            current_pos: 0,
            window: DEFAULT_WINDOW,
            asm: ByteAssembler::new(8),
            history: Vec::with_capacity(DEFAULT_HISTORY_CAP.min(1 << 20)),
            bit_pos: 0,
        }
    }

    #[must_use]
    pub fn with_window(window: usize) -> Self {
        let mut s = Self::new();
        s.window = window;
        s
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

    /// Record that the stream has advanced to `pos`. Inserts the new 4-gram
    /// ending at `pos` at the head of its chain.
    pub fn train_at(&mut self, data: &[u8], pos: usize) {
        if pos < 4 {
            return;
        }
        let idx = Self::hash(&data[..pos]);
        let p = pos as u32;
        let old = self.heads[idx];
        self.heads[idx] = p;
        let ni = (p as usize) % NEXT_SIZE;
        self.next[ni] = old;
        self.current_pos = pos;
    }

    /// Record that the stream has advanced by one completed byte.
    pub fn train_byte(&mut self, byte: u8) {
        self.history.push(byte);
        if self.history.len() > DEFAULT_HISTORY_CAP {
            let drop = self.history.len() - DEFAULT_HISTORY_CAP;
            self.history.drain(0..drop);
        }
        let hlen = self.history.len();
        if hlen >= 4 {
            let idx = Self::hash(&self.history[..hlen]);
            let p = hlen as u32;
            let old = self.heads[idx];
            self.heads[idx] = p;
            let ni = hlen % NEXT_SIZE;
            self.next[ni] = old;
            self.current_pos = hlen;
        }
    }

    /// Walk at most `MAX_CHAIN` candidates and return the longest match length,
    /// or `None` if no match reaches `MIN_MATCH`.
    pub fn longest_match(&self, data: &[u8], pos: usize) -> Option<usize> {
        if pos < 4 || pos >= data.len() {
            return None;
        }
        let idx = Self::hash(&data[..pos]);
        let mut best = 0usize;
        let mut cur = self.heads[idx];
        let mut walked = 0usize;
        let limit = self.current_pos.saturating_sub(self.window);
        while cur != 0 && walked < MAX_CHAIN {
            let p = cur as usize;
            let ni = p % NEXT_SIZE;
            if p >= pos || p < limit {
                cur = self.next[ni];
                walked += 1;
                continue;
            }
            let mut len = 0usize;
            while pos + len < data.len()
                && p + len < data.len()
                && data[pos + len] == data[p + len]
            {
                len += 1;
                if len > 255 {
                    break;
                }
            }
            if len > best {
                best = len;
                if best == 255 {
                    break;
                }
            }
            cur = self.next[ni];
            walked += 1;
        }
        if best >= MIN_MATCH {
            Some(best)
        } else {
            None
        }
    }

    /// Return the byte at the best matched position, for bit prediction.
    fn matched_byte(&self) -> Option<u8> {
        let hlen = self.history.len();
        if hlen < MIN_MATCH + 1 {
            return None;
        }
        let idx = Self::hash(&self.history[..hlen]);
        let mut cur = self.heads[idx];
        let mut best = 0usize;
        let mut best_pos = 0usize;
        let limit = self.current_pos.saturating_sub(self.window);
        let mut walked = 0usize;
        while cur != 0 && walked < MAX_CHAIN {
            let p = cur as usize;
            let ni = p % NEXT_SIZE;
            if p >= hlen || p < limit {
                cur = self.next[ni];
                walked += 1;
                continue;
            }
            if hlen - p > best {
                best = hlen - p;
                best_pos = p;
                if best == 255 {
                    break;
                }
            }
            cur = self.next[ni];
            walked += 1;
        }
        if best >= MIN_MATCH {
            self.history.get(best_pos).copied()
        } else {
            None
        }
    }
}

impl BitModel for Lzp {
    fn predict(&self) -> u16 {
        self.matched_byte().map_or(2048, |b| {
            let bit = (b >> (7 - self.bit_pos)) & 1 == 1;
            if bit {
                3584
            } else {
                512
            }
        })
    }

    fn update(&mut self, bit: bool) {
        self.bit_pos = self.asm.nbits();
        let completed = self.asm.push_bit(bit);
        if let Some(byte) = completed {
            self.train_byte(byte);
            self.bit_pos = 0;
        }
    }

    fn reset(&mut self) {
        self.heads.fill(0);
        self.next.fill(0);
        self.current_pos = 0;
        self.asm.reset();
        self.history.clear();
        self.bit_pos = 0;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Default for Lzp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_finds_longest_repeat() {
        let data = b"abcabcabcabcabcabc";
        let mut lzp = Lzp::new();
        for i in 1..=8 {
            lzp.train_at(data, i);
        }
        let m = lzp.longest_match(data, 9);
        assert!(m.is_some_and(|l| l >= MIN_MATCH));
    }

    #[test]
    fn chain_no_match_on_unique_prefix() {
        let data = b"abcdefgh";
        let lzp = Lzp::new();
        assert!(lzp.longest_match(data, 4).is_none());
    }
}
