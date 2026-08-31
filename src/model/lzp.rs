//! LZP pre-stage (LZ + Prediction).
//!
//! LZP finds repeating byte sequences via a hash table of the last 4 bytes → position.
//! As a [`BitModel`] it contributes a per-bit prediction: when the current 4-gram context
//! has a previous occurrence, it predicts the next bit to equal the bit at that matched
//! position (strong on repetitive data); otherwise it predicts 2048 (uninformative). This
//! lets the logistic mixer down-weight it on non-repetitive blocks and up-weight it on ones
//! where literal bits are predictable from past occurrences — the "LZP pre-stage feeding a
//! shared bit-mixer" design.

use super::ByteAssembler;

const MIN_MATCH: usize = 4; // minimum match length worth emitting
const TABLE_BITS: usize = 18; // 1<<18 entries
const TABLE_SIZE: usize = 1 << TABLE_BITS;

/// LZP matcher / pre-stage bit model.
pub struct Lzp {
    table: Vec<u32>,
    /// Assembled byte stream so we can key the hash on whole bytes, not bits.
    asm: ByteAssembler,
    /// Ring of recent bytes for match confirmation.
    history: Vec<u8>,
    /// Bit index within the current (in-progress) byte, for extracting a matched bit.
    bit_pos: u8,
}

impl Lzp {
    /// Create a matcher with a zeroed hash table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            table: vec![0u32; TABLE_SIZE],
            asm: ByteAssembler::new(8),
            history: Vec::with_capacity(TABLE_SIZE.max(16)),
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

    /// Try to match the next bytes of `data` starting at `pos` against history.
    ///
    /// Returns `Some(len)` if a match of at least `MIN_MATCH` bytes was found, where `len`
    /// is the match length. The caller should advance `pos` by `len` and record the match.
    #[must_use]
    pub fn try_match(&self, data: &[u8], pos: usize) -> Option<usize> {
        if pos < 4 || pos >= data.len() {
            return None;
        }
        let idx = Self::hash(&data[..pos]);
        let prev = self.table[idx] as usize;
        if prev == 0 || prev >= pos {
            return None;
        }
        // confirm the bytes actually match from prev.. against pos..
        let mut len = 0;
        while pos + len < data.len() && data[prev + len] == data[pos + len] {
            len += 1;
            if len > 255 {
                break;
            }
        }
        if len >= MIN_MATCH {
            Some(len)
        } else {
            None
        }
    }

    /// Record that `data[..pos]` ends at `pos` (call after consuming a literal or match).
    pub fn train(&mut self, data: &[u8], pos: usize) {
        if pos >= 4 {
            let idx = Self::hash(&data[..pos]);
            self.table[idx] = pos as u32;
        }
    }

    /// If the last completed 4-gram repeats an earlier one, return the byte at the
    /// matched position that should follow (used to predict the current bit).
    fn matched_byte(&self) -> Option<u8> {
        let hlen = self.history.len();
        if hlen < MIN_MATCH + 1 {
            return None;
        }
        let idx = Self::hash(&self.history[..hlen]);
        let prev = self.table[idx] as usize;
        if prev == 0 || prev >= hlen {
            return None;
        }
        // Confirm the 4-gram actually repeats at `prev`.
        if self.history[hlen - 4..hlen] != self.history[prev - 4..prev] {
            return None;
        }
        // Predict the byte that followed the earlier occurrence.
        (prev < self.history.len()).then(|| self.history[prev])
    }
}

impl super::BitModel for Lzp {
    fn predict(&self) -> u16 {
        // When we can predict the next whole byte from an LZP match, derive the
        // current bit of that byte. Otherwise uninformative.
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
            self.history.push(byte);
            if self.history.len() > 1 << 20 {
                // Cap history growth; we only need recent context for matching.
                let drop = self.history.len() - (1 << 20);
                self.history.drain(0..drop);
            }
            self.bit_pos = 0;
            // Train the hash on the completed byte.
            let hlen = self.history.len();
            if hlen >= 4 {
                let idx = Self::hash(&self.history[..hlen]);
                self.table[idx] = hlen as u32;
            }
        }
    }

    fn reset(&mut self) {
        self.table.fill(0);
        self.asm.reset();
        self.history.clear();
        self.bit_pos = 0;
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
    fn matches_repeated_literal_run() {
        // Six repeats of "abc" (18 bytes). The 4-gram "cabc" first occurs ending at
        // pos=6 (data[2..6]); its next occurrence ends at pos=9.
        let data = b"abcabcabcabcabcabc";
        let mut lzp = Lzp::new();
        for i in 1..=8 {
            lzp.train(data, i); // records the first "cabc" at pos=6
        }
        // At pos=9 the 4-gram "cabc" (data[5..9]) repeats the one seen at pos=6;
        // the following bytes (data[9..15]) match data[6..12] for >= MIN_MATCH.
        let m = lzp.try_match(data, 9);
        assert!(
            m.is_some_and(|l| l >= MIN_MATCH),
            "expected a match at the repeat"
        );
    }

    #[test]
    fn no_match_on_unique_prefix() {
        let data = b"abcdefgh";
        let lzp = Lzp::new();
        assert!(lzp.try_match(data, 4).is_none());
    }
}
