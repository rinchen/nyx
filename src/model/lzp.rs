//! LZP pre-stage (LZ + Prediction).
//!
//! LZP finds repeating byte sequences via a hash table of the last 4 bytes → position.
//! When the upcoming bytes match a previously-seen position, it emits a *match* bit and a
//! match *length* instead of literal bits, giving zstd-like speed on repetitive blocks.
//!
//! This is intentionally a lightweight pre-stage: it is consulted by the codec before the
//! bit models; it is **not** a [`BitModel`](super::BitModel) because its output is a
//! match length, not a per-bit probability.

const MIN_MATCH: usize = 4; // minimum match length worth emitting
const TABLE_BITS: usize = 18; // 1<<18 entries
const TABLE_SIZE: usize = 1 << TABLE_BITS;

/// LZP matcher.
pub struct Lzp {
    table: Vec<u32>,
}

impl Lzp {
    /// Create a matcher with a zeroed hash table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            table: vec![0u32; TABLE_SIZE],
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
        assert!(m.is_some_and(|l| l >= MIN_MATCH), "expected a match at the repeat");
    }

    #[test]
    fn no_match_on_unique_prefix() {
        let data = b"abcdefgh";
        let lzp = Lzp::new();
        assert!(lzp.try_match(data, 4).is_none());
    }
}
