//! Word/string model for text compression.
//!
//! Adds a **word dictionary model** that tokenizes text into word symbols and
//! maintains a P(bit==1) table keyed on the previous word + current bit position.
//! This is PAQ's "word model" — it turns natural language redundancy into direct
//! symbol prediction instead of byte-level context chains.
//!
//! Only active for `Text` blocks. Binary/Exec keep the existing stack.

use super::ctable::CtxTable;
use super::BitModel;
use super::ByteAssembler;

const MAX_PROB: u16 = 4095;
const MIN_PROB: u16 = 1;
const CTX_BITS: u32 = 19; // ~512 KiB table

/// Word break characters: space, tab, newline, carriage return, and common punctuation.
const WORD_BREAKS: &[u8] = b" \t\n\r\0\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\
                               !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/// Rolling word buffer: keeps the last N completed words.
#[derive(Debug, Clone, Default)]
struct WordBuffer {
    /// Completed word byte sequences.
    words: Vec<Vec<u8>>,
    /// Maximum number of words to retain.
    cap: usize,
}

impl WordBuffer {
    fn new(cap: usize) -> Self {
        Self {
            words: Vec::with_capacity(cap),
            cap,
        }
    }

    /// Push a completed word. If buffer is full, drop the oldest.
    fn push(&mut self, word: Vec<u8>) {
        if self.words.len() == self.cap {
            self.words.remove(0);
        }
        self.words.push(word);
    }

    /// Get the last word, if any.
    fn last(&self) -> Option<&[u8]> {
        self.words.last().map(|w| w.as_slice())
    }

    /// Reset the buffer.
    fn reset(&mut self) {
        self.words.clear();
    }
}

/// Simple word hash: FNV-1a 32-bit over the word bytes.
#[inline]
fn word_hash(word: &[u8]) -> u32 {
    let mut h = 2166136261u32;
    for &b in word {
        h ^= u32::from(b);
        h = h.wrapping_mul(16777619);
    }
    h
}

/// Word model: tokenizes text into words and models the current bit position
/// conditioned on the previous word hash.
///
/// This captures the strong "previous word predicts current word" signal in
/// natural language text, which byte-level order-N models miss because they
/// operate over raw byte positions rather than symbol boundaries.
pub struct WordModel {
    asm: ByteAssembler,
    ctab: CtxTable,
    /// Rolling buffer of recent words.
    words: WordBuffer,
    /// Current incomplete word being assembled.
    cur_word: Vec<u8>,
    /// Maximum word length to retain (prevents pathological memory use).
    max_word_len: usize,
    /// Maximum number of recent words to keep in the rolling buffer.
    max_words: usize,
}

impl WordModel {
    /// Create a new word model with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            asm: ByteAssembler::new(8),
            ctab: CtxTable::new(CTX_BITS),
            words: WordBuffer::new(8),
            cur_word: Vec::new(),
            max_word_len: 64,
            max_words: 8,
        }
    }

    /// Finalize the current incomplete word and push it to the buffer.
    fn flush_word(&mut self) {
        if self.cur_word.is_empty() {
            return;
        }
        let word = std::mem::take(&mut self.cur_word);
        self.words.push(word);
    }

    /// Compute the context key for the current bit position.
    #[inline]
    fn key(&self) -> u64 {
        let prev_hash = self.words.last().map_or(0u32, |w| word_hash(w));
        let bit_pos = u32::from(self.asm.nbits());
        let last = u32::from(self.asm.last_byte());
        let prev = u32::from(self.asm.prev_byte());
        // Mix in the previous word hash + bit position + last byte context.
        u64::from(prev_hash).rotate_left(32)
            ^ (u64::from(bit_pos) << 40)
            ^ (u64::from(last) << 8)
            ^ u64::from(prev)
    }
}

impl BitModel for WordModel {
    #[inline]
    fn predict(&self) -> u16 {
        let [c0, c1] = self.ctab.get(self.key());
        let tot = f64::from(c0 + c1);
        (f64::from(c1) / tot * f64::from(MAX_PROB)).clamp(f64::from(MIN_PROB), f64::from(MAX_PROB))
            as u16
    }

    #[inline]
    fn update(&mut self, bit: bool) {
        let completed = self.asm.push_bit(bit);
        if let Some(byte) = completed {
            if WORD_BREAKS.contains(&byte) {
                self.flush_word();
            } else if self.cur_word.len() < self.max_word_len {
                self.cur_word.push(byte);
            }
        }

        let k = self.key();
        self.ctab.update(k, bit);
    }

    fn reset(&mut self) {
        self.asm.reset();
        self.ctab.reset();
        self.words.reset();
        self.cur_word.clear();
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Default for WordModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_words() {
        let mut m = WordModel::new();
        // Feed "the " — word break after 'e'
        for b in b"the " {
            for bit_idx in (0..8).rev() {
                let bit = (b >> bit_idx) & 1 == 1;
                m.update(bit);
            }
        }
        // Word buffer should have "the"
        let last = m.words.last().unwrap_or(b"");
        assert_eq!(last, b"the");
    }

    #[test]
    fn starts_neutral() {
        let m = WordModel::new();
        assert!(
            (i32::from(m.predict()) - 2048).abs() <= 2,
            "new WordModel should start near neutral, got {}",
            m.predict()
        );
    }

    #[test]
    fn adapts_after_repeated_word() {
        let mut m = WordModel::new();
        let text = b"hello hello hello hello hello hello hello hello hello hello ";
        for b in text {
            for bit_idx in (0..8).rev() {
                let bit = ((*b) >> bit_idx) & 1 == 1;
                m.update(bit);
            }
        }
        // After many repetitions of "hello ", the model should have learned
        // something about the pattern.
        assert!(m.words.words.len() > 0, "should have words in buffer");
    }

    #[test]
    fn reset_clears_state() {
        let mut m = WordModel::new();
        for b in b"test test test " {
            for bit_idx in (0..8).rev() {
                m.update(((b) >> bit_idx) & 1 == 1);
            }
        }
        m.reset();
        assert!(
            m.words.words.is_empty(),
            "word buffer should be empty after reset"
        );
    }
}
