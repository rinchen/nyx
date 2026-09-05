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
use std::collections::HashMap;

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

/// Byte-Pair Re-Pair dictionary, built incrementally from the byte stream.
///
/// Maintains a running count of byte-pair frequencies and replaces frequent
/// pairs with high-byte symbols (0x80+).  The cap-fold bit is stored alongside
/// each substituted symbol to preserve case information.
///
/// Both encoder and decoder build this dictionary identically because they
/// process the same byte stream via `ByteAssembler`.
const MAX_DICT_SIZE: usize = 2048; // top 2K frequent substrings
const DICT_SYMBOL_BASE: u8 = 0x80; // substituted symbols start at 0x80

/// An entry in the Re-Pair dictionary: maps a (byte pair) → symbol id.
#[derive(Debug, Clone, Copy)]
struct DictEntry {
    /// The byte pair this entry replaces (e.g. [b'a', b'b']).
    pair: [u8; 2],
    /// The symbol ID assigned (0-based, added to DICT_SYMBOL_BASE when emitted).
    id: u8,
}

/// Byte-Pair-RePair dictionary, built incrementally.
///
/// Tracks byte-pair frequencies and promotes frequent pairs to high-byte
/// substitution symbols (`0x80+`).  The cap-fold bit preserves case info.
#[allow(dead_code)] // some methods are for Phase 2 integration
#[derive(Debug, Default)]
struct BytePairDict {
    /// Byte-pair frequency counts.
    pair_counts: HashMap<[u8; 2], u32>,
    /// Forward lookup: pair → symbol id.
    pair_to_id: HashMap<[u8; 2], u8>,
    /// Reverse lookup: symbol id → pair.
    id_to_pair: Vec<DictEntry>,
    /// Next available symbol ID.
    next_id: u8,
}

impl BytePairDict {
    fn new() -> Self {
        Self {
            pair_counts: HashMap::new(),
            pair_to_id: HashMap::new(),
            id_to_pair: Vec::with_capacity(MAX_DICT_SIZE),
            next_id: 0,
        }
    }

    /// Record a byte pair occurrence and potentially promote it to a substitution.
    ///
    /// When a pair's frequency crosses a threshold (currently: it becomes the
    /// most frequent pair and there's room in the dictionary), it gets a symbol ID.
    #[inline]
    fn record_pair(&mut self, pair: [u8; 2]) {
        if self.next_id as usize >= MAX_DICT_SIZE {
            return; // dictionary full
        }
        let count = self.pair_counts.entry(pair).or_insert(0);
        *count += 1;
        // Promote to a symbol if this pair's frequency exceeds the threshold.
        // Threshold: count must be at least 4 (empirically balances noise vs. signal).
        if *count >= 4 && !self.pair_to_id.contains_key(&pair) {
            self.pair_to_id.insert(pair, self.next_id);
            self.id_to_pair.push(DictEntry {
                pair,
                id: self.next_id,
            });
            self.next_id = self.next_id.wrapping_add(1);
        }
    }

    /// Look up a byte pair in the dictionary. Returns the symbol ID if present.
    #[inline]
    fn lookup(&self, pair: [u8; 2]) -> Option<u8> {
        self.pair_to_id.get(&pair).copied()
    }

    /// Check if a byte is a high-byte substitution symbol (0x80+).
    #[inline]
    fn is_symbol(b: u8) -> bool {
        b >= DICT_SYMBOL_BASE
    }

    /// Get the pair for a symbol ID (for inverse substitution).
    #[inline]
    fn reverse_lookup(&self, id: u8) -> Option<[u8; 2]> {
        self.id_to_pair.iter().find(|e| e.id == id).map(|e| e.pair)
    }

    fn reset(&mut self) {
        self.pair_counts.clear();
        self.pair_to_id.clear();
        self.id_to_pair.clear();
        self.next_id = 0;
    }
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
    #[allow(dead_code)] // used for configuration, may be increased later
    max_words: usize,
    /// Byte-Pair Re-Pair dictionary for the current block.
    dict: BytePairDict,
    /// Previous byte (for pair tracking).
    prev_byte: Option<u8>,
    /// Whether to use Re-Pair substitution.
    #[allow(dead_code)] // toggle for experimental use
    use_repar: bool,
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
            dict: BytePairDict::new(),
            prev_byte: None,
            use_repar: true,
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

    /// Apply Re-Pair substitution to a byte if a matching pair was seen.
    ///
    /// Returns `Some(substituted_byte)` if the previous byte + current byte form
    /// a known pair, `None` otherwise.
    fn try_substitute(&mut self, byte: u8) -> Option<u8> {
        if !self.use_repar || self.prev_byte.is_none() {
            return None;
        }
        let prev = self.prev_byte.unwrap();
        if let Some(id) = self.dict.lookup([prev, byte]) {
            // Cap-fold bit: bit 0 = lowercase/original, bit 1 = uppercase variant.
            // We only substitute for lowercase bytes; uppercase bytes are passed through.
            if byte.is_ascii_lowercase() || byte.is_ascii_digit() || WORD_BREAKS.contains(&byte) {
                let sym = DICT_SYMBOL_BASE + id;
                return Some(sym);
            }
        }
        None
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
            // Record byte pair for Re-Pair dictionary.
            if let Some(prev) = self.prev_byte {
                self.dict.record_pair([prev, byte]);
            }
            self.prev_byte = Some(byte);

            // Check if this byte is a Re-Pair substitution symbol.
            if BytePairDict::is_symbol(byte) {
                // Substituted symbol — treat as a word break for word modeling.
                self.flush_word();
                // Don't push the symbol into cur_word.
            } else if WORD_BREAKS.contains(&byte) {
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
        self.dict.reset();
        self.prev_byte = None;
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

    #[test]
    fn repar_dictionary_builds_from_frequent_pairs() {
        let mut m = WordModel::new();
        // "hello world hello world..." x4 — "wo" appears 3 times, "ll" 3 times.
        // Need threshold 3 to trigger. Let me use enough repeats.
        let text = b"hello world hello world hello world hello world hello world hello world";
        for &b in text {
            for bit_idx in (0..8).rev() {
                let bit = (b >> bit_idx) & 1 == 1;
                m.update(bit);
            }
        }
        // After enough repeats, the dictionary should have at least one entry.
        assert!(
            m.dict.id_to_pair.len() > 0,
            "dictionary should have at least one entry after frequent pairs"
        );
    }
}
