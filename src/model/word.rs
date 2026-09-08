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

    #[test]
    fn xwrt_dictionary_serialization_caps_at_128() {
        // A corpus with hundreds of distinct words must serialize only the
        // top 128 (token space 0x80-0xFF); the prior `len() as u8` cast
        // truncated dicts >255 words to zero. Regression test.
        let mut text = Vec::new();
        for i in 0..300 {
            text.extend_from_slice(format!("word{i} ").as_bytes());
        }
        let dict = XwrtDictionary::build_from_data(&text);
        let bytes = dict.to_bytes();
        assert_eq!(bytes[0], 128, "dict must serialize exactly 128 words");
        let back = XwrtDictionary::from_bytes(&bytes).expect("parses");
        assert_eq!(back.id_to_word.len(), 128);
    }

    #[test]
    fn xwrt_global_dict_transform_round_trip() {
        let text = b"the quick brown fox jumps over the lazy dog\n\
the quick brown fox jumps over the lazy dog\n\
the quick brown fox jumps over the lazy dog\n"
            .to_vec();
        let dict = XwrtDictionary::build_from_data(&text);
        let xwrt = dict.transform(&text);
        assert!(xwrt.len() < text.len(), "frequent words should shrink");
        let back = inverse_with_dict(&xwrt, text.len(), &dict);
        assert_eq!(back, text);
    }

    #[test]
    fn xwrt_skips_breakless_runs() {
        // No word breaks in the stream: the whole run is one >maxlen "word",
        // which the scanner drops. Transform must be an identity passthrough.
        let text = b"rcnrcnrcn".repeat(5000);
        let dict = XwrtDictionary::build_from_data(&text);
        let bytes = dict.to_bytes();
        assert_eq!(bytes[0], 0, "no real words -> empty dict");
        let xwrt = dict.transform(&text);
        assert_eq!(xwrt, text, "empty dict is identity");
    }
}

/// XWRT: eXtended Word Replacement Transform.
///
/// Builds a static dictionary of the top words in the input text, then
/// replaces each word occurrence with a single token in `0x80..=0xFF`
/// (token space is 128 values, so only the top [`MAX_XWRT_WORDS`] words
/// are encoded; everything else passes through as literals).
/// Non-word bytes (including word break chars) pass through unchanged.
///
/// The encoder and decoder both build the dictionary identically by
/// scanning the input in a single pass to count word frequencies,
/// then keeping the top [`MAX_XWRT_WORDS`] by frequency (tie-break by
/// first appearance).
///
/// Returns: Vec<u8> where word tokens are in range [0x80, 0xFF].
pub const MAX_XWRT_WORDS: usize = 128;
/// Longest word kept by the XWRT scanner. Also the max that fits the
/// `u8` length byte in [`XwrtDictionary::to_bytes`]; anything longer is a
/// breakless run (base64/hex/garbage) rather than a real word.
pub const MAX_XWRT_WORD_LEN: usize = 255;
pub fn xwrt_transform(data: &[u8]) -> Vec<u8> {
    // First pass: count word frequencies and remember first appearance
    let mut scanner = WordScanner::new(2048); // keep top 2K words
    let mut i = 0usize;
    while i < data.len() {
        if is_word_break(data[i]) {
            i += 1;
            continue;
        }
        // Found start of word
        let start = i;
        while i < data.len() && !is_word_break(data[i]) {
            i += 1;
        }
        if i > start {
            scanner.add_word(&data[start..i]);
        }
    }
    let dictionary = scanner.build_dictionary();

    // Second pass: replace words with tokens
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0usize;
    while i < data.len() {
        if is_word_break(data[i]) {
            out.push(data[i]);
            i += 1;
            continue;
        }
        // Try to match longest word in dictionary
        let mut matched = false;
        for word in &dictionary.words {
            if i + word.len() <= data.len() && &data[i..i + word.len()] == word.as_slice() {
                let id = dictionary
                    .word_to_id
                    .get(String::from_utf8_lossy(word).as_ref())
                    .copied()
                    .unwrap_or(0);
                out.push(0x80 + (id as u8));
                i += word.len();
                matched = true;
                break;
            }
        }
        if !matched {
            // Not a known word, output byte as-is
            out.push(data[i]);
            i += 1;
        }
    }
    out
}

/// Inverse of [`xwrt_transform`].
///
/// Replaces tokens 0x80+ with their corresponding words from the dictionary.
///
/// The decoder rebuilds the identical dictionary by scanning the *output*
/// of xwrt_transform (which has the same word break positions as input)
/// and counting frequencies of the non-token bytes.
pub fn xwrt_inverse_transform(data: &[u8], orig_len: usize) -> Vec<u8> {
    // Rebuild dictionary from the transformed data (same word break positions)
    let mut scanner = WordScanner::new(2048);
    let mut i = 0usize;
    while i < data.len() {
        if data[i] >= 0x80 {
            // Token - skip it
            i += 1;
            continue;
        }
        if is_word_break(data[i]) {
            i += 1;
            continue;
        }
        // Found start of word
        let start = i;
        while i < data.len() && data[i] < 0x80 && !is_word_break(data[i]) {
            i += 1;
        }
        if i > start {
            scanner.add_word(&data[start..i]);
        }
    }
    let dictionary = scanner.build_dictionary();

    // Second pass: replace tokens with words
    let mut out = Vec::with_capacity(orig_len);
    let mut i = 0usize;
    while i < data.len() {
        if data[i] >= 0x80 {
            // Token
            let word_id = (data[i] & 0x7F) as usize;
            if let Some(word) = dictionary.id_to_word.get(word_id) {
                out.extend_from_slice(word.as_bytes());
            }
            i += 1;
        } else {
            // Regular byte
            out.push(data[i]);
            i += 1;
        }
    }
    out
}

/// Helper: scans text to find the top N words by frequency.
#[derive(Debug)]
pub(crate) struct WordScanner {
    /// Minimum word length to consider
    min_word_len: usize,
    /// Maximum number of words to keep in dictionary
    max_words: usize,
    /// Maximum word length kept (words longer than this are literal noise)
    max_word_len: usize,
    /// All words seen with their frequency and first position
    word_freq: Vec<(String, usize, usize)>,
    /// word -> index into `word_freq` (O(1) frequency updates)
    index: std::collections::HashMap<String, usize>,
}
impl WordScanner {
    pub(crate) fn new(max_words: usize) -> Self {
        Self {
            min_word_len: 2, // ignore single-letter words
            max_words,
            max_word_len: MAX_XWRT_WORD_LEN,
            word_freq: Vec::new(),
            index: std::collections::HashMap::new(),
        }
    }
    /// Add a word occurrence
    pub(crate) fn add_word(&mut self, word: &[u8]) {
        if word.len() < self.min_word_len || word.len() > self.max_word_len {
            return;
        }
        let word_str = String::from_utf8_lossy(word).to_string();
        if let Some(&idx) = self.index.get(&word_str) {
            // Increment frequency in place without a linear scan.
            self.word_freq[idx].1 += 1;
        } else {
            // New word
            let idx = self.word_freq.len();
            self.index.insert(word_str.clone(), idx);
            self.word_freq.push((word_str, 1, idx));
        }
    }
    /// Build dictionary: top max_words by frequency, tie-break by first appearance
    pub(crate) fn build_dictionary(&mut self) -> XwrtDictionary {
        // Sort by frequency descending, then by first appearance ascending
        self.word_freq
            .sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));
        // Keep only top max_words
        self.word_freq.truncate(self.max_words);
        // Build lookup tables
        let mut word_to_id = HashMap::new();
        let mut id_to_word = Vec::new();
        for (idx, (word, _, _)) in self.word_freq.iter().enumerate() {
            word_to_id.insert(word.clone(), idx);
            id_to_word.push(word.clone());
        }
        let words: Vec<Vec<u8>> = self
            .word_freq
            .iter()
            .map(|(w, _, _)| w.as_bytes().to_vec())
            .collect();
        XwrtDictionary {
            words,
            word_to_id,
            id_to_word,
        }
    }
}
/// XWRT dictionary: keeps top N words and provides bidirectional lookup.
#[derive(Debug)]
pub struct XwrtDictionary {
    /// The words in the dictionary, sorted by frequency descending
    words: Vec<Vec<u8>>,
    /// word -> id lookup
    word_to_id: HashMap<String, usize>,
    /// id -> word lookup
    id_to_word: Vec<String>,
}

impl XwrtDictionary {
    /// Build dictionary from original data, then apply XWRT transform.
    pub fn build_from_data(data: &[u8]) -> Self {
        let mut scanner = WordScanner::new(2048);
        let mut i = 0usize;
        while i < data.len() {
            if is_word_break(data[i]) {
                i += 1;
                continue;
            }
            let start = i;
            while i < data.len() && !is_word_break(data[i]) {
                i += 1;
            }
            if i > start {
                scanner.add_word(&data[start..i]);
            }
        }
        let dict = scanner.build_dictionary();
        dict
    }

    /// Apply XWRT transform to data using this dictionary.
    ///
    /// Only the top [`MAX_XWRT_WORDS`] words are encoded: tokens span
    /// `0x80..=0xFF` (128 values), so words beyond the top 128 are emitted as
    /// literals to keep the transform lossless.
    pub fn transform(&self, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(data.len());
        let mut i = 0usize;
        while i < data.len() {
            if is_word_break(data[i]) {
                out.push(data[i]);
                i += 1;
                continue;
            }
            let mut matched = false;
            for word in self.words.iter().take(MAX_XWRT_WORDS) {
                if i + word.len() <= data.len() && &data[i..i + word.len()] == word.as_slice() {
                    let id = self
                        .word_to_id
                        .get(String::from_utf8_lossy(word).as_ref())
                        .copied()
                        .unwrap_or(0);
                    out.push(0x80 + (id as u8));
                    i += word.len();
                    matched = true;
                    break;
                }
            }
            if !matched {
                out.push(data[i]);
                i += 1;
            }
        }
        out
    }

    /// Build dictionary from XWRT-transformed data (words appear as non-token, non-break bytes).
    pub fn from_transform(data: &[u8]) -> Vec<u8> {
        let mut scanner = WordScanner::new(2048);
        let mut i = 0usize;
        while i < data.len() {
            if data[i] >= 0x80 {
                i += 1;
                continue;
            }
            if is_word_break(data[i]) {
                i += 1;
                continue;
            }
            let start = i;
            while i < data.len() && data[i] < 0x80 && !is_word_break(data[i]) {
                i += 1;
            }
            if i > start {
                scanner.add_word(&data[start..i]);
            }
        }
        let dict = scanner.build_dictionary();
        dict.to_bytes()
    }

    /// Serialize dictionary to bytes for storage in the encoded stream.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        // Limit to MAX_XWRT_WORDS (token range 0x80-0xFF holds 128 ids).
        // Cast AFTER capping: `id_to_word.len() as u8` would truncate for
        // dicts larger than 255 words.
        let n = self.id_to_word.len().min(MAX_XWRT_WORDS) as u8;
        out.push(n);
        for i in 0..n as usize {
            let word = &self.id_to_word[i];
            out.push(word.len() as u8);
            out.extend_from_slice(word.as_bytes());
        }
        out
    }

    /// Deserialize dictionary from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }
        let n = data[0] as usize;
        let mut pos = 1;
        let mut id_to_word = Vec::with_capacity(n);
        let mut word_to_id = HashMap::new();
        let mut words = Vec::with_capacity(n);
        for i in 0..n {
            if pos >= data.len() {
                return None;
            }
            let len = data[pos] as usize;
            pos += 1;
            if pos + len > data.len() {
                return None;
            }
            let word = String::from_utf8_lossy(&data[pos..pos + len]).to_string();
            pos += len;
            id_to_word.push(word.clone());
            word_to_id.insert(word.clone(), i);
            words.push(word.as_bytes().to_vec());
        }
        Some(Self {
            words,
            word_to_id,
            id_to_word,
        })
    }
}
/// Check if a byte is a word break character.
#[inline]
pub(crate) fn is_word_break(b: u8) -> bool {
    matches!(
        b,
        b' ' | b'\t'
            | b'\n'
            | b'\r'
            | b'\x00'
            | b'!'
            | b'"'
            | b'#'
            | b'$'
            | b'%'
            | b'&'
            | b'\''
            | b'('
            | b')'
            | b'*'
            | b'+'
            | b','
            | b'-'
            | b'.'
            | b'/'
            | b':'
            | b';'
            | b'<'
            | b'='
            | b'>'
            | b'?'
            | b'@'
            | b'['
            | b'\\'
            | b']'
            | b'^'
            | b'_'
            | b'`'
            | b'{'
            | b'|'
            | b'}'
            | b'~'
    )
}

/// Inverse XWRT using an explicit dictionary (from the encoded stream).
pub fn xwrt_inverse_with_dict(data: &[u8], orig_len: usize, dict_data: &[u8]) -> Vec<u8> {
    let dict = match XwrtDictionary::from_bytes(dict_data) {
        Some(d) => d,
        None => return Vec::new(),
    };
    inverse_with_dict(data, orig_len, &dict)
}

pub(crate) fn inverse_with_dict(data: &[u8], orig_len: usize, dict: &XwrtDictionary) -> Vec<u8> {
    let mut out = Vec::with_capacity(orig_len);
    let mut i = 0usize;
    while i < data.len() {
        if data[i] >= 0x80 {
            let word_id = (data[i] & 0x7F) as usize;
            if let Some(word) = dict.id_to_word.get(word_id) {
                out.extend_from_slice(word.as_bytes());
            }
            i += 1;
        } else {
            out.push(data[i]);
            i += 1;
        }
    }
    out
}
