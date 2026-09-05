//! Two-level context-mixing hierarchy: bank mixers → global mixer → master mixer.
//!
//! CMIX / PAQ8 use a multi-layer mixer stack:
//!   1. **Bank mixers** (4096 instances): each selected by a context hash of
//!      order-1 / order-2 / word-context + bit position. One logistic mixer
//!      saturates at ~50% on repetitive corpora (dickens); per-context banks
//!      avoid this by specializing weights per context.
//!   2. **Global mixer**: a single shared logistic mixer over the same models,
//!      providing a context-agnostic fallback prediction.
//!   3. **Master mixer**: blends `[p_bank, p_global, p_lzp]` in logistic space.
//!      Only the master + the selected bank are trained per bit — never all 4096.
//!
//! Cross-block persistence: bank and master weights are **decayed** (not reset)
//! at block boundaries, preserving learned structure across the stream.
//!
//! Memory: 4096 mixers × 8 models × (1 base + 8 pos) × 4 bytes ≈ 1.1 MB for the
//! banks, plus 2 mixers (global + master) ≈ 1.1 MB total.

use crate::model::mixer::LogisticMixer;
use crate::model::ByteAssembler;

/// Number of mixer instances in the bank. 4096 gives 12 bits of context selection.
pub const NUM_MIXERS: usize = 4096;

/// Byte classes for mixer selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteClass {
    Space = 0,
    Vowel = 1,
    Consonant = 2,
    Digit = 3,
    Punct = 4,
    Other = 5,
}

impl ByteClass {
    /// Classify a byte into one of six text-relevant classes.
    #[must_use]
    pub fn classify(b: u8) -> Self {
        match b {
            b' ' | b'\t' | b'\n' | b'\r' => ByteClass::Space,
            b'a' | b'e' | b'i' | b'o' | b'u' | b'A' | b'E' | b'I' | b'O' | b'U' => ByteClass::Vowel,
            b'0'..=b'9' => ByteClass::Digit,
            b'.' | b',' | b';' | b':' | b'!' | b'?' | b'"' | b'\'' | b'-' | b'(' | b')' | b'['
            | b']' | b'{' | b'}' => ByteClass::Punct,
            _ if b.is_ascii_alphabetic() => ByteClass::Consonant,
            _ => ByteClass::Other,
        }
    }
}

/// Compute the bank ID from context.
///
/// Uses a 12-bit hash: `(byte_class << 9) | (bit_pos << 6) | (order1 << 3) | word_hash`
///
/// The bit position is folded into the bank ID so each (context, bit-position) pair
/// gets its own weight vector — this is why the per-bit-position win worked:
/// text bytes have very different bit distributions per position.
///
/// `order2_byte` is the byte before `order1_byte`, providing two levels of
/// backward context for bank selection.
pub fn mixer_id(
    byte_class: ByteClass,
    bit_pos: u8,
    order1_byte: u8,
    order2_byte: u8,
    word_hash: usize,
) -> usize {
    let class_bits = (byte_class as usize) & 0x7;
    let bp = (bit_pos as usize) & 0x7;
    let o1 = (order1_byte as usize) & 0x7;
    let o2 = (order2_byte as usize) & 0x7;
    let wh = (word_hash as usize) & 0x3;
    ((class_bits << 9) | (bp << 6) | (o1 << 4) | (o2 << 2) | wh) & 0xFFF
}

/// Two-level mixer: 4096 bank mixers + a global mixer + a master mixer.
///
/// - `mixers`: context-specific bank selected by `mixer_id`. Only the selected
///   bank is trained per bit.
/// - `global_mixer`: context-agnostic fallback over the same models.
/// - `master_mixer`: blends `[p_bank, p_global, p_lzp_conf]` (3 inputs).
///
/// Encoder and decoder must call `mix`/`update`/`push_byte` with identical
/// inputs and in the same order. The `ByteAssembler` tracks the byte stream
/// for context hashing; it is deterministic from the coded bit stream, so
/// both sides stay in sync.
pub struct MixerBank {
    /// 4096 context-specific logistic mixers.
    mixers: Vec<LogisticMixer>,
    /// Global context-agnostic mixer (same n_models).
    global_mixer: LogisticMixer,
    /// Master mixer: blends bank + global + lzp_conf (3 inputs).
    master_mixer: LogisticMixer,
    n_models: usize,
    /// Tracks byte history for bank context selection.
    asm: ByteAssembler,
}

impl MixerBank {
    /// Create a two-level mixer hierarchy.
    ///
    /// `n_models` is the number of base bit-model probabilities fed into each
    /// bank and global mixer. The master mixer takes 3 inputs (bank, global,
    /// lzp_confidence).
    #[must_use]
    pub fn new(n_models: usize) -> Self {
        Self {
            mixers: (0..NUM_MIXERS)
                .map(|_| LogisticMixer::new(n_models))
                .collect(),
            global_mixer: LogisticMixer::new(n_models),
            master_mixer: LogisticMixer::new(3), // bank, global, lzp_conf
            n_models,
            asm: ByteAssembler::new(8),
        }
    }

    /// Set a per-model learning rate scale (index corresponds to model position).
    pub fn set_lr_scale(&mut self, idx: usize, scale: f32) {
        if idx < self.n_models {
            for m in &mut self.mixers {
                m.set_lr_scale(idx, scale);
            }
            self.global_mixer.set_lr_scale(idx, scale);
        }
    }

    /// Number of base models.
    #[must_use]
    pub fn n_models(&self) -> usize {
        self.n_models
    }

    /// Compute the bank ID for the current context.
    #[inline]
    fn current_bank_id(&self, bit_pos: u8) -> usize {
        let order1 = self.asm.prev_byte();
        let order2 = self.asm.last(2).first().copied().unwrap_or(0);
        let byte_class = if self.asm.has_byte() {
            ByteClass::classify(self.asm.last_byte())
        } else {
            ByteClass::Other
        };
        mixer_id(byte_class, bit_pos, order1, order2, self.asm.word_hash())
    }

    /// Mix the model probabilities and return a fused 12-bit P(bit==1).
    ///
    /// `probs` — one probability per base model (in `[1,4095]`).
    /// `bit_pos` — MSB-first bit position within the current byte (0..7).
    /// `lzp_conf` — the LZP model's current prediction (high-confidence match
    ///   signal), used as the third input to the master mixer.
    #[must_use]
    #[inline(always)]
    pub fn mix(&self, probs: &[u16], bit_pos: u8, lzp_conf: u16) -> u16 {
        let bank_id = self.current_bank_id(bit_pos);
        let p_bank = self.mixers[bank_id].mix(probs, bit_pos);
        let p_global = self.global_mixer.mix(probs, bit_pos);
        // Master blends bank + global + lzp confidence.
        self.master_mixer.mix(&[p_bank, p_global, lzp_conf], 0)
    }

    /// Train the selected bank mixer, the global mixer, and the master mixer.
    ///
    /// Only three mixers are touched per bit: the context-selected bank, the
    /// global, and the master. Not all 4096. This is the key performance
    /// property of the two-level hierarchy.
    #[inline]
    pub fn update(&mut self, probs: &[u16], bit: bool, bit_pos: u8, lzp_conf: u16) {
        let bank_id = self.current_bank_id(bit_pos);
        // Train bank and global with the full model stack.
        self.mixers[bank_id].update(probs, bit, bit_pos);
        self.global_mixer.update(probs, bit, bit_pos);
        // Master blends bank + global + lzp_conf.
        let p_bank = self.mixers[bank_id].mix(probs, bit_pos);
        let p_global = self.global_mixer.mix(probs, bit_pos);
        self.master_mixer
            .update(&[p_bank, p_global, lzp_conf], bit, 0);
    }

    /// Feed a completed byte so byte-history context advances.
    ///
    /// Called by the codec after each full byte is assembled (8 bits done).
    /// This updates the byte-level context used for bank selection on the
    /// *next* byte, mirroring how the model `ByteAssembler`s work.
    #[inline]
    pub fn push_byte(&mut self, byte: u8) {
        self.asm.push_byte(byte);
    }

    /// Decay all mixer weights toward init (1.0) by `factor`.
    ///
    /// Unlike `reset`, this preserves learned structure: weights shrink toward
    /// 1.0 but never hard-clear. The 4096 bank vectors and the master/global
    /// mixers all decay uniformly. Called at block boundaries.
    pub fn decay(&mut self, factor: f32) {
        for m in &mut self.mixers {
            m.decay(factor);
        }
        self.global_mixer.decay(factor);
        self.master_mixer.decay(factor);
    }

    /// Hard reset (kept for API compatibility, but `decay` is preferred).
    pub fn reset(&mut self) {
        for m in &mut self.mixers {
            m.reset();
        }
        self.global_mixer.reset();
        self.master_mixer.reset();
        self.asm.reset();
    }

    /// Approximate memory footprint in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.mixers.len() * std::mem::size_of::<LogisticMixer>()
            + 2 * std::mem::size_of::<LogisticMixer>()
    }
}

impl Default for MixerBank {
    fn default() -> Self {
        Self::new(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixer_id_in_range() {
        let id = mixer_id(ByteClass::Vowel, 3, b'a', b'b', 42);
        assert!(id < NUM_MIXERS, "mixer_id {} out of range", id);
    }

    #[test]
    fn mixer_id_deterministic() {
        let a = mixer_id(ByteClass::Space, 5, b'x', b'y', 99);
        let b = mixer_id(ByteClass::Space, 5, b'x', b'y', 99);
        assert_eq!(a, b);
    }

    #[test]
    fn bank_mix_returns_valid_prob() {
        let bank = MixerBank::new(2);
        let p = bank.mix(&[2000, 3000], 0, 2048);
        assert!((1..=4095).contains(&p), "probability out of range: {}", p);
    }

    #[test]
    fn bank_learns() {
        let mut bank = MixerBank::new(2);
        let probs = [2000u16, 3000u16];
        for _ in 0..200 {
            bank.update(&probs, true, 0, 2048);
        }
        let p = bank.mix(&probs, 0, 2048);
        // Master should have learned toward 1 after 200 ones.
        assert!(p > 2048, "master mixer should learn toward 1, got {}", p);
    }

    #[test]
    fn bank_decay_preserves_structure() {
        let mut bank = MixerBank::new(2);
        let probs = [2000u16, 3000u16];
        for _ in 0..200 {
            bank.update(&probs, true, 0, 2048);
        }
        let before = bank.mix(&probs, 0, 2048);
        bank.decay(0.99);
        let after = bank.mix(&probs, 0, 2048);
        // After decay, the prediction should still be biased (not reset to 2048).
        assert!(
            after > 2048,
            "decay should preserve learned structure: got {} (before {})",
            after,
            before
        );
    }

    #[test]
    fn bank_decay_zero_matches_reset() {
        let mut bank = MixerBank::new(2);
        let probs = [2000u16, 3000u16];
        for _ in 0..100 {
            bank.update(&probs, true, 0, 2048);
        }
        bank.decay(0.0);
        let p_decayed = bank.mix(&probs, 0, 2048);
        let fresh = MixerBank::new(2);
        let p_fresh = fresh.mix(&probs, 0, 2048);
        assert_eq!(
            p_decayed, p_fresh,
            "decay(0.0) should match fresh mixer: got {}, fresh {}",
            p_decayed, p_fresh
        );
    }

    #[test]
    fn push_byte_advances_context() {
        let mut bank = MixerBank::new(2);
        bank.push_byte(b'a');
        bank.push_byte(b'b');
        assert_eq!(bank.asm.last_byte(), b'b');
        assert_eq!(bank.asm.prev_byte(), b'a');
    }

    #[test]
    fn context_changes_bank_selection() {
        let bank = MixerBank::new(2);
        // Same bit_pos but different preceding byte → different bank_id.
        let id1 = bank.current_bank_id(0);
        let mut bank2 = MixerBank::new(2);
        bank2.push_byte(b'x');
        let id2 = bank2.current_bank_id(0);
        assert_ne!(
            id1, id2,
            "different byte context should select different bank"
        );
    }

    #[test]
    fn memory_under_2mb() {
        let bank = MixerBank::new(8);
        let bytes = bank.memory_bytes();
        assert!(
            bytes < 2_000_000,
            "MixerBank with 8 models should be <2 MB: got {} bytes",
            bytes
        );
    }
}
