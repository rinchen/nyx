//! Context-selected mixer bank.
//!
//! Top compressors (PAQ8, CMIX) use 4k+ mixer instances selected by a hash of
//! higher-order context. Nyx previously had one global mixer with per-bit-position
//! deltas. This module adds 4096 mixer instances, each with its own weight vector,
//! selected by a context hash of order-1/order-2/word context + bit position.
//!
//! Memory: 4096 mixers × 8 models × (1 base weight + 8 bit-position deltas) × 4 bytes
//! ≈ 1.1 MB. Shared stretch/squash tables live in [`super::LogisticMixer`].

use crate::model::mixer::LogisticMixer;

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

/// A bank of 4096 independent logistic mixers, selected by context hash.
///
/// Each mixer in the bank is a full `LogisticMixer` with its own weight vector
/// and per-bit-position deltas. The `mix()` call selects the mixer by ID and
/// delegates to it; `update()` trains the same mixer. Both encoder and decoder
/// must use the same selection function and see the same byte history.
pub struct MixerBank {
    mixers: Vec<LogisticMixer>,
    n_models: usize,
}

impl MixerBank {
    /// Create a bank of `num_mixers` instances, each for `n_models` base models.
    #[must_use]
    pub fn new(n_models: usize, num_mixers: usize) -> Self {
        let mixers = (0..num_mixers)
            .map(|_| LogisticMixer::new(n_models))
            .collect();
        Self { mixers, n_models }
    }

    /// Compute the mixer ID from context.
    ///
    /// Uses a 12-bit hash: `(byte_class << 3) | bit_pos` for the basic variant,
    /// or the extended hash below when order-1/order-2/word context is available.
    #[must_use]
    pub fn mixer_id(
        byte_class: ByteClass,
        bit_pos: u8,
        order1_byte: u8,
        order2_byte: u8,
        word_hash: usize,
    ) -> usize {
        // 12-bit selection:
        //   bits 11..9 = byte class (0..5, 3 bits)
        //   bits 8..6  = bit position (0..7, 3 bits)
        //   bits 5..3  = order1 low 3 bits (3 bits)
        //   bits 2..0  = word_hash low 3 bits (3 bits)
        let class_bits = (byte_class as usize) & 0x7;
        let bp = (bit_pos as usize) & 0x7;
        let o1 = (order1_byte as usize) & 0x7;
        let wh = (word_hash as usize) & 0x7;
        (class_bits << 9) | (bp << 6) | (o1 << 3) | wh
    }

    /// Mix `probs` using the mixer selected by `mixer_id`.
    /// Returns a 12-bit probability in `[1, 4095]`.
    #[must_use]
    #[inline(always)]
    pub fn mix(&self, probs: &[u16], mixer_id: usize) -> u16 {
        let mixer = &self.mixers[mixer_id];
        mixer.mix(probs, 0) // bit_pos is already folded into mixer_id
    }

    /// Update the mixer selected by `mixer_id` with the observed `bit`.
    #[inline]
    pub fn update(&mut self, probs: &[u16], bit: bool, mixer_id: usize) {
        let mixer = &mut self.mixers[mixer_id];
        mixer.update(probs, bit, 0);
    }

    /// Reset all mixers in the bank to initial state.
    pub fn reset(&mut self) {
        for mixer in &mut self.mixers {
            mixer.reset();
        }
    }

    /// Approximate memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.mixers.len() * std::mem::size_of::<LogisticMixer>()
    }
}

impl Default for MixerBank {
    fn default() -> Self {
        Self::new(8, NUM_MIXERS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixer_id_in_range() {
        let id = MixerBank::mixer_id(ByteClass::Vowel, 3, b'a', b'b', 42);
        assert!(id < NUM_MIXERS, "mixer_id {} out of range", id);
    }

    #[test]
    fn mixer_id_deterministic() {
        let a = MixerBank::mixer_id(ByteClass::Space, 5, b'x', b'y', 99);
        let b = MixerBank::mixer_id(ByteClass::Space, 5, b'x', b'y', 99);
        assert_eq!(a, b);
    }

    #[test]
    fn bank_mix_returns_valid_prob() {
        let bank = MixerBank::new(2, NUM_MIXERS);
        let p = bank.mix(&[2000, 3000], 42);
        assert!((1..=4095).contains(&p), "probability out of range: {}", p);
    }

    #[test]
    fn bank_learns() {
        let mut bank = MixerBank::new(2, NUM_MIXERS);
        let probs = [2000u16, 3000u16];
        for _ in 0..100 {
            bank.update(&probs, true, 1);
        }
        let p = bank.mix(&probs, 1);
        // Mixer 1 should have learned toward 1 after 100 ones.
        assert!(p > 2048, "bank mixer should learn toward 1, got {}", p);
    }

    #[test]
    fn bank_reset_clears_weights() {
        let mut bank = MixerBank::new(2, NUM_MIXERS);
        let probs = [2000u16, 3000u16];
        for _ in 0..100 {
            bank.update(&probs, true, 1);
        }
        bank.reset();
        let p = bank.mix(&probs, 1);
        // After reset, weights are neutral again.
        assert!(
            (p as i32 - 2048i32).abs() < 100,
            "post-reset prob should be near-neutral: {}",
            p
        );
    }
}
