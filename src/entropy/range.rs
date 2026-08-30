//! Entropy backend for `nyx`: a **bit coder** built on the audited [`ans`] rANS crate.
//!
//! `nyx` models data bit-by-bit (the logistic mixer emits a probability per bit), so
//! this module exposes a stream of bits where each bit is entropy-coded against a 2-symbol
//! rANS frequency table derived from the predicted probability. Using `ans` for the
//! primitive keeps the entropy stage correct and battle-tested; the novel work lives in
//! the models + mixer + classifier, not in re-deriving rANS.
//!
//! Wire format (per `ans::RansEncoder`/`RansDecoder`): encoder feeds symbols in
//! **reverse** order, decoder reads forward, final state is the 4-byte LE tail.

use ans::{AnsError, FrequencyTable, RansDecoder, RansEncoder};

const PRECISION: u32 = 12; // total = 1<<12 = 4096 frequency mass, 12-bit probabilities
const TOTAL: u32 = 1 << PRECISION;

/// Build a 2-symbol frequency table for a predicted P(bit==1) in `[1, TOTAL-1]`.
///
/// Symbol 1 (bit set) gets frequency `p`; symbol 0 gets the complement `TOTAL - p`.
#[inline]
#[must_use]
fn bit_table(p1: u16) -> FrequencyTable {
    let max = u16::try_from(TOTAL - 1).expect("TOTAL-1 fits in u16");
    let p1 = u32::from(p1.clamp(1, max));
    let p0 = TOTAL - p1;
    // `from_counts` normalizes to sum == TOTAL exactly; a valid 2-symbol table
    // cannot fail, so unwrap is safe.
    FrequencyTable::from_counts(&[p0, p1], PRECISION).expect("bit table always valid")
}

/// Bit-level arithmetic encoder over rANS.
pub struct BitEncoder {
    pending: Vec<(bool, u16)>, // (bit, predicted P(bit==1)); fed in reverse on finish()
}

impl BitEncoder {
    /// Create a new bit encoder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Buffer one bit for encoding.
    ///
    /// `p` is the predicted probability of bit==1, in `[1, 4095]` (12-bit). Bits are
    /// fed to the underlying rANS encoder in reverse order, so they are buffered here
    /// and flushed by [`finish`](Self::finish).
    pub fn encode_bit(&mut self, bit: bool, p: u16) {
        self.pending.push((bit, p));
    }

    /// Finalize, returning the entropy-coded byte stream.
    ///
    /// # Panics
    ///
    /// Panics only if the underlying rANS primitive fails on a valid 2-symbol table,
    /// which cannot happen for the tables constructed here.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        let mut enc = RansEncoder::new();
        // Feed in reverse: the last buffered bit is encoded first.
        for &(bit, p) in self.pending.iter().rev() {
            let table = bit_table(p);
            enc.put(u32::from(bit), &table)
                .expect("rANS put cannot fail for a valid 2-symbol table");
        }
        enc.finish()
    }
}

impl Default for BitEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Bit-level decoder (forward order) over rANS.
pub struct BitDecoder<'a> {
    dec: RansDecoder<'a>,
}

impl<'a> BitDecoder<'a> {
    /// Create a decoder over `data`, the byte stream from [`BitEncoder::finish`].
    ///
    /// # Errors
    ///
    /// Returns [`AnsError`] if the stream is too short (< 4 bytes) or its trailing
    /// final-state word is below the rANS lower bound.
    pub fn new(data: &'a [u8]) -> Result<Self, AnsError> {
        Ok(BitDecoder {
            dec: RansDecoder::new(data)?,
        })
    }

    /// Decode one bit given the predicted P(bit==1) `p` in `[1, 4095]`.
    ///
    /// Must be called in the same forward order as encoding.
    ///
    /// # Errors
    ///
    /// Returns [`AnsError`] on a corrupt or truncated stream.
    pub fn decode_bit(&mut self, p: u16) -> Result<bool, AnsError> {
        let table = bit_table(p);
        let sym = self.dec.get(&table)?; // 0 or 1
        Ok(sym == 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_bits() {
        // Sequence of (bit, p) pairs. p is the predicted prob of bit==1.
        let pairs: Vec<(bool, u16)> = (0..256)
            .map(|i| (i % 3 == 0, if i % 3 == 0 { 3000 } else { 1095 }))
            .collect();
        let mut e = BitEncoder::new();
        for &(b, p) in &pairs {
            e.encode_bit(b, p);
        }
        let buf = e.finish();
        let mut d = BitDecoder::new(&buf).expect("decode init");
        for &(b, p) in &pairs {
            assert_eq!(d.decode_bit(p).expect("decode bit"), b, "bit mismatch");
        }
    }
}
