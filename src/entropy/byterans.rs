//! 256-symbol rANS entropy coder for the byte-level codec path.
//!
//! The bit-level path codes each bit against a 2-symbol table (see
//! [`super::range`]). The byte path codes a whole byte against a 256-symbol
//! distribution produced by the byte mixer. One rANS symbol per byte — 8× fewer
//! arithmetic-coding steps than the bit path.
//!
//! Implementation follows the canonical rANS scheme (Fabian Giesen, *Arithmetic
//! Coding Revisited*): a 32-bit state `x`, 12-bit frequency scale (`SCALE =
//! 4096`), and byte-wise renormalization. The encoder feeds symbols in reverse
//! order and flushes the final state as a 4-byte LE tail; the decoder reads
//! forward from the tail.
//!
//! Frequencies: the caller hands a `[u16; 256]` distribution summing to `SCALE`
//! (each symbol ≥ 1). Cumulative frequencies are computed on the fly — the same
//! deterministic table on both sides.

/// Total frequency mass (12-bit). Must be a power of two for the `x & (SCALE-1)`
/// symbol search in the decoder.
pub const BYTE_SCALE: u32 = 1 << 12;
/// Lower bound for the decoder state: state is kept in `[BYTE_L, 2^32)`.
const BYTE_L: u32 = 1 << 23;

/// Byte-level rANS encoder. Buffers symbols; [`finish`](Self::finish) encodes in
/// reverse (the rANS convention) and returns the byte stream.
pub struct RansByteEncoder {
    state: u32,
    pending: Vec<u8>, // renormalized low bytes, in reverse order already
    // Per-symbol (freq, cum) records — 8 bytes instead of the full 256-way
    // distribution the caller handed us. Compressed forward but decoded
    // reverse; the compact record is all the reverse pass needs.
    syms: Vec<(u32, u32)>,
}

impl Default for RansByteEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RansByteEncoder {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: BYTE_L,
            pending: Vec::new(),
            syms: Vec::new(),
        }
    }

    /// Encode `symbol` against distribution `probs` (sum = [`BYTE_SCALE`]).
    ///
    /// rANS encodes *reverse* order: the last symbol encoded is decoded first.
    /// To keep the caller's natural forward order, symbols are buffered here and
    /// emitted in reverse by [`finish`](Self::finish).
    pub fn encode_symbol(&mut self, symbol: u8, probs: [u16; 256]) {
        let freq = u32::from(probs[usize::from(symbol)]);
        let cum = probs[..usize::from(symbol)]
            .iter()
            .map(|&p| u32::from(p))
            .sum::<u32>();
        self.syms.push((freq, cum));
    }

    /// Encode one symbol whose `freq`/`cum` are already known (computed in the
    /// caller's distribution-build pass).
    ///
    /// rANS encodes *reverse* order: the last symbol encoded is decoded first.
    /// To keep the caller's natural forward order, symbols are buffered here and
    /// emitted in reverse by [`finish`](Self::finish).
    pub fn encode_fc(&mut self, freq: u32, cum: u32) {
        self.syms.push((freq, cum));
    }

    /// Flush and return the encoded byte stream.
    ///
    /// rANS writes the stream as `[final state (4B)][renorm bytes, in reverse
    /// encode order]` matching Fabian Giesen's `rans_byte.h` layout: the
    /// encoder works backwards through a buffer, so reading forward yields the
    /// final state first.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        // Feed symbols in reverse order (rANS stack order).
        for &(freq, cum) in self.syms.iter().rev() {
            // Renorm: keep the state below ((L >> 12) << 8) * freq so the encode
            // step below stays within 32 bits.
            let x_max = ((BYTE_L >> 12) << 8) * freq;
            while self.state >= x_max {
                self.pending.push((self.state & 0xFF) as u8);
                self.state >>= 8;
            }
            self.state = (self.state / freq) * BYTE_SCALE + cum + (self.state % freq);
        }
        // Final stream: state tail first, then the renorm bytes reversed
        // (reversing flips both per-symbol and across-symbol order).
        let mut out = Vec::with_capacity(4 + self.pending.len());
        out.extend_from_slice(&self.state.to_le_bytes());
        out.extend(self.pending.iter().rev());
        out
    }
}

/// Byte-level rANS decoder over a stream from [`RansByteEncoder::finish`].
pub struct RansByteDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    state: u32,
}

impl<'a> RansByteDecoder<'a> {
    /// Create a decoder. The stream must be at least 4 bytes (the state tail).
    ///
    /// # Errors
    ///
    /// Returns `()` if the stream is too short to hold the 4-byte final state.
    pub fn new(data: &'a [u8]) -> Result<Self, ()> {
        if data.len() < 4 {
            return Err(());
        }
        let state = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        Ok(Self {
            data,
            pos: 4,
            state,
        })
    }

    /// Decode one symbol against `probs` (must match the encoder's distribution).
    ///
    /// Mirrors `RansDecAdvance`: decode first, then renormalize — the inverse
    /// of the encoder, which renormalizes *before* encoding.
    #[must_use]
    pub fn decode_symbol(&mut self, probs: &[u16; 256]) -> u8 {
        let (sym, freq, cum) = {
            let cdf = self.state & (BYTE_SCALE - 1);
            find_symbol(probs, cdf)
        };
        self.advance(freq, cum);
        sym
    }

    /// Return the current rANS `state & (BYTE_SCALE - 1)` (the 12-bit
    /// cumulative symbol index embedded in the state).
    #[inline]
    pub fn cdf(&self) -> u32 {
        self.state & (BYTE_SCALE - 1)
    }

    /// Advance the rANS state for a symbol whose `freq`/`cum` are already known
    /// (the caller found them in its distribution-build pass): decode first,
    /// then renormalize — the inverse of the encoder, which renormalizes
    /// *before* encoding.
    pub fn advance(&mut self, freq: u32, cum: u32) {
        let cdf = self.state & (BYTE_SCALE - 1);
        // Inverse of encode: state_out = (state_in/freq)*SCALE + cum + state_in%freq.
        // With state_out = (state_out>>12)*SCALE + cdf, D = freq*(state>>12) + cdf - cum.
        self.state = freq * (self.state >> 12) + (cdf - cum);
        while self.state < BYTE_L {
            let byte = self.data.get(self.pos).copied().unwrap_or(0);
            self.pos += 1;
            self.state = (self.state << 8) | u32::from(byte);
        }
    }
}

/// Find the symbol whose cumulative range `[cum, cum+freq)` contains `cdf`, and
/// return `(symbol, freq_of_symbol, cum_before_symbol)`. The distribution sums
/// to [`BYTE_SCALE`] with every symbol ≥ 1, so the linear scan terminates
/// correctly.
fn find_symbol(probs: &[u16; 256], cdf: u32) -> (u8, u32, u32) {
    let mut running = 0u32;
    for (s, &p) in probs.iter().enumerate() {
        running += u32::from(p);
        if running > cdf {
            return (s as u8, u32::from(p), running - u32::from(p));
        }
    }
    (255, u32::from(probs[255]), running - u32::from(probs[255]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(symbols: &[u8], probs: impl Fn(usize) -> [u16; 256]) {
        let mut enc = RansByteEncoder::new();
        for (i, &s) in symbols.iter().enumerate() {
            enc.encode_symbol(s, probs(i));
        }
        let buf = enc.finish();
        let mut dec = RansByteDecoder::new(&buf).expect("4-byte tail");
        for (i, &s) in symbols.iter().enumerate() {
            let got = dec.decode_symbol(&probs(i));
            assert_eq!(got, s, "mismatch at index {i}");
        }
    }

    fn uniform() -> [u16; 256] {
        // 4096/256 = 16 each.
        [16u16; 256]
    }

    fn biased(byte: u8) -> [u16; 256] {
        let mut p = [1u16; 256];
        p[usize::from(byte)] = 4096 - 255;
        p
    }

    #[test]
    fn roundtrip_uniform() {
        let data: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        roundtrip(&data, |_| uniform());
    }

    #[test]
    fn roundtrip_biased() {
        let data: Vec<u8> = vec![0x41; 64];
        roundtrip(&data, |_| biased(0x41));
    }

    #[test]
    fn roundtrip_mixed() {
        let data: Vec<u8> = (0u8..=255).collect();
        let variants: Vec<[u16; 256]> = (0..8).map(|k| biased(k * 7)).collect();
        roundtrip(&data, |i| variants[i % variants.len()]);
    }

    #[test]
    fn decoder_rejects_short_stream() {
        assert!(RansByteDecoder::new(&[1, 2, 3]).is_err());
        assert!(RansByteDecoder::new(&[1, 2, 3, 4]).is_ok());
    }

    #[test]
    fn scale_is_power_of_two() {
        assert_eq!(BYTE_SCALE, 4096);
        assert!(BYTE_SCALE.is_power_of_two());
    }

    // Minimal deterministic PRNG so the fuzz test needs no dependencies.
    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    fn random_dist(rng: &mut XorShift) -> [u16; 256] {
        // Every symbol ≥ 1; then scatter the remaining SCALE-256 mass randomly.
        let mut p = [1u16; 256];
        for _ in 256..BYTE_SCALE {
            let s = (rng.next() % 256) as usize;
            p[s] += 1;
        }
        let sum: u64 = p.iter().map(|&w| u64::from(w)).sum();
        debug_assert_eq!(sum, BYTE_SCALE as u64, "distribution must sum to SCALE");
        p
    }

    #[test]
    fn roundtrip_fuzz() {
        let mut rng = XorShift(0x9E3779B97F4A7C15);
        for trial in 0..8 {
            let n = 16 + (rng.next() % 400) as usize;
            let data: Vec<u8> = (0..n).map(|_| (rng.next() % 256) as u8).collect();
            let dists: Vec<[u16; 256]> = (0..n).map(|_| random_dist(&mut rng)).collect();
            let mut enc = RansByteEncoder::new();
            for (i, &s) in data.iter().enumerate() {
                enc.encode_symbol(s, dists[i]);
            }
            let buf = enc.finish();
            let mut dec = RansByteDecoder::new(&buf).expect("4-byte state");
            for (i, &s) in data.iter().enumerate() {
                let got = dec.decode_symbol(&dists[i]);
                assert_eq!(got, s, "trial {trial} mismatch at {i}");
            }
        }
    }
}