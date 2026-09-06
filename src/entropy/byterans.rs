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

// ---------------------------------------------------------------------------
// 32-way interleaved byte rANS
// ---------------------------------------------------------------------------
//
// The single-stream coder above chains every symbol through one state: decode
// of byte _n_ cannot start until byte _n-1_ renormed. Splitting the block into
// 32 independent lane states breaks that dependency chain: a lane's state only
// updates once every 32 symbols, so the 32 state paths run independently
// (category-vectorizable cdf extraction + independent renorm byte fetches).
//
//   Lane assignment:   position p → lane p & 31, group p >> 5
//   Decode order:      group 0 lane 0..31, group 1 lane 0..31, ... (forward
//                      byte order — required so the caller's byte model context
//                      advances forward).
//   Encode order:      the exact reverse, per lane (groups descending, lanes
//                      descending within a group).
//
// Stream layout (all LE):
//   [total_symbols: u32]
//   [32 × lane state: u32]      (post-encode state of each lane, = decoder's
//                                initial state before group 0)
//   [32 × lane run_len: u32]    (renorm bytes for that lane)
//   [lane 0 run][lane 1 run]...
//
// A lane's run, read forward, holds exactly that lane's renorm bytes in decode
// (group-ascending) order — the per-lane mirror of the single-stream layout.

/// Number of interleaved rANS lanes. A power of two keeps `p & (LANES-1)` cheap.
pub const RANS32_LANES: usize = 32;

/// 32-way interleaved byte rANS encoder.
pub struct RansByteEncoder32 {
    states: [u32; RANS32_LANES],
    pending: [Vec<u8>; RANS32_LANES],
    /// (freq, cum) per symbol, in input byte order.
    syms: Vec<(u32, u32)>,
}

impl Default for RansByteEncoder32 {
    fn default() -> Self {
        Self::new()
    }
}

impl RansByteEncoder32 {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            states: [BYTE_L; RANS32_LANES],
            pending: [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
                      Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
                      Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
                      Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
                      Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
                      Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
                      Vec::new(), Vec::new()],
            syms: Vec::new(),
        }
    }

    /// Buffer one symbol's `(freq, cum)`. The caller computes the distribution.
    pub fn encode_fc(&mut self, freq: u32, cum: u32) {
        self.syms.push((freq, cum));
    }

    /// Flush and return the encoded byte stream (see module docs for layout).
    ///
    /// Symbols are encoded in the reverse of decode order: groups descending,
    /// lanes descending within a group — the per-lane inverse of the decoder's
    /// forward (group, lane) walk.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        let total = self.syms.len();
        for p in (0..total).rev() {
            let (freq, cum) = self.syms[p];
            let lane = p & (RANS32_LANES - 1);
            let st = &mut self.states[lane];
            // Renorm before encode (same bound as the single-stream coder so
            // the encode step below stays within 32 bits).
            let x_max = ((BYTE_L >> 12) << 8) * freq;
            while *st >= x_max {
                self.pending[lane].push((*st & 0xFF) as u8);
                *st >>= 8;
            }
            *st = (*st / freq) * BYTE_SCALE + cum + (*st % freq);
        }

        // Header: total symbols, 32 states, 32 run lens.
        let mut out = Vec::with_capacity(4 + 128 + 128 + total + 512);
        out.extend_from_slice(&(total as u32).to_le_bytes());
        for st in &self.states {
            out.extend_from_slice(&st.to_le_bytes());
        }
        for pl in &self.pending {
            out.extend_from_slice(&(pl.len() as u32).to_le_bytes());
        }
        for pl in &self.pending {
            out.extend(pl.iter().rev());
        }
        out
    }
}

/// 32-way interleaved byte rANS decoder.
pub struct RansByteDecoder32<'a> {
    data: &'a [u8],
    states: [u32; RANS32_LANES],
    /// Absolute offset of each lane's run within `data`.
    run_base: [u32; RANS32_LANES],
    /// Remaining renorm bytes in each lane's run.
    run_rem: [u32; RANS32_LANES],
    /// Total symbols to decode.
    total: usize,
    /// Position of the next symbol to decode.
    next: usize,
}

impl<'a> RansByteDecoder32<'a> {
    /// Create a decoder. `total` is the number of symbols the payload encodes
    /// (the caller knows it from the container).
    ///
    /// # Errors
    ///
    /// Returns `()` if the stream can't hold the header (4 + 128 + 128 bytes).
    pub fn new(data: &'a [u8], total: usize) -> Result<Self, ()> {
        const HDR: usize = 4 + RANS32_LANES * 4 * 2;
        if data.len() < HDR {
            return Err(());
        }
        let stored = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if stored != total {
            return Err(());
        }
        let mut states = [0u32; RANS32_LANES];
        for (i, st) in states.iter_mut().enumerate() {
            let o = 4 + i * 4;
            *st = u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
        }
        let mut run_len = [0u32; RANS32_LANES];
        for (i, rl) in run_len.iter_mut().enumerate() {
            let o = 4 + RANS32_LANES * 4 + i * 4;
            *rl = u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
        }
        let mut run_base = [0u32; RANS32_LANES];
        let mut base = (4 + RANS32_LANES * 4 * 2) as u32;
        for i in 0..RANS32_LANES {
            run_base[i] = base;
            base += run_len[i];
        }
        if (base as usize) > data.len() {
            return Err(());
        }
        Ok(Self {
            data,
            states,
            run_base,
            run_rem: run_len,
            total,
            next: 0,
        })
    }

    /// Number of symbols remaining.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.total - self.next
    }

    /// Current `state & (SCALE-1)` for every lane — the 12-bit cumulative
    /// symbol index of the *next* symbol each lane will decode.
    ///
    /// Callers decoding a group should take this array once, then use the
    /// entries in lane order (`lane` = within-group position `& 31`). The 32
    /// state reads are independent — the interleaving win.
    #[inline]
    pub fn cdf_batch(&self) -> [u32; RANS32_LANES] {
        let mut c = [0u32; RANS32_LANES];
        for (i, st) in self.states.iter().enumerate() {
            c[i] = *st & (BYTE_SCALE - 1);
        }
        c
    }

    /// Advance lane `lane`'s state for a symbol whose `freq`/`cum` are already
    /// known: inverse encode step, then renorm from the lane's own run.
    #[inline]
    pub fn lane_advance(&mut self, lane: usize, freq: u32, cum: u32) {
        self.next += 1;
        let st = &mut self.states[lane];
        let cdf = *st & (BYTE_SCALE - 1);
        *st = freq * (*st >> 12) + (cdf - cum);
        if *st < BYTE_L {
            // Renorm reads bytes from this lane's run at its own cursor.
            let mut base = self.run_base[lane] as usize;
            let rem = &mut self.run_rem[lane];
            while *st < BYTE_L {
                let byte = if *rem > 0 {
                    let b = self.data.get(base).copied().unwrap_or(0);
                    base += 1;
                    *rem -= 1;
                    b
                } else {
                    0
                };
                *st = (*st << 8) | u32::from(byte);
            }
            self.run_base[lane] = base as u32;
        }
    }
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

    fn roundtrip32(symbols: &[u8], probs: impl Fn(usize) -> [u16; 256]) {
        let mut enc = RansByteEncoder32::new();
        for (i, &s) in symbols.iter().enumerate() {
            let p = probs(i);
            let freq = u32::from(p[usize::from(s)]);
            let cum: u32 = p[..usize::from(s)].iter().map(|&x| u32::from(x)).sum();
            enc.encode_fc(freq, cum);
        }
        let buf = enc.finish();
        let mut dec = RansByteDecoder32::new(&buf, symbols.len()).expect("header");
        let mut next = 0usize;
        while dec.remaining() > 0 {
            let cdfs = dec.cdf_batch();
            let in_group = dec.remaining().min(RANS32_LANES);
            for lane in 0..in_group {
                let p = probs(next);
                let (sym, freq, cum) = find_symbol(&p, cdfs[lane]);
                assert_eq!(sym, symbols[next], "32-way mismatch at {next}");
                dec.lane_advance(lane, freq, cum);
                next += 1;
            }
        }
    }

    #[test]
    fn roundtrip32_uniform() {
        let data: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        roundtrip32(&data, |_| uniform());
    }

    #[test]
    fn roundtrip32_biased() {
        let data: Vec<u8> = vec![0x41; 64];
        roundtrip32(&data, |_| biased(0x41));
    }

    #[test]
    fn roundtrip32_mixed() {
        let data: Vec<u8> = (0u8..=255).collect();
        let variants: Vec<[u16; 256]> = (0..8).map(|k| biased(k * 7)).collect();
        roundtrip32(&data, |i| variants[i % variants.len()]);
    }

    #[test]
    fn roundtrip32_fuzz() {
        let mut rng = XorShift(0x243F6A8885A308D3);
        for trial in 0..8 {
            let n = 16 + (rng.next() % 700) as usize;
            let data: Vec<u8> = (0..n).map(|_| (rng.next() % 256) as u8).collect();
            let dists: Vec<[u16; 256]> = (0..n).map(|_| random_dist(&mut rng)).collect();
            let mut enc = RansByteEncoder32::new();
            for (i, &s) in data.iter().enumerate() {
                let p = dists[i];
                enc.encode_fc(u32::from(p[usize::from(s)]), {
                    p[..usize::from(s)].iter().map(|&x| u32::from(x)).sum()
                });
            }
            let buf = enc.finish();
            let mut dec = RansByteDecoder32::new(&buf, n).expect("header");
            let mut next = 0usize;
            while dec.remaining() > 0 {
                let cdfs = dec.cdf_batch();
                let in_group = dec.remaining().min(RANS32_LANES);
                for lane in 0..in_group {
                    let p = dists[next];
                    let (sym, freq, cum) = find_symbol(&p, cdfs[lane]);
                    assert_eq!(sym, data[next], "trial {trial} 32-way mismatch at {next}");
                    dec.lane_advance(lane, freq, cum);
                    next += 1;
                }
            }
        }
    }

    #[test]
    fn roundtrip32_vs_single_stream_output_size_only() {
        // Sanity: 32-way payload ≈ single-stream payload + 256-byte header.
        let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let mut enc = RansByteEncoder::new();
        for &s in &data {
            enc.encode_symbol(s, uniform());
        }
        let buf = enc.finish();
        let mut enc32 = RansByteEncoder32::new();
        for &s in &data {
            let p = uniform();
            enc32.encode_fc(
                u32::from(p[usize::from(s)]),
                p[..usize::from(s)].iter().map(|&x| u32::from(x)).sum(),
            );
        }
        let buf32 = enc32.finish();
        // 4-byte state vs 4+128+128 header.
        assert!(buf32.len() <= buf.len() + 256, "buf32 {} vs buf {}", buf32.len(), buf.len());
    }
}