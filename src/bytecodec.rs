//! Byte-level PPM-style entropy codec (the "fast" path).
//!
//! The bit path (see [`crate::codec`]) runs eight model-predict + mixer-mix +
//! eight model-update passes *per bit* (~16M bit-steps per 2MB block, with
//! text-trained models that are poorly calibrated on transformed streams).
//! This module is a true byte-level coder:
//!
//!   per byte = deterministic context selection → one 256-way distribution
//!              built from the winning count row → one rANS symbol.
//!
//! No `[u16; 256]` distribution array is materialized.  The 256-symbol
//! distribution walk is fused directly with the encode cumulative capture
//! (encode) or the decode cumulative search (decode), so both sides are
//! single-pass ALU-only per byte:
//!
//!   per byte = order selection (2 integer comparisons) + 256× (load + 2 mul +
//!              2 shift + fixed-point cumulative rounding + add) + 1 division
//!              (rANS state step) + model update (~20 ops).
//!
//! Contexts:
//!
//! * **order-0** (1 context), **order-1** (256 contexts), **order-2** (hashed
//!   4 096 / 1 024 contexts) — `u16` count tables with row-halving when any
//!   count exceeds 256 (keeps row totals ≤ 65 536).
//! * A **deterministic single-context selector** (no escapes, no softmax
//!   mixing): pick the order whose context is best concentrated, so both
//!   encode and decode agree on the distribution without any decoding of the
//!   symbol itself. Selection uses only context statistics (`total` and
//!   `distinct`), maintained incrementally.
//!
//! The encoder and decoder run byte-identical deterministic code, so the
//! distribution fed to [`RansByteEncoder::encode_fc`] on the encode side
//! equals the one the decoder walks in its cumulative search on the decode side.
//!
//! The block framing is handled by the caller ([`crate::codec`]): this module
//! only produces/consumes the rANS byte stream for one block. Uncompressed
//! length travels in the container `BlockEntry`.

use crate::entropy::byterans::{RansByteDecoder, RansByteEncoder, BYTE_SCALE};
use crate::error::{NyxError, Result};

/// Hashed order-2 contexts. 4 096 rows × 256 × u16 = 2 MB.
const ORDER2_CTX: usize = 1 << 12;
const ORDER2_MASK: usize = ORDER2_CTX - 1;
/// Hashed order-2 contexts for small (binary/exec) blocks.
const ORDER2_CTX_SMALL: usize = 1 << 10;

/// Halve a row when any count exceeds this → row total ≤ 256 · 256 = 65 536.
const COUNT_CAP: u16 = 256;
const MAX_TOTAL: usize = COUNT_CAP as usize * 256;

/// Minimum distinct-probability mass before an order is eligible, so fresh
/// contexts don't win by accident. order-1: 2 bytes seen, order-2: 4 bytes.
const MIN_T1: u32 = 2;
const MIN_T2: u32 = 4;

/// Distribution scale of the normalized count row (movable mass, before the
/// base-1 per symbol).
const BUDGET: u32 = BYTE_SCALE - 256;

/// Fixed-point fractional bits for cumulative rounding (guarantees the 256-symbol
/// sum = SCALE-1 or SCALE exactly, at most one unit of deficit patched to
/// symbol 255). `idx` is a 12-bit probability, so `exact = idx·BUDGET/4096`
/// accumulates directly as `idx·BUDGET` fixed-point at 2^12.
const FRAC_BITS: u32 = 12;
const FRAC_ROUND: u64 = 1 << (FRAC_BITS - 1);

/// `floor((1 << 32) / t)` for `t` in 1..=MAX_TOTAL.
fn reciprocal_table() -> &'static [u32; MAX_TOTAL + 1] {
    static T: std::sync::OnceLock<[u32; MAX_TOTAL + 1]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0u32; MAX_TOTAL + 1];
        for (i, v) in t.iter_mut().enumerate().skip(1) {
            *v = ((1u64 << 32) / i as u64) as u32;
        }
        t
    })
}

fn hash2(a: u8, b: u8) -> usize {
    let x = u32::from(a) | (u32::from(b) << 8);
    (x.wrapping_mul(0x9E37_79B1) >> 20) as usize
}

/// One count-table byte model. Row = 256 `u16` counts.
///
/// `totals`/`distinct` are maintained incrementally and recomputed only on
/// row-halving, so [`ByteCountModel::total`] and [`ByteCountModel::distinct`]
/// are O(1) for the context selector.
struct ByteCountModel {
    counts: Vec<u16>,
    num_ctx: usize,
    totals: Vec<u32>,
    distinct: Vec<u32>,
}

impl ByteCountModel {
    fn new(num_ctx: usize) -> Self {
        Self {
            counts: vec![0; num_ctx * 256],
            num_ctx,
            totals: vec![0; num_ctx],
            distinct: vec![0; num_ctx],
        }
    }

    #[inline]
    fn total(&self, ctx: usize) -> u32 {
        self.totals[ctx]
    }

    #[inline]
    fn distinct(&self, ctx: usize) -> u32 {
        self.distinct[ctx]
    }

    /// Walk the count row for `ctx` and, in a single pass:
    ///
    /// - **encode** (`target = Some(byte)`): capture `(freq, cum)` for that byte.
    /// - **decode** (`cdf = Some(state_cdf)`): find the byte whose cumulative
    ///   range contains `cdf`, and return it.
    ///
    /// The walk builds the distribution on-the-fly using cumulative fixed-point
    /// rounding (no sort, no extra materialisation). The sum is at most SCALE;
    /// symbol 255 absorbs any leftover mass so encode and decode agree on
    /// range [cum_before_255, SCALE) for it.
    #[inline(always)]
    fn walk_dist(
        &self,
        ctx: usize,
        target: Option<u8>,
        cdf: Option<u32>,
    ) -> WalkResult {
        let base = ctx * 256;
        let t = u64::from(self.totals[ctx].max(1));
        let inv = u64::from(reciprocal_table()[t as usize]);

        let mut acc = 0u64;
        let mut prev = 0u64;
        let mut cum = 0u32;
        let mut cum_before_255 = 0u32;
        let mut f255 = 0u32;
        let mut found_sym = -1i32;
        let mut found_freq = 0u32;
        let mut found_cum = 0u32;

        for s in 0..256 {
            let cnt = u64::from(self.counts[base + s]);
            let idx = (cnt * inv) >> 20;
            // exact = idx·BUDGET/4096, accumulated in FRAC_BITS-bit fixed point.
            acc += idx * u64::from(BUDGET);
            // base_s = R(s) - R(s-1) where R accumulates rounded exacts; the
            // delta is always ≥ 0 and the total ≤ BUDGET.
            let r = (acc + FRAC_ROUND) >> FRAC_BITS;
            let base_s = r - prev;
            prev = r;
            let f = 1u32 + base_s as u32;

            if s == 255 {
                f255 = f;
                cum_before_255 = cum;
            }

            if let Some(b) = target {
                if s == usize::from(b) {
                    found_cum = cum;
                    found_freq = f;
                }
            }
            cum += f;

            if let Some(c) = cdf {
                if found_sym < 0 && cum > c {
                    found_sym = s as i32;
                    found_freq = f;
                    found_cum = cum - f;
                }
            }
        }

        // Cumulative rounding caps the total at SCALE but can fall short when
        // the row is sparse (indices don't sum to 4096). Symbol 255 absorbs
        // the leftover mass, giving it the padded range [cum_before, SCALE) on
        // BOTH sides — the walk uses the real values up to the total, and cdfs
        // in the padded region ([total, SCALE)) decode to 255 by fallback.
        let total = cum;
        let sym = if found_sym >= 0 {
            found_sym as u8
        } else if cdf.is_some() {
            255
        } else {
            target.unwrap_or(0)
        };
        // Cumulative rounding caps the total at SCALE but can fall short when
        // the row is sparse (index shares don't sum to 4096). Symbol 255
        // absorbs all leftover mass deterministically on BOTH sides: whether
        // found by the walk (cdf < total) or by fallback (cdf in [total,
        // SCALE)), its range is [cum_before_255, SCALE). This matters for
        // sparse order-1/2 rows with few symbols and low counts.
        if sym == 255 {
            found_freq = f255 + (BYTE_SCALE - total);
            found_cum = cum_before_255;
        }

        WalkResult {
            sym,
            freq: found_freq,
            cum: found_cum,
        }
    }

    fn update(&mut self, ctx: usize, byte: u8) {
        let base = ctx * 256;
        let s = usize::from(byte);
        let c = &mut self.counts[base + s];
        *c = c.saturating_add(1);
        if *c == 1 {
            self.distinct[ctx] += 1;
        }
        self.totals[ctx] += 1;
        if *c > COUNT_CAP {
            let mut t = 0u32;
            let mut d = 0u32;
            for v in &mut self.counts[base..base + 256] {
                *v /= 2;
                t += u32::from(*v);
                if *v > 0 {
                    d += 1;
                }
            }
            self.totals[ctx] = t;
            self.distinct[ctx] = d;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct WalkResult {
    sym: u8,
    freq: u32,
    cum: u32,
}

/// PPM-style deterministic order selection: pick the highest-order context
/// whose concentration (`total/(distinct+1)`-style score) beats the lower
/// orders. Only context statistics are consulted — never the symbol being
/// encoded — so encode and decode agree without any escape mechanism.
#[inline(always)]
fn pick_order(t1: u32, d1: u32, t2: u32, d2: u32) -> Order {
    if t2 >= MIN_T2
        && (t1 < MIN_T1
            || u64::from(t2) * u64::from(d1 + 1) >= u64::from(t1) * u64::from(d2 + 1))
    {
        Order::Order2
    } else if t1 >= MIN_T1 {
        Order::Order1
    } else {
        Order::Order0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Order {
    Order0,
    Order1,
    Order2,
}

/// Compress one block's worth of bytes into an rANS byte stream.
///
/// `data.len()` must be ≥ 1; empty input should be stored via the container's
/// copy method instead.
#[must_use]
pub fn compress_block(data: &[u8]) -> Vec<u8> {
    let big = data.len() >= 256 * 1024;
    let order2_ctx = if big { ORDER2_CTX } else { ORDER2_CTX_SMALL };
    let order2_mask = order2_ctx - 1;
    let mut o0 = ByteCountModel::new(1);
    let mut o1 = ByteCountModel::new(256);
    let mut o2 = ByteCountModel::new(order2_ctx);
    let mut enc = RansByteEncoder::new();

    let mut p_2 = 0u8;
    let mut p_1 = 0u8;

    for &b in data {
        let c1 = usize::from(p_1);
        let c2 = hash2(p_1, p_2) & order2_mask;
        let ord = pick_order(o1.total(c1), o1.distinct(c1), o2.total(c2), o2.distinct(c2));
        let wr = match ord {
            Order::Order0 => o0.walk_dist(0, Some(b), None),
            Order::Order1 => o1.walk_dist(c1, Some(b), None),
            Order::Order2 => o2.walk_dist(c2, Some(b), None),
        };
        enc.encode_fc(wr.freq, wr.cum);

        o0.update(0, b);
        o1.update(c1, b);
        o2.update(c2, b);

        p_2 = p_1;
        p_1 = b;
    }
    enc.finish()
}

/// Decompress a block stream from [`compress_block`].
///
/// # Errors
///
/// Returns [`NyxError::CorruptBlock`] if the stream is truncated (fewer than
/// the 4-byte rANS state tail).
pub fn decompress_block(comp: &[u8], orig_len: usize) -> Result<Vec<u8>> {
    let big = orig_len >= 256 * 1024;
    let order2_ctx = if big { ORDER2_CTX } else { ORDER2_CTX_SMALL };
    let order2_mask = order2_ctx - 1;
    let mut o0 = ByteCountModel::new(1);
    let mut o1 = ByteCountModel::new(256);
    let mut o2 = ByteCountModel::new(order2_ctx);
    let mut dec =
        RansByteDecoder::new(comp).map_err(|_| NyxError::CorruptBlock("short rANS stream".into()))?;

    let mut p_2 = 0u8;
    let mut p_1 = 0u8;

    let mut out = Vec::with_capacity(orig_len);
    while out.len() < orig_len {
        let c1 = usize::from(p_1);
        let c2 = hash2(p_1, p_2) & order2_mask;
        let ord = pick_order(o1.total(c1), o1.distinct(c1), o2.total(c2), o2.distinct(c2));
        let wr = {
            let cdf = Some(dec.cdf());
            match ord {
                Order::Order0 => o0.walk_dist(0, None, cdf),
                Order::Order1 => o1.walk_dist(c1, None, cdf),
                Order::Order2 => o2.walk_dist(c2, None, cdf),
            }
        };
        dec.advance(wr.freq, wr.cum);
        o0.update(0, wr.sym);
        o1.update(c1, wr.sym);
        o2.update(c2, wr.sym);

        out.push(wr.sym);
        p_2 = p_1;
        p_1 = wr.sym;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &[u8]) {
        let comp = compress_block(data);
        let back = decompress_block(&comp, data.len()).expect("decompress");
        assert_eq!(back, data, "byte-codec round-trip mismatch");
    }

    #[test]
    fn walk_roundtrip_exact() {
        let mut v = vec![0u8; 8192];
        let mut x = 0x9E37_79B9u32;
        for b in &mut v {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        let n = 256;
        let big = n >= 256 * 1024;
        let order2_ctx = if big { ORDER2_CTX } else { ORDER2_CTX_SMALL };
        let order2_mask = order2_ctx - 1;
        let mut o0 = ByteCountModel::new(1);
        let mut o1 = ByteCountModel::new(256);
        let mut o2 = ByteCountModel::new(order2_ctx);
        let mut enc = RansByteEncoder::new();
        let mut p_2 = 0u8;
        let mut p_1 = 0u8;
        let mut e_triples = Vec::new();
        for &b in &v[..n] {
            let c1 = usize::from(p_1);
            let c2 = hash2(p_1, p_2) & order2_mask;
            let ord = pick_order(o1.total(c1), o1.distinct(c1), o2.total(c2), o2.distinct(c2));
            let wr = match ord {
                Order::Order0 => o0.walk_dist(0, Some(b), None),
                Order::Order1 => o1.walk_dist(c1, Some(b), None),
                Order::Order2 => o2.walk_dist(c2, Some(b), None),
            };
            enc.encode_fc(wr.freq, wr.cum);
            e_triples.push((b, wr.freq, wr.cum));
            o0.update(0, b);
            o1.update(c1, b);
            o2.update(c2, b);
            p_2 = p_1;
            p_1 = b;
        }
        let comp = enc.finish();

        let mut d0 = ByteCountModel::new(1);
        let mut d1 = ByteCountModel::new(256);
        let mut d2 = ByteCountModel::new(order2_ctx);
        let mut dec = RansByteDecoder::new(&comp).unwrap();
        let mut q_2 = 0u8;
        let mut q_1 = 0u8;
        for i in 0..n {
            let c1 = usize::from(q_1);
            let c2 = hash2(q_1, q_2) & order2_mask;
            let ord = pick_order(d1.total(c1), d1.distinct(c1), d2.total(c2), d2.distinct(c2));
            let cdf = dec.cdf();
            let wr = match ord {
                Order::Order0 => d0.walk_dist(0, None, Some(cdf)),
                Order::Order1 => d1.walk_dist(c1, None, Some(cdf)),
                Order::Order2 => d2.walk_dist(c2, None, Some(cdf)),
            };
            let (eb, ef, ec) = e_triples[i];
            assert_eq!(
                (wr.sym, wr.freq, wr.cum),
                (eb, ef, ec),
                "walk divergence at i={i} (ord={ord:?}, cdf={cdf})"
            );
            dec.advance(wr.freq, wr.cum);
            d0.update(0, wr.sym);
            d1.update(c1, wr.sym);
            d2.update(c2, wr.sym);
            q_2 = q_1;
            q_1 = wr.sym;
        }
    }

    #[test]
    fn roundtrip_deterministic_streams() {
        roundtrip(b"");
        roundtrip(b"a");
        roundtrip(&vec![0x41; 64]);
        roundtrip(&(0u8..=255).collect::<Vec<_>>());
        roundtrip(&(0u8..=255).cycle().take(4096).collect::<Vec<_>>());
    }

    #[test]
    fn roundtrip_text_like() {
        let text = b"the quick brown fox jumps over the lazy dog. ";
        let mut v = Vec::new();
        for _ in 0..50 {
            v.extend_from_slice(text);
        }
        roundtrip(&v);
    }

    #[test]
    fn roundtrip_random() {
        let mut v = vec![0u8; 8192];
        let mut x = 0x9E37_79B9u32;
        for b in &mut v {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        roundtrip(&v);
    }

    #[test]
    fn compress_is_smaller_on_redundant() {
        let data: Vec<u8> = std::iter::repeat(b'x').take(4096).collect();
        let comp = compress_block(&data);
        assert!(comp.len() < data.len(), "redundant data should shrink");
    }

    #[test]
    fn order_selection_uses_concentration() {
        assert_eq!(pick_order(0, 0, 0, 0), Order::Order0);
        assert_eq!(pick_order(1, 1, 0, 0), Order::Order0, "t1 below MIN_T1");
        assert_eq!(pick_order(3, 2, 4, 4), Order::Order1, "order-2 too scattered");
        assert_eq!(
            pick_order(3, 2, 4, 1),
            Order::Order2,
            "concentrated order-2 wins at equal scale"
        );
    }
}