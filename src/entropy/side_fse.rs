//! Order-0 FSE-family entropy coding for the DP-LZP match side-stream.
//!
//! Uses the existing byte rANS (`RansByteEncoder` / `RansByteDecoder`) as an
//! order-0 tANS/FSE-style coder: one stationary byte distribution over the
//! packed varint blob. The outer codec stores a flag so we can fall back to
//! raw bytes when entropy coding does not shrink the payload.

use crate::entropy::byterans::{RansByteDecoder, RansByteEncoder, BYTE_SCALE};

/// Side-stream stored as raw varint bytes.
pub const FLAG_RAW: u8 = 0;
/// Side-stream entropy-coded with order-0 byte rANS.
pub const FLAG_FSE: u8 = 1;

/// Compress `data` with order-0 rANS.
///
/// Payload layout:
/// `[orig_len:u32 LE][present_bitmap: 32 bytes][freq:u16 LE per present symbol][rANS]`.
fn compress_order0(data: &[u8]) -> Vec<u8> {
    let mut counts = [0u32; 256];
    for &b in data {
        counts[usize::from(b)] += 1;
    }
    let freqs = normalize_freqs(&counts);
    let mut enc = RansByteEncoder::new();
    for &b in data {
        enc.encode_symbol(b, freqs);
    }
    let stream = enc.finish();

    let mut bitmap = [0u8; 32];
    let mut sparse = Vec::new();
    for (sym, &f) in freqs.iter().enumerate() {
        if f > 0 {
            bitmap[sym / 8] |= 1 << (sym % 8);
            sparse.extend_from_slice(&f.to_le_bytes());
        }
    }

    let mut out = Vec::with_capacity(4 + 32 + sparse.len() + stream.len());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&bitmap);
    out.extend_from_slice(&sparse);
    out.extend_from_slice(&stream);
    out
}

/// Inverse of [`compress_order0`].
fn decompress_order0(payload: &[u8]) -> Result<Vec<u8>, ()> {
    if payload.len() < 4 + 32 {
        return Err(());
    }
    let orig_len =
        u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let bitmap = &payload[4..36];
    let mut freqs = [0u16; 256];
    let mut pos = 36usize;
    for sym in 0..256 {
        if bitmap[sym / 8] & (1 << (sym % 8)) != 0 {
            if pos + 2 > payload.len() {
                return Err(());
            }
            freqs[sym] = u16::from_le_bytes([payload[pos], payload[pos + 1]]);
            pos += 2;
        }
    }
    let stream = &payload[pos..];
    let mut dec = RansByteDecoder::new(stream)?;
    let mut out = Vec::with_capacity(orig_len);
    for _ in 0..orig_len {
        out.push(dec.decode_symbol(&freqs));
    }
    Ok(out)
}

/// Normalize histogram to sum exactly [`BYTE_SCALE`], with every observed
/// symbol getting at least frequency 1.
fn normalize_freqs(counts: &[u32; 256]) -> [u16; 256] {
    let total: u32 = counts.iter().sum();
    if total == 0 {
        // Unused empty path — uniform so rANS stays well-defined.
        return [BYTE_SCALE as u16 / 256; 256];
    }
    let mut freqs = [0u16; 256];
    let mut present = 0u32;
    for &c in counts {
        if c > 0 {
            present += 1;
        }
    }
    // Reserve 1 slot per present symbol, distribute the rest by share.
    let rest = (BYTE_SCALE as u32).saturating_sub(present);
    let mut assigned = 0u32;
    let mut max_i = 0usize;
    let mut max_c = 0u32;
    for (i, &c) in counts.iter().enumerate() {
        if c == 0 {
            continue;
        }
        if c > max_c {
            max_c = c;
            max_i = i;
        }
        let share = 1 + ((u64::from(c) * u64::from(rest)) / u64::from(total)) as u32;
        freqs[i] = share as u16;
        assigned += share;
    }
    // Fix rounding so frequencies sum to BYTE_SCALE.
    let target = BYTE_SCALE as u32;
    if assigned < target {
        freqs[max_i] = freqs[max_i].saturating_add((target - assigned) as u16);
    } else if assigned > target {
        let over = assigned - target;
        freqs[max_i] = freqs[max_i].saturating_sub(over as u16).max(1);
        // If clamp left the sum high, peel from other present symbols.
        let mut sum: u32 = freqs.iter().map(|&f| u32::from(f)).sum();
        if sum > target {
            for f in freqs.iter_mut() {
                if sum <= target {
                    break;
                }
                if *f > 1 {
                    let take = ((*f as u32) - 1).min(sum - target);
                    *f -= take as u16;
                    sum -= take;
                }
            }
        }
    }
    freqs
}

/// Pack a varint side-stream blob: try order-0 entropy coding, fall back to raw
/// when it is not smaller.
///
/// Returns `(flag, payload)` where `payload` is either the raw bytes or the
/// FSE/rANS frame from [`compress_order0`].
#[must_use]
pub fn pack_side_stream(data: &[u8]) -> (u8, Vec<u8>) {
    if data.is_empty() {
        return (FLAG_RAW, Vec::new());
    }
    let compressed = compress_order0(data);
    if compressed.len() < data.len() {
        (FLAG_FSE, compressed)
    } else {
        (FLAG_RAW, data.to_vec())
    }
}

/// Unpack a side-stream payload produced by [`pack_side_stream`].
///
/// # Errors
///
/// Returns `Err(())` on unknown flag or corrupt FSE payload.
pub fn unpack_side_stream(flag: u8, payload: &[u8]) -> Result<Vec<u8>, ()> {
    match flag {
        FLAG_RAW => Ok(payload.to_vec()),
        FLAG_FSE => decompress_order0(payload),
        _ => Err(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_raw_preferred_on_random() {
        let mut data = vec![0u8; 64];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i.wrapping_mul(37) ^ 0xA5) as u8;
        }
        let (flag, payload) = pack_side_stream(&data);
        let back = unpack_side_stream(flag, &payload).expect("unpack");
        assert_eq!(back, data);
    }

    #[test]
    fn round_trip_fse_on_repetitive() {
        let data = vec![1u8, 2, 3, 1, 2, 3, 1, 2, 3]
            .into_iter()
            .cycle()
            .take(512)
            .collect::<Vec<_>>();
        let (flag, payload) = pack_side_stream(&data);
        assert_eq!(flag, FLAG_FSE, "repetitive data should prefer FSE");
        let back = unpack_side_stream(flag, &payload).expect("unpack");
        assert_eq!(back, data);
    }

    #[test]
    fn empty_is_raw() {
        let (flag, payload) = pack_side_stream(&[]);
        assert_eq!(flag, FLAG_RAW);
        assert!(payload.is_empty());
        assert_eq!(unpack_side_stream(flag, &payload).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn order0_direct_round_trip() {
        let data: Vec<u8> = (0..200).map(|i| (i % 7) as u8).collect();
        let packed = compress_order0(&data);
        let back = decompress_order0(&packed).expect("decompress");
        assert_eq!(back, data);
    }
}
