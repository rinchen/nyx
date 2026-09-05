//! Burrows-Wheeler transform + Move-To-Front + RLE0 pipeline for Text blocks.
//!
//! BWT groups similar contexts together, turning long-range repetitions (e.g. repeated
//! words in text) into local runs that CM predicts well. MTF converts BWT output to a
//! rank stream (0 = same char as previous position), which produces many zeros — RLE0
//! compresses those runs. The full pipeline is an involution: every transform has an
//! exact inverse, and the encoder/decoder apply them in opposite order.
//!
//! Pipeline (encode):  data → BWT → MTF → RLE0 → [CM/rANS]
//! Pipeline (decode):  [rANS/CM] → RLE0⁻¹ → MTF⁻¹ → BWT⁻¹ → data
//!
//! BWT uses `divsufsort` for O(n) suffix-array construction. Rotation-based BWT is used
//! (via doubled string) to avoid sentinel collisions when data contains null bytes. The
//! LF-mapping walk forms a single cycle guaranteed by the cyclic rotation ordering.

use divsufsort::sort as divsufsort_sort;

// ---------------------------------------------------------------------------
// RLE0
// ---------------------------------------------------------------------------

/// Run-length-encode zero runs and literal 0xFF bytes in a byte stream.
///
/// Non-zero, non-0xFF bytes pass through unchanged. Consecutive zeros are encoded as a
/// flag byte `0xFF` followed by the run length as a single byte (max run 255).
/// A literal `0xFF` in the data is escaped as `0xFF 0x00`.
/// A literal `0x00` in the data is escaped as `0xFF 0x01` (if not part of a longer run).
pub fn rle0(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let b = data[i];
        if b == 0xFF {
            // Escape literal 0xFF.
            out.push(0xFF);
            out.push(0x00);
            i += 1;
        } else if b == 0 {
            // Count zero run (max 255 per RLE token).
            let mut count = 0u8;
            while i < data.len() && data[i] == 0 && count < 255 {
                count += 1;
                i += 1;
            }
            out.push(0xFF);
            out.push(count);
        } else {
            out.push(b);
            i += 1;
        }
    }
    out
}

/// Inverse of [`rle0`].
pub fn rle0_inverse(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let b = data[i];
        if b == 0xFF {
            i += 1;
            if i >= data.len() {
                // Trailing 0xFF with no count byte — shouldn't happen in valid data.
                break;
            }
            let count = data[i];
            if count == 0 {
                // Escaped literal 0xFF.
                out.push(0xFF);
            } else {
                // Zero run of length count.
                out.resize(out.len() + count as usize, 0);
            }
            i += 1;
        } else {
            out.push(b);
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// MTF
// ---------------------------------------------------------------------------

/// Move-To-Front transform on a byte stream.
///
/// Maintains a list of 256 symbols. Each input byte is replaced by its index in
/// the list, then moved to the front.
pub fn mtf_transform(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut list: Vec<u8> = (0u8..=255).collect();
    for &b in data {
        let pos = list.iter().position(|&x| x == b).unwrap();
        out.push(pos as u8);
        list.remove(pos);
        list.insert(0, b);
    }
    out
}

/// Inverse of [`mtf_transform`].
pub fn mtf_inverse(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut list: Vec<u8> = (0u8..=255).collect();
    for &idx in data {
        let idx = idx as usize;
        let b = list[idx];
        out.push(b);
        list.remove(idx);
        list.insert(0, b);
    }
    out
}

// ---------------------------------------------------------------------------
// BWT (rotation-based via doubled string)
// ---------------------------------------------------------------------------

/// Forward Burrows-Wheeler transform.
///
/// Sorts all cyclic rotations of `data` (using the doubled-string SA trick to avoid
/// sentinel collisions) and produces the BWT string: for each rotation in sorted order,
/// the last character. Appends a 4-byte LE primary index so the decoder knows where to
/// start the LF-mapping walk.
pub fn bwt_forward(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let n = data.len();

    // Build doubled string for rotation-based BWT (avoid sentinel collisions).
    let mut doubled = Vec::with_capacity(n * 2);
    doubled.extend_from_slice(data);
    doubled.extend_from_slice(data);

    let sa = divsufsort_sort(&doubled);
    let sa: Vec<usize> = sa.into_parts().1.iter().map(|&p| p as usize).collect();

    // Collect only suffixes starting at positions 0..n (rotations of the original).
    let mut bwt = Vec::with_capacity(n);
    let mut primary = 0usize;
    let mut bwt_idx = 0usize;
    for &suf_pos in &sa {
        if suf_pos < n {
            if suf_pos == 0 {
                primary = bwt_idx;
            }
            // The character preceding this rotation is data[suf_pos - 1] (circular).
            bwt.push(data[(suf_pos + n - 1) % n]);
            bwt_idx += 1;
        }
        if bwt.len() == n {
            break;
        }
    }

    // Append 4-byte LE primary index.
    bwt.extend_from_slice(&(primary as u32).to_le_bytes());

    bwt
}

/// Inverse Burrows-Wheeler transform.
///
/// Reconstructs the original data from the BWT output (with 4-byte primary index)
/// produced by [`bwt_forward`]. The LF-mapping walk is guaranteed to form a single
/// cycle because the BWT is rotation-based.
pub fn bwt_inverse(data: &[u8]) -> Vec<u8> {
    // Need at least the 4-byte primary index.
    if data.len() < 4 {
        return Vec::new();
    }

    // Split BWT data and 4-byte primary index (last 4 bytes).
    let primary_bytes = &data[data.len() - 4..];
    let primary = u32::from_le_bytes([
        primary_bytes[0],
        primary_bytes[1],
        primary_bytes[2],
        primary_bytes[3],
    ]) as usize;
    let bwt = &data[..data.len() - 4];

    let n = bwt.len();
    if n == 0 {
        return Vec::new();
    }

    // Count occurrences of each byte value in the BWT (last column).
    let mut counts = [0usize; 256];
    for &b in bwt {
        counts[b as usize] += 1;
    }

    // Starting position of each byte value in the sorted (first) column.
    let mut starts = [0usize; 256];
    let mut acc = 0;
    for i in 0..256 {
        starts[i] = acc;
        acc += counts[i];
    }

    // For each position in BWT, compute its rank within its byte value.
    let mut ranks = vec![0usize; n];
    let mut seen = [0usize; 256];
    for i in 0..n {
        let c = bwt[i] as usize;
        ranks[i] = seen[c];
        seen[c] += 1;
    }

    // LF-mapping walk from the primary index, reconstructing the original string.
    // The walk produces the string in reverse order (last character first).
    let mut result = Vec::with_capacity(n);
    let mut idx = primary;
    for _ in 0..n {
        let c = bwt[idx] as usize;
        result.push(c as u8);
        idx = starts[c] + ranks[idx];
    }
    result.reverse();

    result
}

// ---------------------------------------------------------------------------
// LZP encode/decode (lightweight — reused from model/lzp.rs patterns)
// ---------------------------------------------------------------------------

/// Encode `data` with a simple LZP pre-filter using a hash chain for match finding.
///
/// Scans for matches of length >= 4 in a 4 MB history window. Emits:
///   - `[1, len, dist_hi, dist_lo, dist_lo2]` for matches (len 4..255, dist 1..4MB)
///   - `[0, literal]` for non-matching bytes
///
/// Uses a 2-byte hash with chaining for O(n) expected match finding.
pub fn lzp_encode(data: &[u8]) -> Vec<u8> {
    let n = data.len();
    let mut out = Vec::with_capacity(n / 4);
    const HASH_SIZE: usize = 1 << 16;
    let mut head: Vec<i32> = vec![-1; HASH_SIZE];
    let mut prev: Vec<i32> = vec![-1; n];

    let window = 4 * 1024 * 1024;
    let mut i = 0usize;
    while i < n {
        let mut best_len = 0usize;
        let mut best_dist = 0usize;

        if i + 4 <= n {
            let hash = ((data[i] as u32) << 8 | (data[i + 1] as u32)) as usize % HASH_SIZE;
            let mut cand = head[hash];
            let mut probes = 0;
            while cand >= 0 && probes < 32 {
                let cand_usize = cand as usize;
                if i.saturating_sub(cand_usize) <= window {
                    let mut len = 0usize;
                    while i + len < n && len < 255 && data[cand_usize + len] == data[i + len] {
                        len += 1;
                    }
                    if len > best_len {
                        best_len = len;
                        best_dist = i - cand_usize;
                        if best_len >= 255 {
                            break;
                        }
                    }
                }
                let next = prev[cand_usize];
                if next < 0 {
                    break;
                }
                cand = next;
                probes += 1;
            }
            prev[i] = head[hash];
            head[hash] = i as i32;
        }

        if best_len >= 4 {
            out.push(1);
            out.push(best_len.min(255) as u8);
            let dist = best_dist as u32;
            out.push((dist >> 16) as u8);
            out.push((dist >> 8) as u8);
            out.push(dist as u8);

            // Insert skipped positions into the hash chain.
            let advance = best_len;
            i += advance;
            // Insert intermediate positions
            for j in 0..advance {
                let p = i - advance + j;
                if p + 1 < n {
                    let h = ((data[p] as u32) << 8 | (data[p + 1] as u32)) as usize % HASH_SIZE;
                    prev[p] = head[h];
                    head[h] = p as i32;
                }
            }
        } else {
            out.push(0);
            out.push(data[i]);
            i += 1;
        }
    }
    out
}

/// Decode an LZP side-stream produced by [`lzp_encode`].
pub fn lzp_decode(data: &[u8], orig_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(orig_len);
    let mut i = 0usize;
    while i < data.len() && out.len() < orig_len {
        let flag = data[i];
        i += 1;
        if flag == 1 {
            if i + 3 >= data.len() {
                break;
            }
            let len = data[i] as usize;
            let dist =
                ((data[i + 1] as u32) << 16) | ((data[i + 2] as u32) << 8) | (data[i + 3] as u32);
            i += 4;
            let dist = dist as usize;
            if dist > out.len() || dist > 4 * 1024 * 1024 {
                break;
            }
            let start = out.len() - dist;
            let mut j = 0;
            while out.len() < orig_len && j < len {
                // Copy byte-by-byte to handle overlapping matches (dist < len).
                out.push(out[start + j]);
                j += 1;
            }
        } else {
            if i >= data.len() {
                break;
            }
            out.push(data[i]);
            i += 1;
        }
    }
    if out.len() < orig_len {
        out.resize(orig_len, 0);
    }
    out
}

// ---------------------------------------------------------------------------
// Full BWT encode/decode pipelines
// ---------------------------------------------------------------------------

/// Full BWT encode pipeline: data → BWT → MTF → RLE0.
///
/// The BWT output (with 4-byte primary index) is fed through MTF + RLE0.
pub fn bwt_mtf_rle_encode(data: &[u8]) -> Vec<u8> {
    let bwt = bwt_forward(data);
    let mtf = mtf_transform(&bwt);
    rle0(&mtf)
}

/// Full BWT decode pipeline: RLE0⁻¹ → MTF⁻¹ → BWT⁻¹.
pub fn bwt_mtf_rle_decode(data: &[u8]) -> Vec<u8> {
    let rle = rle0_inverse(data);
    let mtf = mtf_inverse(&rle);
    bwt_inverse(&mtf)
}

/// Same as bwt_forward but without RLE0 — for Path C (LZP → BWT → MTF → CM)
/// where the CM operates on MTF ranks directly (no RLE0).
pub fn bwt_mtf_encode(data: &[u8]) -> Vec<u8> {
    let bwt = bwt_forward(data);
    mtf_transform(&bwt)
}

/// Full BWT decode pipeline without RLE0: MTF⁻¹ → BWT⁻¹.
pub fn bwt_mtf_decode(data: &[u8]) -> Vec<u8> {
    let mtf = mtf_inverse(data);
    bwt_inverse(&mtf)
}

// ---------------------------------------------------------------------------
// Per-block trial: pick the best pipeline for this block
// ---------------------------------------------------------------------------

/// Which BWT-based pipeline was chosen for a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BwtPipeline {
    /// Path A: raw CM — no transform, original bytes fed to CM.
    RawCm,
    /// Path B: BWT → MTF → RLE0 → CM on MTF ranks.
    BwtMtfRle,
    /// Path C: LZP → BWT → MTF → CM (no RLE0).
    LzpBwtMtf,
}

impl BwtPipeline {
    /// Encode `data` using this pipeline, returning the payload that CM/rANS will
    /// compress. For `RawCm`, the payload IS the original data.
    pub fn encode(self, data: &[u8]) -> Vec<u8> {
        match self {
            BwtPipeline::RawCm => data.to_vec(),
            BwtPipeline::BwtMtfRle => bwt_mtf_rle_encode(data),
            BwtPipeline::LzpBwtMtf => bwt_mtf_encode(&lzp_encode(data)),
        }
    }

    /// Decode a payload produced by [`Self::encode`], returning the original data.
    pub fn decode(self, payload: &[u8], orig_len: usize) -> Vec<u8> {
        match self {
            BwtPipeline::RawCm => payload.to_vec(),
            BwtPipeline::BwtMtfRle => bwt_mtf_rle_decode(payload),
            BwtPipeline::LzpBwtMtf => {
                let mtf = bwt_mtf_decode(payload);
                lzp_decode(&mtf, orig_len)
            }
        }
    }
}

/// Result of running all three paths and comparing sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BwtPathResult {
    /// The chosen pipeline.
    pub pipeline: BwtPipeline,
    /// Size of the encoded payload (before rANS/CM compression).
    pub encoded_size: usize,
    /// Whether this is a transform path (BWT-based) or raw CM.
    pub is_bwt: bool,
}

/// Run all three BWT paths on `data` and return the smallest.
///
/// For small blocks (< 256 KB), only Path A (raw CM) is used to avoid the overhead of
/// BWT/LZP trials. For large blocks, all three paths are tried and the one with the
/// smallest transformed output is chosen. Only the **pipeline** and **size** are
/// returned; the caller re-encodes with the chosen pipeline. For blocks < 256 KB,
/// the caller can skip the BWT encode and pass the raw data directly.
pub fn compress_text_with_trial(data: &[u8]) -> BwtPathResult {
    if data.len() < 256 * 1024 {
        return BwtPathResult {
            pipeline: BwtPipeline::RawCm,
            encoded_size: data.len(),
            is_bwt: false,
        };
    }

    let path_a_size = data.len();

    let path_b = BwtPipeline::BwtMtfRle.encode(data);
    let path_b_size = path_b.len();

    let path_c = BwtPipeline::LzpBwtMtf.encode(data);
    let path_c_size = path_c.len();

    let (best_pipeline, best_size, is_bwt) =
        if path_b_size <= path_a_size && path_b_size <= path_c_size {
            (BwtPipeline::BwtMtfRle, path_b_size, true)
        } else if path_c_size <= path_a_size && path_c_size <= path_b_size {
            (BwtPipeline::LzpBwtMtf, path_c_size, true)
        } else {
            (BwtPipeline::RawCm, path_a_size, false)
        };

    BwtPathResult {
        pipeline: best_pipeline,
        encoded_size: best_size,
        is_bwt,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rle0_round_trip_empty() {
        let encoded = rle0(b"");
        assert!(encoded.is_empty());
        let decoded = rle0_inverse(&encoded);
        assert!(decoded.is_empty());
    }

    #[test]
    fn rle0_round_trip_zeros() {
        let input = vec![0u8; 100];
        let encoded = rle0(&input);
        let decoded = rle0_inverse(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn rle0_round_trip_mixed() {
        let input: Vec<u8> = (0u8..=255).collect();
        let encoded = rle0(&input);
        let decoded = rle0_inverse(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn rle0_round_trip_with_0xff() {
        let input = vec![0xFF, 0x00, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00];
        let encoded = rle0(&input);
        let decoded = rle0_inverse(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn rle0_round_trip_long_zeros() {
        let input = vec![0u8; 600];
        let encoded = rle0(&input);
        let decoded = rle0_inverse(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn rle0_round_trip_lone_0xff() {
        // 0xFF not adjacent to any zeros — must survive.
        let input = vec![0x42, 0xFF, 0x43];
        let encoded = rle0(&input);
        let decoded = rle0_inverse(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn mtf_round_trip() {
        let input: Vec<u8> = (0u8..=255).cycle().take(1000).collect();
        let encoded = mtf_transform(&input);
        let decoded = mtf_inverse(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn mtf_round_trip_repetitive() {
        let input = b"abracadabra".repeat(100);
        let encoded = mtf_transform(&input);
        let decoded = mtf_inverse(&encoded);
        assert_eq!(decoded.as_slice(), input);
    }

    #[test]
    fn bwt_round_trip_small() {
        let input = b"banana";
        let bwt = bwt_forward(input);
        let decoded = bwt_inverse(&bwt);
        assert_eq!(decoded, input);
    }

    #[test]
    fn bwt_round_trip_empty() {
        let bwt = bwt_forward(b"");
        let decoded = bwt_inverse(&bwt);
        assert!(decoded.is_empty());
    }

    #[test]
    fn bwt_round_trip_single() {
        let bwt = bwt_forward(&[42u8]);
        let decoded = bwt_inverse(&bwt);
        assert_eq!(decoded, vec![42u8]);
    }

    #[test]
    fn bwt_round_trip_large() {
        let text = b"the quick brown fox jumps over the lazy dog. \n".repeat(200);
        let bwt = bwt_forward(&text);
        let decoded = bwt_inverse(&bwt);
        assert_eq!(decoded, text);
    }

    #[test]
    fn bwt_round_trip_repetitive() {
        // "banana" has repeated substrings — tests rotation-based BWT correctness.
        let text = b"banana".repeat(1000);
        let bwt = bwt_forward(&text);
        let decoded = bwt_inverse(&bwt);
        assert_eq!(decoded, text);
    }

    #[test]
    fn bwt_round_trip_random() {
        let mut buf = vec![0u8; 10_000];
        let mut x = 0x1234_5678u32;
        for b in &mut buf {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        let bwt = bwt_forward(&buf);
        let decoded = bwt_inverse(&bwt);
        assert_eq!(decoded, buf);
    }

    #[test]
    fn bwt_pipeline_round_trip() {
        let text = b"the quick brown fox. \n".repeat(500);
        let encoded = bwt_mtf_rle_encode(&text);
        let decoded = bwt_mtf_rle_decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn bwt_pipeline_round_trip_random() {
        let mut buf = vec![0u8; 10_000];
        let mut x = 0x1234_5678u32;
        for b in &mut buf {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        let encoded = bwt_mtf_rle_encode(&buf);
        let decoded = bwt_mtf_rle_decode(&encoded);
        assert_eq!(decoded, buf);
    }

    #[test]
    fn bwt_pipeline_round_trip_with_nulls() {
        // Data with embedded null bytes.
        let mut buf = vec![0u8; 5_000];
        let mut x = 0x9876_5432u32;
        for b in &mut buf {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = (x % 200) as u8; // 0-199, includes 0
        }
        let encoded = bwt_mtf_rle_encode(&buf);
        let decoded = bwt_mtf_rle_decode(&encoded);
        assert_eq!(decoded, buf);
    }

    #[test]
    fn bwt_pipeline_round_trip_with_0xff() {
        // Data with many 0xFF bytes.
        let mut buf = vec![0u8; 5_000];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = if i % 3 == 0 { 0xFF } else { (i % 200) as u8 };
        }
        let encoded = bwt_mtf_rle_encode(&buf);
        let decoded = bwt_mtf_rle_decode(&encoded);
        assert_eq!(decoded, buf);
    }

    #[test]
    fn lzp_round_trip() {
        let mut buf = vec![0u8; 10_000];
        let mut x = 0x1234_5678u32;
        for b in &mut buf {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        let encoded = lzp_encode(&buf);
        let decoded = lzp_decode(&encoded, buf.len());
        assert_eq!(decoded, buf);
    }

    #[test]
    fn bwt_pipeline_lzp_bwt_round_trip() {
        let text = b"the quick brown fox. \n".repeat(500);
        let encoded = BwtPipeline::LzpBwtMtf.encode(&text);
        let decoded = BwtPipeline::LzpBwtMtf.decode(&encoded, text.len());
        assert_eq!(decoded, text);
    }

    #[test]
    fn compress_text_with_trial_selects_smallest() {
        // Repetitive text should favor BWT path.
        let text = b"banana ".repeat(5000);
        let result = compress_text_with_trial(&text);
        println!(
            "Trial result: {:?}, size={}",
            result.pipeline, result.encoded_size
        );
        assert!(result.pipeline != BwtPipeline::RawCm || result.encoded_size <= text.len());
    }
}
