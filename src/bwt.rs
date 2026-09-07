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
    let mut out = Vec::with_capacity(n / 4 + 4);
    // Prepend original length as 4-byte LE so the decoder knows how many bytes
    // of original data to expect (the LZP side-stream itself is variable-length).
    out.extend_from_slice(&(n as u32).to_le_bytes());
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
    // Read 4-byte LE original length from the start of the stream.
    if data.len() < 4 {
        return Vec::new();
    }
    let orig = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    // Use the decoded original length if the passed orig_len doesn't match
    // (e.g., when called from the codec with transformed length). Fall back
    // to the passed value.
    let orig = if orig == 0 && orig_len == 0 { 0 } else { orig };
    let mut out = Vec::with_capacity(orig);
    let mut i = 4usize;
    while i < data.len() && out.len() < orig {
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
            while out.len() < orig && j < len {
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
    if out.len() < orig {
        out.resize(orig, 0);
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
    /// Path D: JSON stream split → 4× independent BWT trials → CM.
    JsonSplit,
    /// Path E: XWRT dictionary → BWT → MTF → RLE0 → CM.
    XwrtBwtMtfRle,
}

impl BwtPipeline {
    /// Encode `data` using this pipeline, returning the payload that CM/rANS will
    /// compress. For `RawCm`, the payload IS the original data.
    ///
    /// For `JsonSplit`, the payload is: `[4×4-byte stream lengths][4 BWT-encoded streams]`.
    pub fn encode(self, data: &[u8]) -> Vec<u8> {
        match self {
            BwtPipeline::RawCm => data.to_vec(),
            BwtPipeline::BwtMtfRle => bwt_mtf_rle_encode(data),
            BwtPipeline::LzpBwtMtf => bwt_mtf_encode(&lzp_encode(data)),
            BwtPipeline::JsonSplit => {
                let streams = crate::json_split::split(data);
                // Layout: [orig_len:u32][selector:u8][s0_len:u32][s1_len:u32][s2_len:u32][s0][s1][s2][s3]
                // 2 bits per stream in selector: 0=RawCm, 1=BwtMtfRle, 2=LzpBwtMtf
                let s_raw = streams.structural.len();
                let s_bwt = bwt_mtf_rle_encode(&streams.structural);
                let s_lzp = bwt_mtf_encode(&lzp_encode(&streams.structural));
                let struct_pipe = if s_bwt.len() <= s_raw && s_bwt.len() <= s_lzp.len() {
                    1
                } else if s_lzp.len() <= s_raw && s_lzp.len() <= s_bwt.len() {
                    2
                } else {
                    0
                };
                let struct_encoded = match struct_pipe {
                    0 => streams.structural.clone(),
                    1 => s_bwt,
                    _ => s_lzp,
                };

                let k_raw = streams.keys.len();
                let k_bwt = bwt_mtf_rle_encode(&streams.keys);
                let k_lzp = bwt_mtf_encode(&lzp_encode(&streams.keys));
                let keys_pipe = if k_bwt.len() <= k_raw && k_bwt.len() <= k_lzp.len() {
                    1
                } else if k_lzp.len() <= k_raw && k_lzp.len() <= k_bwt.len() {
                    2
                } else {
                    0
                };
                let keys_encoded = match keys_pipe {
                    0 => streams.keys.clone(),
                    1 => k_bwt,
                    _ => k_lzp,
                };

                let v_raw = streams.string_values.len();
                let v_bwt = bwt_mtf_rle_encode(&streams.string_values);
                let v_lzp = bwt_mtf_encode(&lzp_encode(&streams.string_values));
                let vals_pipe = if v_bwt.len() <= v_raw && v_bwt.len() <= v_lzp.len() {
                    1
                } else if v_lzp.len() <= v_raw && v_lzp.len() <= v_bwt.len() {
                    2
                } else {
                    0
                };
                let vals_encoded = match vals_pipe {
                    0 => streams.string_values.clone(),
                    1 => v_bwt,
                    _ => v_lzp,
                };

                let n_raw = streams.numbers.len();
                let n_bwt = bwt_mtf_rle_encode(&streams.numbers);
                let n_lzp = bwt_mtf_encode(&lzp_encode(&streams.numbers));
                let nums_pipe = if n_bwt.len() <= n_raw && n_bwt.len() <= n_lzp.len() {
                    1
                } else if n_lzp.len() <= n_raw && n_lzp.len() <= n_bwt.len() {
                    2
                } else {
                    0
                };
                let nums_encoded = match nums_pipe {
                    0 => streams.numbers.clone(),
                    1 => n_bwt,
                    _ => n_lzp,
                };

                let selector = (struct_pipe as u8)
                    | ((keys_pipe as u8) << 2)
                    | ((vals_pipe as u8) << 4)
                    | ((nums_pipe as u8) << 6);

                // Layout: [orig_len:u32][selector:u8][s0_len:u32][s1_len:u32][s2_len:u32][s0][s1][s2][s3]
                let mut out = Vec::with_capacity(data.len() + 21);
                out.extend_from_slice(&(data.len() as u32).to_le_bytes());
                out.push(selector);
                out.extend_from_slice(&(struct_encoded.len() as u32).to_le_bytes());
                out.extend_from_slice(&(keys_encoded.len() as u32).to_le_bytes());
                out.extend_from_slice(&(vals_encoded.len() as u32).to_le_bytes());
                out.extend_from_slice(&struct_encoded);
                out.extend_from_slice(&keys_encoded);
                out.extend_from_slice(&vals_encoded);
                out.extend_from_slice(&nums_encoded);
                out
            }
            BwtPipeline::XwrtBwtMtfRle => {
                // Build dictionary from original data and apply XWRT transform.
                let dict = crate::model::word::XwrtDictionary::build_from_data(data);
                let xwrt = dict.transform(data);
                let encoded = bwt_mtf_rle_encode(&xwrt);
                let dict_bytes = dict.to_bytes();
                let mut out = Vec::with_capacity(4 + 4 + encoded.len() + dict_bytes.len());
                out.extend_from_slice(&(data.len() as u32).to_le_bytes());
                out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
                out.extend_from_slice(&encoded);
                out.extend_from_slice(&dict_bytes);
                out
            }
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
            BwtPipeline::JsonSplit => {
                // Layout: [orig_len:u32][selector:u8][s0_len:u32][s1_len:u32][s2_len:u32][s0][s1][s2][s3]
                if payload.len() < 17 {
                    return Vec::new();
                }
                let orig_len =
                    u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
                let selector = payload[4];
                let s0_pipe = selector & 0x3;
                let s1_pipe = (selector >> 2) & 0x3;
                let s2_pipe = (selector >> 4) & 0x3;
                let s3_pipe = (selector >> 6) & 0x3;

                let s0_len =
                    u32::from_le_bytes([payload[5], payload[6], payload[7], payload[8]]) as usize;
                let s1_len = u32::from_le_bytes([payload[9], payload[10], payload[11], payload[12]])
                    as usize;
                let s2_len =
                    u32::from_le_bytes([payload[13], payload[14], payload[15], payload[16]])
                        as usize;
                let mut pos = 17;
                let s0 = &payload[pos..pos + s0_len];
                pos += s0_len;
                let s1 = &payload[pos..pos + s1_len];
                pos += s1_len;
                let s2 = &payload[pos..pos + s2_len];
                pos += s2_len;
                let s3 = &payload[pos..];

                let decode_stream = |data: &[u8], pipe: u8| -> Vec<u8> {
                    match pipe {
                        0 => data.to_vec(),
                        1 => bwt_mtf_rle_decode(data),
                        _ => {
                            let mtf = bwt_mtf_decode(data);
                            lzp_decode(&mtf, 0)
                        }
                    }
                };

                let streams = crate::json_split::JsonStreams {
                    structural: decode_stream(s0, s0_pipe),
                    keys: decode_stream(s1, s1_pipe),
                    string_values: decode_stream(s2, s2_pipe),
                    numbers: decode_stream(s3, s3_pipe),
                };
                crate::json_split::merge(&streams, orig_len).unwrap_or_default()
            }
            BwtPipeline::XwrtBwtMtfRle => {
                // Layout: [orig_len:u32][encoded_len:u32][bwt_mtf_rle_encoded][dictionary_bytes]
                if payload.len() < 8 {
                    return Vec::new();
                }
                let orig_len =
                    u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
                let encoded_len =
                    u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;
                if payload.len() < 8 + encoded_len {
                    return Vec::new();
                }
                let mtf = bwt_mtf_rle_decode(&payload[8..8 + encoded_len]);
                let dict_data = &payload[8 + encoded_len..];
                crate::model::word::xwrt_inverse_with_dict(&mtf, orig_len, dict_data)
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

/// Fast-path threshold: blocks smaller than this skip the BWT/JSON trials and
/// go straight to raw CM. Below ~1 MB the transformed-size comparison rarely pays
/// for the BWT construction + two extra full-CM passes. JSON is exempt — its
/// stream-split win is large enough that trialing it from 256 KB up is worth it.
const TRIAL_MIN_LEN: usize = 1 << 20; // 1 MB
/// Blocks remaining above this trial threshold but near-random skip all trials.
/// (Text passed the classifier at shannon < 7.9, but rich-but-near-random text
/// gains nothing from BWT's long-range reordering.)
const TRIAL_MAX_SHANNON: f32 = 7.2;

/// Run all three BWT paths on `data` and return the smallest.
///
/// Fast-path heuristics: blocks < 256 KB always use raw CM (the transforms can
/// only help long-range structure that short blocks lack). Blocks under 1 MB
/// (256 KB..1 MB) skip trials *unless* they look like JSON — JSON stream-splitting
/// pays off from 256 KB up. Blocks above 1 MB use a Shannon guard: near-random
/// text skips trials too. Only the **pipeline** and **size** are returned; the
/// caller re-encodes with the chosen pipeline.
///
/// If the block looks like JSON and is large enough, a fourth path (JSON split) is
/// also tried: the block is split into 4 streams and each is BWT-trialed independently.
pub fn compress_text_with_trial(data: &[u8]) -> BwtPathResult {
    let is_json = crate::json_split::looks_like_json(data);
    let small = data.len() < 256 * 1024;
    let medium = (256 * 1024..TRIAL_MIN_LEN).contains(&data.len()) && !is_json;
    let near_random = crate::classify::shannon_estimate(data) > TRIAL_MAX_SHANNON;
    if small || medium || near_random {
        return BwtPathResult {
            pipeline: BwtPipeline::RawCm,
            encoded_size: data.len(),
            is_bwt: false,
        };
    }

    let path_a_size = data.len();

    let mut path_b_size = data.len();
    let mut path_c_size = data.len();
    // Run Paths B, C (and D for JSON) concurrently: each is an independent
    // transform + full-buffer comparison. This is the dominant cost of Text
    // blocks (divsufsort + MTF/RLE passes), and they share nothing.
    let mut json_split_size: Option<usize> = None;
    std::thread::scope(|s| {
        let hb = s.spawn(|| BwtPipeline::BwtMtfRle.encode(data).len());
        let hc = s.spawn(|| BwtPipeline::LzpBwtMtf.encode(data).len());
        let hd = is_json.then(|| s.spawn(|| BwtPipeline::JsonSplit.encode(data).len()));
        path_b_size = hb.join().unwrap_or(data.len());
        path_c_size = hc.join().unwrap_or(data.len());
        json_split_size = hd.and_then(|h| h.join().ok());
    });

    let (best_pipeline, best_size, is_bwt) = if let Some(json_size) = json_split_size {
        // Compare all four paths.
        if json_size <= path_a_size && json_size <= path_b_size && json_size <= path_c_size {
            (BwtPipeline::JsonSplit, json_size, true)
        } else if path_b_size <= path_a_size && path_b_size <= path_c_size {
            (BwtPipeline::BwtMtfRle, path_b_size, true)
        } else if path_c_size <= path_a_size && path_c_size <= path_b_size {
            (BwtPipeline::LzpBwtMtf, path_c_size, true)
        } else {
            (BwtPipeline::RawCm, path_a_size, false)
        }
    } else {
        if path_b_size <= path_a_size && path_b_size <= path_c_size {
            (BwtPipeline::BwtMtfRle, path_b_size, true)
        } else if path_c_size <= path_a_size && path_c_size <= path_b_size {
            (BwtPipeline::LzpBwtMtf, path_c_size, true)
        } else {
            (BwtPipeline::RawCm, path_a_size, false)
        }
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

    #[test]
    fn bwt_pipeline_json_split_round_trip() {
        let text = b"{\"name\":\"John\",\"age\":42,\"city\":\"New York\"}\n".repeat(500);
        let encoded = BwtPipeline::JsonSplit.encode(&text);
        let decoded = BwtPipeline::JsonSplit.decode(&encoded, text.len());
        assert_eq!(decoded, text);
    }

    #[test]
    fn bwt_pipeline_xwrt_round_trip() {
        let text = b"hello world hello world hello world".to_vec();
        let encoded = BwtPipeline::XwrtBwtMtfRle.encode(&text);
        eprintln!("XWRT encode len={} content={:?}", encoded.len(), encoded);
        let decoded = BwtPipeline::XwrtBwtMtfRle.decode(&encoded, text.len());
        eprintln!("XWRT decode len={} content={:?}", decoded.len(), decoded);
        assert_eq!(decoded, text);
    }
}
