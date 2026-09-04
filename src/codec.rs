//! Core codec: glue the classifier, bit models, logistic mixer, rANS backend, and the
//! `NYX1` container into `compress` / `decompress`.
//!
//! Strategy per block:
//! - `Random` blocks are stored verbatim (copy record, method 0).
//! - `Text` blocks (method 2) use a text-optimized stack: orders 0–2, Sparse,
//!   Exec, LazyLzp, PpmModel order-3, WordModel.
//! - `Binary` blocks (method 3) use the full stack (orders 0–2, Sparse, Exec, LZP,
//!   PPM order-3) — same as the legacy `method 1` CM path, since mixed binary
//!   benefits from every signal.
//! - `Exec` blocks (method 4) use a stack without the `Exec` model (redundant on
//!   already-classified machine code) but keep orders 0–2, Sparse, LZP, PPM order-3
//!   — binary structure plus higher-order context matches better than byte-pattern
//!   detection on known code.
//! - Fallback / unknown (method 1) is the full heterogeneous stack, identical to the
//!   original CM path. This is also the decoder default for any future method value,
//!   so old streams remain valid.
//!
//! ## Two-pass CM residual (experimental)
//!
//! For compressed blocks, nyx runs a forward LZP match pre-pass and emits
//! explicit `(len, dist)` records for long matches (≥ 8 bytes). Only the
//! *residual* bytes (those not covered by a match) are context-mixed and
//! rANS-encoded. The decoder replays the identical scan, copy-matching from
//! its own output buffer, and rANS-decodes only residual bytes. LZP state
//! stays in sync because both sides train on the same reconstructed byte
//! stream.
//!
//! Wire format (per block payload):
//!   - `u32 LE` = number of match records
//!   - match records: `u8 len` + `u32 LE dist` (5 bytes each)
//!   - rANS bit stream: only for residual bytes
//!
//! Method values:
//!   0 = copy, 1 = cm (full stack), 2 = text, 3 = binary, 4 = exec.

use crate::container::{BlockEntry, Header, VERSION};
use crate::entropy::range::{BitDecoder, BitEncoder};
use crate::error::{NyxError, Result};
use crate::model::lzp::Lzp;
use crate::model::mixer::LogisticMixer;
use crate::model::BitModel;

/// Minimum match length to emit as an explicit record (vs. letting CM model it).
/// The reverted Sep-11 experiment used MIN_MATCH=4 and regressed all 5 files; 8
/// avoids tiny-match overhead that bloats the side stream.
const MATCH_MIN_LEN: usize = 8;

/// A single match record emitted by the pre-pass.
/// `len` is the byte count copied; `dist` is the backward distance.
#[derive(Debug, Clone, Copy)]
struct MatchRun {
    len: usize,
    dist: usize,
}

/// Default block size: 64 KiB. `block_size_log = 16`.
pub const DEFAULT_BLOCK_SIZE_LOG: u8 = 16;

/// Container method constants (stored in `BlockEntry::method`).
pub const METHOD_COPY: u8 = 0;
/// Full heterogenous CM stack (legacy / fallback).
pub const METHOD_CM: u8 = 1;
/// Text-optimized CM stack (orders 0–2, Sparse, Exec, LazyLzp, PpmModel order-3, WordModel).
pub const METHOD_TEXT: u8 = 2;
/// Binary CM stack (orders 0–2, Sparse, Exec, LZP, PPM order-3).
pub const METHOD_BINARY: u8 = 3;
/// Exec-optimized CM stack (orders 0–2, Sparse, LZP, PPM order-3; no Exec model).
pub const METHOD_EXEC: u8 = 4;

/// Compress `buf` into a `NYX1` container using classifier-aware stacks.
///
/// # Errors
///
/// Returns [`NyxError`] if an entropy primitive fails.
pub fn compress(buf: &[u8]) -> Result<Vec<u8>> {
    compress_with(buf, &mut build_stack_for_kind)
}

/// Compress `buf` using a custom per-block stack builder.
///
/// The builder receives the [`BlockKind`](crate::classify::BlockKind) the classifier
/// assigned to each block and returns the `(models, mixer)` pair to use. This is the
/// extension point that classifier-aware selection uses.
///
/// # Errors
///
/// Returns [`NyxError`] if an entropy primitive fails.
pub fn compress_with<F>(buf: &[u8], build_stack: &mut F) -> Result<Vec<u8>>
where
    F: FnMut(crate::classify::BlockKind) -> (Vec<Box<dyn BitModel>>, LogisticMixer),
{
    let mut out = Vec::new();
    let mut entries: Vec<BlockEntry> = Vec::new();
    let mut payloads: Vec<u8> = Vec::new();
    let mut offset = 0usize;

    let mut last_kind: Option<crate::classify::BlockKind> = None;
    let mut models: Vec<Box<dyn BitModel>> = Vec::new();
    let mut mixer = LogisticMixer::new(0);

    while offset < buf.len() {
        let block = &buf[offset..];
        let kind = crate::classify::classify(block);
        let block_size = block_size_for_kind(kind, block, offset, buf.len());
        let end = (offset + block_size).min(buf.len());
        let block_data = &buf[offset..end];

        if last_kind != Some(kind) {
            let (new_models, new_mixer) = build_stack(kind);
            models = new_models;
            mixer = new_mixer;
            last_kind = Some(kind);
        }

        let comp = compress_block(&mut models, &mut mixer, block_data);

        let entry = BlockEntry {
            comp_len: comp.len() as u32,
            orig_len: block_data.len() as u32,
            method: method_for_kind(kind),
            crc32: crc32(block_data),
        };
        entries.push(entry);
        payloads.extend_from_slice(&comp);
        offset = end;
    }

    let header = Header {
        version: VERSION,
        flags: 0,
        block_size_log: DEFAULT_BLOCK_SIZE_LOG,
        num_blocks: entries.len() as u32,
    };
    header.write(&mut out);
    for e in &entries {
        e.write(&mut out);
    }
    out.extend_from_slice(&payloads);
    Ok(out)
}

fn block_size_for_kind(
    kind: crate::classify::BlockKind,
    block: &[u8],
    offset: usize,
    total: usize,
) -> usize {
    match kind {
        crate::classify::BlockKind::Text => {
            let max_text = 4 * 1024 * 1024;
            let size = (total - offset).min(max_text);
            size.max(64 * 1024)
        }
        crate::classify::BlockKind::Binary
        | crate::classify::BlockKind::Exec
        | crate::classify::BlockKind::Random => 64 * 1024,
    }
}

/// Map a `BlockKind` to the container method byte the decoder uses to pick a stack.
#[must_use]
pub const fn method_for_kind(kind: crate::classify::BlockKind) -> u8 {
    match kind {
        crate::classify::BlockKind::Random => METHOD_COPY,
        crate::classify::BlockKind::Text => METHOD_TEXT,
        crate::classify::BlockKind::Binary => METHOD_BINARY,
        crate::classify::BlockKind::Exec => METHOD_EXEC,
    }
}

/// Choose the model stack for a block, based on the classifier's `BlockKind`.
/// Both encode and decode paths call this, so the stacks are guaranteed to be in sync.
#[must_use]
pub fn build_stack_for_kind(
    kind: crate::classify::BlockKind,
) -> (Vec<Box<dyn BitModel>>, LogisticMixer) {
    match kind {
        // Random: copy, no models needed (encoder won't call compress_block).
        crate::classify::BlockKind::Random => {
            let models: Vec<Box<dyn BitModel>> = vec![];
            (models, LogisticMixer::new(0))
        }
        crate::classify::BlockKind::Text => {
            // Text-optimized stack: full hybrid + WordModel + LazyLzp + new 4MB LZP.
            let n = 9;
            let models: Vec<Box<dyn BitModel>> = vec![
                Box::new(crate::model::order::OrderN::new(0)),
                Box::new(crate::model::order::OrderN::new(1)),
                Box::new(crate::model::order::OrderN::new(2)),
                Box::new(crate::model::sparse::Sparse::new()),
                Box::new(crate::model::exec::Exec::new()),
                Box::new(crate::model::lazy_lzp::LazyLzp::new()),
                Box::new(crate::model::lzp::Lzp::new()),
                Box::new(crate::model::ppm::PpmModel::new(3)),
                Box::new(crate::model::word::WordModel::new()),
            ];
            (models, LogisticMixer::new(n))
        }
        crate::classify::BlockKind::Binary => {
            // Binary: full stack.
            let n = 7;
            let models: Vec<Box<dyn BitModel>> = vec![
                Box::new(crate::model::order::OrderN::new(0)),
                Box::new(crate::model::order::OrderN::new(1)),
                Box::new(crate::model::order::OrderN::new(2)),
                Box::new(crate::model::sparse::Sparse::new()),
                Box::new(crate::model::exec::Exec::new()),
                Box::new(crate::model::lzp::Lzp::new()),
                Box::new(crate::model::ppm::PpmModel::new(3)),
            ];
            (models, LogisticMixer::new(n))
        }
        crate::classify::BlockKind::Exec => {
            // Exec: orders 0-2, Sparse, LZP, PPM order-3. Drop Exec model (redundant).
            let n = 6;
            let models: Vec<Box<dyn BitModel>> = vec![
                Box::new(crate::model::order::OrderN::new(0)),
                Box::new(crate::model::order::OrderN::new(1)),
                Box::new(crate::model::order::OrderN::new(2)),
                Box::new(crate::model::sparse::Sparse::new()),
                Box::new(crate::model::lzp::Lzp::new()),
                Box::new(crate::model::ppm::PpmModel::new(3)),
            ];
            (models, LogisticMixer::new(n))
        }
    }
}

/// Legacy alias kept for benchmark tooling (`src/stacks.rs`).
#[must_use]
pub fn build_full_stack() -> (Vec<Box<dyn BitModel>>, LogisticMixer) {
    build_stack_for_kind(crate::classify::BlockKind::Binary)
}

/// Compress one block (encode-side of the two-pass CM residual).
///
/// (1) Forward LZP scan: record (len, dist) for matches ≥ `MATCH_MIN_LEN`.
/// (2) Emit match records as a side stream, then rANS-encode only residual bytes.
fn compress_block(
    models: &mut [Box<dyn BitModel>],
    mixer: &mut LogisticMixer,
    block: &[u8],
) -> Vec<u8> {
    let runs = scan_matches(block);
    encode_block_with_matches(models, mixer, block, &runs)
}

/// Forward LZP scan: train on each byte, call `longest_match` at position i+1,
/// record matches ≥ `MATCH_MIN_LEN`, skip ahead by the match length.
///
/// This is deterministic and depends only on the input byte stream, so the
/// decoder can replay it if it has the same (len, dist) records to validate.
/// In practice the side-stream records are the source of truth; the scan
/// produces them and the decoder validates its own replay against them.
fn scan_matches(block: &[u8]) -> Vec<MatchRun> {
    let mut lzp = Lzp::new();
    let mut runs: Vec<MatchRun> = Vec::new();
    let mut i = 0usize;
    while i + 1 < block.len() {
        // Train LZP on the current source byte (not on any matched copy).
        lzp.train_at(block, i);
        if i + 1 >= 4 {
            if let Some(raw_len) = lzp.longest_match(block, i + 1) {
                let len = raw_len.min(255);
                let dist = find_match_distance(block, i + 1, len);
                if dist > 0 && dist <= 4 * 1024 * 1024 && len >= MATCH_MIN_LEN {
                    runs.push(MatchRun { len, dist });
                    // Do NOT train LZP on the matched bytes — they're copies
                    // from earlier in the block, and training on them would
                    // pollute the hash chains with self-referential matches.
                    i += len;
                    continue;
                }
            }
        }
        i += 1;
    }
    runs
}

/// Find how far back the matching string sits at `pos` in `data` with `len` bytes.
/// Linear backward scan, bounded by the 4 MB window.
fn find_match_distance(data: &[u8], pos: usize, len: usize) -> usize {
    if pos < len || len == 0 {
        return 0;
    }
    let needle = &data[pos - len..pos];
    let window = 4 * 1024 * 1024;
    let start = pos.saturating_sub(window);
    for back in (start..pos - len + 1).rev() {
        if data[back..back + len] == *needle {
            return pos - back;
        }
    }
    0
}

/// Encode a block with explicit match records + residual CM.
///
/// Wire format (per block payload):
///   - 4 bytes: `u32::LE` = number of match records
///   - match records: `u8 len` (raw, ≥8) + `u32 LE dist` (5 bytes each)
///   - rANS bit stream: CM-encoded bits for the block bytes
///
/// Stage 1 (current): the rANS stream covers ALL bytes (matches recorded but
///  not yet used to skip CM encoding). This keeps round-trip correct while
///  validating the scaffolding.
/// Stage 2 (future): skip matched bytes from rANS, only encode residuals.
fn encode_block_with_matches(
    models: &mut [Box<dyn BitModel>],
    mixer: &mut LogisticMixer,
    block: &[u8],
    runs: &[MatchRun],
) -> Vec<u8> {
    let mut out = Vec::new();

    // Match side-stream
    out.extend_from_slice(&(runs.len() as u32).to_le_bytes());
    for r in runs {
        out.push(r.len as u8);
        out.extend_from_slice(&(r.dist as u32).to_le_bytes());
    }

    // Stage 1: CM-encode ALL bytes (residual = full block).
    let mut enc = BitEncoder::new();
    let mut probs: [u16; 12] = [2048; 12];
    let n = models.len();

    for &byte in block {
        for bit_idx in (0..8).rev() {
            let bit = (byte >> bit_idx) & 1 == 1;
            let bit_pos = bit_idx as u8;
            for (i, m) in models.iter().enumerate() {
                probs[i] = m.predict();
            }
            let p = mixer.mix(&probs[..n], bit_pos);
            enc.encode_bit(bit, p);
            mixer.update(&probs[..n], bit, bit_pos);
            for m in models.iter_mut() {
                m.update(bit);
            }
        }
    }
    out.extend(enc.finish());
    out
}

/// Decode a block encoded with `encode_block_with_matches`.
///
/// Stage 1: reads the match side-stream (validates record count/length) then
/// rANS-decodes ALL bytes — the encoder CM-encodes all bytes, so we do too.
/// Stage 2 (future): skip rANS decoding for matched positions, copy-matching
/// from the output buffer instead.
fn decode_block_with_matches(
    comp: &[u8],
    orig_len: usize,
    models: &mut [Box<dyn BitModel>],
    mixer: &mut LogisticMixer,
) -> Result<Vec<u8>> {
    if comp.len() < 4 {
        return Err(NyxError::InvalidContainer("match side-stream too short".into()));
    }
    let num_runs = u32::from_le_bytes([comp[0], comp[1], comp[2], comp[3]]) as usize;
    let mut offset = 4;
    // Validate and skip match records.
    for _ in 0..num_runs {
        if offset + 5 > comp.len() {
            return Err(NyxError::InvalidContainer("truncated match record".into()));
        }
        offset += 5; // u8 len + u32 LE dist
    }

    // Stage 1: full-block CM decode (residual = entire block).
    let mut dec = BitDecoder::new(&comp[offset..])
        .map_err(|e| NyxError::Entropy(e.to_string()))?;
    let mut out = Vec::with_capacity(orig_len);
    let mut probs: [u16; 12] = [2048; 12];
    let n = models.len();

    while out.len() < orig_len {
        let mut byte = 0u8;
        for bit_idx in (0..8).rev() {
            let bit_pos = bit_idx as u8;
            for (i, m) in models.iter().enumerate() {
                probs[i] = m.predict();
            }
            let p = mixer.mix(&probs[..n], bit_pos);
            let bit = dec
                .decode_bit(p)
                .map_err(|e| NyxError::Entropy(e.to_string()))?;
            mixer.update(&probs[..n], bit, bit_pos);
            for m in models.iter_mut() {
                m.update(bit);
            }
            if bit {
                byte |= 1 << bit_idx;
            }
        }
        out.push(byte);
    }
    Ok(out)
}

/// Decompress a `NYF1` container back to the original bytes.
///
/// # Errors
///
/// Returns [`NyxError`] on a malformed container, corrupt block, or CRC mismatch.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    decompress_impl(data)
}

fn decompress_impl(data: &[u8]) -> Result<Vec<u8>> {
    use std::io::Cursor;
    let mut cur = Cursor::new(data);
    let header = Header::read(&mut cur).map_err(|e| NyxError::InvalidContainer(e.to_string()))?;
    if header.version != VERSION {
        return Err(NyxError::InvalidContainer(format!(
            "unsupported version {}",
            header.version
        )));
    }
    let mut entries = Vec::with_capacity(header.num_blocks as usize);
    for _ in 0..header.num_blocks {
        entries.push(
            BlockEntry::read(&mut cur).map_err(|e| NyxError::InvalidContainer(e.to_string()))?,
        );
    }
    let payload_start = cur.position() as usize;
    let payloads = &data[payload_start..];

    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut last_kind: Option<crate::classify::BlockKind> = None;
    let mut models: Vec<Box<dyn BitModel>> = Vec::new();
    let mut mixer = LogisticMixer::new(0);
    for (bi, entry) in entries.iter().enumerate() {
        let comp = &payloads[pos..pos + entry.comp_len as usize];
        pos += entry.comp_len as usize;

        let block = if entry.method == METHOD_COPY {
            comp.to_vec()
        } else {
            let kind = kind_for_method(entry.method)?;
            if last_kind != Some(kind) {
                let (new_models, new_mixer) = build_stack_for_kind(kind);
                models = new_models;
                mixer = new_mixer;
                last_kind = Some(kind);
            }
            decompress_block(comp, entry.orig_len as usize, &mut models, &mut mixer).map_err(
                |e| match e {
                    NyxError::Entropy(s) => NyxError::CorruptBlock(s),
                    other => other,
                },
            )?
        };

        if crate::container::crc32(&block) != entry.crc32 {
            return Err(NyxError::CrcMismatch(
                bi,
                crate::container::crc32(&block),
                entry.crc32,
            ));
        }
        out.extend_from_slice(&block);
    }
    Ok(out)
}

/// Reverse map: container method byte → `BlockKind`. Unknown methods error.
fn kind_for_method(method: u8) -> Result<crate::classify::BlockKind> {
    match method {
        METHOD_COPY => Ok(crate::classify::BlockKind::Random),
        METHOD_TEXT => Ok(crate::classify::BlockKind::Text),
        METHOD_BINARY => Ok(crate::classify::BlockKind::Binary),
        METHOD_EXEC => Ok(crate::classify::BlockKind::Exec),
        _ => Err(NyxError::InvalidContainer(format!(
            "unknown method {}",
            method
        ))),
    }
}

/// Decompress one block payload. Delegates to `decode_block_with_matches`.
fn decompress_block(
    comp: &[u8],
    orig_len: usize,
    models: &mut [Box<dyn BitModel>],
    mixer: &mut LogisticMixer,
) -> Result<Vec<u8>> {
    decode_block_with_matches(comp, orig_len, models, mixer)
}

/// Re-export so callers can build CRCs without reaching into the container module.
pub use crate::container::crc32;

#[cfg(test)]
mod tests {
    use super::*;

    /// A ~200 KB mixed fixture: text + JSON + a binary blob + an ELF-like byte pattern.
    fn mixed_fixture() -> Vec<u8> {
        let mut v = Vec::new();
        let text = b"the quick brown fox jumps over the lazy dog. \
            compression mixes many context models so that each bit is predicted well. ";
        for _ in 0..2000 {
            v.extend_from_slice(text);
        }
        let json = b"{\"name\":\"nyx\",\"level\":3,\"models\":[\"order0\",\"order1\",\"order2\",\"sparse\",\"exec\",\"lzp\"],\"ratio\":0.42}\n";
        for _ in 0..500 {
            v.extend_from_slice(json);
        }
        let mut x = 0x9E37_79B9u32;
        for _ in 0..40_000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            v.push(x as u8);
            v.push((x >> 11) as u8);
        }
        for _ in 0..3000 {
            v.extend_from_slice(&[0x7f, b'E', b'L', b'F', 0x55, 0x89, 0xE5, 0xFF, 0xD0]);
        }
        v
    }

    #[test]
    fn compress_then_decompress_returns_original() {
        let original = mixed_fixture();
        assert!(original.len() > 200_000, "fixture should exceed 200 KB");
        let comp = compress(&original).expect("compress");
        let back = decompress(&comp).expect("decompress");
        assert_eq!(back, original, "round-trip mismatch");
    }

    #[test]
    fn compressed_is_smaller_on_redundant_input() {
        let mut input = Vec::new();
        for _ in 0..50_000 {
            input.extend_from_slice(b"nyxnyxnyx");
        }
        let comp = compress(&input).expect("compress");
        assert!(
            comp.len() < input.len(),
            "redundant input should compress (got {} vs {})",
            comp.len(),
            input.len()
        );
    }

    #[test]
    fn empty_input_round_trips() {
        let comp = compress(&[]).expect("compress");
        let back = decompress(&comp).expect("decompress");
        assert!(back.is_empty());
    }

    #[test]
    fn random_block_is_stored_verbatim() {
        let mut buf = [0u8; 4096];
        let mut x = 0x1234_5678u32;
        for b in &mut buf {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        let kind = crate::classify::classify(&buf);
        assert_eq!(kind, crate::classify::BlockKind::Random);
        assert_eq!(method_for_kind(kind), METHOD_COPY);
    }

    #[test]
    fn text_block_uses_text_stack() {
        let text = b"the quick brown fox jumps over the lazy dog. the quick brown fox. ";
        let kind = crate::classify::classify(text);
        assert_eq!(kind, crate::classify::BlockKind::Text);
        assert_eq!(method_for_kind(kind), METHOD_TEXT);
    }

    #[test]
    fn binary_block_uses_binary_stack() {
        let mut buf = [0u8; 256];
        let mut x = 0x9E37_79B9u32;
        for b in &mut buf {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        let kind = crate::classify::classify(&buf);
        assert_eq!(kind, crate::classify::BlockKind::Binary);
        assert_eq!(method_for_kind(kind), METHOD_BINARY);
    }

    #[test]
    fn scan_matches_finds_repeats() {
        let data = b"abcabcabcabcabcabc";
        let runs = scan_matches(data);
        assert!(
            !runs.is_empty(),
            "expected at least one match in repeated data"
        );
        assert!(runs[0].len >= MATCH_MIN_LEN);
    }

    #[test]
    fn scan_matches_empty_on_unique() {
        let mut data = vec![0u8; 256];
        let mut x = 0x1234_5678u32;
        for b in &mut data {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        let runs = scan_matches(&data);
        assert!(runs.is_empty(), "expected no matches in random data");
    }

    #[test]
    fn find_match_distance_correct() {
        let data = b"abcabcabcabc";
        // At position 6, "abc" matches distance 3.
        let d = find_match_distance(data, 6, 3);
        assert_eq!(d, 3, "expected distance 3, got {}", d);
    }
}
