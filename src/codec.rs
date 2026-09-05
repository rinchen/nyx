//! Core codec: glue the classifier, bit models, logistic mixer, rANS backend, and the
//! `NYX1` container into `compress` / `decompress`.
//!
//! Strategy per block:
//! - `Random` blocks are stored verbatim (copy record, method 0).
//! - `Text` blocks (method 2) use a text-optimized stack: orders 0–2, Sparse,
//!   Exec, LazyLzp, PpmModel order-3, WordModel.
//! - `Binary` blocks (method 3) use the full stack (orders 0–2, Sparse, Exec,
//!   LZP, PPM order-3) — same as the legacy `method 1` CM path, since mixed binary
//!   benefits from every signal.
//! - `Exec` blocks (method 4) use a stack without the `Exec` model (redundant on
//!   already-classified machine code) but keep orders 0–2, Sparse, LZP, PPM order-3.
//! - Fallback / unknown (method 1) is the full heterogeneous stack, identical to the
//!   original CM path. This is also the decoder default for any future method value,
//!   so old streams remain valid.
//!
//! ## Two-pass CM residual (experimental, behind `two_pass` feature)
//!
//! When the `two_pass` feature is enabled, nyx runs a forward LZP match pre-pass
//! and emits explicit `(len, dist)` records for long matches (≥ 8 bytes).
//! The `Two-pass CM residual` feature adds an SsmMixer (8-dim Mamba-style state-space
//! model) and a Byte-Pair Re-Pair dictionary to the word model as additional base
//! models. NOTE: measured on the 5-file Silesia subset, the SSM + Re-Pair + match
//! side-stream combination caused a net regression (nci 20.9%→33.3%, webster 45.1%→57.6%,
//! etc.), so it is behind a feature flag and off by default.
//!
//! Method values:
//!   0 = copy, 1 = cm (full stack), 2 = text, 3 = binary, 4 = exec.

use crate::container::{BlockEntry, Header, VERSION};
use crate::entropy::range::{BitDecoder, BitEncoder};
use crate::error::{NyxError, Result};
use crate::model::mixer::LogisticMixer;
use crate::model::BitModel;

#[cfg(feature = "two_pass")]
use crate::model::lzp::Lzp;

#[cfg(feature = "two_pass")]
use crate::model::ssm::SsmMixer;

#[cfg(feature = "two_pass")]
const MATCH_MIN_LEN: usize = 8;

#[cfg(feature = "two_pass")]
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
            #[cfg(feature = "two_pass")]
            {
                // Text-optimized stack WITH SSM + Re-Pair word model (experimental).
                let n = 10;
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
                    Box::new(crate::model::ssm::SsmMixer::new()),
                ];
                (models, LogisticMixer::new(n))
            }
            #[cfg(not(feature = "two_pass"))]
            {
                // Text-optimized stack (best configuration, no SSM/Re-Pair).
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
        }
        crate::classify::BlockKind::Binary => {
            #[cfg(feature = "two_pass")]
            {
                // Binary stack WITH SSM (experimental).
                let n = 8;
                let models: Vec<Box<dyn BitModel>> = vec![
                    Box::new(crate::model::order::OrderN::new(0)),
                    Box::new(crate::model::order::OrderN::new(1)),
                    Box::new(crate::model::order::OrderN::new(2)),
                    Box::new(crate::model::sparse::Sparse::new()),
                    Box::new(crate::model::exec::Exec::new()),
                    Box::new(crate::model::lzp::Lzp::new()),
                    Box::new(crate::model::ppm::PpmModel::new(3)),
                    Box::new(crate::model::ssm::SsmMixer::new()),
                ];
                (models, LogisticMixer::new(n))
            }
            #[cfg(not(feature = "two_pass"))]
            {
                // Binary stack (best configuration, no SSM).
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
        }
        crate::classify::BlockKind::Exec => {
            #[cfg(feature = "two_pass")]
            {
                // Exec stack WITH SSM (experimental).
                let n = 7;
                let models: Vec<Box<dyn BitModel>> = vec![
                    Box::new(crate::model::order::OrderN::new(0)),
                    Box::new(crate::model::order::OrderN::new(1)),
                    Box::new(crate::model::order::OrderN::new(2)),
                    Box::new(crate::model::sparse::Sparse::new()),
                    Box::new(crate::model::lzp::Lzp::new()),
                    Box::new(crate::model::ppm::PpmModel::new(3)),
                    Box::new(crate::model::ssm::SsmMixer::new()),
                ];
                (models, LogisticMixer::new(n))
            }
            #[cfg(not(feature = "two_pass"))]
            {
                // Exec stack (best configuration, no SSM).
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
}

/// Legacy alias kept for benchmark tooling (`src/stacks.rs`).
#[must_use]
pub fn build_full_stack() -> (Vec<Box<dyn BitModel>>, LogisticMixer) {
    build_stack_for_kind(crate::classify::BlockKind::Binary)
}

/// Compress one block.
///
/// With `two_pass` feature: runs a match pre-pass, emits (len, dist) side-stream, then
/// rANS-encodes ALL bytes (Stage 1 — match records present but not yet used for residual
/// skipping, since Stage 2 decoder is blocked on state synchronization).
/// Without `two_pass`: plain CM encoding of all bytes.
fn compress_block(
    models: &mut [Box<dyn BitModel>],
    mixer: &mut LogisticMixer,
    block: &[u8],
) -> Vec<u8> {
    #[cfg(feature = "two_pass")]
    {
        let runs = scan_matches(block);
        encode_block_with_matches(models, mixer, block, &runs)
    }
    #[cfg(not(feature = "two_pass"))]
    {
        encode_block_plain(models, mixer, block)
    }
}

/// Plain CM encoding (no match side-stream).
fn encode_block_plain(
    models: &mut [Box<dyn BitModel>],
    mixer: &mut LogisticMixer,
    block: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();

    // Pre-build per-block dictionaries.
    for m in models.iter_mut() {
        m.prepare_block(block);
    }

    let mut enc = BitEncoder::new();
    let mut probs: [u16; 12] = [2048; 12];
    let n = models.len();

    for &byte in block {
        for bit_idx in (0..8).rev() {
            let bit = (byte >> bit_idx) & 1u8 == 1u8;
            let bit_pos = bit_idx as u8;
            for (j, m) in models.iter().enumerate() {
                probs[j] = m.predict();
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

#[cfg(feature = "two_pass")]
fn encode_block_with_matches(
    models: &mut [Box<dyn BitModel>],
    mixer: &mut LogisticMixer,
    block: &[u8],
    runs: &[MatchRun],
) -> Vec<u8> {
    let mut out = Vec::new();

    for m in models.iter_mut() {
        m.prepare_block(block);
    }

    // Match side-stream
    out.extend_from_slice(&(runs.len() as u32).to_le_bytes());
    for r in runs {
        out.push(r.len as u8);
        out.extend_from_slice(&(r.dist as u32).to_le_bytes());
    }

    let mut enc = BitEncoder::new();
    let mut probs: [u16; 12] = [2048; 12];
    let n = models.len();

    for &byte in block {
        for bit_idx in (0..8).rev() {
            let bit = (byte >> bit_idx) & 1u8 == 1u8;
            let bit_pos = bit_idx as u8;
            for (j, m) in models.iter().enumerate() {
                probs[j] = m.predict();
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

#[cfg(feature = "two_pass")]
fn scan_matches(block: &[u8]) -> Vec<MatchRun> {
    let mut lzp = Lzp::new();
    let mut runs: Vec<MatchRun> = Vec::new();
    let mut i = 0usize;
    while i + 1 < block.len() {
        lzp.train_at(block, i);
        if i + 1 >= 16 {
            if let Some(raw_len) = lzp.longest_match(block, i + 1) {
                let len = raw_len.min(255);
                let dist = find_match_distance(block, i + 1, len);
                if dist > 0 && dist <= 4 * 1024 * 1024 && len >= MATCH_MIN_LEN {
                    runs.push(MatchRun { len, dist });
                    i += len;
                    continue;
                }
            }
        }
        i += 1;
    }
    runs
}

#[cfg(feature = "two_pass")]
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

/// Decode a block.
///
/// With `two_pass`: reads match side-stream (validates records), then rANS-decodes all bytes.
/// Without `two_pass`: plain CM decode of all bytes.
fn decode_block(
    comp: &[u8],
    orig_len: usize,
    models: &mut [Box<dyn BitModel>],
    mixer: &mut LogisticMixer,
) -> Result<Vec<u8>> {
    #[cfg(feature = "two_pass")]
    {
        decode_block_with_matches(comp, orig_len, models, mixer)
    }
    #[cfg(not(feature = "two_pass"))]
    {
        decode_block_plain(comp, orig_len, models, mixer)
    }
}

fn decode_block_plain(
    comp: &[u8],
    orig_len: usize,
    models: &mut [Box<dyn BitModel>],
    mixer: &mut LogisticMixer,
) -> Result<Vec<u8>> {
    let mut dec = BitDecoder::new(comp)
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

#[cfg(feature = "two_pass")]
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
    for _ in 0..num_runs {
        if offset + 5 > comp.len() {
            return Err(NyxError::InvalidContainer("truncated match record".into()));
        }
        offset += 5;
    }

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
            decode_block(comp, entry.orig_len as usize, &mut models, &mut mixer).map_err(
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
    fn json_round_trips() {
        let json = b"{\"name\":\"nyx\",\"level\":3,\"models\":[\"order0\",\"order1\",\"order2\",\"sparse\",\"exec\",\"lzp\"],\"ratio\":0.42}\n";
        let original: Vec<u8> = std::iter::repeat(json.as_ref())
            .take(4000)
            .flatten()
            .copied()
            .collect();
        let comp = compress(&original).expect("compress");
        let back = decompress(&comp).expect("decompress");
        assert_eq!(back, original, "JSON round-trip mismatch");
    }

    #[cfg(feature = "two_pass")]
    #[test]
    fn scan_matches_finds_repeats() {
        let data = b"abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";
        let runs = scan_matches(data);
        assert!(
            !runs.is_empty(),
            "expected at least one match in repeated data"
        );
        assert!(runs[0].len >= MATCH_MIN_LEN);
    }

    #[cfg(feature = "two_pass")]
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

    #[cfg(feature = "two_pass")]
    #[test]
    fn find_match_distance_correct() {
        let data = b"abcabcabcabc";
        let d = find_match_distance(data, 6, 3);
        assert_eq!(d, 3, "expected distance 3, got {}", d);
    }
}
