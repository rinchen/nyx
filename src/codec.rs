//! Core codec: glue the classifier, bit models, two-level logistic mixer, rANS backend,
//! and the `RCN1` container into `compress` / `decompress`.
//!
//! ## Two-level mixer hierarchy
//!
//! The mixer uses a CMIX/PAQ8-style three-layer stack:
//!
//! 1. **Bank mixers** (16384 instances): selected by a context hash of
//!    order-1 / order-2 / word-context + bit position. Each bank specializes
//!    its weights to a specific context, avoiding the ~50% saturation that a
//!    single logistic mixer hits on repetitive corpora like dickens.
//! 2. **Global mixer**: a single context-agnostic mixer over the same models.
//! 3. **Master mixer**: blends `[p_bank, p_global, p_lzp_conf]` in logistic space.
//!
//! Only the selected bank + global + master are trained per bit — never all 16384.
//! At block boundaries, weights are **decayed** (not reset), preserving learned
//! structure across the stream.
//!
//! Strategy per block:
//! - `Random` blocks are stored verbatim (copy record, method 0).
//! - `Text` blocks (method 2) use a text-optimized stack: orders 0–2, Sparse,
//!   Exec, Lzp, PpmModel order-3, WordModel, IndirectModel.
//! - `Binary` blocks (method 3) use the full stack (orders 0–2, Sparse, Exec,
//!   LZP, PPM order-3) — same as the legacy `method 1` CM path, since mixed binary
//!   benefits from every signal.
//! - `Exec` blocks (method 4) use a stack without the `Exec` model (redundant on
//!   already-classified machine code) but keep orders 0–2, Sparse, LZP, PPM order-3.
//! - Fallback / unknown (method 1) is the full heterogeneous stack, identical to the
//!   original CM path. This is also the decoder default for any future method value,
//!   so old streams remain valid.
//!
//! ## DP-optimal LZP match pre-pass (default)
//!
//! rcn runs a forward LZP match pre-pass
//! with **DP optimal parsing** and emits explicit `(len, dist)` records for long
//! matches (adaptive min length: Text ≥12, Binary/Exec ≥24, Random/other ≥16).
//! Matched bytes are **skipped** in the rANS stream —
//! only literals (non-matched bytes) are CM-encoded. The decoder reconstructs
//! matched bytes by copying from history.
//!
//! Match selection cost = bits(match_flag) + bits(len) + bits(dist) + residual_cost,
//! where residual_cost is estimated as len × AVG_BITS_PER_BYTE (CM cost per byte).
//! A match is taken when its overhead (33 bits) is less than the CM cost of the
//! matched bytes (len × 4 bits), i.e., len > 8.25. With min length ≥12, matches
//! always save net bits.
//!
//! Method values:
//!   0 = copy, 1 = cm (full stack), 2 = text, 3 = binary, 4 = exec.

use crate::bwt::{self};
use crate::container::{
    read_global_dict, write_global_dict, BlockEntry, Header, FLAG_GLOBAL_DICT, VERSION,
};
use crate::entropy::range::{BitDecoder, BitEncoder};
use crate::entropy::side_fse;
use crate::error::{RcnError, Result};
use crate::model::lzp::Lzp;
use crate::model::mixer_bank::MixerBank;
use crate::model::sse_apm::SseApmCascade;
use crate::model::word::XwrtDictionary;
use crate::model::BitModel;

/// Varint encoding: read/write unsigned LEB128. Most values in the match
/// side-stream (pos deltas, lengths, distances) are small, so varint
/// encoding saves space over fixed-width fields.
fn read_varint(data: &[u8], mut offset: usize) -> Option<(usize, usize)> {
    let mut result: usize = 0;
    let mut shift: usize = 0;
    while offset < data.len() {
        let byte = data[offset];
        result |= ((byte & 0x7F) as usize) << shift;
        offset += 1;
        if (byte & 0x80) == 0 {
            return Some((result, offset));
        }
        shift += 7;
        // Safety: prevent infinite loop on malformed data.
        if shift > 35 {
            return None;
        }
    }
    None
}

fn write_varint(out: &mut Vec<u8>, mut value: usize) {
    loop {
        if value < 0x80 {
            out.push(value as u8);
            return;
        }
        out.push(((value & 0x7F) | 0x80) as u8);
        value >>= 7;
    }
}

/// Adaptive DP-LZP minimum match length by block kind.
/// Text uses a lower gate (more matches); Binary/Exec keep 16 after a ≥24
/// trial regressed `mr` (~20.6%→27.3%).
#[inline]
fn match_min_len(kind: crate::classify::BlockKind) -> usize {
    match kind {
        crate::classify::BlockKind::Text => 12,
        crate::classify::BlockKind::Binary
        | crate::classify::BlockKind::Exec
        | crate::classify::BlockKind::Random => 16,
    }
}

#[derive(Debug, Clone, Copy)]
struct MatchRun {
    pos: usize,
    len: usize,
    dist: usize,
}

/// Default block size: 64 KiB. `block_size_log = 16`.
pub const DEFAULT_BLOCK_SIZE_LOG: u8 = 16;

/// Container method constants (stored in `BlockEntry::method`).
pub const METHOD_COPY: u8 = 0;
/// Full heterogenous CM stack (legacy / fallback).
pub const METHOD_CM: u8 = 1;
/// Text-optimized CM stack (orders 0–2, Sparse, Exec, Lzp, PpmModel order-3, WordModel).
pub const METHOD_TEXT: u8 = 2;
/// Binary CM stack (orders 0–2, Sparse, Exec, LZP, PPM order-3).
pub const METHOD_BINARY: u8 = 3;
/// Exec-optimized CM stack (orders 0–2, Sparse, LZP, PPM order-3; no Exec model).
pub const METHOD_EXEC: u8 = 4;
/// Text BWT+B path: BWT(1MB) → MTF → RLE0 → CM on MTF ranks.
pub const METHOD_BWT_MTF_RLE: u8 = 5;
/// Text BWT path: LZP(4MB) → BWT → MTF → CM (no RLE0).
pub const METHOD_LZP_BWT_MTF: u8 = 6;
/// Text JSON split path: split JSON into 4 streams → 4× BWT → CM.
pub const METHOD_JSON_SPLIT: u8 = 7;
/// Text XWRT path: XWRT dictionary → BWT → MTF → RLE0 → CM.
pub const METHOD_XWRT_BWT_MTF_RLE: u8 = 13;
/// Byte-level CM (fast path): orders 0–2 count models + byte rANS.
pub const METHOD_BYTE_CM: u8 = 8;
/// Text BWT+RLE0 path, byte-coded (fast).
pub const METHOD_BYTE_BWT_MTF_RLE: u8 = 9;
/// Text LZP→BWT→MTF path, byte-coded (fast).
pub const METHOD_BYTE_LZP_BWT_MTF: u8 = 10;
/// Text JSON split path, byte-coded (fast).
pub const METHOD_BYTE_JSON_SPLIT: u8 = 11;
/// Text XWRT→BWT→MTF→RLE0 path, byte-coded (fast).
pub const METHOD_BYTE_XWRT_BWT_MTF_RLE: u8 = 12;

/// One block's compression summary, reported by [`compress_mode_diag`] for
/// `--verbose` progress output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockDiag {
    /// Classifier label for the block.
    pub kind: String,
    /// Raw [`METHOD_*`] byte written to the container.
    pub method: u8,
    /// Uncompressed size of the block.
    pub size_in: usize,
    /// Compressed size of the block.
    pub size_out: usize,
}

/// Human-readable label for a [`METHOD_*`] byte.
#[must_use]
pub fn method_label(method: u8) -> &'static str {
    match method {
        METHOD_COPY => "COPY",
        METHOD_CM => "CM",
        METHOD_TEXT => "CM(text)",
        METHOD_BINARY => "CM(binary)",
        METHOD_EXEC => "CM(exec)",
        METHOD_BWT_MTF_RLE => "BWT→MTF→RLE0→CM",
        METHOD_LZP_BWT_MTF => "LZP→BWT→MTF→CM",
        METHOD_JSON_SPLIT => "JSON-split→BWT→CM",
        METHOD_XWRT_BWT_MTF_RLE => "XWRT→BWT→MTF→RLE0→CM",
        METHOD_BYTE_CM => "byte-CM",
        METHOD_BYTE_BWT_MTF_RLE => "byte-BWT→MTF→RLE0",
        METHOD_BYTE_LZP_BWT_MTF => "byte-LZP→BWT→MTF",
        METHOD_BYTE_JSON_SPLIT => "byte-JSON-split→BWT",
        METHOD_BYTE_XWRT_BWT_MTF_RLE => "byte-XWRT→BWT→MTF→RLE0",
        _ => "?",
    }
}
/// Exec E8E9 transform, byte-coded (fast).
pub const METHOD_BYTE_EXEC_E8E9: u8 = 14;

/// Encoding strategy for [`compress_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecMode {
    /// Bit-level CM: 8–9 models + two-level bank mixer + bit rANS (current default).
    Slow,
    /// Byte-level CM: orders 0–2 count models + byte rANS.
    Fast,
}

/// Decay factor for cross-block weight persistence. 0.995 keeps 99.5% of learned
/// weight structure per block boundary, smoothly transferring context without
/// hard-clearing (which would defeat the 8k-bank specialization).
const BLOCK_DECAY: f32 = 0.995;

/// Compress `buf` into a `RCN1` container using classifier-aware stacks.
///
/// # Errors
///
/// Returns [`RcnError`] if an entropy primitive fails.
pub fn compress(buf: &[u8]) -> Result<Vec<u8>> {
    compress_mode(buf, CodecMode::Slow)
}

/// Compress `buf` using the byte-level (fast) path.
///
/// # Errors
///
/// Returns [`RcnError`] if an entropy primitive fails.
pub fn compress_fast(buf: &[u8]) -> Result<Vec<u8>> {
    compress_mode(buf, CodecMode::Fast)
}

pub fn compress_with<F>(buf: &[u8], build_stack: &mut F) -> Result<Vec<u8>>
where
    F: FnMut(crate::classify::BlockKind) -> (Vec<Box<dyn BitModel>>, MixerBank, Option<usize>),
{
    Ok(compress_impl(buf, CodecMode::Slow, build_stack)?.0)
}

/// Compress with an explicit [`CodecMode`]. Both encoder sides of a given
/// mode decode correctly from the method byte in the container.
///
/// # Errors
///
/// Returns [`RcnError`] if an entropy primitive fails.
pub fn compress_mode(buf: &[u8], mode: CodecMode) -> Result<Vec<u8>> {
    Ok(compress_mode_diag(buf, mode)?.0)
}

/// Compress with an explicit [`CodecMode`], also returning a per-block
/// [`BlockDiag`] summary suitable for `--verbose` progress output.
///
/// # Errors
///
/// Returns [`RcnError`] if an entropy primitive fails.
pub fn compress_mode_diag(buf: &[u8], mode: CodecMode) -> Result<(Vec<u8>, Vec<BlockDiag>)> {
    compress_impl(buf, mode, &mut build_stack_for_kind)
}

fn compress_impl<F>(
    buf: &[u8],
    mode: CodecMode,
    build_stack: &mut F,
) -> Result<(Vec<u8>, Vec<BlockDiag>)>
where
    F: FnMut(crate::classify::BlockKind) -> (Vec<Box<dyn BitModel>>, MixerBank, Option<usize>),
{
    let mut out = Vec::new();
    let mut entries: Vec<BlockEntry> = Vec::new();
    let mut payloads: Vec<u8> = Vec::new();
    let mut diags: Vec<BlockDiag> = Vec::new();
    let mut offset = 0usize;

    let mut last_kind: Option<crate::classify::BlockKind> = None;
    let mut models: Vec<Box<dyn BitModel>> = Vec::new();
    let mut mixer = MixerBank::new(0);
    let mut lzp_idx: Option<usize> = None;

    // First pass: build a global XWRT dictionary from the entire corpus.
    // Stored once in the container header and shared by all Text-block XWRT
    // trials (replaces the per-block dictionary, which costs ~0.5-1pt).
    let global_dict: Option<XwrtDictionary> =
        (mode == CodecMode::Slow).then(|| XwrtDictionary::build_from_data(buf));

    // Prefetched (end, kind) for the next Slow-mode block — filled by overlapping
    // classify of N+1 while encoding N (bit-identical: encode stays serial).
    let mut pending_next: Option<(usize, crate::classify::BlockKind)> = None;

    while offset < buf.len() {
        let (end, kind) = if let Some((pend_end, pend_kind)) = pending_next.take() {
            (pend_end, pend_kind)
        } else {
            let kind = crate::classify::classify(&buf[offset..]);
            let block_size = block_size_for_kind(kind, &buf[offset..], offset, buf.len());
            ((offset + block_size).min(buf.len()), kind)
        };
        let block_data = &buf[offset..end];

        if mode == CodecMode::Slow && last_kind != Some(kind) {
            let (new_models, new_mixer, new_lzp_idx) = build_stack(kind);
            models = new_models;
            mixer = new_mixer;
            lzp_idx = new_lzp_idx;
            last_kind = Some(kind);
        }

        let next_start = end;
        let (comp, method, store_orig_len) = match mode {
            CodecMode::Fast => encode_block_fast(block_data, kind),
            CodecMode::Slow if next_start < buf.len() => {
                // Overlap classify+size of block N+1 with encode of block N.
                let ((c, m, o), next) = rayon::join(
                    || {
                        encode_block_slow(
                            block_data,
                            kind,
                            &mut models,
                            &mut mixer,
                            lzp_idx,
                            global_dict.as_ref(),
                        )
                    },
                    || {
                        let next_kind = crate::classify::classify(&buf[next_start..]);
                        let next_size = block_size_for_kind(
                            next_kind,
                            &buf[next_start..],
                            next_start,
                            buf.len(),
                        );
                        ((next_start + next_size).min(buf.len()), next_kind)
                    },
                );
                pending_next = Some(next);
                (c, m, o)
            }
            CodecMode::Slow => encode_block_slow(
                block_data,
                kind,
                &mut models,
                &mut mixer,
                lzp_idx,
                global_dict.as_ref(),
            ),
        };

        let entry = BlockEntry {
            comp_len: comp.len() as u32,
            orig_len: store_orig_len as u32,
            method,
            crc32: crc32(block_data),
        };
        diags.push(BlockDiag {
            kind: format!("{:?}", kind),
            method,
            size_in: block_data.len(),
            size_out: comp.len(),
        });
        entries.push(entry);
        payloads.extend_from_slice(&comp);
        offset = end;

        // Decay (not reset) at block boundaries: preserve learned weight
        // structure across same-kind blocks in the stream. Skip for copy blocks
        // (no models were trained, no mixer state to decay) — mirrors the
        // decoder's `entry.method != METHOD_COPY` guard.
        if mode == CodecMode::Slow && method != METHOD_COPY {
            mixer.decay(BLOCK_DECAY);
        }
    }

    // Write container: magic + header + entries + payloads
    let header = Header {
        version: VERSION,
        flags: if global_dict.is_some() {
            FLAG_GLOBAL_DICT
        } else {
            0
        },
        block_size_log: DEFAULT_BLOCK_SIZE_LOG,
        num_blocks: entries.len() as u32,
    };
    header.write(&mut out);
    if let Some(ref dict) = global_dict {
        let dict_bytes = dict.to_bytes();
        write_global_dict(&mut out, &dict_bytes);
    }
    for e in &entries {
        e.write(&mut out);
    }
    out.extend_from_slice(&payloads);
    Ok((out, diags))
}

/// Fast-mode per-block encode: byte-level CM, or copy for random blocks.
fn encode_block_fast(block_data: &[u8], kind: crate::classify::BlockKind) -> (Vec<u8>, u8, usize) {
    if kind == crate::classify::BlockKind::Random {
        (block_data.to_vec(), METHOD_COPY, block_data.len())
    } else if kind == crate::classify::BlockKind::Text {
        // Same BWT trial as the bit path; the chosen pipeline's output is then
        // byte-coded (one rANS symbol per byte instead of per bit).
        let trial = bwt::compress_text_with_trial(block_data, None);
        let transformed = trial.pipeline.encode(block_data, None);
        let (comp, method) = match trial.pipeline {
            bwt::BwtPipeline::RawCm => (
                crate::bytecodec::compress_block(&transformed),
                METHOD_BYTE_CM,
            ),
            bwt::BwtPipeline::BwtMtfRle => (
                crate::bytecodec::compress_block(&transformed),
                METHOD_BYTE_BWT_MTF_RLE,
            ),
            bwt::BwtPipeline::LzpBwtMtf => (
                crate::bytecodec::compress_block(&transformed),
                METHOD_BYTE_LZP_BWT_MTF,
            ),
            bwt::BwtPipeline::JsonSplit => (
                crate::bytecodec::compress_block(&transformed),
                METHOD_BYTE_JSON_SPLIT,
            ),
            bwt::BwtPipeline::XwrtBwtMtfRle => (
                crate::bytecodec::compress_block(&transformed),
                METHOD_BYTE_XWRT_BWT_MTF_RLE,
            ),
        };
        (comp, method, transformed.len())
    } else if kind == crate::classify::BlockKind::Exec {
        // Exec: apply E8E9 transform to convert x86 relative offsets to absolute,
        // making them much more compressible.
        let transformed = crate::model::e8e9::e8e9_transform(block_data);
        (
            crate::bytecodec::compress_block(&transformed),
            METHOD_BYTE_EXEC_E8E9,
            transformed.len(),
        )
    } else {
        // Binary: raw byte CM.
        (
            crate::bytecodec::compress_block(block_data),
            METHOD_BYTE_CM,
            block_data.len(),
        )
    }
}

/// Slow-mode per-block encode: bit-level CM with the classifier-aware stacks.
fn encode_block_slow(
    block_data: &[u8],
    kind: crate::classify::BlockKind,
    models: &mut [Box<dyn BitModel>],
    mixer: &mut MixerBank,
    lzp_idx: Option<usize>,
    global_dict: Option<&crate::model::word::XwrtDictionary>,
) -> (Vec<u8>, u8, usize) {
    // Random blocks: store verbatim (method COPY). Don't run CM on
    // entropy-poor data — the rANS path would inflate and the decoder
    // treats METHOD_COPY as a passthrough anyway.
    if kind == crate::classify::BlockKind::Random {
        (block_data.to_vec(), METHOD_COPY, block_data.len())
    } else if kind == crate::classify::BlockKind::Text {
        // Per-block trial: pick the best BWT pipeline for this Text block.
        let trial = bwt::compress_text_with_trial(block_data, global_dict);
        let method = match trial.pipeline {
            bwt::BwtPipeline::RawCm => METHOD_TEXT,
            bwt::BwtPipeline::BwtMtfRle => METHOD_BWT_MTF_RLE,
            bwt::BwtPipeline::LzpBwtMtf => METHOD_LZP_BWT_MTF,
            bwt::BwtPipeline::JsonSplit => METHOD_JSON_SPLIT,
            bwt::BwtPipeline::XwrtBwtMtfRle => METHOD_XWRT_BWT_MTF_RLE,
        };
        // Transform the block data through the chosen pipeline, then CM-encode.
        let transformed = trial.pipeline.encode(block_data, global_dict);
        // For BWT paths, `orig_len` stores the *transformed* length (what the
        // decoder must decode from rANS). The original length is recovered
        // during BWT reversal; correctness is verified by CRC.
        let comp = compress_block(models, mixer, lzp_idx, &transformed, kind);
        (comp, method, transformed.len())
    } else if kind == crate::classify::BlockKind::Exec {
        // Exec: apply E8E9 transform to convert x86 relative offsets to absolute,
        // making them much more compressible.
        let transformed = crate::model::e8e9::e8e9_transform(block_data);
        let comp = compress_block(models, mixer, lzp_idx, &transformed, kind);
        (comp, METHOD_EXEC, transformed.len())
    } else {
        // Binary: raw CM with the existing stack.
        let comp = compress_block(models, mixer, lzp_idx, block_data, kind);
        (comp, method_for_kind(kind), block_data.len())
    }
}

fn block_size_for_kind(
    kind: crate::classify::BlockKind,
    _block: &[u8],
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
///
/// Returns `(models, mixer, lzp_idx)` where `lzp_idx` is the index of the LZP
/// model in `models` (if present).
#[must_use]
pub fn build_stack_for_kind(
    kind: crate::classify::BlockKind,
) -> (Vec<Box<dyn BitModel>>, MixerBank, Option<usize>) {
    match kind {
        // Random: copy, no models needed (encoder won't call compress_block).
        crate::classify::BlockKind::Random => {
            let models: Vec<Box<dyn BitModel>> = vec![];
            (models, MixerBank::new(0), None)
        }
        crate::classify::BlockKind::Text => {
            // Text stack: orders 0–2, Sparse, Exec, LZP, PPMd+SSM, Word.
            // IndirectModel was tried with 16k banks (C8); kept in-tree but not
            // in the default stack after prior dickens regressions.
            let n = 8;
            let models: Vec<Box<dyn BitModel>> = vec![
                Box::new(crate::model::order::OrderN::new(0)),
                Box::new(crate::model::order::OrderN::new(1)),
                Box::new(crate::model::order::OrderN::new(2)),
                Box::new(crate::model::sparse::Sparse::new()),
                Box::new(crate::model::exec::Exec::new()),
                Box::new(crate::model::lzp::Lzp::new()),
                Box::new(crate::model::ppmd_ssm::PpmdSsm::new()),
                Box::new(crate::model::word::WordModel::new()),
            ];
            (models, MixerBank::new(n), Some(5))
        }
        crate::classify::BlockKind::Binary => {
            // Binary stack (best configuration, no SSM).
            // with orders 0-2, Sparse, Exec, LZP, PPM order-3
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
            (models, MixerBank::new(n), Some(5))
        }
        crate::classify::BlockKind::Exec => {
            // Exec stack (best configuration, no SSM).
            // with orders 0-2, Sparse, LZP, PPM order-3; no Exec model
            let n = 6;
            let models: Vec<Box<dyn BitModel>> = vec![
                Box::new(crate::model::order::OrderN::new(0)),
                Box::new(crate::model::order::OrderN::new(1)),
                Box::new(crate::model::order::OrderN::new(2)),
                Box::new(crate::model::sparse::Sparse::new()),
                Box::new(crate::model::lzp::Lzp::new()),
                Box::new(crate::model::ppm::PpmModel::new(3)),
            ];
            (models, MixerBank::new(n), Some(4))
        }
    }
}

/// Legacy alias kept for benchmark tooling (`src/stacks.rs`).
#[must_use]
pub fn build_full_stack() -> (Vec<Box<dyn BitModel>>, MixerBank, Option<usize>) {
    build_stack_for_kind(crate::classify::BlockKind::Binary)
}

/// Compress one block.
///
/// Runs DP-optimal LZP match pre-pass, emits (len, dist, pos)
/// side-stream records, then rANS-encodes only **literal** (non-matched) bytes.
/// Matched bytes are reconstructed by the decoder from the side-stream.
fn compress_block(
    models: &mut [Box<dyn BitModel>],
    mixer: &mut MixerBank,
    lzp_idx: Option<usize>,
    block: &[u8],
    kind: crate::classify::BlockKind,
) -> Vec<u8> {
    let runs = scan_matches(block, kind);
    encode_block_with_matches(models, mixer, lzp_idx, block, &runs)
}

/// Plain CM encoding (no match side-stream).
fn encode_block_plain(
    models: &mut [Box<dyn BitModel>],
    mixer: &mut MixerBank,
    lzp_idx: Option<usize>,
    block: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    let mut cascade = SseApmCascade::new();

    // Pre-build per-block dictionaries.
    for m in models.iter_mut() {
        m.prepare_block(block);
    }

    let mut enc = BitEncoder::new();
    let mut probs: [u16; 12] = [2048; 12];
    let n = models.len();
    let lzp_conf_default = 2048u16;

    let mut prev_byte = 0u8;
    for &byte in block {
        cascade.set_context(prev_byte); // Use PREVIOUS byte as context (decoder also sees prev)
        for bit_idx in (0..8).rev() {
            let bit = (byte >> bit_idx) & 1u8 == 1u8;
            let bit_pos = bit_idx as u8;
            for (j, m) in models.iter().enumerate() {
                probs[j] = m.predict();
            }
            let lzp_conf = lzp_idx.map(|i| probs[i]).unwrap_or(lzp_conf_default);

            mixer.mix_and_update(
                &probs[..n],
                bit,
                bit_pos,
                lzp_conf,
                &mut |encoded_bit, p_mixer| {
                    let p_refined = cascade.refine(p_mixer, bit_pos);
                    enc.encode_bit(encoded_bit, p_refined);
                    cascade.update(encoded_bit, p_mixer, bit_pos);
                },
            );
            for m in models.iter_mut() {
                m.update(bit);
            }
        }
        // Feed completed byte to the mixer bank's byte assembler for context.
        mixer.push_byte(byte);
        prev_byte = byte; // Update for next iteration
    }
    out.extend(enc.finish());
    out
}

fn encode_block_with_matches(
    models: &mut [Box<dyn BitModel>],
    mixer: &mut MixerBank,
    lzp_idx: Option<usize>,
    block: &[u8],
    runs: &[MatchRun],
) -> Vec<u8> {
    let mut out = Vec::new();

    for m in models.iter_mut() {
        m.prepare_block(block);
    }

    // Match side-stream: [num_runs:u32][flag:u8][payload_len:u32][payload...]
    // where payload is varint records (delta_pos, len, dist), optionally
    // order-0 FSE/rANS-compressed (flag=1) when that shrinks the blob.
    let mut raw_side = Vec::new();
    let mut prev_pos: usize = 0;
    for r in runs {
        let delta_pos = r.pos.wrapping_sub(prev_pos);
        write_varint(&mut raw_side, delta_pos);
        write_varint(&mut raw_side, r.len);
        write_varint(&mut raw_side, r.dist);
        prev_pos = r.pos;
    }
    let (flag, payload) = side_fse::pack_side_stream(&raw_side);
    out.extend_from_slice(&(runs.len() as u32).to_le_bytes());
    out.push(flag);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);

    // Build a position→length map for matched regions.
    let mut match_len_at: Vec<usize> = vec![0; block.len()];
    for r in runs {
        if r.pos < block.len() && match_len_at[r.pos] < r.len {
            match_len_at[r.pos] = r.len;
        }
    }

    // rANS-encode only literal bytes (skip matched regions).
    let mut enc = BitEncoder::new();
    let mut probs: [u16; 12] = [2048; 12];
    let n = models.len();
    let lzp_conf_default = 2048u16;
    let mut enc_bit = |bit: bool, p: u16| enc.encode_bit(bit, p);

    // Helper closure: feed a byte through models+mixer WITHOUT rANS encoding.
    // Used for matched bytes — they must update context state but not produce
    // rANS bits (the decoder reconstructs them from match records).
    macro_rules! skip_byte {
        ($byte:expr) => {{
            let byte = $byte;
            for bit_idx in (0..8).rev() {
                let bit = (byte >> bit_idx) & 1u8 == 1u8;
                let bit_pos = bit_idx as u8;
                for (j, m) in models.iter().enumerate() {
                    probs[j] = m.predict();
                }
                let lzp_conf = lzp_idx.map(|i| probs[i]).unwrap_or(lzp_conf_default);
                mixer.update(&probs[..n], bit, bit_pos, lzp_conf);
                for m in models.iter_mut() {
                    m.update(bit);
                }
            }
            mixer.push_byte(byte);
        }};
    }

    let mut i = 0usize;
    while i < block.len() {
        if match_len_at[i] > 0 {
            // Matched region: feed bytes for context, skip rANS encoding.
            let len = match_len_at[i];
            for j in 0..len {
                skip_byte!(block[i + j]);
            }
            i += len;
        } else {
            // Literal: encode through rANS.
            let byte = block[i];
            let mut cascade = SseApmCascade::new();
            cascade.set_context(byte); // Set byte context for SSE/APM cascade
            for bit_idx in (0..8).rev() {
                let bit = (byte >> bit_idx) & 1u8 == 1u8;
                let bit_pos = bit_idx as u8;
                for (j, m) in models.iter().enumerate() {
                    probs[j] = m.predict();
                }
                let lzp_conf = lzp_idx.map(|i| probs[i]).unwrap_or(lzp_conf_default);

                mixer.mix_and_update(
                    &probs[..n],
                    bit,
                    bit_pos,
                    lzp_conf,
                    &mut |encoded_bit, p_mixer| {
                        let p_refined = cascade.refine(p_mixer, bit_pos);
                        enc.encode_bit(encoded_bit, p_refined);
                        cascade.update(encoded_bit, p_mixer, bit_pos);
                    },
                );
                for m in models.iter_mut() {
                    m.update(bit);
                }
            }
            mixer.push_byte(byte);
            i += 1;
        }
    }
    out.extend(enc.finish());
    out
}

fn scan_matches(block: &[u8], kind: crate::classify::BlockKind) -> Vec<MatchRun> {
    let mut lzp = Lzp::new();
    let n = block.len();
    let min_len = match_min_len(kind);

    // AVG_BITS_PER_BYTE: estimated CM cost per byte for residual cost calculation.
    // A well-trained CM predicts ~3-4 bits/byte on structured data. We use 4.0
    // as a conservative estimate: matches must save more than this in CM cost.
    const AVG_BITS_PER_BYTE: f64 = 4.0;
    // Match record overhead: 1 (flag) + 8 (len) + 24 (dist) = 33 bits.
    const MATCH_OVERHEAD_BITS: f64 = 33.0;
    let window = 4 * 1024 * 1024;

    // Phase 1: pre-compute the best match at every position.
    // best_match[i] = Some((len, dist)) if a match of >= min_len exists at position i.
    // The LZP chain walk reports (len, dist) in one pass — no O(window) backward
    // re-scan per position, which was the pathological blow-up on large text blocks.
    let mut best_match: Vec<Option<(usize, usize)>> = vec![None; n];
    for i in 0..n {
        lzp.train_at(block, i);
        if i + 1 >= min_len && i + min_len <= n {
            if let Some((len, dist)) = lzp.best_match(block, i) {
                let len = len.min(255);
                if len >= min_len && dist > 0 && dist <= window {
                    best_match[i] = Some((len, dist));
                }
            }
        }
    }

    // Phase 2: DP optimal parse.
    // dp[i] = minimum total cost to encode from position i to the end.
    // cost(literal) = AVG_BITS_PER_BYTE (1 byte × predicted bits)
    // cost(match len,dist) = MATCH_OVERHEAD_BITS + (len * AVG_BITS_PER_BYTE)
    //   — the matched bytes still get CM-encoded (for now), so residual_cost = len * AVG_BITS_PER_BYTE
    //   — but the match flag/len/dist overhead is constant per match.
    // A match is chosen when:
    //   MATCH_OVERHEAD_BITS + len * AVG_BITS_PER_BYTE < len * AVG_BITS_PER_BYTE (literal cost)
    //   i.e., when the match doesn't add overhead compared to literals.
    // Actually: literal cost = len * AVG_BITS_PER_BYTE
    // Match cost = MATCH_OVERHEAD_BITS + 0 (skip CM for matched bytes)
    // So match is better when: MATCH_OVERHEAD_BITS < len * AVG_BITS_PER_BYTE
    // i.e., len > MATCH_OVERHEAD_BITS / AVG_BITS_PER_BYTE = 33/4 = 8.25
    // With threshold 16, matches of 16+ bytes save 16*4 - 33 = 31 bits. Take them.
    let mut dp: Vec<f64> = vec![f64::INFINITY; n + 1];
    let mut choice: Vec<bool> = vec![false; n]; // true = match taken, false = literal
    dp[n] = 0.0;

    for i in (0..n).rev() {
        // Option 1: literal (cost = residual cost of 1 byte)
        let literal_cost = AVG_BITS_PER_BYTE + dp[i + 1];
        dp[i] = literal_cost;
        choice[i] = false;

        // Option 2: match (if available)
        if let Some((len, dist)) = best_match[i] {
            let match_cost = MATCH_OVERHEAD_BITS + dp[i + len];
            if match_cost < dp[i] {
                dp[i] = match_cost;
                choice[i] = true;
            }
        }
    }

    // Phase 3: backtrack to extract match runs (with positions).
    let mut runs: Vec<MatchRun> = Vec::new();
    let mut i = 0usize;
    while i < n {
        if choice[i] {
            if let Some((len, dist)) = best_match[i] {
                runs.push(MatchRun { pos: i, len, dist });
                i += len;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    runs
}

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
/// Reads match side-stream (validates records), then rANS-decodes all bytes.
fn decode_block(
    comp: &[u8],
    orig_len: usize,
    models: &mut [Box<dyn BitModel>],
    mixer: &mut MixerBank,
    lzp_idx: Option<usize>,
) -> Result<Vec<u8>> {
    decode_block_with_matches(comp, orig_len, models, mixer, lzp_idx)
}

fn decode_block_plain(
    comp: &[u8],
    orig_len: usize,
    models: &mut [Box<dyn BitModel>],
    mixer: &mut MixerBank,
    lzp_idx: Option<usize>,
) -> Result<Vec<u8>> {
    let mut dec = BitDecoder::new(comp).map_err(|e| RcnError::Entropy(e.to_string()))?;
    let mut out = Vec::with_capacity(orig_len);
    let mut probs: [u16; 12] = [2048; 12];
    let n = models.len();
    let lzp_conf_default = 2048u16;
    let mut cascade = SseApmCascade::new();

    while out.len() < orig_len {
        let mut byte = 0u8;
        for bit_idx in (0..8).rev() {
            let bit_pos = bit_idx as u8;
            for (i, m) in models.iter().enumerate() {
                probs[i] = m.predict();
            }
            let lzp_conf = lzp_idx.map(|i| probs[i]).unwrap_or(lzp_conf_default);
            let (p_mixer, pacc) = mixer.mix_acc(&probs[..n], bit_pos, lzp_conf);
            let p_refined = cascade.refine(p_mixer, bit_pos);
            let bit = dec
                .decode_bit(p_refined)
                .map_err(|e| RcnError::Entropy(e.to_string()))?;
            mixer.update_acc(&probs[..n], bit, bit_pos, pacc);
            cascade.update(bit, p_mixer, bit_pos);
            for m in models.iter_mut() {
                m.update(bit);
            }
            if bit {
                byte |= 1 << bit_idx;
            }
        }
        out.push(byte);
        cascade.set_context(byte);
        mixer.push_byte(byte);
    }
    Ok(out)
}

fn decode_block_with_matches(
    comp: &[u8],
    orig_len: usize,
    models: &mut [Box<dyn BitModel>],
    mixer: &mut MixerBank,
    lzp_idx: Option<usize>,
) -> Result<Vec<u8>> {
    // Header: [num_runs:u32][flag:u8][payload_len:u32] = 9 bytes minimum.
    if comp.len() < 9 {
        return Err(RcnError::InvalidContainer(
            "match side-stream too short".into(),
        ));
    }
    let num_runs = u32::from_le_bytes([comp[0], comp[1], comp[2], comp[3]]) as usize;
    let flag = comp[4];
    let payload_len = u32::from_le_bytes([comp[5], comp[6], comp[7], comp[8]]) as usize;
    let payload_start: usize = 9;
    let payload_end = payload_start
        .checked_add(payload_len)
        .ok_or_else(|| RcnError::InvalidContainer("match side-stream length overflow".into()))?;
    if payload_end > comp.len() {
        return Err(RcnError::InvalidContainer(
            "match side-stream truncated".into(),
        ));
    }
    let raw_side = side_fse::unpack_side_stream(flag, &comp[payload_start..payload_end])
        .map_err(|()| RcnError::InvalidContainer("corrupt match side-stream entropy".into()))?;

    // Read all match records (varint-encoded delta positions).
    let mut runs: Vec<MatchRun> = Vec::with_capacity(num_runs);
    let mut prev_pos: usize = 0;
    let mut offset = 0usize;
    for _ in 0..num_runs {
        let (delta_pos, new_offset) = read_varint(&raw_side, offset).ok_or_else(|| {
            RcnError::InvalidContainer("truncated match record (delta_pos)".into())
        })?;
        offset = new_offset;
        let (len, new_offset) = read_varint(&raw_side, offset)
            .ok_or_else(|| RcnError::InvalidContainer("truncated match record (len)".into()))?;
        offset = new_offset;
        let (dist, new_offset) = read_varint(&raw_side, offset)
            .ok_or_else(|| RcnError::InvalidContainer("truncated match record (dist)".into()))?;
        offset = new_offset;
        let pos = prev_pos.wrapping_add(delta_pos);
        prev_pos = pos;
        runs.push(MatchRun { pos, len, dist });
    }

    let mut dec =
        BitDecoder::new(&comp[payload_end..]).map_err(|e| RcnError::Entropy(e.to_string()))?;
    let mut out = Vec::with_capacity(orig_len);
    let mut probs: [u16; 12] = [2048; 12];
    let n = models.len();
    let lzp_conf_default = 2048u16;

    // Walk through the original block, interleaving matched and literal bytes.
    let mut match_idx = 0usize;
    let mut i = 0usize; // current position in the output (decoded) stream

    // Helper closure: feed a byte through models+mixer WITHOUT rANS decoding.
    macro_rules! skip_byte {
        ($byte:expr) => {{
            let byte = $byte;
            for bit_idx in (0..8).rev() {
                let bit = (byte >> bit_idx) & 1u8 == 1u8;
                let bit_pos = bit_idx as u8;
                for (j, m) in models.iter().enumerate() {
                    probs[j] = m.predict();
                }
                let lzp_conf = lzp_idx.map(|i| probs[i]).unwrap_or(lzp_conf_default);
                mixer.update(&probs[..n], bit, bit_pos, lzp_conf);
                for m in models.iter_mut() {
                    m.update(bit);
                }
            }
            mixer.push_byte(byte);
        }};
    }

    while i < orig_len {
        // Check if a match starts at the current position.
        if match_idx < runs.len() && runs[match_idx].pos == i {
            let run = runs[match_idx];
            // Copy `len` bytes from history: out.len() - dist .. out.len() - dist + len
            // But we need to be careful about overlapping copies.
            let dist = run.dist;
            for j in 0..run.len {
                let byte = if i + j < orig_len && dist <= out.len() {
                    out[out.len() - dist]
                } else {
                    0u8
                };
                skip_byte!(byte);
                out.push(byte);
            }
            i += run.len;
            match_idx += 1;
        } else {
            // Literal: rANS-decode a byte.
            let mut byte = 0u8;
            let mut cascade = SseApmCascade::new();
            for bit_idx in (0..8).rev() {
                let bit_pos = bit_idx as u8;
                for (j, m) in models.iter().enumerate() {
                    probs[j] = m.predict();
                }
                let lzp_conf = lzp_idx.map(|i| probs[i]).unwrap_or(lzp_conf_default);
                let (p_mixer, pacc) = mixer.mix_acc(&probs[..n], bit_pos, lzp_conf);
                let p_refined = cascade.refine(p_mixer, bit_pos);
                let bit = dec
                    .decode_bit(p_refined)
                    .map_err(|e| RcnError::Entropy(e.to_string()))?;
                mixer.update_acc(&probs[..n], bit, bit_pos, pacc);
                cascade.update(bit, p_mixer, bit_pos);
                for m in models.iter_mut() {
                    m.update(bit);
                }
                if bit {
                    byte |= 1 << bit_idx;
                }
            }
            out.push(byte);
            cascade.set_context(byte);
            mixer.push_byte(byte);
            i += 1;
        }
    }
    Ok(out)
}

/// Decompress a `RCN1` container back to the original bytes.
///
/// # Errors
///
/// Returns [`RcnError`] on a malformed container, corrupt block, or CRC mismatch.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    decompress_impl(data, &mut build_stack_for_kind)
}

/// Decompress a `RCN1` container using a custom model-stack builder.
///
/// This mirrors [`compress_with`] on the decode side: the `build_stack` closure
/// must match the one used for compression, otherwise rANS decoding will produce
/// garbage and the CRC check will fail.
pub fn decompress_with<F>(data: &[u8], build_stack: &mut F) -> Result<Vec<u8>>
where
    F: FnMut(crate::classify::BlockKind) -> (Vec<Box<dyn BitModel>>, MixerBank, Option<usize>),
{
    decompress_impl(data, build_stack)
}

fn decompress_impl<F>(data: &[u8], build_stack: &mut F) -> Result<Vec<u8>>
where
    F: FnMut(crate::classify::BlockKind) -> (Vec<Box<dyn BitModel>>, MixerBank, Option<usize>),
{
    use std::io::Cursor;
    let mut cur = Cursor::new(data);
    let header = Header::read(&mut cur).map_err(|e| RcnError::InvalidContainer(e.to_string()))?;
    if header.version != VERSION {
        return Err(RcnError::InvalidContainer(format!(
            "unsupported version {}",
            header.version
        )));
    }
    // Consume the global XWRT dictionary (if present) from between the header
    // and the BlockEntry table; every XWRT block refers back to it.
    let (global_dict_bytes, dict_end) =
        read_global_dict(data, cur.position() as usize, header.flags);
    cur.set_position(dict_end as u64);
    let global_dict = XwrtDictionary::from_bytes(&global_dict_bytes);

    let mut entries = Vec::with_capacity(header.num_blocks as usize);
    for _ in 0..header.num_blocks {
        entries.push(
            BlockEntry::read(&mut cur).map_err(|e| RcnError::InvalidContainer(e.to_string()))?,
        );
    }
    let payload_start = cur.position() as usize;
    let payloads = &data[payload_start..];

    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut last_kind: Option<crate::classify::BlockKind> = None;
    let mut models: Vec<Box<dyn BitModel>> = Vec::new();
    let mut mixer = MixerBank::new(0);
    let mut lzp_idx: Option<usize> = None;
    for (bi, entry) in entries.iter().enumerate() {
        let comp = &payloads[pos..pos + entry.comp_len as usize];
        pos += entry.comp_len as usize;

        let block = if entry.method == METHOD_COPY {
            comp.to_vec()
        } else if matches!(
            entry.method,
            METHOD_BYTE_CM
                | METHOD_BYTE_BWT_MTF_RLE
                | METHOD_BYTE_LZP_BWT_MTF
                | METHOD_BYTE_JSON_SPLIT
                | METHOD_BYTE_XWRT_BWT_MTF_RLE
                | METHOD_BYTE_EXEC_E8E9
        ) {
            // Byte-level (fast) path: one rANS symbol per byte.
            let decoded = crate::bytecodec::decompress_block(comp, entry.orig_len as usize)
                .map_err(|e| match e {
                    RcnError::CorruptBlock(s) => RcnError::CorruptBlock(s),
                    other => other,
                })?;
            match entry.method {
                METHOD_BYTE_BWT_MTF_RLE => bwt::bwt_mtf_rle_decode(&decoded),
                METHOD_BYTE_LZP_BWT_MTF => {
                    let mtf = bwt::bwt_mtf_decode(&decoded);
                    bwt::lzp_decode(&mtf, entry.orig_len as usize)
                }
                METHOD_BYTE_JSON_SPLIT => bwt::BwtPipeline::JsonSplit.decode(
                    &decoded,
                    entry.orig_len as usize,
                    global_dict.as_ref(),
                ),
                METHOD_BYTE_XWRT_BWT_MTF_RLE => bwt::BwtPipeline::XwrtBwtMtfRle.decode(
                    &decoded,
                    entry.orig_len as usize,
                    global_dict.as_ref(),
                ),
                METHOD_BYTE_EXEC_E8E9 => crate::model::e8e9::e8e9_inverse(&decoded),
                _ => decoded,
            }
        } else {
            let kind = kind_for_method(entry.method)?;
            if last_kind != Some(kind) {
                let (new_models, new_mixer, new_lzp_idx) = build_stack(kind);
                models = new_models;
                mixer = new_mixer;
                lzp_idx = new_lzp_idx;
                last_kind = Some(kind);
            }
            let decoded = decode_block(
                comp,
                entry.orig_len as usize,
                &mut models,
                &mut mixer,
                lzp_idx,
            )
            .map_err(|e| match e {
                RcnError::Entropy(s) => RcnError::CorruptBlock(s),
                other => other,
            })?;
            // Reverse BWT transforms for method 5/6, mirroring the encoder's trial.
            match entry.method {
                METHOD_BWT_MTF_RLE => bwt::bwt_mtf_rle_decode(&decoded),
                METHOD_LZP_BWT_MTF => {
                    let mtf = bwt::bwt_mtf_decode(&decoded);
                    bwt::lzp_decode(&mtf, entry.orig_len as usize)
                }
                METHOD_JSON_SPLIT => bwt::BwtPipeline::JsonSplit.decode(
                    &decoded,
                    entry.orig_len as usize,
                    global_dict.as_ref(),
                ),
                METHOD_XWRT_BWT_MTF_RLE => bwt::BwtPipeline::XwrtBwtMtfRle.decode(
                    &decoded,
                    entry.orig_len as usize,
                    global_dict.as_ref(),
                ),
                METHOD_EXEC => crate::model::e8e9::e8e9_inverse(&decoded),
                _ => decoded,
            }
        };

        if crate::container::crc32(&block) != entry.crc32 {
            return Err(RcnError::CrcMismatch(
                bi,
                crate::container::crc32(&block),
                entry.crc32,
            ));
        }
        out.extend_from_slice(&block);

        // Decay at block boundaries (mirrors compressor).
        if entry.method != METHOD_COPY {
            mixer.decay(BLOCK_DECAY);
        }
    }
    Ok(out)
}

/// Reverse map: container method byte → `BlockKind`. Unknown methods error.
fn kind_for_method(method: u8) -> Result<crate::classify::BlockKind> {
    match method {
        METHOD_COPY => Ok(crate::classify::BlockKind::Random),
        METHOD_CM | METHOD_TEXT => Ok(crate::classify::BlockKind::Text),
        METHOD_BINARY => Ok(crate::classify::BlockKind::Binary),
        METHOD_EXEC => Ok(crate::classify::BlockKind::Exec),
        METHOD_BWT_MTF_RLE | METHOD_LZP_BWT_MTF | METHOD_JSON_SPLIT | METHOD_XWRT_BWT_MTF_RLE => {
            Ok(crate::classify::BlockKind::Text)
        }
        METHOD_BYTE_CM
        | METHOD_BYTE_BWT_MTF_RLE
        | METHOD_BYTE_LZP_BWT_MTF
        | METHOD_BYTE_JSON_SPLIT
        | METHOD_BYTE_XWRT_BWT_MTF_RLE => Ok(crate::classify::BlockKind::Text),
        METHOD_BYTE_EXEC_E8E9 => Ok(crate::classify::BlockKind::Exec),
        _ => Err(RcnError::InvalidContainer(format!(
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
        let text = b"the quick brown fox jumps over the lazy dog. \\\
            compression mixes many context models so that each bit is predicted well. ";
        for _ in 0..2000 {
            v.extend_from_slice(text);
        }
        let json = b"{\"name\":\"rcn\",\"level\":3,\"models\":[\"order0\",\"order1\",\"order2\",\"sparse\",\"exec\",\"lzp\"],\"ratio\":0.42}\n";
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
            input.extend_from_slice(b"rcnrcnrcn");
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
        let json = b"{\"name\":\"rcn\",\"level\":3,\"models\":[\"order0\",\"order1\",\"order2\",\"sparse\",\"exec\",\"lzp\"],\"ratio\":0.42}\n";
        let original: Vec<u8> = std::iter::repeat(json.as_ref())
            .take(4000)
            .flatten()
            .copied()
            .collect();
        let comp = compress(&original).expect("compress");
        let back = decompress(&comp).expect("decompress");
        assert_eq!(back, original, "JSON round-trip mismatch");
    }

    #[test]
    fn scan_matches_finds_repeats() {
        let data = b"abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";
        let runs = scan_matches(data, crate::classify::BlockKind::Text);
        assert!(
            !runs.is_empty(),
            "expected at least one match in repeated data"
        );
        assert!(runs[0].len >= match_min_len(crate::classify::BlockKind::Text));
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
        let runs = scan_matches(&data, crate::classify::BlockKind::Binary);
        assert!(runs.is_empty(), "expected no matches in random data");
    }

    #[test]
    fn find_match_distance_correct() {
        let data = b"abcabcabcabc";
        let d = find_match_distance(data, 6, 3);
        assert_eq!(d, 3, "expected distance 3, got {}", d);
    }
}
