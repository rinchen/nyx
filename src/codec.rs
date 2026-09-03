//! Core codec: glue the classifier, bit models, logistic mixer, rANS backend, and the
//! `NYF1` container into `compress` / `decompress`.
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
//! Because modeling is causal and the per-block method is recorded in the block
//! header, the decoder reconstructs the exact same model state from the decoded stream
//! and round-trips losslessly.
//!
//! Method values:
//!   0 = copy, 1 = cm (full stack), 2 = text, 3 = binary, 4 = exec.

use crate::container::{BlockEntry, Header, VERSION};
use crate::entropy::range::{BitDecoder, BitEncoder};
use crate::error::{NyxError, Result};
use crate::model::mixer::LogisticMixer;
use crate::model::mixer_bank::{ByteClass, MixerBank, NUM_MIXERS};
use crate::model::BitModel;

/// Default block size: 64 KiB. `block_size_log = 16`.
pub const DEFAULT_BLOCK_SIZE_LOG: u8 = 16;

/// Container method constants (stored in `BlockEntry::method`).
pub const METHOD_COPY: u8 = 0;
/// Full heterogeneous CM stack (legacy / fallback).
pub const METHOD_CM: u8 = 1;
/// Text-optimized CM stack (orders 0–2, Sparse, PPM order-4, LZP).
pub const METHOD_TEXT: u8 = 2;
/// Binary CM stack (orders 0–2, Sparse, Exec, LZP, PPM order-3).
pub const METHOD_BINARY: u8 = 3;
/// Exec-optimized CM stack (orders 0–2, Sparse, LZP, PPM order-3; no Exec model).
pub const METHOD_EXEC: u8 = 4;

/// Compress `buf` into a `NYF1` container using classifier-aware stacks.
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

    // Cross-block persisted state: mixer weight decay + LZP table survive across blocks.
    let mut global_mixer_weights: Option<Vec<f32>> = None;
    let mut lzp_table: Option<Vec<u32>> = None;
    let mut lzp_next: Option<Vec<u32>> = None;
    let mut lzp_current_pos: usize = 0;
    let mut lzp_history: Option<Vec<u8>> = None;

    while offset < buf.len() {
        let block = &buf[offset..];
        let kind = crate::classify::classify(block);
        let block_size = block_size_for_kind(kind, block, offset, buf.len());
        let end = (offset + block_size).min(buf.len());
        let block_data = &buf[offset..end];

        let (mut models, mut mixer) = build_stack(kind);
        hydrate_lzp(&mut models, &lzp_table, &lzp_next, lzp_current_pos, &lzp_history);
        hydrate_mixer_weights(&mut mixer, &global_mixer_weights);

        let comp = compress_block(&mut models, &mut mixer, block_data);

        persist_state(&mut global_mixer_weights, &mixer, &mut models, &mut lzp_table, &mut lzp_next, &mut lzp_current_pos, &mut lzp_history);

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
        block_size_log: 0, // variable block sizes, not a power-of-two global
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
            // Text blocks get larger windows for long-range repetition.
            // Scale with remaining data, capped at 4 MB.
            let max_text = 4 * 1024 * 1024;
            let size = (total - offset).min(max_text);
            size.max(64 * 1024)
        }
        crate::classify::BlockKind::Binary
        | crate::classify::BlockKind::Exec
        | crate::classify::BlockKind::Random => 64 * 1024,
    }
}

fn hydrate_lzp(
    models: &mut [Box<dyn BitModel>],
    lzp_table: &Option<Vec<u32>>,
    lzp_next: &Option<Vec<u32>>,
    current_pos: usize,
    history: &Option<Vec<u8>>,
) {
    if let (Some(table), Some(nxt), Some(hist)) = (lzp_table, lzp_next, history) {
        for m in models.iter_mut() {
            if let Some(lzp) = m.as_any_mut().downcast_mut::<crate::model::lzp::Lzp>() {
                lzp.set_state(table.clone(), nxt.clone(), current_pos, hist.clone());
            }
        }
    }
}

fn hydrate_mixer_weights(
    mixer: &mut LogisticMixer,
    saved: &Option<Vec<f32>>,
) {
    if let Some(weights) = saved {
        mixer.set_weights(weights.clone());
    }
}

fn persist_state(
    global_mixer_weights: &mut Option<Vec<f32>>,
    mixer: &LogisticMixer,
    models: &[Box<dyn BitModel>],
    lzp_table: &mut Option<Vec<u32>>,
    lzp_next: &mut Option<Vec<u32>>,
    lzp_current_pos: &mut usize,
    lzp_history: &mut Option<Vec<u8>>,
) {
    // Decay mixer weights instead of clearing them (EWC-style).
    if let Some(weights) = global_mixer_weights {
        for w in weights.iter_mut() {
            *w *= 0.995;
        }
    } else {
        *global_mixer_weights = Some(mixer.weights().to_vec());
    }
    let decayed = global_mixer_weights
        .as_mut()
        .expect("global_mixer_weights is set");
    for (d, s) in decayed.iter_mut().zip(mixer.weights()) {
        *d = *d * 0.995 + s * (1.0 - 0.995);
    }

    for m in models.iter() {
        if let Some(lzp) = m.as_any().downcast_ref::<crate::model::lzp::Lzp>() {
            // retain only positions within the sliding window
            let mut table = lzp.heads().to_vec();
            let mut nxt = lzp.nexts().to_vec();
            let pos = lzp.current_pos();
            let limit = pos.saturating_sub(lzp.window());
            for h in &mut table {
                let mut cur = *h;
                while cur != 0 && (cur >= pos as u32 || cur < limit as u32) {
                    cur = nxt[cur as usize];
                }
                *h = cur;
            }
            *lzp_table = Some(table);
            *lzp_next = Some(nxt);
            *lzp_current_pos = pos;
            *lzp_history = Some(lzp.history().to_vec());
            break;
        }
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
            // Text-optimized stack: full hybrid + WordModel + LazyLzp.
            // LazyLzp adds multi-context hash chains + longest-match selection.
            let n = 8;
            let models: Vec<Box<dyn BitModel>> = vec![
                Box::new(crate::model::order::OrderN::new(0)),
                Box::new(crate::model::order::OrderN::new(1)),
                Box::new(crate::model::order::OrderN::new(2)),
                Box::new(crate::model::sparse::Sparse::new()),
                Box::new(crate::model::exec::Exec::new()),
                Box::new(crate::model::lazy_lzp::LazyLzp::new()),
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

fn compress_block(
    models: &mut [Box<dyn BitModel>],
    _mixer: &mut LogisticMixer,
    block: &[u8],
) -> Vec<u8> {
    let mut enc = BitEncoder::new();
    // Stack-allocate probability buffer (max 9 models in Text stack).
    let mut probs: [u16; 10] = [2048; 10];
    let n = models.len();
    let mut bank = MixerBank::new(n, NUM_MIXERS);
    let mut prev_byte: u8 = 0;
    let mut order1: u8 = 0;
    let mut cascade = crate::model::sse_apm::SseApmCascade::new();
    cascade.set_lr(0.02);

    for &byte in block {
        let cls = ByteClass::classify(byte);
        for bit_idx in (0..8).rev() {
            let bit = (byte >> bit_idx) & 1 == 1;
            let bit_pos = bit_idx as u8;
            for (i, m) in models.iter().enumerate() {
                probs[i] = m.predict();
            }
            let mixer_id = MixerBank::mixer_id(cls, bit_pos, order1, prev_byte, 0);
            let p_mixer = bank.mix(&probs[..n], mixer_id);
            let p = cascade.refine(p_mixer, bit_pos);
            enc.encode_bit(bit, p);
            bank.update(&probs[..n], bit, mixer_id);
            cascade.update(bit, p_mixer, bit_pos);
            for m in models.iter_mut() {
                m.update(bit);
            }
        }
        cascade.set_context(byte);
        order1 = prev_byte;
        prev_byte = byte;
    }
    enc.finish()
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
    for (bi, entry) in entries.iter().enumerate() {
        let comp = &payloads[pos..pos + entry.comp_len as usize];
        pos += entry.comp_len as usize;

        let block = if entry.method == METHOD_COPY {
            comp.to_vec()
        } else {
            let kind = kind_for_method(entry.method)?;
            let (mut models, mut mixer) = build_stack_for_kind(kind);
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

fn decompress_block(
    comp: &[u8],
    orig_len: usize,
    models: &mut [Box<dyn BitModel>],
    mixer: &mut LogisticMixer,
) -> Result<Vec<u8>> {
    let mut dec = BitDecoder::new(comp).map_err(|e| NyxError::Entropy(e.to_string()))?;
    let mut out = Vec::with_capacity(orig_len);
    // Stack-allocate probability buffer (max 9 models in Text stack).
    let mut probs: [u16; 10] = [2048; 10];
    let n = models.len();
    let mut cascade = crate::model::sse_apm::SseApmCascade::new();
    cascade.set_lr(0.02);

    while out.len() < orig_len {
        let mut byte = 0u8;
        for bit_idx in (0..8).rev() {
            let bit_pos = bit_idx as u8;
            for (i, m) in models.iter().enumerate() {
                probs[i] = m.predict();
            }
            let p_mixer = mixer.mix(&probs[..n], bit_pos);
            let p = cascade.refine(p_mixer, bit_pos);
            let bit = dec
                .decode_bit(p)
                .map_err(|e| NyxError::Entropy(e.to_string()))?;
            mixer.update(&probs[..n], bit, bit_pos);
            cascade.update(bit, p_mixer, bit_pos);
            for m in models.iter_mut() {
                m.update(bit);
            }
            if bit {
                byte |= 1 << bit_idx;
            }
        }
        out.push(byte);
        // Mirror encoder: update cascade context AFTER this byte completes,
        // so the next byte's SSE/APM/APM2 uses the same prev/order1/order2
        // as the encoder did.
        cascade.set_context(byte);
    }
    Ok(out)
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
    fn kind_for_method_round_trips() {
        assert_eq!(
            kind_for_method(METHOD_COPY).unwrap(),
            crate::classify::BlockKind::Random
        );
        assert_eq!(
            kind_for_method(METHOD_TEXT).unwrap(),
            crate::classify::BlockKind::Text
        );
        assert_eq!(
            kind_for_method(METHOD_BINARY).unwrap(),
            crate::classify::BlockKind::Binary
        );
        assert_eq!(
            kind_for_method(METHOD_EXEC).unwrap(),
            crate::classify::BlockKind::Exec
        );
    }

    #[test]
    fn unknown_method_errors() {
        assert!(kind_for_method(99).is_err());
    }
}
