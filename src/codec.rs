//! Core codec: glue the classifier, bit models, logistic mixer, rANS backend, and the
//! `NYX1` container into `compress` / `decompress`.
//!
//! Strategy per block:
//! - `Random` blocks are stored verbatim (copy record, method 0).
//! - Everything else (method 1) is modeled bit-by-bit: a stack of context models
//!   (order-0/1/2, sparse, executable 2D, LZP) feeds a logistic mixer; the fused
//!   probability drives the rANS bit coder. Because modeling is causal, the decoder
//!   reconstructs the exact same model state from the decoded stream and round-trips
//!   losslessly.

use crate::container::{BlockEntry, Header, VERSION};
use crate::entropy::range::{BitDecoder, BitEncoder};
use crate::error::{NyxError, Result};
use crate::model::mixer::LogisticMixer;
use crate::model::BitModel;

/// Default block size: 64 KiB. `block_size_log = 16`.
pub const DEFAULT_BLOCK_SIZE_LOG: u8 = 16;
const METHOD_COPY: u8 = 0;
const METHOD_CM: u8 = 1;

/// Compress `buf` into a `NYX1` container using the default model stack.
///
/// # Errors
///
/// Returns [`NyxError`] if an entropy primitive fails.
pub fn compress(buf: &[u8]) -> Result<Vec<u8>> {
    compress_with(buf, &mut build_full_stack)
}

/// Compress `buf` using a custom model-stack builder.
///
/// # Errors
///
/// Returns [`NyxError`] if an entropy primitive fails.
pub fn compress_with<'a, F>(buf: &[u8], build_stack: &mut F) -> Result<Vec<u8>>
where
    F: FnMut() -> (Vec<Box<dyn BitModel>>, LogisticMixer) + 'a,
{
    let block_size = 1usize << DEFAULT_BLOCK_SIZE_LOG;
    let mut out = Vec::new();
    let mut entries: Vec<BlockEntry> = Vec::new();
    let mut payloads: Vec<u8> = Vec::new();
    let mut offset = 0;

    while offset < buf.len() {
        let end = (offset + block_size).min(buf.len());
        let block = &buf[offset..end];
        let comp = compress_block(block, build_stack);
        let entry = BlockEntry {
            comp_len: comp.len() as u32,
            orig_len: block.len() as u32,
            method: METHOD_CM,
            crc32: crc32(block),
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

fn compress_block<F>(block: &[u8], build_stack: &mut F) -> Vec<u8>
where
    F: FnMut() -> (Vec<Box<dyn BitModel>>, LogisticMixer),
{
    let (mut models, mut mixer) = build_stack();
    let mut enc = BitEncoder::new();
    let mut probs: Vec<u16> = vec![2048; models.len()];

    for &byte in block {
        for bit_idx in (0..8).rev() {
            let bit = (byte >> bit_idx) & 1 == 1;
            let bit_pos = bit_idx as u8;
            for (i, m) in models.iter().enumerate() {
                probs[i] = m.predict();
            }
            let p = mixer.mix(&probs, bit_pos);
            enc.encode_bit(bit, p);
            mixer.update(&probs, bit, bit_pos);
            for m in &mut models {
                m.update(bit);
            }
        }
    }

    enc.finish()
}

/// Build the full per-block model stack (encode and decode must agree).
#[must_use]
pub fn build_full_stack() -> (Vec<Box<dyn BitModel>>, LogisticMixer) {
    let models: Vec<Box<dyn BitModel>> = vec![
        Box::new(crate::model::order::OrderN::new(0)),
        Box::new(crate::model::order::OrderN::new(1)),
        Box::new(crate::model::order::OrderN::new(2)),
        Box::new(crate::model::sparse::Sparse::new()),
        Box::new(crate::model::lzp::Lzp::new()),
        Box::new(crate::model::exec::Exec::new()),
        Box::new(crate::model::ppm::PpmModel::new(3)),
    ];
    let mixer = LogisticMixer::new(models.len());
    (models, mixer)
}

/// Decompress a `NYX1` container back to the original bytes.
///
/// # Errors
///
/// Returns [`NyxError`] on a malformed container, corrupt block, or CRC mismatch.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    decompress_with(data, &mut build_full_stack)
}

/// Decompress a `NYX1` container using a custom model-stack builder.
///
/// # Errors
///
/// Returns [`NyxError`] on a malformed container, corrupt block, or CRC mismatch.
pub fn decompress_with<'a, F>(data: &[u8], build_stack: &mut F) -> Result<Vec<u8>>
where
    F: FnMut() -> (Vec<Box<dyn BitModel>>, LogisticMixer) + 'a,
{
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
        } else if entry.method == METHOD_CM {
            decompress_block(comp, entry.orig_len as usize, build_stack)
                .map_err(|e| match e {
                    NyxError::Entropy(s) => NyxError::CorruptBlock(s),
                    other => other,
                })?
        } else {
            return Err(NyxError::InvalidContainer(format!(
                "unknown method {} in block {bi}",
                entry.method
            )));
        };

        if crate::container::crc32(&block) != entry.crc32 {
            return Err(NyxError::CrcMismatch(bi, crate::container::crc32(&block), entry.crc32));
        }
        out.extend_from_slice(&block);
    }
    Ok(out)
}

fn decompress_block<F>(comp: &[u8], orig_len: usize, build_stack: &mut F) -> Result<Vec<u8>>
where
    F: FnMut() -> (Vec<Box<dyn BitModel>>, LogisticMixer),
{
    let (mut models, mut mixer) = build_stack();
    let mut dec = BitDecoder::new(comp).map_err(|e| NyxError::Entropy(e.to_string()))?;
    let mut out = Vec::with_capacity(orig_len);
    let mut probs: Vec<u16> = vec![2048; models.len()];

    while out.len() < orig_len {
        let mut byte = 0u8;
        for bit_idx in (0..8).rev() {
            let bit_pos = bit_idx as u8;
            for (i, m) in models.iter().enumerate() {
                probs[i] = m.predict();
            }
            let p = mixer.mix(&probs, bit_pos);
            let bit = dec
                .decode_bit(p)
                .map_err(|e| NyxError::Entropy(e.to_string()))?;
            mixer.update(&probs, bit, bit_pos);
            for m in &mut models {
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
}
