//! XML-aware stream splitting for specialized modeling.
//!
//! Coarse three-way split (tags / attributes / text) via a single-pass scan —
//! not a full XML parser. Structural markup and length-prefixed fragment
//! markers live in the tags stream; attribute and text payloads are routed to
//! their own channels so BWT sees highly repetitive streams.

use crate::split_common::{
    check_stream_bounds, emit_marked_fragment, parse_marked_fragment, skip_ascii_whitespace,
    MARK_PREFIX,
};

const CHANNEL_ATTR: u8 = b'A';
const CHANNEL_TEXT: u8 = b'T';

/// Three output streams produced by [`split`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XmlStreams {
    /// Tag markup (`<`, `/`, `>`, `?`, `!`, tag names) plus fragment markers
    /// `0xFE` + 4-byte LE length + channel (`A`/`T`).
    pub tags: Vec<u8>,
    /// Attribute name/value content (inside tags, after the name).
    pub attrs: Vec<u8>,
    /// Text node content between tags.
    pub text: Vec<u8>,
}

impl XmlStreams {
    /// Total length of all three streams combined.
    #[must_use]
    pub fn total_len(&self) -> usize {
        self.tags.len() + self.attrs.len() + self.text.len()
    }
}

/// Heuristic: starts with `<` / `<?xml` / has tag-like structure.
#[must_use]
pub fn looks_like_xml(data: &[u8]) -> bool {
    if data.len() < 16 {
        return false;
    }
    let idx = skip_ascii_whitespace(data);
    if idx >= data.len() || data[idx] != b'<' {
        return false;
    }

    // `<?xml` is a strong signal.
    if data[idx..].starts_with(b"<?xml") || data[idx..].starts_with(b"<?XML") {
        return true;
    }

    // Count tag-like `<...>` pairs in the first 8KB.
    let sample = &data[..data.len().min(8192)];
    if !sample.is_ascii() {
        // Allow non-ASCII text content but require ASCII-heavy markup sample.
        let ascii_frac = sample.iter().filter(|b| b.is_ascii()).count() * 100 / sample.len();
        if ascii_frac < 70 {
            return false;
        }
    }

    let lt = sample.iter().filter(|&&b| b == b'<').count();
    let gt = sample.iter().filter(|&&b| b == b'>').count();
    if lt < 2 || gt < 2 {
        return false;
    }
    // Rough balance and density.
    let diff = lt.abs_diff(gt);
    diff <= lt / 2 + 1
}

/// Split `data` into tags / attrs / text streams.
#[must_use]
pub fn split(data: &[u8]) -> XmlStreams {
    let mut out = XmlStreams::default();
    let mut text_buf: Vec<u8> = Vec::new();
    let mut i = 0usize;

    while i < data.len() {
        if data[i] == b'<' {
            // Flush pending text.
            if !text_buf.is_empty() {
                emit_marked_fragment(&mut out.tags, &mut out.text, &mut text_buf, CHANNEL_TEXT);
            }

            out.tags.push(b'<');
            i += 1;
            if i >= data.len() {
                break;
            }

            // Optional `/`, `?`, or `!` after `<`.
            if matches!(data[i], b'/' | b'?' | b'!') {
                out.tags.push(data[i]);
                i += 1;
            }

            // Tag name: continue until whitespace, `>`, or `/`.
            while i < data.len() && !matches!(data[i], b' ' | b'\t' | b'\n' | b'\r' | b'>' | b'/') {
                out.tags.push(data[i]);
                i += 1;
            }

            // Attribute region until `>`. A trailing `/` or `?` before `>`
            // belongs in the tags stream (self-close / PI end), not attrs.
            let mut attr_buf = Vec::new();
            while i < data.len() && data[i] != b'>' {
                attr_buf.push(data[i]);
                i += 1;
            }
            let trailing = match attr_buf.last().copied() {
                Some(b'/') | Some(b'?') => attr_buf.pop(),
                _ => None,
            };
            if !attr_buf.is_empty() {
                emit_marked_fragment(&mut out.tags, &mut out.attrs, &mut attr_buf, CHANNEL_ATTR);
            }
            if let Some(t) = trailing {
                out.tags.push(t);
            }
            if i < data.len() && data[i] == b'>' {
                out.tags.push(b'>');
                i += 1;
            }
        } else {
            text_buf.push(data[i]);
            i += 1;
        }
    }

    if !text_buf.is_empty() {
        emit_marked_fragment(&mut out.tags, &mut out.text, &mut text_buf, CHANNEL_TEXT);
    }

    out
}

/// Merge streams back into the original XML text.
///
/// # Errors
///
/// Returns [`RcnError::XmlSplitError`] on truncated markers or stream underrun.
pub fn join(streams: &XmlStreams, original_len: usize) -> Result<Vec<u8>, crate::error::RcnError> {
    let mut out = Vec::with_capacity(original_len);
    let mut ai = 0usize;
    let mut ti = 0usize;
    let s = &streams.tags;
    let mut si = 0usize;

    while si < s.len() {
        let b = s[si];
        if b == MARK_PREFIX {
            let (len, channel, new_si) = parse_marked_fragment(s, si).ok_or_else(|| {
                crate::error::RcnError::XmlSplitError(format!(
                    "truncated marker at tags offset {si}"
                ))
            })?;
            si = new_si;
            match channel {
                CHANNEL_ATTR => {
                    check_stream_bounds(ai, len, streams.attrs.len(), "attrs")
                        .map_err(crate::error::RcnError::XmlSplitError)?;
                    out.extend_from_slice(&streams.attrs[ai..ai + len]);
                    ai += len;
                }
                CHANNEL_TEXT => {
                    check_stream_bounds(ti, len, streams.text.len(), "text")
                        .map_err(crate::error::RcnError::XmlSplitError)?;
                    out.extend_from_slice(&streams.text[ti..ti + len]);
                    ti += len;
                }
                _ => {
                    return Err(crate::error::RcnError::XmlSplitError(format!(
                        "unknown channel byte {channel}"
                    )));
                }
            }
        } else {
            out.push(b);
            si += 1;
        }
    }

    if out.len() != original_len {
        return Err(crate::error::RcnError::XmlSplitError(format!(
            "merge produced {} bytes, expected {original_len}",
            out.len()
        )));
    }
    Ok(out)
}

/// Alias matching the task naming.
#[must_use]
pub fn split_xml(data: &[u8]) -> Vec<Vec<u8>> {
    let s = split(data);
    vec![s.tags, s.attrs, s.text]
}

/// Alias matching the task naming.
pub fn join_xml(streams: &[Vec<u8>], original_len: usize) -> Result<Vec<u8>, crate::error::RcnError> {
    if streams.len() < 3 {
        return Err(crate::error::RcnError::XmlSplitError(
            "expected 3 streams (tags, attrs, text)".into(),
        ));
    }
    join(
        &XmlStreams {
            tags: streams[0].clone(),
            attrs: streams[1].clone(),
            text: streams[2].clone(),
        },
        original_len,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(data: &[u8]) {
        let streams = split(data);
        let merged = join(&streams, data.len()).expect("join failed");
        assert_eq!(merged.as_slice(), data, "round-trip mismatch");
    }

    #[test]
    fn split_simple_xml() {
        round_trip(b"<root><item id=\"1\">hello</item></root>");
    }

    #[test]
    fn split_xml_declaration() {
        round_trip(b"<?xml version=\"1.0\"?><a>x</a>");
    }

    #[test]
    fn split_self_closing() {
        round_trip(b"<root><br/><img src=\"x.png\"/></root>");
    }

    #[test]
    fn split_with_text_and_attrs() {
        round_trip(b"<person name=\"Ada\" age=\"36\">Lovelace</person>");
    }

    #[test]
    fn looks_like_xml_detects() {
        let xml = b"<?xml version=\"1.0\"?><catalog><book id=\"1\"/><book id=\"2\"/></catalog>";
        assert!(looks_like_xml(xml));
    }

    #[test]
    fn looks_like_xml_detects_plain_tags() {
        let xml = b"<html><body><p>Hello world</p><p>More text</p></body></html>";
        assert!(looks_like_xml(xml));
    }

    #[test]
    fn looks_like_xml_rejects_text() {
        let text = b"The quick brown fox jumps over the lazy dog. ".repeat(10);
        assert!(!looks_like_xml(&text));
    }

    #[test]
    fn streams_separate_attrs_and_text() {
        let streams = split(b"<a href=\"x\">hi</a>");
        assert!(!streams.attrs.is_empty());
        assert_eq!(streams.text, b"hi");
    }
}
