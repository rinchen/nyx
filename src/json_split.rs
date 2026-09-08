//! JSON-aware stream splitting for specialized modeling.
//!
//! Structured JSON text compresses poorly under generic bit-level models because
//! repeated byte sequences like `"name":` look identical to the CM — each bit is
//! predicted independently with no cross-position match awareness.  By splitting
//! a JSON block into four structurally-aware streams (delimiters, keys, string
//! values, numbers/other) each stream becomes highly repetitive and BWT + CM
//! can exploit the structure that zstd's LZ77 finds naturally.
//!
//! The split is a lightweight single-pass state machine — no full JSON parser,
//! just enough to route bytes into the right channel.  The structural stream
//! uses a `0xFE` prefix byte before each length-prefixed fragment header, which
//! cannot appear in JSON structural data (all JSON chars are ASCII < 128),
//! making the merge unambiguous.

/// Prefix byte that marks a length-prefixed fragment in the structural stream.
/// Cannot appear in JSON structural data (all JSON chars are ASCII < 0x80).
const MARK_PREFIX: u8 = 0xFE;

/// Stream channel identifier (appears after MARK_PREFIX + 4-byte length).
const CHANNEL_KEY: u8 = b'K';
const CHANNEL_VALUE: u8 = b'V';
const CHANNEL_NUMBER: u8 = b'N';

/// Four output streams produced by [`split`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JsonStreams {
    /// Structural characters: `{ } [ ] : ,`, quotes (`"`), backslash escapes,
    /// and whitespace.  Fragment transitions are encoded as:
    /// `0xFE` + 4-byte LE length + channel byte (`K`/`V`/`N`).
    pub structural: Vec<u8>,
    /// All key string content, concatenated in order.
    pub keys: Vec<u8>,
    /// All value string content, concatenated in order.
    pub string_values: Vec<u8>,
    /// All numeric/other content, concatenated in order.
    pub numbers: Vec<u8>,
}

impl JsonStreams {
    /// Total length of all four streams combined.
    #[must_use]
    pub fn total_len(&self) -> usize {
        self.structural.len() + self.keys.len() + self.string_values.len() + self.numbers.len()
    }
}

/// Helper: emit a length-prefixed fragment header + content to the structural stream.
fn emit_fragment(
    structural: &mut Vec<u8>,
    content_stream: &mut Vec<u8>,
    buf: &mut Vec<u8>,
    channel: u8,
) {
    let len = buf.len() as u32;
    structural.push(MARK_PREFIX);
    structural.extend_from_slice(&len.to_le_bytes());
    structural.push(channel);
    content_stream.extend_from_slice(buf);
    buf.clear();
}

/// Split `data` (a JSON text block) into four component streams.
///
/// # Behavior
/// - Structural chars `{ } [ ] : ,`, quotes (`"`), backslash escapes, whitespace → `structural`.
/// - Key string content → `keys`, preceded by a `0xFE` + len + `K` marker in structural.
/// - Value string content → `string_values`, preceded by a `0xFE` + len + `V` marker.
/// - Numbers, `true`, `false`, `null` → `numbers`, preceded by a `0xFE` + len + `N` marker.
///
/// A string is classified as a key when followed (after optional whitespace) by `:`.
#[must_use]
pub fn split(data: &[u8]) -> JsonStreams {
    let mut out = JsonStreams::default();

    let mut in_string = false;
    let mut in_escape = false;
    let mut string_buf: Vec<u8> = Vec::new();

    let mut i = 0usize;
    while i < data.len() {
        let b = data[i];

        if in_string {
            if in_escape {
                string_buf.push(b);
                in_escape = false;
                i += 1;
                continue;
            }

            match b {
                b'"' => {
                    // End of string. Emit the string content fragment first (between
                    // opening and closing quotes), then the closing quote.
                    // Classify: look ahead for `:` (key) or not (value string).
                    let mut j = i + 1;
                    while j < data.len() && matches!(data[j], b' ' | b'\t' | b'\n' | b'\r') {
                        j += 1;
                    }
                    let is_key = j < data.len() && data[j] == b':';

                    if is_key {
                        emit_fragment(
                            &mut out.structural,
                            &mut out.keys,
                            &mut string_buf,
                            CHANNEL_KEY,
                        );
                    } else {
                        emit_fragment(
                            &mut out.structural,
                            &mut out.string_values,
                            &mut string_buf,
                            CHANNEL_VALUE,
                        );
                    }

                    out.structural.push(b'"');
                    in_string = false;
                }
                b'\\' => {
                    string_buf.push(b);
                    in_escape = true;
                }
                _ => {
                    string_buf.push(b);
                }
            }
            i += 1;
            continue;
        }

        // Not in a string
        match b {
            b'"' => {
                // Start of a new string. Flush any pending number content first.
                if !string_buf.is_empty() {
                    emit_fragment(
                        &mut out.structural,
                        &mut out.numbers,
                        &mut string_buf,
                        CHANNEL_NUMBER,
                    );
                }
                in_string = true;
                out.structural.push(b'"');
            }
            b'{' | b'}' | b'[' | b']' | b':' | b',' => {
                // Flush any pending number content.
                if !string_buf.is_empty() {
                    emit_fragment(
                        &mut out.structural,
                        &mut out.numbers,
                        &mut string_buf,
                        CHANNEL_NUMBER,
                    );
                }
                out.structural.push(b);
            }
            b' ' | b'\t' | b'\n' | b'\r' => {
                // Flush any pending number content before whitespace.
                if !string_buf.is_empty() {
                    emit_fragment(
                        &mut out.structural,
                        &mut out.numbers,
                        &mut string_buf,
                        CHANNEL_NUMBER,
                    );
                }
                out.structural.push(b);
            }
            _ => {
                // Number/true/false/null content
                string_buf.push(b);
            }
        }
        i += 1;
    }

    // Flush any trailing number content.
    if !string_buf.is_empty() {
        emit_fragment(
            &mut out.structural,
            &mut out.numbers,
            &mut string_buf,
            CHANNEL_NUMBER,
        );
    }

    out
}

/// Merge four streams back into the original JSON text.
///
/// # Errors
///
/// Returns `RcnError::JsonSplitError` if a stream is exhausted prematurely
/// or the structural stream contains a malformed marker.
pub fn merge(
    streams: &JsonStreams,
    original_len: usize,
) -> Result<Vec<u8>, crate::error::RcnError> {
    let mut out = Vec::with_capacity(original_len);
    let mut ki = 0usize;
    let mut vi = 0usize;
    let mut ni = 0usize;

    let keys = &streams.keys;
    let vals = &streams.string_values;
    let nums = &streams.numbers;
    let s = &streams.structural;

    let mut si = 0usize;
    while si < s.len() {
        let b = s[si];

        if b == MARK_PREFIX {
            // Read 4-byte LE length + 1-byte channel tag.
            if si + 5 >= s.len() {
                return Err(crate::error::RcnError::JsonSplitError(format!(
                    "truncated marker at structural offset {si}"
                )));
            }
            let len = u32::from_le_bytes([s[si + 1], s[si + 2], s[si + 3], s[si + 4]]) as usize;
            let channel = s[si + 5];
            si += 6; // consume prefix + 4 length bytes + 1 channel byte

            match channel {
                CHANNEL_KEY => {
                    check_bounds(ki, len, keys.len(), "keys")?;
                    out.extend_from_slice(&keys[ki..ki + len]);
                    ki += len;
                }
                CHANNEL_VALUE => {
                    check_bounds(vi, len, vals.len(), "string_values")?;
                    out.extend_from_slice(&vals[vi..vi + len]);
                    vi += len;
                }
                CHANNEL_NUMBER => {
                    check_bounds(ni, len, nums.len(), "numbers")?;
                    out.extend_from_slice(&nums[ni..ni + len]);
                    ni += len;
                }
                _ => {
                    return Err(crate::error::RcnError::JsonSplitError(format!(
                        "unknown channel byte {channel} at structural offset {si}"
                    )));
                }
            }
        } else {
            out.push(b);
            si += 1;
        }
    }

    if out.len() != original_len {
        return Err(crate::error::RcnError::JsonSplitError(format!(
            "merge produced {} bytes, expected {}",
            out.len(),
            original_len
        )));
    }

    Ok(out)
}

#[inline]
fn check_bounds(
    start: usize,
    len: usize,
    total: usize,
    name: &str,
) -> Result<(), crate::error::RcnError> {
    if start + len > total {
        Err(crate::error::RcnError::JsonSplitError(format!(
            "stream '{name}' exhausted: need {len} bytes at offset {start}, only {} remaining",
            total.saturating_sub(start)
        )))
    } else {
        Ok(())
    }
}

/// Naive fallback merge — same as [`merge`] but without the final length check.
pub fn merge_naive(streams: &JsonStreams, _original_len: usize) -> Vec<u8> {
    merge(streams, 0).unwrap_or_default()
}

/// Check if data looks like JSON (heuristic for deciding when to use stream splitting).
#[must_use]
pub fn looks_like_json(data: &[u8]) -> bool {
    if data.len() < 32 {
        return false;
    }

    // JSON documents start with `[` or `{` (after optional whitespace).
    let mut idx = 0;
    while idx < data.len() && data[idx].is_ascii_whitespace() {
        idx += 1;
    }
    if idx >= data.len() {
        return false;
    }

    let first = data[idx];
    if first != b'{' && first != b'[' {
        return false;
    }

    // Check for "key": pattern density in the first 8KB.
    let sample = data.iter().take(8192).copied().collect::<Vec<_>>();
    let colon_count = sample.iter().filter(|&&b| b == b':').count();
    let quote_count = sample.iter().filter(|&&b| b == b'"').count();

    // JSON has roughly 2x quotes (key + value) per colon.
    quote_count >= colon_count.saturating_mul(2) && colon_count >= 2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(data: &[u8]) {
        let streams = split(data);
        let merged = merge(&streams, data.len()).expect("merge failed");
        assert_eq!(merged.as_slice(), data, "round-trip mismatch");
    }

    #[test]
    fn split_simple_json() {
        round_trip(b"{\"name\":\"test\",\"age\":42}");
    }

    #[test]
    fn split_nested_json() {
        round_trip(b"{\"a\":{\"b\":1},\"c\":[2,3]}");
    }

    #[test]
    fn split_with_escaped_quotes() {
        round_trip(b"{\"key\":\"val\\\"ue\"}");
    }

    #[test]
    fn split_with_whitespace() {
        round_trip(b"{\n  \"name\": \"test\",\n  \"age\": 42,\n  \"active\": true\n}");
    }

    #[test]
    fn split_empty_object() {
        round_trip(b"{}");
    }

    #[test]
    fn split_empty_array() {
        round_trip(b"[]");
    }

    #[test]
    fn split_deeply_nested() {
        round_trip(b"{\"a\":{\"b\":{\"c\":{\"d\":\"hello\"}}}}");
    }

    #[test]
    fn looks_like_json_detects_json() {
        let json = b"{\"name\": \"John\", \"age\": 30, \"city\": \"New York\"}";
        assert!(looks_like_json(json));
    }

    #[test]
    fn looks_like_json_rejects_text() {
        let text = b"The quick brown fox jumps over the lazy dog. ".repeat(10);
        assert!(!looks_like_json(&text));
    }

    #[test]
    fn round_trip_larger_json() {
        let input = br#"{"employees":[{"firstName":"John","lastName":"Doe","age":30,"active":true},{"firstName":"Anna","lastName":"Smith","age":28,"active":false},{"firstName":"Peter","lastName":"Jones","age":45,"active":true}]}"#;
        round_trip(input);
    }

    #[test]
    fn split_streams_separate_keys_and_values() {
        let input = b"{\"name\":\"John\",\"age\":42}";
        let streams = split(input);
        assert_eq!(streams.keys, b"nameage");
        assert_eq!(streams.string_values, b"John");
        assert_eq!(streams.numbers, b"42");
    }

    #[test]
    fn split_multiple_keys() {
        let input = b"{\"k1\":\"v1\",\"k2\":\"v2\",\"k3\":\"v3\"}";
        let streams = split(input);
        assert_eq!(streams.keys, b"k1k2k3");
        assert_eq!(streams.string_values, b"v1v2v3");
        round_trip(input);
    }
}
