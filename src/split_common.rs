//! Shared framing helpers for JSON/CSV/XML stream splitters.

/// Prefix byte that marks a length-prefixed fragment in a structural stream.
/// Must not appear in well-formed ASCII structural markup (JSON/XML).
pub const MARK_PREFIX: u8 = 0xFE;

/// Default sample window for `looks_like_*` heuristics.
pub const LOOKS_LIKE_SAMPLE: usize = 8192;

/// Emit a `0xFE` + u32 LE length + channel marker, then append `buf` to `content`.
pub fn emit_marked_fragment(
    structural: &mut Vec<u8>,
    content: &mut Vec<u8>,
    buf: &mut Vec<u8>,
    channel: u8,
) {
    let len = buf.len() as u32;
    structural.push(MARK_PREFIX);
    structural.extend_from_slice(&len.to_le_bytes());
    structural.push(channel);
    content.extend_from_slice(buf);
    buf.clear();
}

/// Parse a marked fragment header starting at `si` where `s[si] == MARK_PREFIX`.
///
/// Returns `(len, channel, new_si)` or `None` if the marker is truncated.
#[must_use]
pub fn parse_marked_fragment(s: &[u8], si: usize) -> Option<(usize, u8, usize)> {
    // Need prefix + 4 length bytes + 1 channel = 6 bytes total (si..si+5 inclusive).
    if si + 5 >= s.len() {
        return None;
    }
    let len = u32::from_le_bytes([s[si + 1], s[si + 2], s[si + 3], s[si + 4]]) as usize;
    let channel = s[si + 5];
    Some((len, channel, si + 6))
}

/// Ensure `start + len <= total` for a content stream named `name`.
pub fn check_stream_bounds(
    start: usize,
    len: usize,
    total: usize,
    name: &str,
) -> Result<(), String> {
    if start + len > total {
        Err(format!(
            "stream '{name}' exhausted: need {len} bytes at offset {start}, only {} remaining",
            total.saturating_sub(start)
        ))
    } else {
        Ok(())
    }
}

/// Index of the first non-ASCII-whitespace byte, or `data.len()` if all whitespace.
#[must_use]
pub fn skip_ascii_whitespace(data: &[u8]) -> usize {
    let mut idx = 0;
    while idx < data.len() && data[idx].is_ascii_whitespace() {
        idx += 1;
    }
    idx
}

/// First `LOOKS_LIKE_SAMPLE` bytes of `data` (or all of it if shorter).
#[must_use]
pub fn sample_prefix(data: &[u8]) -> &[u8] {
    &data[..data.len().min(LOOKS_LIKE_SAMPLE)]
}

/// Write a u32 LE length prefix followed by `bytes`.
pub fn write_len_prefixed(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// Read a u32 LE length-prefixed blob starting at `pos`.
///
/// Returns `(bytes, new_pos)` or an error string on truncation/overrun.
pub fn read_len_prefixed(buf: &[u8], pos: usize) -> Result<(Vec<u8>, usize), String> {
    if pos + 4 > buf.len() {
        return Err(format!("truncated length prefix at offset {pos}"));
    }
    let len = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
    let start = pos + 4;
    if start + len > buf.len() {
        return Err(format!(
            "length-prefixed blob of {len} bytes at offset {start} overruns buffer"
        ));
    }
    Ok((buf[start..start + len].to_vec(), start + len))
}

/// Read a little-endian `u32` at `pos`, or error if truncated.
pub fn read_u32_le(buf: &[u8], pos: usize) -> Result<u32, String> {
    if pos + 4 > buf.len() {
        return Err(format!("truncated u32 at offset {pos}"));
    }
    Ok(u32::from_le_bytes([
        buf[pos],
        buf[pos + 1],
        buf[pos + 2],
        buf[pos + 3],
    ]))
}

/// Read a little-endian `u16` at `pos`, or error if truncated.
pub fn read_u16_le(buf: &[u8], pos: usize) -> Result<u16, String> {
    if pos + 2 > buf.len() {
        return Err(format!("truncated u16 at offset {pos}"));
    }
    Ok(u16::from_le_bytes([buf[pos], buf[pos + 1]]))
}
