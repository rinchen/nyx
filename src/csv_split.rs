//! CSV-aware column-stream splitting for specialized modeling.
//!
//! Spreadsheet-like CSV/TSV compresses poorly as a single byte stream because
//! values in the same column are far apart. Splitting into per-column streams
//! (padding shorter rows with empty cells) groups like values for BWT + CM.
//!
//! Each column stream stores length-prefixed cells (`u32` LE length + bytes),
//! one per row. [`fields_per_row`] records the true field count before padding
//! so [`join`] can reconstruct the original delimiter layout exactly.

use crate::split_common::{
    read_len_prefixed, sample_prefix, skip_ascii_whitespace, write_len_prefixed,
};

/// Parsed CSV ready for per-column BWT.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CsvStreams {
    /// Field delimiter (`b','` or `b'\t'`).
    pub delim: u8,
    /// One stream per column. Cells are `u32`-LE length-prefixed, padded to
    /// `nrows` entries (empty cells for missing fields).
    pub columns: Vec<Vec<u8>>,
    /// Actual number of fields on each row (before padding).
    pub fields_per_row: Vec<u16>,
    /// Whether the input ended with a newline.
    pub trailing_newline: bool,
}

impl CsvStreams {
    /// Number of rows (== `fields_per_row.len()`).
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.fields_per_row.len()
    }

    /// Number of columns (max field count, after padding).
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    /// Total bytes across all column streams.
    #[must_use]
    pub fn total_len(&self) -> usize {
        self.columns.iter().map(Vec::len).sum()
    }
}

/// Heuristic: many commas/tabs, consistent column counts in the first lines, ASCII.
#[must_use]
pub fn looks_like_csv(data: &[u8]) -> bool {
    if data.len() < 32 {
        return false;
    }
    // Reject obvious JSON / XML.
    let idx = skip_ascii_whitespace(data);
    if idx < data.len() && matches!(data[idx], b'{' | b'[' | b'<') {
        return false;
    }

    let sample = sample_prefix(data);
    if !sample.is_ascii() {
        return false;
    }

    let commas = sample.iter().filter(|&&b| b == b',').count();
    let tabs = sample.iter().filter(|&&b| b == b'\t').count();
    let (delim, delim_count) = if tabs > commas {
        (b'\t', tabs)
    } else {
        (b',', commas)
    };
    if delim_count < 4 {
        return false;
    }

    // Parse first N lines and check column-count consistency.
    let lines: Vec<&[u8]> = sample
        .split(|&b| b == b'\n')
        .map(|l| l.strip_suffix(&[b'\r']).unwrap_or(l))
        .filter(|l| !l.is_empty())
        .take(16)
        .collect();
    if lines.len() < 2 {
        return false;
    }

    let counts: Vec<usize> = lines
        .iter()
        .map(|line| split_row(line, delim).len())
        .collect();
    let first = counts[0];
    if first < 2 {
        return false;
    }
    let consistent = counts.iter().filter(|&&c| c == first).count();
    // At least half of the sampled lines share the same column count.
    consistent * 2 >= counts.len()
}

/// Split a single CSV row on `delim`, respecting double-quoted fields.
///
/// Quoted fields keep their surrounding quotes (and `""` escapes) so [`join`]
/// can emit a bit-identical reconstruction.
fn split_row(line: &[u8], delim: u8) -> Vec<Vec<u8>> {
    let mut fields = Vec::new();
    let mut cur = Vec::new();
    let mut in_quotes = false;
    let mut i = 0;
    while i < line.len() {
        let b = line[i];
        if in_quotes {
            cur.push(b);
            if b == b'"' {
                if i + 1 < line.len() && line[i + 1] == b'"' {
                    // Escaped quote — keep both bytes, stay quoted.
                    cur.push(b'"');
                    i += 2;
                    continue;
                }
                in_quotes = false;
            }
            i += 1;
        } else if b == b'"' {
            in_quotes = true;
            cur.push(b);
            i += 1;
        } else if b == delim {
            fields.push(std::mem::take(&mut cur));
            i += 1;
        } else {
            cur.push(b);
            i += 1;
        }
    }
    fields.push(cur);
    fields
}

fn push_cell(col: &mut Vec<u8>, cell: &[u8]) {
    write_len_prefixed(col, cell);
}

fn detect_delim(data: &[u8]) -> u8 {
    let sample = sample_prefix(data);
    let commas = sample.iter().filter(|&&b| b == b',').count();
    let tabs = sample.iter().filter(|&&b| b == b'\t').count();
    if tabs > commas {
        b'\t'
    } else {
        b','
    }
}

/// Split `data` into per-column streams (shorter rows padded with empty cells).
#[must_use]
pub fn split(data: &[u8]) -> CsvStreams {
    let delim = detect_delim(data);
    let trailing_newline = data.last() == Some(&b'\n');

    let mut rows: Vec<Vec<Vec<u8>>> = Vec::new();
    let mut line_start = 0usize;
    let mut i = 0usize;
    while i < data.len() {
        if data[i] == b'\n' {
            let mut end = i;
            if end > line_start && data[end - 1] == b'\r' {
                end -= 1;
            }
            // Skip a final empty line produced solely by a trailing newline.
            if !(trailing_newline && line_start == i) {
                rows.push(split_row(&data[line_start..end], delim));
            }
            i += 1;
            line_start = i;
        } else {
            i += 1;
        }
    }
    if line_start < data.len() {
        rows.push(split_row(&data[line_start..], delim));
    } else if !trailing_newline && line_start == data.len() && !data.is_empty() {
        // empty trailing row without newline — nothing to add
    }

    let fields_per_row: Vec<u16> = rows
        .iter()
        .map(|r| r.len().min(u16::MAX as usize) as u16)
        .collect();
    let ncols = fields_per_row.iter().map(|&c| c as usize).max().unwrap_or(0);

    let mut columns = vec![Vec::new(); ncols];
    for row in &rows {
        for (c, col) in columns.iter_mut().enumerate() {
            if c < row.len() {
                push_cell(col, &row[c]);
            } else {
                push_cell(col, b"");
            }
        }
    }

    CsvStreams {
        delim,
        columns,
        fields_per_row,
        trailing_newline,
    }
}

/// Merge column streams back into CSV text.
///
/// # Errors
///
/// Returns [`RcnError::CsvSplitError`] if a column stream is truncated or
/// lengths are inconsistent.
pub fn join(streams: &CsvStreams) -> Result<Vec<u8>, crate::error::RcnError> {
    let nrows = streams.nrows();
    let ncols = streams.ncols();
    if streams.fields_per_row.len() != nrows {
        return Err(crate::error::RcnError::CsvSplitError(
            "fields_per_row length mismatch".into(),
        ));
    }

    // Decode length-prefixed cells from each column.
    let mut col_cells: Vec<Vec<Vec<u8>>> = Vec::with_capacity(ncols);
    for (ci, col) in streams.columns.iter().enumerate() {
        let mut cells = Vec::with_capacity(nrows);
        let mut pos = 0usize;
        for _ in 0..nrows {
            let (cell, new_pos) = read_len_prefixed(col, pos).map_err(|e| {
                crate::error::RcnError::CsvSplitError(format!("column {ci}: {e}"))
            })?;
            cells.push(cell);
            pos = new_pos;
        }
        if pos != col.len() {
            return Err(crate::error::RcnError::CsvSplitError(format!(
                "column {ci} has {} trailing bytes",
                col.len() - pos
            )));
        }
        col_cells.push(cells);
    }

    let mut out = Vec::new();
    for row in 0..nrows {
        let nfields = streams.fields_per_row[row] as usize;
        for col in 0..nfields {
            if col > 0 {
                out.push(streams.delim);
            }
            if col < ncols {
                out.extend_from_slice(&col_cells[col][row]);
            }
        }
        if row + 1 < nrows || streams.trailing_newline {
            out.push(b'\n');
        }
    }
    Ok(out)
}

/// Alias matching the task naming.
#[must_use]
pub fn split_csv(data: &[u8]) -> Vec<Vec<u8>> {
    split(data).columns
}

/// Alias matching the task naming. Requires a full [`CsvStreams`] for round-trip;
/// this reconstructs using comma delimiter and inferred field counts (lossy).
/// Prefer [`join`] with the full struct.
pub fn join_csv(columns: &[Vec<u8>]) -> Result<Vec<u8>, crate::error::RcnError> {
    if columns.is_empty() {
        return Ok(Vec::new());
    }
    // Infer nrows from first column's length-prefixed cells.
    let mut nrows = 0usize;
    let mut pos = 0usize;
    while pos + 4 <= columns[0].len() {
        let len = u32::from_le_bytes([
            columns[0][pos],
            columns[0][pos + 1],
            columns[0][pos + 2],
            columns[0][pos + 3],
        ]) as usize;
        pos += 4 + len;
        nrows += 1;
    }
    let fields_per_row = vec![columns.len() as u16; nrows];
    join(&CsvStreams {
        delim: b',',
        columns: columns.to_vec(),
        fields_per_row,
        trailing_newline: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(data: &[u8]) {
        let streams = split(data);
        let merged = join(&streams).expect("join failed");
        assert_eq!(merged.as_slice(), data, "round-trip mismatch");
    }

    #[test]
    fn split_simple_csv() {
        round_trip(b"name,age,city\nJohn,30,NYC\nAnna,28,LA\n");
    }

    #[test]
    fn split_tsv() {
        round_trip(b"a\tb\tc\n1\t2\t3\n4\t5\t6\n");
    }

    #[test]
    fn split_ragged_rows() {
        round_trip(b"a,b,c\n1,2\n3,4,5\n");
    }

    #[test]
    fn split_quoted_commas() {
        round_trip(b"name,note\n\"Smith, John\",\"hello\"\n");
    }

    #[test]
    fn looks_like_csv_detects() {
        let csv = b"name,age,city,country\nJohn,30,NYC,US\nAnna,28,LA,US\nBob,45,CHI,US\n";
        assert!(looks_like_csv(csv));
    }

    #[test]
    fn looks_like_csv_rejects_text() {
        let text = b"The quick brown fox jumps over the lazy dog. ".repeat(10);
        assert!(!looks_like_csv(&text));
    }

    #[test]
    fn looks_like_csv_rejects_json() {
        let json = b"{\"name\": \"John\", \"age\": 30, \"city\": \"NYC\", \"x\": 1}";
        assert!(!looks_like_csv(json));
    }

    #[test]
    fn columns_padded() {
        let streams = split(b"a,b,c\n1,2\n");
        assert_eq!(streams.ncols(), 3);
        assert_eq!(streams.fields_per_row, vec![3, 2]);
        // Each column has 2 length-prefixed cells.
        for col in &streams.columns {
            let mut n = 0;
            let mut pos = 0;
            while pos + 4 <= col.len() {
                let len = u32::from_le_bytes(col[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4 + len;
                n += 1;
            }
            assert_eq!(n, 2);
        }
    }
}
