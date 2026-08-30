//! Per-block data-type classifier.
//!
//! Before compressing a block, `nyx` estimates its character with a cheap order-0 Shannon
//! entropy over the block and a few structural signals. This chooses which predictor stack
//! to run (and whether to just copy random data), which is what keeps speed near `zstd`
//! while allowing CM-quality prediction only where it pays off.

/// The kind of data a block looks like.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// High-entropy, incompressible (already compressed / encrypted): emit a copy record.
    Random,
    /// Predominantly printable text / JSON / logs.
    Text,
    /// Mixed binary with some structure (neither clearly text nor executable).
    Binary,
    /// Looks like machine code (ELF/Mach-O/PE byte patterns).
    Exec,
}

impl BlockKind {
    /// Whether this block should be entropy-coded (vs copied verbatim).
    #[must_use]
    pub const fn is_compressible(self) -> bool {
        !matches!(self, Self::Random)
    }
}

/// Classify a block.
///
/// `buf` may be empty (treated as `Random` so it is copied, never modeled).
#[must_use]
pub fn classify(buf: &[u8]) -> BlockKind {
    if buf.len() < 32 {
        // Too small to estimate reliably; copy tiny blocks.
        return BlockKind::Random;
    }
    let mut hist = [0u32; 256];
    for &b in buf {
        hist[b as usize] += 1;
    }
    let n = buf.len() as f32;
    let mut h = 0.0f32;
    for &c in &hist {
        if c > 0 {
            let p = c as f32 / n;
            h -= p * p.log2();
        }
    }
    let frac_printable = buf
        .iter()
        .filter(|&&b| (0x20..0x7f).contains(&b))
        .count() as f32
        / n;

    if h > 7.9 {
        BlockKind::Random
    } else if frac_printable > 0.85 {
        BlockKind::Text
    } else if has_exec_markers(buf) {
        BlockKind::Exec
    } else {
        BlockKind::Binary
    }
}

/// Heuristic for machine-code: common x86/AMD64 instruction-sequence footprints.
fn has_exec_markers(b: &[u8]) -> bool {
    // Look for function prologue `push rbp; mov rbp, rsp` (55 89 E5) and common
    // call / mov patterns. Require at least 2 distinct hits to avoid false positives.
    let mut hits = 0usize;
    for w in b.windows(3) {
        if w == [0x55, 0x89, 0xE5] || w == [0x48, 0x89, 0xE5] || w == [0xFF, 0xD0] {
            hits += 1;
            if hits >= 2 {
                return true;
            }
        }
    }
    // ELF/Mach-O/PE magic at a common alignment also counts.
    b.windows(4)
        .any(|w| w == b"\x7fELF" || w == b"\xCF\xFA\xED\xFE" || w == b"MZ\x90\x00")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_text() {
        let text = b"hello world, this is plain text that should be recognized as text..............";
        assert_eq!(classify(text), BlockKind::Text);
    }

    #[test]
    fn classifies_random() {
        // All-zero block: entropy 0, but tiny? it's 4096 bytes so not tiny; entropy low -> not random.
        // Use high-entropy pseudo-random-looking data instead.
        let mut buf = [0u8; 4096];
        let mut x = 0x1234_5678u32;
        for b in &mut buf {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        assert_eq!(classify(&buf), BlockKind::Random);
    }

    #[test]
    fn classifies_empty_as_random() {
        assert_eq!(classify(&[]), BlockKind::Random);
    }

    #[test]
    fn classifies_exec() {
        // x86 prologue repeated: push rbp; mov rbp,rsp
        let mut buf = Vec::new();
        for _ in 0..4 {
            buf.extend_from_slice(&[0x55, 0x89, 0xE5, 0x48, 0x89, 0xC7, 0xFF, 0xD0]);
        }
        assert_eq!(classify(&buf), BlockKind::Exec);
    }

    #[test]
    fn random_not_compressible() {
        assert!(!BlockKind::Random.is_compressible());
        assert!(BlockKind::Text.is_compressible());
        assert!(BlockKind::Binary.is_compressible());
        assert!(BlockKind::Exec.is_compressible());
    }
}
