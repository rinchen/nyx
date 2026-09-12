//! User-facing compression levels and the peer class each one is scored against.
//!
//! CLI accepts both `1`/`3`/`9`/`19` and `-1`/`-3`/`-9`/`-19`.

use crate::codec::CodecMode;

/// Named compression level. Default is [`Level::Archive`] (`-9`, hybrid).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Wire: hash-chain LZ + optional order-0 rANS. Target: `lz4 -9`, `zstd -1`.
    Wire,
    /// General: byte CM + DP-LZP, no BWT trials. Target: `gzip -9`.
    General,
    /// Archive (default): hybrid Fast Text + Slow Binary. Target: `zstd -19`.
    Archive,
    /// Max: Slow bit CM on every block. Target: xz / brotli ratio stretch.
    Max,
}

impl Level {
    /// Parse `1`, `3`, `9`, `19` or the negative aliases `-1`, `-3`, `-9`, `-19`.
    #[must_use]
    pub fn from_i32(n: i32) -> Option<Self> {
        match n {
            1 | -1 => Some(Self::Wire),
            3 | -3 => Some(Self::General),
            9 | -9 => Some(Self::Archive),
            19 | -19 => Some(Self::Max),
            _ => None,
        }
    }

    /// Canonical positive level number (`1`, `3`, `9`, `19`).
    #[must_use]
    pub const fn number(self) -> i32 {
        match self {
            Self::Wire => 1,
            Self::General => 3,
            Self::Archive => 9,
            Self::Max => 19,
        }
    }

    /// Engine used to implement this level.
    #[must_use]
    pub const fn mode(self) -> CodecMode {
        match self {
            Self::Wire => CodecMode::Wire,
            Self::General => CodecMode::General,
            Self::Archive => CodecMode::Hybrid,
            Self::Max => CodecMode::Slow,
        }
    }

    /// Peer CLIs this level must beat on both ratio and compress speed.
    #[must_use]
    pub const fn target_peers(self) -> &'static str {
        match self {
            Self::Wire => "lz4 -9, zstd -1",
            Self::General => "gzip -9",
            Self::Archive => "zstd -19 (stretch: xz -9, brotli -11)",
            Self::Max => "xz -9, brotli -11 (ratio stretch)",
        }
    }
}

impl Default for Level {
    fn default() -> Self {
        Self::Archive
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_map() {
        assert_eq!(Level::from_i32(1), Some(Level::Wire));
        assert_eq!(Level::from_i32(-1), Some(Level::Wire));
        assert_eq!(Level::from_i32(3), Some(Level::General));
        assert_eq!(Level::from_i32(-9), Some(Level::Archive));
        assert_eq!(Level::from_i32(19), Some(Level::Max));
        assert_eq!(Level::from_i32(2), None);
    }

    #[test]
    fn default_is_archive() {
        assert_eq!(Level::default().mode(), CodecMode::Hybrid);
        assert_eq!(Level::default().number(), 9);
    }
}
