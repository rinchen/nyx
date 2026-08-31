//! Error type for `nyx` compression/decompression.

use thiserror::Error;

/// Errors produced by the `nyx` codec.
#[derive(Debug, Error)]
pub enum NyxError {
    /// The container magic or structure was not valid.
    #[error("invalid nyx container: {0}")]
    InvalidContainer(String),

    /// A block payload was truncated or the entropy stream was corrupt.
    #[error("corrupt block stream: {0}")]
    CorruptBlock(String),

    /// A decompressed block failed its CRC32 check (data corruption / wrong key).
    #[error("block {0} failed CRC32 check (got {1:#010x}, expected {2:#010x})")]
    CrcMismatch(usize, u32, u32),

    /// The decoder ran out of entropy bytes mid-block.
    #[error("unexpected end of entropy stream in block {0}")]
    TruncatedStream(usize),

    /// An entropy-coding primitive failed.
    #[error("entropy coder error: {0}")]
    Entropy(String),
}

/// Convenience alias for codec results.
pub type Result<T> = std::result::Result<T, NyxError>;
