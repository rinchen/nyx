//! Error type for `rcn` compression/decompression.

use thiserror::Error;

/// Errors produced by the `rcn` codec.
#[derive(Debug, Error)]
pub enum RcnError {
    /// The container magic or structure was not valid.
    #[error("invalid rcn container: {0}")]
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

    /// JSON stream splitting/merging failed (corrupt sub-stream).
    #[error("json split error: {0}")]
    JsonSplitError(String),

    /// CSV stream splitting/merging failed (corrupt sub-stream).
    #[error("csv split error: {0}")]
    CsvSplitError(String),

    /// XML stream splitting/merging failed (corrupt sub-stream).
    #[error("xml split error: {0}")]
    XmlSplitError(String),
}

impl RcnError {
    /// Cold path: construct an invalid-container error (W7).
    #[cold]
    #[inline(never)]
    pub fn invalid_container(msg: impl Into<String>) -> Self {
        Self::InvalidContainer(msg.into())
    }

    /// Cold path: construct a corrupt-block error (W7).
    #[cold]
    #[inline(never)]
    pub fn corrupt_block(msg: impl Into<String>) -> Self {
        Self::CorruptBlock(msg.into())
    }
}

/// Convenience alias for codec results.
pub type Result<T> = std::result::Result<T, RcnError>;
