//! Bit-level predictors (context models) that feed the logistic mixer.
//!
//! Each model emits P(bit==1) in `[1, 4095]` (12-bit) and is updated causally after
//! the true bit is known. The codec uses a stack of these per block, mixed by
//! [`mixer::LogisticMixer`](super::model::mixer::LogisticMixer).

pub mod exec;
pub mod lzp;
pub mod mixer;
pub mod order;
pub mod sparse;

/// Assembles whole bytes from the per-bit [`BitModel::update`] stream so context
/// models can key on *byte* history instead of individual bits.
///
/// Coding is bit-by-bit (MSB→LSB) within each byte. `push_bit` feeds one coded bit;
/// when 8 are collected the completed byte is emitted to `on_byte` (which the model
/// uses to advance its real context), and the bit accumulator resets. Models call
/// this in `update` and key their prediction tables on the assembled byte context.
#[derive(Debug, Clone, Default)]
pub struct ByteAssembler {
    pending: u8,
    nbits: u8,
    /// Most recent completed bytes, most-recent last. Capped at `cap`.
    bytes: Vec<u8>,
    cap: usize,
}

impl ByteAssembler {
    /// Create an assembler retaining up to `cap` completed bytes of context.
    #[must_use]
    pub fn new(cap: usize) -> Self {
        Self {
            pending: 0,
            nbits: 0,
            bytes: Vec::with_capacity(cap),
            cap,
        }
    }

    /// Feed one coded bit. Returns `Some(byte)` when a full byte completes.
    #[inline]
    pub fn push_bit(&mut self, bit: bool) -> Option<u8> {
        self.pending = (self.pending << 1) | u8::from(bit);
        self.nbits += 1;
        if self.nbits == 8 {
            self.nbits = 0;
            let byte = self.pending;
            self.pending = 0;
            if self.bytes.len() == self.cap {
                self.bytes.remove(0);
            }
            self.bytes.push(byte);
            Some(byte)
        } else {
            None
        }
    }

    /// The number of bits accumulated toward the current (in-progress) byte, 0..8.
    #[must_use]
    pub const fn nbits(&self) -> u8 {
        self.nbits
    }

    /// Number of completed bytes currently retained.
    #[must_use]
    pub const fn bytes_len(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// The last `n` completed bytes (fewer if not enough history yet).
    #[must_use]
    pub fn last(&self, n: usize) -> &[u8] {
        let start = self.bytes.len().saturating_sub(n);
        &self.bytes[start..]
    }

    /// The single most-recent completed byte (0 before any).
    #[must_use]
    pub fn last_byte(&self) -> u8 {
        *self.bytes.last().unwrap_or(&0)
    }

    /// The previous completed byte (0 before two bytes exist).
    #[must_use]
    pub fn prev_byte(&self) -> u8 {
        if self.bytes.len() >= 2 {
            self.bytes[self.bytes.len() - 2]
        } else {
            0
        }
    }

    /// True once at least one full byte has been assembled.
    #[must_use]
    pub const fn has_byte(&self) -> bool {
        !self.bytes.is_empty()
    }

    /// Clear all state (called at the start of each block).
    pub fn reset(&mut self) {
        self.pending = 0;
        self.nbits = 0;
        self.bytes.clear();
    }
}

/// A bit-level predictor.
///
/// `predict` returns the model's estimate of P(bit==1) in `[1, 4095]` (12-bit).
/// `update` is called **after** the true bit is known so the model can adapt.
/// `reset` clears per-block state at the start of each new block.
pub trait BitModel {
    /// Predicted probability of bit==1, in `[1, 4095]`.
    fn predict(&self) -> u16;

    /// Adapt the model to the observed `bit`.
    fn update(&mut self, bit: bool);

    /// Reset per-block state (called at the start of each block).
    fn reset(&mut self);
}
