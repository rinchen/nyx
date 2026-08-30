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
