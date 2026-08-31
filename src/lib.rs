// nyx library root

// The bit-probability / entropy math performs deliberate numeric casts (u16/u32 <-> f32/f64,
// usize position -> u32 table index). These are exact at our magnitudes (12-bit probs,
// bounded counts, <4 GiB inputs) and clippy's numeric-cast lints are noise here, not risk.
#![allow(
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops
)]

pub mod classify;
pub mod codec;
pub mod container;
pub mod entropy;
pub mod error;
pub mod model;
