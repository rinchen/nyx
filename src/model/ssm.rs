//! Micro neural mixer — StateSMix / GLN style.
//!
//! A small Mamba-style state-space model (SSM) trained **online** during
//! compression.  Unlike the prior "Micro SSM mixer" experiment (which replaced
//! the `LogisticMixer` entirely and regressed), `SsmMixer` is an *additional
//! base model* that feeds its probability into the existing logistic mixer
//! alongside the order / sparse / exec / LZP models.
//!
//! Architecture (8-dim hidden state):
//! - **Input**  `x[t]`: 8-dim embedding of the last 32 decoded bytes (hash-based).
//! - **State**  `h[t] = A · h[t-1] + B · x[t]`  (A is diagonal-dominant, 0.95 decay).
//! - **Output** `y[t] = C · h[t] + D · x[t]`     (scalar logit).
//! - **Predict** `p = squash(y[t])` mapped to `[1, 4095]`.
//!
//! Training: SGD on the logistic error.  Only `C` and `D` are updated; `A` and
//! `B` stay fixed to avoid the gradient instability that caused the 16-dim
//! replacement-mixer experiment to regress.
//!
//! The model is causal and decoder-safe: `h[t]` is a deterministic function of
//! `h[t-1]`, `x[t]`, and (fixed) parameters.  The SGD step on `C`/`D` is
//! identical in encoder and decoder because both sides observe the same bit
//! stream and the same model probabilities.

use super::BitModel;
use super::ByteAssembler;

const MAX_PROB: u16 = 4095;
const HIDDEN_DIM: usize = 8;
const INPUT_DIM: usize = 8; // embedding dimension
const CONTEXT_BYTES: usize = 32; // how many past bytes to embed

/// Sigmoid (logistic) function, clamped to avoid overflow.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 20.0 {
        1.0
    } else if x <= -20.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Squash a logit into the 12-bit probability range `[1, 4095]`.
#[inline]
fn squash_logit(logit: f32) -> u16 {
    let p = sigmoid(logit);
    let scaled = p * (MAX_PROB as f32 - 1.0) + 1.0; // map (0,1) → (1, 4095)
    scaled as u16
}

/// 8-dim state-space model mixer that acts as a `BitModel` base model.
///
/// The state `h` is a small recurrent vector.  The input embedding `x` is
/// derived from the last `CONTEXT_BYTES` bytes via a fast hash-fold into
/// `INPUT_DIM` floats.  Only the readout weights `C` and `D` are trained;
/// `A` (diagonal, 0.95 decay) and `B` (random-fixed) provide the recurrent
/// structure without gradient instability.
pub struct SsmMixer {
    /// Diagonal state transition: h = A_diag * h (elementwise).
    a_diag: [f32; HIDDEN_DIM],
    /// Input projection: h += B * x (B is HIDDEN_DIM × INPUT_DIM, row-major).
    b: [[f32; INPUT_DIM]; HIDDEN_DIM],
    /// Output projection: y = dot(C, h).
    c: [f32; HIDDEN_DIM],
    /// Direct input term: y += dot(D, x).
    d: [f32; HIDDEN_DIM],
    /// Hidden state vector.
    h: [f32; HIDDEN_DIM],
    /// Learning rate for SGD on C and D.
    lr: f32,
    /// Assembles bytes from the bit stream so we can build the input embedding.
    asm: ByteAssembler,
    /// Rolling byte history (for embedding construction).
    history: [u8; CONTEXT_BYTES],
    /// How many bytes are currently in `history` (≤ CONTEXT_BYTES).
    hist_len: usize,
}

impl SsmMixer {
    /// Create a new SsmMixer.
    #[must_use]
    pub fn new() -> Self {
        // Build a deterministic-but-spread diagonal A (0.95 decay).
        let a_diag = [0.95f32; HIDDEN_DIM];

        // B: pseudo-random fixed input projection (seeded for determinism).
        let mut b = [[0.0f32; INPUT_DIM]; HIDDEN_DIM];
        let mut seed = 0x1234_5678u32;
        for row in &mut b {
            for val in row.iter_mut() {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                // Map to [-0.2, 0.2] for small init.
                *val = ((seed as f32) / (u32::MAX as f32) - 0.5) * 0.4;
            }
        }

        Self {
            a_diag,
            b,
            c: [0.0f32; HIDDEN_DIM],
            d: [0.0f32; HIDDEN_DIM],
            h: [0.0f32; HIDDEN_DIM],
            lr: 0.01,
            asm: ByteAssembler::new(CONTEXT_BYTES),
            history: [0u8; CONTEXT_BYTES],
            hist_len: 0,
        }
    }

    /// Set the learning rate (default 0.01).
    pub const fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    /// Build the 8-dim input embedding from the last `CONTEXT_BYTES` bytes.
    ///
    /// Uses a fast multiplicative hash-fold: each byte is mixed into all 8 dims
    /// via a different hash constant.  This is deterministic (same input → same
    /// output) and cheap (≈32 multiplies), and provides enough signal for the
    /// SSM to start learning sequence structure.
    #[inline]
    fn make_embedding(&self) -> [f32; INPUT_DIM] {
        let mut x = [0.0f32; INPUT_DIM];
        let len = self.hist_len.min(CONTEXT_BYTES);
        if len == 0 {
            return x;
        }
        for i in 0..len {
            let b = self.history[i] as u32;
            let base = b.wrapping_mul(2654435761).wrapping_rem(u32::MAX);
            for d in 0..INPUT_DIM {
                // Different rotation per dimension.
                let mixed = base.rotate_left((d * 3 + 7) as u32);
                x[d] += ((mixed as f32) / (u32::MAX as f32) - 0.5) * 2.0;
            }
        }
        // Normalize by length so the embedding doesn't blow up.
        let scale = 1.0 / (len as f32);
        for v in &mut x {
            *v *= scale;
        }
        x
    }
}

impl Default for SsmMixer {
    fn default() -> Self {
        Self::new()
    }
}

impl BitModel for SsmMixer {
    #[inline]
    fn predict(&self) -> u16 {
        // Forward pass: compute h[t] and y[t] from current state and embedding.
        let x = self.make_embedding();

        // h[t] = A * h[t-1] + B * x  (A is diagonal)
        let mut new_h = [0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            new_h[i] = self.a_diag[i] * self.h[i];
            for j in 0..INPUT_DIM {
                new_h[i] += self.b[i][j] * x[j];
            }
        }

        // y[t] = C · h[t] + D · x[t]
        let mut y = 0.0f32;
        for i in 0..HIDDEN_DIM {
            y += self.c[i] * new_h[i];
            y += self.d[i] * x[i];
        }

        squash_logit(y)
    }

    #[inline]
    fn update(&mut self, bit: bool) {
        // Reconstruct the same forward pass that `predict` did, so we can
        // compute the gradient.
        let x = self.make_embedding();
        let mut new_h = [0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            new_h[i] = self.a_diag[i] * self.h[i];
            for j in 0..INPUT_DIM {
                new_h[i] += self.b[i][j] * x[j];
            }
        }

        // Compute prediction (same as predict).
        let mut y = 0.0f32;
        for i in 0..HIDDEN_DIM {
            y += self.c[i] * new_h[i];
            y += self.d[i] * x[i];
        }
        let p = sigmoid(y);

        // SGD step on C and D.
        // err = target - p  (target is 1 for bit=true, 0 for bit=false)
        let target = if bit { 1.0f32 } else { 0.0 };
        let err = target - p;

        // ∂y/∂C = h[t], ∂y/∂D = x[t]
        for i in 0..HIDDEN_DIM {
            self.c[i] += self.lr * err * new_h[i];
            self.d[i] += self.lr * err * x[i];
        }

        // Commit the new state.
        self.h = new_h;

        // Assemble bytes from the bit stream and update byte history.
        if let Some(byte) = self.asm.push_bit(bit) {
            if self.hist_len < CONTEXT_BYTES {
                self.history[self.hist_len] = byte;
                self.hist_len += 1;
            } else {
                self.history.copy_within(1..CONTEXT_BYTES, 0);
                self.history[CONTEXT_BYTES - 1] = byte;
            }
        }
    }

    fn reset(&mut self) {
        self.h = [0.0f32; HIDDEN_DIM];
        self.c = [0.0f32; HIDDEN_DIM];
        self.d = [0.0f32; HIDDEN_DIM];
        self.asm.reset();
        self.history = [0u8; CONTEXT_BYTES];
        self.hist_len = 0;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_neutral() {
        let m = SsmMixer::new();
        // Fresh SSM has zero C/D, so y=0, sigmoid(0)=0.5 → prob ≈ 2048.
        let p = m.predict();
        assert!(
            (i32::from(p) - 2048).abs() <= 2,
            "fresh SsmMixer should start near neutral (2048), got {}",
            p
        );
    }

    #[test]
    fn learns_bias() {
        let mut m = SsmMixer::new();
        // Feed many 1-bits. The SSM should learn to predict >50% for 1-bits
        // via SGD on C and D.
        for _ in 0..500 {
            m.predict();
            m.update(true);
        }
        let p = m.predict();
        assert!(
            p > 2048,
            "after 500 1-bits, SsmMixer should learn to predict >50%, got {}",
            p
        );
    }

    #[test]
    fn learns_bias_zero() {
        let mut m = SsmMixer::new();
        // Feed many 0-bits. The SSM should learn to predict <50% for 0-bits.
        for _ in 0..500 {
            m.predict();
            m.update(false);
        }
        let p = m.predict();
        assert!(
            p < 2048,
            "after 500 0-bits, SsmMixer should learn to predict <50%, got {}",
            p
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut m = SsmMixer::new();
        for _ in 0..200 {
            m.predict();
            m.update(true);
        }
        assert!(m.predict() > 2048, "should have learned bias");
        m.reset();
        let p = m.predict();
        assert!(
            (i32::from(p) - 2048).abs() <= 2,
            "after reset, should be neutral, got {}",
            p
        );
    }

    #[test]
    fn prediction_in_range() {
        let m = SsmMixer::new();
        let p = m.predict();
        assert!(
            (1u16..=MAX_PROB).contains(&p),
            "prediction {} out of range",
            p
        );
    }
}
