//! Online logistic mixer — the novel core of `nyx`.
//!
//! Each base model emits a 12-bit probability of bit==1. The mixer fuses them with a
//! weighted sum in the logistic domain (stretch → linear combination → squash) and adapts
//! the weights online via stochastic gradient descent on the logistic loss. This is the
//! "context mixing" step that lets heterogeneous models (order, sparse, exec, LZP) cover
//! for each other per bit, rather than picking a single best model.

use super::BitModel;

const MAX_PROB: u16 = 4095;
const MIN_PROB: u16 = 1;

/// Logistic mixer over `n` model probabilities.
pub struct LogisticMixer {
    weights: Vec<f32>,
    lr: f32,
    // Precomputed stretch/squash tables (12-bit probability in, logit out / prob out).
    stretch: [f32; 4096],
    squash: [u16; 4096],
}

impl LogisticMixer {
    /// Create a mixer for `n` models, with learning rate `lr`.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // builds Vecs via loops; not const-evaluable
    pub fn new(n: usize) -> Self {
        let mut stretch = [0.0f32; 4096];
        for (p, slot) in stretch.iter_mut().enumerate() {
            let pr = (p as f32 + 0.5) / 4096.0;
            *slot = if pr <= 1e-6 {
                -7.0
            } else if pr >= 1.0 - 1e-6 {
                7.0
            } else {
                (pr / (1.0 - pr)).ln()
            };
        }
        let mut squash = [0u16; 4096];
        for (x, slot) in squash.iter_mut().enumerate() {
            let v = x as f32 / 4095.0 * 14.0 - 7.0; // map [0,4095] -> [-7, 7]
            let s = 1.0 / (1.0 + (-v).exp());
            *slot = (s * MAX_PROB as f32).clamp(MIN_PROB as f32, MAX_PROB as f32) as u16;
        }
        Self {
            // Start each weight at 1.0 so the mix is a sensible average of the
            // (stretched) model probabilities from the first bit. A zero
            // initialization makes `mix` return 50/50 until SGD slowly learns to
            // trust the models — which costs most of the compression on small/early
            // blocks. Positive weights also keep the mix grounded in the models'
            // evidence rather than the prior.
            weights: vec![1.0; n],
            lr: 0.02,
            stretch,
            squash,
        }
    }

    /// Set the learning rate (default 0.02).
    pub const fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    #[inline]
    fn stretch_of(&self, p: u16) -> f32 {
        self.stretch[(p as usize).clamp(1, 4095)]
    }

    #[inline]
    fn squash_of(&self, acc: f32) -> u16 {
        // map logistic accumulator to a table index in [0,4095]
        let idx = ((acc + 7.0) / 14.0 * 4095.0).clamp(0.0, 4095.0) as usize;
        self.squash[idx]
    }

    /// Mix `probs` (one P(bit==1) per model, each in `[1,4095]`) → fused P in `[1,4095]`.
    #[must_use]
    pub fn mix(&self, probs: &[u16]) -> u16 {
        let mut acc = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            acc += self.weights[i] * self.stretch_of(p);
        }
        self.squash_of(acc)
    }

    /// Online update after the true `bit` is known.
    pub fn update(&mut self, probs: &[u16], bit: bool) {
        let target = if bit { 1.0f32 } else { 0.0 };
        let mut acc = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            acc += self.weights[i] * self.stretch_of(p);
        }
        let pred = 1.0 / (1.0 + (-acc).exp());
        let err = target - pred;
        for (i, &p) in probs.iter().enumerate() {
            self.weights[i] += self.lr * err * self.stretch_of(p);
        }
    }
}

impl BitModel for LogisticMixer {
    fn predict(&self) -> u16 {
        // Only meaningful when driven via `mix`; standalone predict returns mid.
        2048
    }

    fn update(&mut self, _bit: bool) {
        // The mixer is updated via `update(probs, bit)`, not this trait method.
    }

    fn reset(&mut self) {
        self.weights.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::order::OrderN;

    #[test]
    fn mixer_favors_correct_model() {
        // Biased data: feed bytes that are mostly 1-bits (0xFF 3 of every 4 steps).
        // The models learn the bias and the mixer's fused prob should track it.
        let mut m0 = OrderN::new(0);
        let mut m1 = OrderN::new(0);
        let mut mixer = LogisticMixer::new(2);
        let mut last_probs = [2048u16; 2];
        for step in 0..200 {
            let byte: u8 = if step % 4 != 0 { 0xFF } else { 0x00 };
            for bit_idx in (0..8).rev() {
                let bit = (byte >> bit_idx) & 1 == 1;
                last_probs[0] = m0.predict();
                last_probs[1] = m1.predict();
                let fused = mixer.mix(&last_probs);
                m0.update(bit);
                m1.update(bit);
                mixer.update(&last_probs, bit);
                if step > 150 && bit {
                    assert!(i32::from(fused) > 2048, "mixer should learn the bias");
                }
            }
        }
    }

    #[test]
    fn mix_in_range() {
        let mixer = LogisticMixer::new(3);
        let p = mixer.mix(&[1000, 2048, 3000]);
        assert!((1..=4095).contains(&p));
    }
}
