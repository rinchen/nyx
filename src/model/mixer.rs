//! Online logistic mixer — the novel core of `nyx`.
//!
//! Each base model emits a 12-bit probability of bit==1. The mixer fuses them with a
//! weighted sum in the logistic domain (stretch → linear combination → squash) and adapts
//! the weights online via stochastic gradient descent on the logistic loss. This is the
//! "context mixing" step that lets heterogeneous models (order, sparse, exec, LZP) cover
//! for each other per bit, rather than picking a single best model.
//!
//! For higher compression, see [`super::sse_apm::SseApmCascade`] which adds SSE/APM/APM2
//! refinement stages after the mixer, and [`super::mixer_bank::MixerBank`] which selects
//! from 4096 context-specific mixer instances.

use super::BitModel;

const MAX_PROB: u16 = 4095;
const MIN_PROB: u16 = 1;

/// Shared stretch/squash tables (12-bit probability ↔ logit).
/// These are identical for every mixer instance, so we allocate once globally.
static STRETCH: std::sync::OnceLock<[f32; 4096]> = std::sync::OnceLock::new();
static SQUASH: std::sync::OnceLock<[u16; 4096]> = std::sync::OnceLock::new();

fn stretch_table() -> &'static [f32; 4096] {
    STRETCH.get_or_init(|| {
        let mut t = [0.0f32; 4096];
        for (p_slot, slot) in t.iter_mut().enumerate() {
            let pr = (p_slot as f32 + 0.5) / 4096.0;
            *slot = if pr <= 1e-6 {
                -7.0
            } else if pr >= 1.0 - 1e-6 {
                7.0
            } else {
                (pr / (1.0 - pr)).ln()
            };
        }
        t
    })
}

fn squash_table() -> &'static [u16; 4096] {
    SQUASH.get_or_init(|| {
        let mut t = [0u16; 4096];
        for (x_slot, slot) in t.iter_mut().enumerate() {
            let v = x_slot as f32 / 4095.0 * 14.0 - 7.0; // map [0,4095] -> [-7, 7]
            let s = 1.0 / (1.0 + (-v).exp());
            *slot = (s * MAX_PROB as f32).clamp(MIN_PROB as f32, MAX_PROB as f32) as u16;
        }
        t
    })
}

/// Logistic mixer over `n` model probabilities, with per-bit-position context.
///
/// The bit position within a byte (0 = MSB, 7 = LSB) is a cheap, decoder-safe signal:
/// text bytes have very different bit distributions per position (e.g. ASCII high
/// bits are nearly always 0), so conditioning the mixer weights on `bit_pos`
/// lets it specialize without changing the container format.
pub struct LogisticMixer {
    weights: Vec<f32>,
    lr: f32,
    lr_scales: Vec<f32>,
    // Per (model, bit_position) weight deltas. `bit_pos` ∈ [0,7].
    // The effective weight for model `i` at bit position `b` is `base[i] + pos_weights[i][b]`.
    pos_weights: Vec<[f32; 8]>,
    // Adam state for base weights.
    adam_m: Vec<f32>,
    adam_v: Vec<f32>,
    adam_t: u32,
    beta1: f32,
    beta2: f32,
    eps: f32,
}

impl LogisticMixer {
    /// Create a mixer for `n` models, with learning rate `lr`.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // builds Vecs via loops; not const-evaluable
    pub fn new(n: usize) -> Self {
        Self {
            // Start each weight at 1.0 so the mix is a sensible average of the
            // (stretched) model probabilities from the first bit. A zero
            // initialization makes `mix` return 50/50 until SGD slowly learns to
            // trust the models — which costs most of the compression on small/early
            // blocks. Positive weights also keep the mix grounded in the models'
            // evidence rather than the prior.
            weights: vec![1.0; n],
            // All models start with lr_scale=1.0 (SGD).
            lr_scales: vec![1.0; n],
            // Position deltas start at 0 so the initial mix is identical to the
            // non-context-aware version (pure 1.0 base weights).
            pos_weights: (0..n).map(|_| [0.0f32; 8]).collect(),
            lr: 0.02,
            // Adam state initialized in new_adam; zeroed here for plain SGD.
            adam_m: vec![0.0; n],
            adam_v: vec![0.0; n],
            adam_t: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// Create a mixer with Adam optimizer (adaptive per-weight learning rates).
    /// Base weights use Adam; per-bit-position deltas use plain SGD.
    #[must_use]
    pub fn new_adam(n: usize, lr: f32) -> Self {
        Self {
            weights: vec![1.0; n],
            lr_scales: vec![1.0; n],
            pos_weights: (0..n).map(|_| [0.0f32; 8]).collect(),
            lr,
            adam_m: vec![0.0; n],
            adam_v: vec![0.0; n],
            adam_t: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// Set the learning rate (default 0.02).
    pub const fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    /// Set per-model learning rate scale. Index corresponds to model position in the stack.
    pub fn set_lr_scale(&mut self, idx: usize, scale: f32) {
        if idx < self.lr_scales.len() {
            self.lr_scales[idx] = scale;
        }
    }

    /// Set all per-model learning rate scales.
    pub fn set_lr_scales(&mut self, scales: Vec<f32>) {
        if scales.len() == self.lr_scales.len() {
            self.lr_scales = scales;
        }
    }

    /// Return a copy of the per-model learning rate scales.
    #[must_use]
    pub fn lr_scales(&self) -> Vec<f32> {
        self.lr_scales.clone()
    }

    /// Replace base weights. Keeps per-bit-position deltas unchanged.
    pub fn set_weights(&mut self, weights: Vec<f32>) {
        self.weights = weights;
    }

    /// Return a copy of the base weights.
    #[must_use]
    pub fn weights(&self) -> Vec<f32> {
        self.weights.clone()
    }

    /// Reset weights to initial state (called at block boundaries).
    ///
    /// NOTE: Prefer `decay` for cross-block continuity — it shrinks weights
    /// toward their init value (1.0) instead of hard-clearing, preserving
    /// learned structure across block boundaries.
    pub fn reset(&mut self) {
        self.weights.fill(1.0);
        for pw in &mut self.pos_weights {
            pw.fill(0.0);
        }
        self.lr_scales.fill(1.0);
        self.adam_t = 0;
        self.adam_m.fill(0.0);
        self.adam_v.fill(0.0);
        self.lr = 0.02;
    }

    /// Decay all learned weights toward their init values by `factor`.
    ///
    /// Base weights decay toward 1.0, pos_weights decay toward 0.0.
    /// `weight = init + (weight - init) * f`
    ///
    /// A factor of 0.0 restores init; 1.0 leaves unchanged. This preserves
    /// learned structure across block boundaries without hard-clearing, which
    /// would throw away the per-context weight vectors that the 4096-bank
    /// hierarchy depends on.
    pub fn decay(&mut self, factor: f32) {
        for w in &mut self.weights {
            *w = 1.0 + (*w - 1.0) * factor;
        }
        for pw in &mut self.pos_weights {
            for dw in pw.iter_mut() {
                *dw = 0.0 + (*dw - 0.0) * factor;
            }
        }
    }

    #[inline]
    fn stretch_of(&self, p: u16) -> f32 {
        stretch_table()[(p as usize).clamp(1, 4095)]
    }

    #[inline]
    fn squash_of(&self, acc: f32) -> u16 {
        // map logistic accumulator to a table index in [0,4095]
        let idx = ((acc + 7.0) / 14.0 * 4095.0).clamp(0.0, 4095.0) as usize;
        squash_table()[idx]
    }

    /// Mix `probs` (one P(bit==1) per model, each in `[1,4095]`) → fused P in `[1,4095]`.
    /// `bit_pos` is the 0-based MSB-first bit position within the current byte.
    #[must_use]
    #[inline(always)]
    pub fn mix(&self, probs: &[u16], bit_pos: u8) -> u16 {
        let b = usize::from(bit_pos.min(7));
        let mut acc = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            let w = self.weights[i] + self.pos_weights[i][b];
            acc += w * self.stretch_of(p);
        }
        self.squash_of(acc)
    }

    /// Online update after the true `bit` is known.
    ///
    /// Base weights use Adam when `adam_t > 0`; pos_weights always use SGD.
    pub fn update(&mut self, probs: &[u16], bit: bool, bit_pos: u8) {
        let b = usize::from(bit_pos.min(7));
        let target = if bit { 1.0f32 } else { 0.0 };
        let mut acc = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            let w = self.weights[i] + self.pos_weights[i][b];
            acc += w * self.stretch_of(p);
        }
        let pred = 1.0 / (1.0 + (-acc).exp());
        let err = target - pred;

        for (i, &p) in probs.iter().enumerate() {
            let scale = self.lr_scales[i];
            let grad = self.lr * scale * err * self.stretch_of(p);

            if self.adam_t > 0 {
                // Adam update for base weights.
                self.adam_m[i] = self.beta1 * self.adam_m[i] + (1.0 - self.beta1) * grad;
                self.adam_v[i] = self.beta2 * self.adam_v[i] + (1.0 - self.beta2) * grad * grad;
                let m_hat = self.adam_m[i] / (1.0 - self.beta1.powi(self.adam_t as i32));
                let v_hat = self.adam_v[i] / (1.0 - self.beta2.powi(self.adam_t as i32));
                self.weights[i] += m_hat / (v_hat.sqrt() + self.eps);
            } else {
                // Plain SGD.
                self.weights[i] += grad;
            }

            // Per-bit-position deltas always use SGD.
            self.pos_weights[i][b] += grad;
        }

        if self.adam_t > 0 {
            self.adam_t = self.adam_t.saturating_add(1);
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
        self.weights.fill(1.0);
        for pw in &mut self.pos_weights {
            pw.fill(0.0);
        }
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
    use crate::model::order::OrderN;

    #[test]
    fn mixer_favors_correct_model() {
        // Biased data: feed bytes that are mostly 1-bits (0xFF 3 of every 4 steps).
        // The linear logistic mix learns the bias, so the fused prob should track it.
        let mut m0 = OrderN::new(0);
        let mut m1 = OrderN::new(0);
        let mut mixer = LogisticMixer::new(2);
        let mut last_probs = [2048u16; 2];
        for step in 0..200 {
            let byte: u8 = if step % 4 != 0 { 0xFF } else { 0x00 };
            for bit_idx in (0..8).rev() {
                let bit = (byte >> bit_idx) & 1 == 1;
                let bit_pos = bit_idx as u8;
                last_probs[0] = m0.predict();
                last_probs[1] = m1.predict();
                let fused = mixer.mix(&last_probs, bit_pos);
                m0.update(bit);
                m1.update(bit);
                mixer.update(&last_probs, bit, bit_pos);
                if step > 150 && bit {
                    assert!(i32::from(fused) > 2048, "mixer should learn the bias");
                }
            }
        }
    }

    #[test]
    fn mix_in_range() {
        let mixer = LogisticMixer::new(3);
        let p = mixer.mix(&[1000, 2048, 3000], 0);
        assert!((1..=4095).contains(&p));
    }
}
