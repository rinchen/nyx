//! Executable 2D-context bit model.
//!
//! Executables (x86/ELF/Mach-O) have strong local structure: the current byte and the
//! signed delta to the previous byte predict the next. This model keys on
//! `(prev_byte, delta = cur - prev)` using the **assembled byte** stream, which plain
//! bit-context models cannot see.

use super::ctable::CtxTable;
use super::BitModel;
use super::ByteAssembler;

const MAX_PROB: u16 = 4095;
const MIN_PROB: u16 = 1;
const CTX_BITS: u32 = 20;

/// Executable-oriented 2D-context bit model.
pub struct Exec {
    asm: ByteAssembler,
    ctab: CtxTable,
}

impl Exec {
    /// Create an executable context model.
    #[must_use]
    pub fn new() -> Self {
        Self {
            asm: ByteAssembler::new(2),
            ctab: CtxTable::new(CTX_BITS),
        }
    }

    #[inline]
    fn key(&self) -> u64 {
        let prev = self.asm.prev_byte();
        let cur = self.asm.last_byte();
        let delta = cur.wrapping_sub(prev);
        // pack (prev, delta) into a 64-bit context
        (u64::from(prev) << 8) | u64::from(delta)
    }
}

impl BitModel for Exec {
    fn predict(&self) -> u16 {
        if !self.asm.has_byte() {
            return 2048; // no byte context yet
        }
        let [c0, c1] = self.ctab.get(self.key());
        let tot = f64::from(c0 + c1);
        (f64::from(c1) / tot * f64::from(MAX_PROB)).clamp(f64::from(MIN_PROB), f64::from(MAX_PROB))
            as u16
    }

    fn update(&mut self, bit: bool) {
        // Use the pre-push context (matches predict), then advance the assembler.
        let k = self.key();
        self.asm.push_bit(bit);
        self.ctab.update(k, bit);
    }

    fn reset(&mut self) {
        self.asm.reset();
        self.ctab.reset();
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Default for Exec {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicts_uniform_before_context() {
        let m = Exec::new();
        assert_eq!(m.predict(), 2048);
    }

    #[test]
    fn adapts_after_run() {
        let mut m = Exec::new();
        for _ in 0..12 {
            m.update(true);
        }
        assert!(m.predict() > 2048);
    }

    #[test]
    fn reset_clears() {
        let mut m = Exec::new();
        m.update(true);
        m.reset();
        assert_eq!(m.predict(), 2048);
    }
}
