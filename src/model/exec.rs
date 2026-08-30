//! Executable 2D-context bit model.
//!
//! Executables (x86/ELF/Mach-O) have strong local structure: the current byte and the
//! signed delta to the previous byte predict the next. This model keys on
//! `(prev_byte, delta = cur - prev)` to exploit that, which plain order models overlook.

use super::BitModel;

const INIT: u32 = 32;
const MAX_PROB: u16 = 4095;
const MIN_PROB: u16 = 1;

/// Executable-oriented 2D-context bit model.
pub struct Exec {
    prev: u8,
    cur: u8,
    tables: std::collections::HashMap<u32, [u32; 2]>,
    seen: bool,
}

impl Exec {
    /// Create an executable context model.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prev: 0,
            cur: 0,
            tables: std::collections::HashMap::new(),
            seen: false,
        }
    }

    fn key(&self) -> u32 {
        // key = prev_byte<<8 | (delta & 0xFF), delta = cur.wrapping_sub(prev)
        let delta = self.cur.wrapping_sub(self.prev);
        (u32::from(self.prev) << 8) | u32::from(delta)
    }

    fn entry(&self) -> [u32; 2] {
        self.tables.get(&self.key()).copied().unwrap_or([INIT, INIT])
    }
}

impl BitModel for Exec {
    fn predict(&self) -> u16 {
        if !self.seen {
            return 2048; // no context yet
        }
        let [c0, c1] = self.entry();
        let tot = f64::from(c0 + c1);
        (f64::from(c1) / tot * f64::from(MAX_PROB))
            .clamp(f64::from(MIN_PROB), f64::from(MAX_PROB)) as u16
    }

    fn update(&mut self, bit: bool) {
        if self.seen {
            let k = self.key();
            let e = self.tables.entry(k).or_insert([INIT, INIT]);
            e[bit as usize] += 1;
        }
        self.prev = self.cur;
        self.cur = bit as u8;
        self.seen = true;
    }

    fn reset(&mut self) {
        self.prev = 0;
        self.cur = 0;
        self.seen = false;
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
