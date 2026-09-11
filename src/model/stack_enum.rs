//! Concrete model enum for the default Slow stacks (V3).
//!
//! Replaces `Box<dyn BitModel>` in the hot predict→update loop with a closed
//! enum so the compiler can monomorphize / inline per-variant calls.

use super::exec::Exec;
use super::indirect_dmc::IndirectModel;
use super::lzp::Lzp;
use super::order::OrderN;
use super::ppm::PpmModel;
use super::ppmd_ssm::PpmdSsm;
use super::sparse::Sparse;
use super::word::WordModel;
use super::BitModel;

/// One entry in a default Slow-mode model stack.
pub enum StackModel {
    Order(OrderN),
    Sparse(Sparse),
    Exec(Exec),
    Lzp(Lzp),
    Ppmd(PpmdSsm),
    Word(WordModel),
    Ppm(PpmModel),
    /// Binary-only (W5); not used on Text after prior regressions.
    Indirect(IndirectModel),
}

impl StackModel {
    #[inline(always)]
    pub fn predict(&self) -> u16 {
        match self {
            Self::Order(m) => m.predict(),
            Self::Sparse(m) => m.predict(),
            Self::Exec(m) => m.predict(),
            Self::Lzp(m) => m.predict(),
            Self::Ppmd(m) => m.predict(),
            Self::Word(m) => m.predict(),
            Self::Ppm(m) => m.predict(),
            Self::Indirect(m) => m.predict(),
        }
    }

    #[inline(always)]
    pub fn update(&mut self, bit: bool) {
        match self {
            Self::Order(m) => m.update(bit),
            Self::Sparse(m) => m.update(bit),
            Self::Exec(m) => m.update(bit),
            Self::Lzp(m) => m.update(bit),
            Self::Ppmd(m) => m.update(bit),
            Self::Word(m) => m.update(bit),
            Self::Ppm(m) => m.update(bit),
            Self::Indirect(m) => m.update(bit),
        }
    }

    #[inline]
    pub fn prepare_block(&mut self, block: &[u8]) {
        match self {
            Self::Order(m) => m.prepare_block(block),
            Self::Sparse(m) => m.prepare_block(block),
            Self::Exec(m) => m.prepare_block(block),
            Self::Lzp(m) => m.prepare_block(block),
            Self::Ppmd(m) => m.prepare_block(block),
            Self::Word(m) => m.prepare_block(block),
            Self::Ppm(m) => m.prepare_block(block),
            Self::Indirect(m) => m.prepare_block(block),
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        match self {
            Self::Order(m) => m.reset(),
            Self::Sparse(m) => m.reset(),
            Self::Exec(m) => m.reset(),
            Self::Lzp(m) => m.reset(),
            Self::Ppmd(m) => m.reset(),
            Self::Word(m) => m.reset(),
            Self::Ppm(m) => m.reset(),
            Self::Indirect(m) => m.reset(),
        }
    }
}
