//! Benchmark-only model-stack builders for config comparison.
//!
//! These builders are intentionally public so comparison binaries can exercise the
//! same stacks without duplicating construction logic.

use crate::model::exec::Exec;
use crate::model::lzp::Lzp;
use crate::model::mixer_bank::MixerBank;
use crate::model::order::OrderN;
use crate::model::ppm::PpmModel;
use crate::model::ppmd_ssm::PpmdSsm;
use crate::model::sparse::Sparse;
use crate::model::stack_enum::StackModel;

pub struct BaselineBuilder;
pub struct PpmBuilder {
    pub max_order: usize,
}
pub struct HybridPpm3Builder;
pub struct PpmdSsmBuilder;

impl BaselineBuilder {
    #[must_use]
    pub fn build() -> (Vec<StackModel>, MixerBank, Option<usize>) {
        crate::codec::build_full_stack()
    }
}

impl PpmBuilder {
    #[must_use]
    pub const fn new(max_order: usize) -> Self {
        Self { max_order }
    }

    #[must_use]
    pub fn build(&self) -> (Vec<StackModel>, MixerBank, Option<usize>) {
        let models = vec![StackModel::Ppm(PpmModel::new(self.max_order))];
        let mixer = MixerBank::new(models.len());
        (models, mixer, None)
    }
}

impl HybridPpm3Builder {
    #[must_use]
    pub fn build() -> (Vec<StackModel>, MixerBank, Option<usize>) {
        let models = vec![
            StackModel::Order(OrderN::new(0)),
            StackModel::Order(OrderN::new(1)),
            StackModel::Order(OrderN::new(2)),
            StackModel::Sparse(Sparse::new()),
            StackModel::Exec(Exec::new()),
            StackModel::Lzp(Lzp::new()),
            StackModel::Ppm(PpmModel::new(3)),
        ];
        let mixer = MixerBank::new(models.len());
        (models, mixer, Some(5))
    }
}

/// Stack using the PpmdSsm model (orders 0-8 + SEE + sparse de Bruijn contexts)
/// instead of PpmModel(3). Useful for comparing PPMd vs vanilla PPM on text.
/// Unlike the original PpmdSsmBuilder, this version keeps WordModel in the
/// stack, matching the hybrid_ppm3 configuration. (LazyLzp removed for speed —
/// its drain-based history was O(n²) on large blocks.)
impl PpmdSsmBuilder {
    #[must_use]
    pub fn build() -> (Vec<StackModel>, MixerBank, Option<usize>) {
        let models = vec![
            StackModel::Order(OrderN::new(0)),
            StackModel::Order(OrderN::new(1)),
            StackModel::Order(OrderN::new(2)),
            StackModel::Sparse(Sparse::new()),
            StackModel::Exec(Exec::new()),
            StackModel::Lzp(Lzp::new()),
            StackModel::Ppmd(PpmdSsm::new()),
            StackModel::Word(crate::model::word::WordModel::new()),
        ];
        let mixer = MixerBank::new(models.len());
        (models, mixer, Some(5))
    }
}
