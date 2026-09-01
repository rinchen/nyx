//! Benchmark-only model-stack builders for config comparison.
//!
//! These builders are intentionally public so comparison binaries can exercise the
//! same stacks without duplicating construction logic.

use crate::model::exec::Exec;
use crate::model::lzp::Lzp;
use crate::model::mixer::LogisticMixer;
use crate::model::order::OrderN;
use crate::model::ppm::PpmModel;
use crate::model::sparse::Sparse;
use crate::model::BitModel;

pub struct BaselineBuilder;
pub struct PpmBuilder {
    pub max_order: usize,
}
pub struct HybridPpm3Builder;

impl BaselineBuilder {
    #[must_use]
    pub fn build() -> (Vec<Box<dyn BitModel>>, LogisticMixer) {
        crate::codec::build_full_stack()
    }
}

impl PpmBuilder {
    #[must_use]
    pub const fn new(max_order: usize) -> Self {
        Self { max_order }
    }

    #[must_use]
    pub fn build(&self) -> (Vec<Box<dyn BitModel>>, LogisticMixer) {
        let models: Vec<Box<dyn BitModel>> = vec![Box::new(PpmModel::new(self.max_order))];
        let mixer = LogisticMixer::new(models.len());
        (models, mixer)
    }
}

impl HybridPpm3Builder {
    #[must_use]
    pub fn build() -> (Vec<Box<dyn BitModel>>, LogisticMixer) {
        let models: Vec<Box<dyn BitModel>> = vec![
            Box::new(OrderN::new(0)),
            Box::new(OrderN::new(1)),
            Box::new(OrderN::new(2)),
            Box::new(Sparse::new()),
            Box::new(Lzp::new()),
            Box::new(Exec::new()),
            Box::new(PpmModel::new(3)),
        ];
        let mixer = LogisticMixer::new(models.len());
        (models, mixer)
    }
}
