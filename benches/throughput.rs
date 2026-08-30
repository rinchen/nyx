use criterion::{criterion_group, criterion_main, Criterion};

#[allow(clippy::missing_const_for_fn)]
fn bench_placeholder(_c: &mut Criterion) {}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
