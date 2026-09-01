# Nyx — Stage 2.1 measurement log (complete)

## Harness
- Binary: `cargo run --release --bin bench_configs -- bench-configs <corpus>`
- Configs: `baseline`, `ppm3`, `ppm4`, `hybrid_ppm3`
- Subset: `dickens`, `webster`, `nci`, `mr`, `json`

## Final results

| file   | baseline | ppm3 | ppm4 | hybrid_ppm3 | marker vs baseline |
|--------|----------|------|------|-------------|-------------------|
| dickens| 58.4%    | 64.8%↓ | 64.8%↓ | 57.5% | ↑ |
| webster| 54.7%    | 59.2%↓ | 59.2%↓ | 51.8% | ↑ |
| nci    | 30.7%    | 35.2%↓ | 35.2%↓ | 30.4% | ↑ |
| mr     | 30.6%    | 39.0%↓ | 39.0%↓ | 30.5% | ~ |
| json   | 13.6%    | 8.9%  | 8.9%  | 7.8% | ↑ |

## Verdict
- PPM-only is regressive on text/binary files.
- `hybrid_ppm3` improves all five subset files and preserves existing wins.
- `hybrid_ppm3` is now the default stack in `src/codec.rs::build_full_stack()`.
- README and BENCH updated with measured numbers.
- All round-trips passed; tests/clippy/build verified.

## Next stage candidates
- Adaptive LZP for `mr`/`json`
- Classifier-aware model selection
- Richer mixer context
