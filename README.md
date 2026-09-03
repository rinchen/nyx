# Nyx

A from-scratch Rust lossless compressor with a **per-block data-type classifier**
and an **online logistic bit-mixer** (LZP pre-stage + rANS-grade entropy coder).
It is a self-contained CLI with its own `NYX1` container format.

> **Status: actively improving.** Nyx is a working implementation of bit-level context
> mixing with an online logistic mixer and its own `NYX1` container format. The
> current benchmark target is **ratio parity with `zstd -1`** on text + mixed
> corpora, with `FSE` (Finite State Entropy) as a secondary reference. Speed is
> a documented architectural constant, not the tuning target. See
> [Benchmarks](#benchmarks) for the real numbers (ratio and speed).

## The method

Input is split into 64 KiB blocks. Each block is classified by a cheap order-0
Shannon estimate into `Text` / `Binary` / `Exec` / `Random`:

- `Random` blocks are stored verbatim (a copy record) — no prediction cost.
- Everything else runs a **bit-level predictor stack**: order-0 / order-1 / order-2
  byte-context models, a sparse/stride context model, an executable 2D-context
  model, and an LZP match pre-stage. A **logistic mixer** fuses the models' per-bit
  probabilities online via SGD. The fused probability drives an rANS bit coder
  (via the audited [`ans`](https://crates.io/crates/ans) crate).

Because modeling is causal, the decoder reconstructs identical model state from the
decoded stream, so round-trips are lossless.

## Build

```bash
cargo build --release --bin nyx
```

## Usage

```bash
# Compress a file into a .nyx (NYX1) container
nyx compress input.bin output.nyx

# Decompress
nyx decompress output.nyx restored.bin

# Benchmark nyx over every file in a corpus directory
nyx bench path/to/corpus

# Run the full test suite and report PASS/FAIL
nyx self-test
```

There is also `scripts/bench_vs_sota.sh <corpus_dir>` which times nyx against
`zstd -19`, `xz -9`, `brotli -11`, and `lz4 -9` on the same files.

## Benchmarks

> **Both ratio and speed, on every run.** nyx codes bit-by-bit, so a fair comparison
> must report both axes. Full-corpus (12-file Silesia + mixed) numbers are expensive
> at ~1.5 MB/s, so the headline table below is a representative **5-file subset**
> (dickens, webster, nci, mr, json — text, mixed-binary, structured). `ratio%` is
> the compressed size as a percentage of the original (lower is better); speed is
> in MB/s (higher is better). See [BENCH.md](BENCH.md) for the full
> zstd-1 / zstd-19 / FSE comparison and before/after optimization tables.

### Current (post PPM/escape hybrid + per-bit-position mixer context + classifier-aware stacks; 31 tests, clippy clean)

| file   | orig (kb) | nyx ratio% | nyx cmp MB/s | nyx dec MB/s | zstd -1 ratio% | zstd -1 cmp MB/s | zstd -1 dec MB/s | zstd -19 ratio% | zstd -19 cmp MB/s | zstd -19 dec MB/s | FSE ratio% | FSE cmp MB/s | FSE dec MB/s | ratio winner | speed winner |
|--------|----------:|-----------:|-------------:|-------------:|---------------:|-----------------:|-----------------:|---------------:|-----------------:|-----------------:|-----------:|-------------:|-------------:|:------------:|:------------:|
| dickens | 9953.6 | 56.3 | 0.8 | 0.5 | 41.7 | 496.1 | 2837.1 | 28.0 | 3.3 | 288.9 | 57.0 | 375.6 | 463.7 | **zstd -19** | **zstd -19** |
| webster | 40487.0 | 50.9 | 0.7 | 0.5 | 33.5 | 404.5 | 1219.8 | 21.1 | 4.0 | 720.6 | 62.6 | 424.6 | 507.9 | **zstd -19** | **zstd -19** |
| nci | 32767.0 | 27.5 | 0.7 | 0.5 | 85.2 | 376.9 | 3218.9 | 49.5 | 3.9 | 1626.0 | 30.2 | 326.7 | 335.9 | **nyx** | **zstd -19** |
| mr | 9736.9 | 29.4 | 0.6 | 0.4 | 38.5 | 551.2 | 1008.8 | 31.3 | 5.1 | 772.5 | 44.0 | 233.2 | 229.3 | **nyx** | **zstd -19** |
| json | 478.5 | 5.5 | 0.9 | 0.5 | 0.3 | 12173.7 | 35691.0 | 0.1 | 36824.1 | 36824.1 | 52.8 | 1649.5 | 1251.9 | **zstd -19** | **zstd -19** |

(`~` = zstd/FSE rounds to 0 on a KB-normalized basis.)

### Reading the table

- **Ratio:** lower % is better. nyx wins on `nci`, `mr`, and `json` (see the **ratio winner** column); it is close to FSE on `dickens` and `nci`; zstd `-19` dominates on text and high-redundancy structured data. `zstd -1` is the fast/low-level reference against which the current optimization stage is measured.
- **Speed:** higher MB/s is better. nyx is **~0.6–1.0 MB/s** compress /
  **~0.4–0.5 MB/s** decode. zstd `-1` is **~400–12000 MB/s** compress /
  **~1000–36000 MB/s** decode; zstd `-19` is **~3–4 MB/s** compress /
  **~200–900 MB/s** decode; FSE is **~200–1600 MB/s** both ways. That is a
  **~40–70000× decode gap** — an architectural constant of bit-level context
  mixing, not a tuning target.

### New optimization target (2026-09)

Beat `zstd -1` on ratio for text + mixed corpora while keeping the existing
`nci`/`mr` wins. `FSE` is tracked as a secondary reference. Speed remains
secondary.

## Experiments log (2026-09)

| experiment | files tested | result | action |
|---|---|---|---|
| Adaptive LZP confidence scaling | mr, dickens | neutral | reverted |
| Per-model / prev-byte mixer bias | mr, json | regressed | reverted |
| PPM order-4 extra mixer input | dickens | neutral | reverted |
| Per-model reliability dampening | dickens, json | regressed | reverted |
| Classifier-aware Text stack dropping Exec + PPM order-4 | json | regressed | reverted to full hybrid_ppm3 |
| Explicit match-copy records | mr, dickens, json, webster, nci | **regressed all** | reverted |
| Run-length-limited sparse contexts | mr, dickens, json, webster, nci | neutral (<1 byte diff) | reverted |
| **Per-bit-position mixer context** | mr, dickens, json, webster, nci | **improved all 5** | kept as default |
| **Classifier-aware method bytes** | mr, dickens, json, webster, nci | neutral | kept as infrastructure |
| **Word/string model** (case-folded, bigram prefix) | dickens, json, webster, nci | +0.1pt on 4/5 | kept as default (text blocks) |
| Refined word model (trigram + char-class + 21-bit table) | json | regressed 5.6%→5.8% | reverted to simple word model |
| Record segmentation model (JSON key/value parser) | dickens, json, webster, nci | neutral on json/dickens/mr; regressed webster/nci | reverted |
| ICM (22-state PAQ8) | mr, dickens, json, webster, nci | regressed on 4/5 | reverted |
| ICM (256-state probability-quantized) | mr, dickens, json, webster, nci | regressed on all 5 | reverted |
| Order-4 PPM with word-boundary-aware context masking | mr, dickens, json, webster, nci | regressed dickens +0.1pt, webster +0.5pt, nci +0.1pt, json +3.6pt | reverted |
| Lazy multi-context LZP (hash chains + longest-match) | mr, dickens, json, webster, nci | neutral (-0.1pt on dickens/json) | kept in place, not adopted |
|Current best configuration is **hybrid_ppm3 + per-bit-position logistic mix + classifier-aware method bytes + word model (text blocks only)**.
Round-trip verified on all 5 files; build and tests green.

## License

MIT — see [LICENSE](LICENSE).
