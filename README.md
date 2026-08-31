# Nyx

A from-scratch Rust lossless compressor with a **per-block data-type classifier**
and an **online logistic bit-mixer** (LZP pre-stage + rANS-grade entropy coder).
It is a self-contained CLI with its own `NYX1` container format.

> **Status: research/learning codec, actively improving.** Nyx is a working, honest
> implementation of bit-level context mixing with an online logistic mixer. It is **not**
> a production competitor to `zstd` — it is dominated by `zstd` on both ratio and speed
> because it entropy-codes bit-by-bit. The project goal is to close the **ratio** gap
> against `zstd -19` on text/binary data while keeping speed as a secondary concern.
> See [Benchmarks](#benchmarks) for the real numbers (both ratio and speed).

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
> must report both axes. Full-corpus (12-file Silesia + mixed) numbers are expensive at
> ~1.5 MB/s, so the headline table below is a representative **5-file subset**
> (dickens, webster, nci, mr, json — text, mixed-binary, structured). `ratio%` is the
> compressed size as a percentage of the original (lower is better); speed is in MB/s
> (higher is better). See [BENCH.md](BENCH.md) for the before/after optimization tables
> and full notes.

### Current (post causal-fix; 27 tests, clippy clean)

| file   | nyx ratio% | nyx cmp MB/s | nyx dec MB/s | zstd-19 ratio% | zstd-19 cmp MB/s | zstd-19 dec MB/s | xz-9 ratio% | lz4-9 ratio% | vs zstd ratio |
|--------|-----------:|-------------:|-------------:|---------------:|-----------------:|-----------------:|------------:|------------:|:-------------:|
| dickens | 58.4       | 1.5          | 0.7          | 28.0           | 3.3              | 252.0            | 27.8        | 43.6        | → zstd        |
| webster | 54.7       | 1.5          | 0.6          | 69.0           | 3.3              | 666.6            | 60.9        | 79.1        | **← nyx**     |
| nci     | 30.6       | 1.3          | 0.7          | 31.2           | 3.7              | 226.7            | 27.6        | 42.6        | **← nyx**     |
| mr      | 30.7       | 1.5          | 0.9          | 5.0            | 3.5              | 754.4            | 5.2         | 11.0        | → zstd        |
| json    | 13.6       | 1.6          | 1.1          | ~0             | 15.4             | 16.6             | ~0          | 0.4         | → zstd        |

(`~0` = zstd compresses json to ~0.1 KB; the harness percentage rounds to 0 on a
KB-normalized basis.)

### Reading the table

- **Ratio:** lower % is better. `→ zstd` means zstd-19 beats nyx on ratio for that file;
  `← nyx` means nyx beats zstd-19.
- **Speed:** higher MB/s is better. nyx is **~1.5 MB/s** compress / **~0.7 MB/s** decode
  across the board; `zstd -19` is **~3–4 MB/s** compress / **200–880 MB/s** decode.
  That is a **~100–500× decode gap** — an architectural constant of bit-level context
  mixing, not a tuning target.

### Honest assessment

- **Ratio:** the 2026-08-30 optimization pass (direct-addressed context tables + a causal
  predict/update fix) dropped dickens 68.6%→58.4%, webster 87.6%→54.7%, json 38.9%→13.6%.
  nyx now **beats `zstd -19` on `nci` and `webster`** (30.6 vs 31.2; 54.7 vs 69.0). But on
  text it is still **~2× worse** than zstd (dickens 58.4 vs 28.0). Reaching zstd text
  parity requires PPM-style adaptive higher-order modeling with escape (a planned future
  stage), not a tuning knob.
- **Speed:** nyx compresses at **~1.5 MB/s** vs zstd's ~3–9 MB/s, and decodes at
  **~0.7 MB/s** vs zstd's **200–880 MB/s**. That is a ~100–500× decode gap — an
  architectural constant of bit-level context mixing, not a tuning target.

What nyx *does* demonstrate correctly: a per-block classifier, a heterogeneous model
stack fused by an online logistic mixer, and causal, lossless round-trips. Hyperparameter
sweeps (block size, mixer learning rate, model set, classifier thresholds) were run and
confirmed the remaining gap is structural, not a knob.

## License

MIT — see [LICENSE](LICENSE).
