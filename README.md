# Nyx

A from-scratch Rust lossless compressor with a **per-block data-type classifier**
and an **online logistic bit-mixer** (LZP pre-stage + rANS-grade entropy coder).
It is a self-contained CLI with its own `NYX1` container format.

> **Status: research/learning codec.** Nyx is a working, honest implementation of
> context mixing, but it is **not** a production competitor to `zstd`. See
> [Benchmarks](#benchmarks) for the real numbers — nyx is dominated by `zstd` on
> both ratio and speed. The point was to build the method correctly, not to beat
> the frontier.

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

Measured on the **Silesia + Mixed** corpus (12 Silesia files + a 5,000-line JSON
file, ~211 MB total) with `nyx` at its current defaults (rANS backend, full model
stack). Compression speed is in MB/s; lower `ratio%` is better.

| file       | nyx % | zstd-19 % | xz-9 % | lz4-9 % | nyx cmp MB/s | nyx dec MB/s |
|------------|------:|----------:|-------:|--------:|-------------:|-------------:|
| dickens    | 68.6  | 28.0      | 27.8   | 43.6    | 0.8          | 0.6          |
| mozilla    | 65.6  | 29.4      | 26.1   | 43.1    | 0.8          | 0.6          |
| nci        | 34.0  | 31.2      | 27.6   | 42.6    | 0.8          | 0.7          |
| osdb       | 32.4  | 5.0       | 5.2    | 11.0    | 0.9          | 0.7          |
| ooffice    | 81.6  | 42.2      | 39.4   | 57.6    | 0.7          | 0.6          |
| reymont    | 80.6  | 30.7      | 28.3   | 39.5    | 0.7          | 0.5          |
| samba      | 61.4  | 20.3      | 19.9   | 31.9    | 0.8          | 0.6          |
| sao        | 62.6  | 18.0      | 17.4   | 28.5    | 0.8          | 0.6          |
| webster    | 87.6  | 69.0      | 60.9   | 79.1    | 0.7          | 0.5          |
| x-ray      | 67.6  | 20.9      | 20.2   | 33.8    | 0.8          | 0.6          |
| xml        | 74.4  | 60.5      | 53.0   | 84.8    | 0.7          | 0.5          |
| json       | 60.0  | 8.5       | 8.5    | 14.4    | 0.8          | 0.6          |

Reference speeds for the SOTA tools on the same corpus: `zstd -19` compresses at
~3–9 MB/s and decompresses at **100–900 MB/s**; `lz4 -9` at ~50–250 MB/s compress
and ~200–1000 MB/s decompress.

### Honest assessment

Nyx does **not** sit on the Pareto frontier. It is roughly **2–3× worse on ratio**
and **~100–1000× slower** than `zstd` on decompression. This is architectural, not
a tuning gap:

- Nyx entropy-codes **bit-by-bit** (8× the symbol rate of byte-level coders), and
  each model keeps a per-bit `HashMap` context — so it is inherently slow.
- The count-based context models are crude next to `zstd`'s FSE + LDM + literal
  modeling.

What nyx *does* demonstrate correctly: a per-block classifier, a heterogeneous
model stack, and an online logistic mixer producing real (if uncompetitive)
compression. Hyperparameter sweeps (block size, mixer learning rate, model set,
classifier thresholds) were run and confirmed the gap is structural.

## License

MIT — see [LICENSE](LICENSE).
