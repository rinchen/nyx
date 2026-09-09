# Rcn

`rcn` — **R**ust **C**ompressor, **N**ew — is a ratio-first command-line
compressor written in Rust. It combines context mixing, Burrows–Wheeler
Transform (BWT), and DP-LZP matching. The benchmark target is beating
`zstd -19` on text. The `.rcn` (RCN1) container and CLI may still change.

Optimization tickets, A/B history, and CI notes:
[DEVELOPMENT.md](DEVELOPMENT.md) · [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md).

## Design methodology

`rcn` is built around a simple thesis: *weight above all* — squeeze the last bit
of redundancy out of the input rather than prioritizing throughput. Its pipeline
is a sequence of independent, replaceable stages, each optimized and benchmarked
in isolation:

- **Classify first.** Input is split into variable-size blocks by data type so
  each block gets the transform it deserves (see [The method](#the-method)).
- **Transform locally.** Text blocks run BWT trials to turn long-range word
  repeats into local runs, and a word transform interpolates unknown symbols.
- **Match with DP-LZP.** A dynamic-programming LZP pre-pass removes the obvious
  redundancy before entropy coding, leaving the residual for the models.
- **Mix probabilistically.** An online logistic mixer blends many context models
  (order-0/1/2, sparse, executable, word, LZP) bit-by-bit to drive a rANS coder.

Stages are modular: swap a model, re-run a benchmark, keep or discard based on
measured ratio. The current target is beating `zstd -19` on text and mixed corpora.

## Projected use cases

- **Learning reference** — a readable, self-contained pipeline combining BWT,
  LZP matching, and context-mixing entropy coding.
- **Text-centric archival** — corpora with lots of redundancy (dickens, webster,
  source trees, JSON dumps) where ratio matters more than speed.
- **Baseline for further work** — a clean staging ground for trying new models,
  transforms, or entropy backends against a fixed benchmark harness.

It is not aimed at general-purpose, at-rest or on-the-wire compression where
`zstd -1`/`gzip` speed is the deciding factor — expect `rcn` to be much slower
per byte than those, in exchange for better ratio on the right inputs.

## The method

Rcn ships two entropy paths:

- `--mode slow` (default): bit-level CM with a two-level 16k-bank mixer.
- `--mode fast`: PPM-style byte CM + interleaved rANS.

On the headline set, **fast beats slow on ratio for text/nci** (and matches on
json); **slow beats fast on `mr`**. Against `zstd -19`, slow wins only `nci`/`mr`
(and ties json) while losing text; fast wins text/`nci` (ties json) but loses
`mr`. Default stays `slow` until fast also ties or beats `zstd -19` on every
headline file (the `mr` gap). See [Benchmarks](#benchmarks).

Input is split into variable-size blocks by data type. Each block is classified
by a cheap order-0 Shannon estimate into `Text` / `Binary` / `Exec` / `Random`:

- `Random` blocks are stored verbatim — no prediction cost.
- `Text` blocks can be up to 4 MB (enabling BWT trials that turn long-range
  word repeats into local runs).
- `Binary`, `Exec`, and `Random` use the default 64 KiB chunk size.

The bit-level path runs an online logistic mixer hierarchy:

1. **Bank mixers** (16384 instances): selected by a context hash of byte-class,
   bit-position, order-1/order-2 bytes, and word-hash. Each bank specializes
   weights to its context, avoiding the ~50% saturation a single mixer hits
   on repetitive corpora.
2. **Global mixer**: a context-agnostic fallback over the same models.
3. **Master mixer**: blends `[p_bank, p_global, p_lzp_conf]` in logistic space.

Only the selected bank + global + master are trained per bit — never all 16384.
At block boundaries, weights are decayed (not reset), preserving learned
structure across the stream. The fused probability drives an rANS bit coder
(via the audited [`ans`](https://crates.io/crates/ans) crate).

Because modeling is causal, the decoder reconstructs identical model state from
the coded stream, so round-trips are lossless.

## Build

```bash
cargo build --release --bin rcn
```

## Usage

```bash
# Help and version
rcn -h
rcn --version

# Compress a file into a .rcn (RCN1) container
rcn compress input.bin output.rcn

# Per-block method/size on stderr
rcn compress --verbose input.bin output.rcn

# Faster byte-CM path (better text ratio; loses to zstd -19 on mr)
rcn compress --mode fast input.bin output.rcn

# Decompress
rcn decompress output.rcn restored.bin

# Benchmark rcn over every file in a corpus directory
rcn bench path/to/corpus

# Run the full test suite and report PASS/FAIL
rcn self-test
```

A section-1 man page lives at [`man/rcn.1`](man/rcn.1). Preview it from the
source tree with `man ./man/rcn.1`. To install it system-wide (optional):

```bash
install -m 644 man/rcn.1 "$(manpath | cut -d: -f1)/man1/rcn.1"
```

`cargo install` installs the binary only, not the man page.

### Installation

`rcn` is installed via `cargo install --locked rcn` or downloaded as a binary
release from [crates.io](https://crates.io/crates/rcn). The package requires
Rust toolchain ≥1.85 (`rust-version` in `Cargo.toml`) and an x86_64 (AVX2) or
arm64 (scalar) processor. On x86_64 (Linux), the AVX2 code path can be disabled
at build time with `--no-default-features --features no_avx2` — see
[CI / testing](DEVELOPMENT.md#ci--testing).

### Compression modes

- `--mode slow` (default): bit-level CM. Beats fast on `mr`; loses text to both
  fast and `zstd -19`. Much slower than fast.
- `--mode fast`: byte-level CM. Beats slow (and usually `zstd -19`) on text/`nci`;
  loses to `zstd -19` on `mr`. ~50–100× throughput vs slow.

Default stays `slow` until fast also beats or ties `zstd -19` on the full
headline 5-file set. Named `-1`…`-9` presets remain out of scope.

### Streaming / `--stdout`

The `compress` and `decompress` subcommands accept `-` as input/output path for
stdin/stdout piping. Compress accepts `--verbose` for per-block method and size.

## Benchmarks

Both ratio and speed, on every run. The tables use a representative 5-file
subset (dickens, webster, nci, mr, json). `ratio%` is compressed size as a
percentage of the original (lower is better); speed is in MB/s (higher is
better). **ratio winner** / **speed winner** name the best codec in that row
among rcn and the reference columns — not “rcn always wins.” A/B history:
[DEVELOPMENT.md](DEVELOPMENT.md).

### Slow path (`--mode slow`, default)

Stack: hybrid_ppm3, two-level 16k-bank mixer, classifier-aware method bytes,
word model, cross-block decay, 32MB LZP window, BWT text trial,
JSON/CSV/XML stream splitting, DP-optimal LZP parse, Exec E8E9, global
XWRT-512 ESC, SSE/APM/APM2, adaptive DP thresholds, side-stream FSE-family,
classify-ahead, MixerAcc stretch/prefetch, wider BWT trials, mimalloc.

| file | orig (KB) | rcn ratio% | rcn cmp MB/s | rcn dec MB/s | zstd -1 ratio% | zstd -1 cmp MB/s | zstd -1 dec MB/s | zstd -19 ratio% | zstd -19 cmp MB/s | zstd -19 dec MB/s | FSE ratio% | FSE cmp MB/s | FSE dec MB/s | ratio winner | speed winner |
|------|----------:|-----------:|-------------:|-------------:|---------------:|-----------------:|-----------------:|---------------:|-----------------:|-----------------:|-----------:|-------------:|-------------:|:------------:|:------------:|
| dickens | 9953.6 | 40.2 | 0.04 | 0.04 | 41.7 | 496.1 | 2837.1 | 28.0 | 3.3 | 288.9 | 57.0 | 375.6 | 463.7 | zstd -19 | zstd -19 |
| webster | 40487.0 | 29.3 | 0.1 | 0.1 | 33.5 | 404.5 | 1219.8 | 21.1 | 4.0 | 720.6 | 62.6 | 424.6 | 507.9 | zstd -19 | zstd -19 |
| nci | 32767.0 | 7.3 | 0.2 | 0.2 | 85.2 | 376.9 | 3218.9 | 49.5 | 3.9 | 1626.0 | 30.2 | 326.7 | 335.9 | rcn | zstd -19 |
| mr | 9736.9 | 27.4 | 0.04 | 0.04 | 38.5 | 551.2 | 1008.8 | 31.2 | 5.6 | 291.2 | 44.0 | 233.2 | 229.3 | rcn | zstd -19 |
| json | 478.5 | 0.1 | 6.3 | 30.7 | 0.3 | 12173.7 | 35691.0 | 0.1 | 36824.1 | 36824.1 | 52.8 | 1649.5 | 1251.9 | rcn/tie | zstd -19 |

Notes: Slow 5-file hygiene 2026-09-09. Bench MB/s prints two decimals. zstd/FSE
columns unchanged.

### Fast path (`--mode fast`)

Re-bench 2026-09-09. zstd/FSE ratio columns reused from the slow-path reference
set. Speed winner uses decode MB/s.

| file | orig (KB) | rcn ratio% | cmp MB/s | dec MB/s | zstd -1 ratio% | zstd -19 ratio% | FSE ratio% | ratio winner | speed winner |
|------|----------:|-----------:|---------:|---------:|---------------:|---------------:|-----------:|:------------:|:------------:|
| dickens | 9953.6 | 26.9 | 2.0 | 4.3 | 41.7 | 28.0 | 57.0 | rcn | zstd -1 |
| webster | 40487.0 | 20.3 | 2.7 | 5.5 | 33.5 | 21.1 | 62.6 | rcn | zstd -1 |
| nci | 32767.0 | 5.1 | 5.1 | 10.6 | 85.2 | 49.5 | 30.2 | rcn | zstd -1 |
| mr | 9736.9 | 35.0 | 2.6 | 4.0 | 38.5 | 31.2 | 44.0 | zstd -19 | zstd -1 |
| json | 478.5 | 0.1 | 5.2 | 59.1 | 0.3 | 0.1 | 52.8 | rcn/tie | zstd -1 |

### Reading the tables

- **Slow vs `zstd -19` (ratio):** rcn wins `nci` and `mr`, ties `json`; loses
  dickens (40.2% vs 28.0%) and webster (29.3% vs 21.1%). Speed winner is always
  zstd.
- **Fast vs `zstd -19` (ratio):** rcn wins dickens/webster/`nci`, ties `json`;
  loses `mr` (35.0% vs 31.2%). That `mr` loss is why fast is not the default.
  Speed winner is always zstd `-1`.
- **Fast vs slow (rcn only):** fast is better on dickens/webster/`nci` (and
  matched on json); slow is better on `mr` (27.4% vs 35.0%).
- **Throughput:** slow ~0.04 MB/s on large text; fast ~2–5 MB/s compress /
  ~4–10 MB/s decode — still short of 20+ MB/s. zstd `-1` is
  hundreds–thousands of MB/s; zstd `-19` is ~3–4 MB/s compress.

## License

MIT — see [LICENSE](LICENSE).
