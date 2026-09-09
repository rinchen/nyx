# Rcn

`rcn` — **R**ust **C**ompressor, **N**ew — is a ratio-first command-line
compressor written in Rust. It combines context mixing, Burrows–Wheeler
Transform (BWT), and DP-LZP matching. The `.rcn` (RCN1) container and CLI may
still change.

Optimization tickets, A/B history, and CI notes:
[DEVELOPMENT.md](DEVELOPMENT.md) · [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md).

## Goal

**Primary success metric:** beat [`zstd`](https://crates.io/crates/zstd) **`-19`
on ratio** (lower `ratio%` wins) on text and mixed corpora — the headline
5-file set (dickens, webster, nci, mr, json).

**Not the goal:** matching `zstd -1` (or gzip/LZ4) on throughput. Those remain
*fast baselines* for context. Losing the speed column to `zstd -1` is expected
and does not mean the project failed.

| file | slow vs -19 | fast vs -19 | slow vs -1 | fast vs -1 |
|------|:-----------:|:-----------:|:----------:|:----------:|
| dickens | lose | win | win | win |
| webster | lose | win | win | win |
| nci | lose | lose | win | win |
| mr | win | lose | win | win |
| json | tie | tie | win | win |

Fast clears dickens/webster vs `-19` but not `nci` (5.1% vs 5.0%) or `mr`.
Slow holds `mr` and ties `json`. Closing the goal still needs better text/`nci`
ratio (or a default that clears every file). Details: [Benchmarks](#benchmarks).

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
measured ratio against the `zstd -19` goal.

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
json); **slow beats fast on `mr`**. Against the goal (`zstd -19`), slow wins
`mr` and ties `json` while losing text/`nci`; fast wins dickens/webster (ties
json) but loses `nci`/`mr`. Default stays `slow` until a mode clears every
headline file vs `-19`. See [Benchmarks](#benchmarks).

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

- `--mode slow` (default): bit-level CM. Beats fast on `mr`; loses text/`nci` to
  `zstd -19`. Much slower than fast.
- `--mode fast`: byte-level CM. Beats `zstd -19` on dickens/webster; loses on
  `nci`/`mr`. ~50–100× throughput vs slow.

Default stays `slow` until some mode beats or ties `zstd -19` on the full
headline 5-file set. Named `-1`…`-9` presets remain out of scope.

### Streaming / `--stdout`

The `compress` and `decompress` subcommands accept `-` as input/output path for
stdin/stdout piping. Compress accepts `--verbose` for per-block method and size.

## Peers (crates.io and CLI equivalents)

Headline benches use system CLIs (and an order-0 FSE column). These crates.io
packages are the usual Rust counterparts for the same algorithms:

| Peer | Typical CLI / level | crates.io | Role vs rcn |
|------|---------------------|-----------|-------------|
| Zstandard | `zstd -19` (goal), `zstd -1` (fast ref) | [`zstd`](https://crates.io/crates/zstd) | Primary goal / speed baseline |
| Brotli | `brotli -11` | [`brotli`](https://crates.io/crates/brotli) | High-ratio text peer |
| LZMA/XZ | `xz -9` | [`xz2`](https://crates.io/crates/xz2) | Archival max-ratio peer |
| LZ4 | `lz4 -9` | [`lz4_flex`](https://crates.io/crates/lz4_flex) | Speed-oriented; expect rcn to win ratio |
| DEFLATE/gzip | `gzip -9` | [`flate2`](https://crates.io/crates/flate2) | Ubiquitous baseline |
| Snappy | — | [`snap`](https://crates.io/crates/snap) | Speed-oriented |
| FSE / order-0 | FiniteStateEntropy-style | (see FSE columns below) | Entropy-only floor; rcn should win |

Re-run peers locally: `SKIP_RCN=1 scripts/bench_vs_sota.sh <corpus_dir>`
(also includes `zstd -1` / `-19`, xz, brotli, lz4, gzip when installed).

## Benchmarks

Progress toward the goal is **“does rcn beat `zstd -19` on ratio?”** Other
columns (`zstd -1`, FSE, peer CLIs) are supporting context. Tables use a
representative 5-file subset. `ratio%` is compressed size as a percentage of
the original (lower is better); speed is in MB/s (higher is better).

The **ratio winner** / **speed winner** columns name the best codec among the
columns in that row — informational only. They are *not* the project success
metric. A/B history: [DEVELOPMENT.md](DEVELOPMENT.md).

### Slow path (`--mode slow`, default)

Stack: hybrid_ppm3, two-level 16k-bank mixer, classifier-aware method bytes,
word model, cross-block decay, 32MB LZP window, BWT text trial,
JSON/CSV/XML stream splitting, DP-optimal LZP parse, Exec E8E9, global
XWRT-512 ESC, SSE/APM/APM2, adaptive DP thresholds, side-stream FSE-family,
classify-ahead, MixerAcc stretch/prefetch, wider BWT trials, mimalloc.

| file | orig (KB) | rcn ratio% | rcn cmp MB/s | rcn dec MB/s | zstd -1 ratio% | zstd -1 cmp MB/s | zstd -1 dec MB/s | zstd -19 ratio% | zstd -19 cmp MB/s | zstd -19 dec MB/s | FSE ratio% | FSE cmp MB/s | FSE dec MB/s | ratio winner | speed winner |
|------|----------:|-----------:|-------------:|-------------:|---------------:|-----------------:|-----------------:|---------------:|-----------------:|-----------------:|-----------:|-------------:|-------------:|:------------:|:------------:|
| dickens | 9953.6 | 40.2 | 0.04 | 0.04 | 41.8 | 228.4 | 158.4 | 28.0 | 2.4 | 267.6 | 57.0 | 375.6 | 463.7 | zstd -19 | zstd -19 |
| webster | 40487.0 | 29.3 | 0.1 | 0.1 | 33.0 | 692.9 | 691.5 | 20.9 | 3.0 | 665.6 | 62.6 | 424.6 | 507.9 | zstd -19 | zstd -19 |
| nci | 32767.0 | 7.3 | 0.2 | 0.2 | 8.5 | 893.4 | 792.1 | 5.0 | 3.5 | 828.2 | 30.2 | 326.7 | 335.9 | zstd -19 | zstd -19 |
| mr | 9736.9 | 27.4 | 0.04 | 0.04 | 38.3 | 291.9 | 256.0 | 31.2 | 3.8 | 269.4 | 44.0 | 233.2 | 229.3 | rcn | zstd -19 |
| json | 478.5 | 0.1 | 6.3 | 30.7 | 0.0 | 18.6 | 18.3 | 0.0 | 16.8 | 18.2 | 52.8 | 1649.5 | 1251.9 | rcn/tie | zstd -1 |

Notes: rcn slow hygiene 2026-09-09. zstd -1/-19 ratio and speed refreshed same day via
`scripts/bench_vs_sota.sh` (prior nci zstd columns were stale). FSE columns unchanged.
**Goal column:** vs `zstd -19`, slow rcn wins `mr` and ties `json`; loses dickens,
webster, and `nci` (7.3% vs 5.0%).

### Fast path (`--mode fast`)

Re-bench 2026-09-09. zstd/FSE ratio columns reused from the slow-path reference
set. Speed winner uses decode MB/s (almost always `zstd -1` — not the goal).

| file | orig (KB) | rcn ratio% | cmp MB/s | dec MB/s | zstd -1 ratio% | zstd -19 ratio% | FSE ratio% | ratio winner | speed winner |
|------|----------:|-----------:|---------:|---------:|---------------:|---------------:|-----------:|:------------:|:------------:|
| dickens | 9953.6 | 26.9 | 2.0 | 4.3 | 41.8 | 28.0 | 57.0 | rcn | zstd -1 |
| webster | 40487.0 | 20.3 | 2.7 | 5.5 | 33.0 | 20.9 | 62.6 | rcn | zstd -1 |
| nci | 32767.0 | 5.1 | 5.1 | 10.6 | 8.5 | 5.0 | 30.2 | zstd -19 | zstd -1 |
| mr | 9736.9 | 35.0 | 2.6 | 4.0 | 38.3 | 31.2 | 44.0 | zstd -19 | zstd -1 |
| json | 478.5 | 0.1 | 5.2 | 59.1 | 0.0 | 0.0 | 52.8 | rcn/tie | zstd -1 |

### Other high-ratio peers (CLI ratio%)

Measured 2026-09-09 with `SKIP_RCN=1 scripts/bench_vs_sota.sh` (system CLIs;
`brotli -q 11`). rcn columns reuse the headline benches above.

| file | rcn slow | rcn fast | zstd -19 | xz -9 | brotli -11 | gzip -9 | lz4 -9 | zstd -1 |
|------|---------:|---------:|---------:|------:|-----------:|--------:|-------:|--------:|
| dickens | 40.2 | 26.9 | 28.0 | 27.8 | 27.7 | 37.8 | 43.6 | 41.8 |
| webster | 29.3 | 20.3 | 20.9 | 20.2 | 20.3 | 29.1 | 33.8 | 33.0 |
| nci | 7.3 | 5.1 | 5.0 | 5.2 | 4.5 | 8.9 | 11.0 | 8.5 |
| mr | 27.4 | 35.0 | 31.2 | 27.6 | 28.3 | 36.7 | 42.6 | 38.3 |
| json | 0.1 | 0.1 | ~0.0 | 0.1 | ~0.0 | 0.4 | 0.4 | ~0.0 |

Takeaways vs these peers (ratio only):

- **rcn fast** beats `zstd -19` / xz / brotli on dickens and webster; on `nci`,
  brotli -11 (4.5%) and `zstd -19` (5.0%) still lead rcn fast (5.1%); on `mr`,
  rcn slow (27.4%) leads this set.
- **rcn** (both modes) beats `zstd -1`, gzip -9, and lz4 -9 on every headline
  file in this table.
- crates.io counterparts: [`zstd`](https://crates.io/crates/zstd),
  [`xz2`](https://crates.io/crates/xz2), [`brotli`](https://crates.io/crates/brotli),
  [`flate2`](https://crates.io/crates/flate2),
  [`lz4_flex`](https://crates.io/crates/lz4_flex).

### Reading the tables

- **vs `zstd -19` (goal):** slow wins `mr`, ties `json`, loses dickens (40.2% vs
  28.0%), webster (29.3% vs 20.9%), and `nci` (7.3% vs 5.0%). Fast wins
  dickens/webster, ties `json`, loses `nci` (5.1% vs 5.0%) and `mr` (35.0% vs
  31.2%) — those gaps keep fast from being the default.
- **vs `zstd -1` (fast baseline, not the goal):** both slow and fast beat `-1`
  on ratio for all five headline files. Speed still belongs to `-1`.
- **vs FSE:** rcn wins ratio on every headline row in the tables above.
- **vs xz / brotli / gzip / lz4:** see [Other high-ratio peers](#other-high-ratio-peers-cli-ratio);
  rcn fast leads dickens/webster; brotli -11 leads `nci`; rcn slow leads `mr`.
- **rcn fast vs slow:** mode choice only — fast better on text/`nci`; slow
  better on `mr`. Neither mode alone clears every file vs `-19` yet.
- **Throughput:** slow ~0.04 MB/s on large text; fast ~2–5 / ~4–10 MB/s. Losing
  speed to `zstd -1` is expected and outside the success criterion.

## License

MIT — see [LICENSE](LICENSE).
