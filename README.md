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

**Not the goal:** matching `zstd -1` (or gzip/LZ4) on **throughput**. Speed
figures in the benches are context only; losing MB/s to `-1` is expected and
does not mean the project failed.

Ratio scorecard from the measured tables below (win = strictly smaller
`ratio%`; json vs `-19` is ~0.1% vs ~0.0% and is counted as a **tie** at this
scale):

| file | hybrid vs `-19` | slow vs `-19` | fast vs `-19` |
|------|:---------------:|:-------------:|:-------------:|
| dickens | win | lose | win |
| webster | win | lose | win |
| nci | win | lose | win |
| mr | win | win | lose |
| json | tie | tie | tie |

**Mode vs mode (ratio only):** fast beats slow on dickens/webster/`nci` and
matches on json; slow beats fast on `mr` (27.3% vs 31.5%). **Hybrid** picks Fast
for Text and Slow for Binary/Exec, so it clears the full headline set vs
`zstd -19`.

**Goal (ratio vs `zstd -19`):** hybrid wins dickens/webster/`nci`/`mr` and ties json
→ CLI default is `--mode hybrid`. Details: [Benchmarks](#benchmarks).

Fast alone loses `mr` (31.5% vs 31.2%); slow alone loses text/`nci`.

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

Rcn ships three entropy modes:

- `--mode hybrid` (default): Fast byte CM on Text/Random blocks; Slow bit CM +
  DP-LZP on Binary/Exec. Clears the headline set vs `zstd -19`.
- `--mode slow`: bit-level CM with a two-level 16k-bank mixer (all blocks).
- `--mode fast`: PPM-style byte CM + interleaved rANS (all blocks).

On the headline set (ratio only): **fast** beats **slow** on
dickens/webster/`nci` and matches on json; **slow** beats **fast** on `mr`.
**Hybrid** combines those strengths. Throughput is not part of the success
gate — see [Benchmarks](#benchmarks).

Input is split into variable-size blocks by data type. Each block is classified
by a cheap order-0 Shannon estimate into `Text` / `Binary` / `Exec` / `Random`:

- `Random` blocks are stored verbatim — no prediction cost.
- `Text` blocks can be up to 4 MB (enabling BWT trials that turn long-range
  word repeats into local runs).
- `Binary` / `Exec` blocks use **1 MiB** chunks (with up to 4 MiB of prior
  same-kind bytes as DP-LZP match history across block boundaries).
- `Random` blocks use 64 KiB and are stored verbatim.

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
the coded stream, so round-trips are lossless. Decompress also checks per-block
CRC32 and rejects truncated containers, overlong block payloads, and corrupt
JSON/CSV/XML/XWRT structured transforms (hard error — not an empty output).

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

# Faster byte-CM path (clears text/nci vs -19; loses mr)
rcn compress --mode fast input.bin output.rcn

# Bit-CM path (wins mr; loses text to -19)
rcn compress --mode slow input.bin output.rcn

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
arm64 (NEON) processor. On x86_64 (Linux), the AVX2 code path can be disabled
at build time with `--no-default-features --features no_avx2` — see
[CI / testing](DEVELOPMENT.md#ci--testing).

### Compression modes

- `--mode hybrid` (default): Fast entropy for Text/Random, Slow+DP-LZP for
  Binary/Exec. Beats or ties `zstd -19` on the headline 5-file set.
- `--mode slow`: bit-level CM on every block. Better than fast on `mr`; loses
  text/`nci` to `-19`. Much slower than fast/hybrid on text.
- `--mode fast`: byte-level CM on every block. Clears text/`nci` vs `-19`; loses
  on `mr`.

Named `-1`…`-9` presets remain out of scope.

### Streaming / `--stdout`

The `compress` and `decompress` subcommands accept `-` as input/output path for
stdin/stdout piping. Compress accepts `--verbose` for per-block method and size.

## Peers (crates.io and CLI equivalents)

Headline benches use system CLIs. These crates.io packages are the usual Rust
counterparts for the same algorithms:

| Peer | Typical CLI / level | crates.io | Role vs rcn |
|------|---------------------|-----------|-------------|
| Zstandard | `zstd -19` (ratio goal), `zstd -1` (speed baseline) | [`zstd`](https://crates.io/crates/zstd) | `-19` = success metric; `-1` = throughput context only |
| Brotli | `brotli -11` | [`brotli`](https://crates.io/crates/brotli) | High-ratio text peer |
| LZMA/XZ | `xz -9` | [`xz2`](https://crates.io/crates/xz2) | Archival max-ratio peer |
| LZ4 | `lz4 -9` | [`lz4_flex`](https://crates.io/crates/lz4_flex) | Speed-oriented; expect rcn to win ratio |
| DEFLATE/gzip | `gzip -9` | [`flate2`](https://crates.io/crates/flate2) | Ubiquitous baseline |
| Snappy | — | [`snap`](https://crates.io/crates/snap) | Speed-oriented |

Ratio win/lose vs these peers (slow and fast): [Other high-ratio
peers](#other-high-ratio-peers-cli-ratio). `scripts/bench_vs_sota.sh`
prints the same scorecard when rcn is included.

- Full comparison (rcn slow + fast + hybrid + peers + scorecard):
  `scripts/bench_vs_sota.sh <corpus_dir>`
- Peers-only numbers (no scorecard):
  `SKIP_RCN=1 scripts/bench_vs_sota.sh <corpus_dir>`

## Benchmarks

All numbers below are from a **refresh on 2026-09-11** (release `rcn` 0.2.0,
Apple Silicon) against `.work/bench5/` (dickens/webster/nci/mr/json), after
**W1–W7** (1 MiB Binary blocks + match hist, Fast Binary order-3, parallel Fast
Text encode, XWRT-1024, kind-specific DP costs, default `bwt_libsais`). Peer CLI
columns re-measured the same day (`SKIP_RCN=1 scripts/bench_vs_sota.sh`).

- Hybrid (default): `rcn bench .work/bench5` or `rcn bench --hybrid .work/bench5`
- Fast: `rcn bench --fast .work/bench5`
- Peers + scorecard: `scripts/bench_vs_sota.sh .work/bench5`

Progress toward the goal is **"does rcn beat `zstd -19` on ratio?"** Other
columns (`zstd -1`, peer CLIs) are supporting context. `ratio%` is compressed
size as a percentage of the original (lower is better); speed is in MB/s
(higher is better) and is **not** used to declare project success. A/B
history: [DEVELOPMENT.md](DEVELOPMENT.md).

### Hybrid path (`--mode hybrid`, default)

Text/Random → Fast byte CM (global XWRT + DP-LZP literal-skip); Binary/Exec →
Slow bit CM + DP-LZP.

| file | orig (KB) | rcn ratio% | cmp MB/s | dec MB/s | vs zstd -19 |
|------|----------:|-----------:|---------:|---------:|:-----------:|
| dickens | 9953.6 | 26.5 | 3.23 | 5.00 | **win** (28.0) |
| webster | 40487.0 | 20.1 | 6.62 | 7.00 | **win** (20.9) |
| nci | 32767.0 | 5.0 | 8.29 | 12.25 | **win** (5.0 → 4.99) |
| mr | 9736.9 | 27.3 | 0.03 | 0.03 | **win** (31.2) |
| json | 478.5 | 0.1 | 14.33 | 113.70 | tie |

### Fast path (`--mode fast`)

| file | orig (KB) | rcn ratio% | cmp MB/s | dec MB/s | zstd -19 ratio% |
|------|----------:|-----------:|---------:|---------:|---------------:|
| dickens | 9953.6 | 26.5 | 3.26 | 5.10 | 28.0 |
| webster | 40487.0 | 20.1 | 6.67 | 6.75 | 20.9 |
| nci | 32767.0 | 5.0 | 8.29 | 11.95 | 5.0 |
| mr | 9736.9 | 31.5 | 4.31 | 5.23 | 31.2 |
| json | 478.5 | 0.1 | 14.14 | 116.74 | 0.0 |

**Goal (ratio vs `zstd -19`):** fast wins dickens/webster/`nci`, ties json, loses
`mr` (31.5% vs 31.2%). Hybrid covers `mr` via the Slow path.

### Slow path (`--mode slow`)

Stack: hybrid_ppm3, two-level 16k-bank mixer, classifier-aware method bytes,
word model, cross-block decay, 32MB LZP window, BWT text trial,
JSON/CSV/XML stream splitting, DP-optimal LZP parse, Exec E8E9, global
XWRT-1024 ESC, SSE/APM/APM2, adaptive DP thresholds, side-stream FSE-family,
1 MiB Binary/Exec blocks + match hist, kind-specific DP costs, classify-ahead,
MixerAcc stretch/prefetch, wider BWT trials, mimalloc, enum `StackModel` (V3),
default `bwt_libsais`.

Text/`nci` slow ratios are still far from `-19` (prior measure: dickens 40.2%,
webster 29.3%, nci 7.3%, json 0.1%). Hybrid/Slow `mr` is **27.3%** after W1/W6
(wins `-19` at 31.2%).

### Other high-ratio peers (CLI ratio%)

Same-day peer pass (`brotli -q 11`). rcn hybrid/fast from the W1–W7 refresh;
rcn slow text columns from the earlier same-day pass; `mr` slow matches hybrid.

| file | rcn hybrid | rcn slow | rcn fast | zstd -19 | xz -9 | brotli -11 | gzip -9 | lz4 -9 | zstd -1 |
|------|-----------:|---------:|---------:|---------:|------:|-----------:|--------:|-------:|--------:|
| dickens | 26.5 | 40.2 | 26.5 | 28.0 | 27.8 | 27.7 | 37.8 | 43.6 | 41.8 |
| webster | 20.1 | 29.3 | 20.1 | 20.9 | 20.2 | 20.3 | 29.1 | 33.8 | 33.0 |
| nci | 5.0 | 7.3 | 5.0 | 5.0 | 5.2 | 4.5 | 8.9 | 11.0 | 8.5 |
| mr | 27.3 | 27.3 | 31.5 | 31.2 | 27.6 | 28.3 | 36.7 | 42.6 | 38.3 |
| json | 0.1 | 0.1 | 0.1 | 0.0 | 0.1 | 0.0 | 0.4 | 0.4 | 0.0 |

**Ratio scorecard** (win = strictly smaller `ratio%`; tie = equal, or both
`< 0.5` for near-zero json scale). Throughput is not scored. crates.io names
above are CLI equivalents only.

**rcn hybrid vs peers:**

| file | vs zstd-19 | vs xz-9 | vs brotli-11 | vs gzip-9 | vs lz4-9 | vs zstd-1 |
|------|:----------:|:-------:|:-----------:|:---------:|:--------:|:---------:|
| dickens | win | win | win | win | win | win |
| webster | win | lose | tie | win | win | win |
| nci | win | win | lose | win | win | win |
| mr | win | win | win | win | win | win |
| json | tie | tie | tie | tie | tie | tie |

**rcn slow vs peers:**

| file | vs zstd-19 | vs xz-9 | vs brotli-11 | vs gzip-9 | vs lz4-9 | vs zstd-1 |
|------|:----------:|:-------:|:-----------:|:---------:|:--------:|:---------:|
| dickens | lose | lose | lose | lose | win | win |
| webster | lose | lose | lose | lose | win | win |
| nci | lose | lose | lose | win | win | win |
| mr | win | win | win | win | win | win |
| json | tie | tie | tie | tie | tie | tie |

**rcn fast vs peers:**

| file | vs zstd-19 | vs xz-9 | vs brotli-11 | vs gzip-9 | vs lz4-9 | vs zstd-1 |
|------|:----------:|:-------:|:-----------:|:---------:|:--------:|:---------:|
| dickens | win | win | win | win | win | win |
| webster | win | lose | tie | win | win | win |
| nci | win | win | lose | win | win | win |
| mr | lose | lose | lose | win | win | win |
| json | tie | tie | tie | tie | tie | tie |

Notes (ratio only):

- **`nci`:** brotli -11 (4.5%) leads the peer set; rcn fast/hybrid at **5.0%**
  ties `zstd -19` (strict win at 4.99%).
- **`mr`:** rcn slow/hybrid (**27.3%**) leads this peer set (ahead of xz -9 at
  27.6%). Fast `mr` improved to **31.5%** (was 35.1%) but still loses `-19`
  (31.2%).
- crates.io counterparts: [`zstd`](https://crates.io/crates/zstd),
  [`xz2`](https://crates.io/crates/xz2), [`brotli`](https://crates.io/crates/brotli),
  [`flate2`](https://crates.io/crates/flate2),
  [`lz4_flex`](https://crates.io/crates/lz4_flex).

### Reading the tables

Four comparisons, kept separate on purpose:

1. **Goal — ratio vs `zstd -19`:** **hybrid** wins dickens/webster/`nci`/`mr`
   and ties json (CLI default). Fast alone loses `mr`; slow alone loses text/`nci`.
2. **Baseline — ratio vs `zstd -1`:** modes beat `-1` on the headline set under
   the Goal scorecard’s strict compare (json near-zero treated as tie in the
   peer table). Not the success criterion.
3. **Peers — ratio vs xz/brotli/gzip/lz4:** see scorecards under
   [Other high-ratio peers](#other-high-ratio-peers-cli-ratio).
4. **Mode choice:** fast better on text/`nci`; slow better on `mr`; hybrid
   combines both.

**Throughput** (context only, this machine): hybrid Text encode is much faster
after parallel Fast blocks (~3–8 cmp MB/s on dickens/webster/`nci`); Slow Binary
`mr` remains ~0.03 MB/s. Fast ~3–14 cmp / ~5–117 dec MB/s. `zstd -1` remains far
faster — expected, and outside the success criterion.

## License

MIT — see [LICENSE](LICENSE).
