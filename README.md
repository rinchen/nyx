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

| file | slow ratio vs `-19` | fast ratio vs `-19` | slow ratio vs `-1` | fast ratio vs `-1` |
|------|:-------------------:|:-------------------:|:------------------:|:------------------:|
| dickens | lose | win | win | win |
| webster | lose | win | win | win |
| nci | lose | lose | win | win |
| mr | win | lose | win | win |
| json | tie | tie | win | win |

**Mode vs mode (ratio only):** fast beats slow on dickens/webster/`nci` and
matches on json; slow beats fast on `mr`.

**vs goal (`zstd -19`):** fast clears dickens/webster but not `nci` (5.1% vs
5.0%) or `mr` (35.0% vs 31.2%). Slow clears `mr` and ties json; loses
dickens/webster/`nci`. Neither mode alone clears every headline file, so the
CLI default stays `--mode slow`. Details: [Benchmarks](#benchmarks).

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

On the headline set (ratio only): **fast** beats **slow** on
dickens/webster/`nci` and matches on json; **slow** beats **fast** on `mr`.
Against the goal (`zstd -19`): slow wins `mr` and ties json, loses
dickens/webster/`nci`; fast wins dickens/webster and ties json, loses
`nci`/`mr`. Default stays `slow` until a mode clears every headline file vs
`-19`. Throughput is not part of that gate — see [Benchmarks](#benchmarks).

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

# Faster byte-CM path (better text/nci ratio; loses to zstd -19 on nci/mr)
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

- `--mode slow` (default): bit-level CM. Better ratio than fast on `mr`; vs
  `zstd -19` wins `mr`, ties json, loses dickens/webster/`nci`. Much slower
  than fast.
- `--mode fast`: byte-level CM. Better ratio than slow on dickens/webster/`nci`;
  vs `zstd -19` wins dickens/webster, ties json, loses `nci`/`mr`. ~50–100×
  throughput vs slow (still far behind `zstd -1` speed).

Default stays `slow` until some mode beats or ties `zstd -19` on the full
headline 5-file set. Named `-1`…`-9` presets remain out of scope.

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

- Full comparison (rcn slow + fast + peers + scorecard):
  `scripts/bench_vs_sota.sh <corpus_dir>`
- Peers-only numbers (no scorecard):
  `SKIP_RCN=1 scripts/bench_vs_sota.sh <corpus_dir>`

## Benchmarks

All numbers below are from a **single refresh on 2026-09-11** (release
`rcn` 0.2.0, Apple Silicon) against `.work/bench5/`
(dickens/webster/nci/mr/json):

- Slow: `rcn bench .work/bench5`
- Fast: `rcn bench --fast .work/bench5`
- Peers + scorecard: `scripts/bench_vs_sota.sh .work/bench5`
- Peers-only numbers: `SKIP_RCN=1 scripts/bench_vs_sota.sh .work/bench5`

Progress toward the goal is **"does rcn beat `zstd -19` on ratio?"** Other
columns (`zstd -1`, peer CLIs) are supporting context. `ratio%` is compressed
size as a percentage of the original (lower is better); speed is in MB/s
(higher is better) and is **not** used to declare project success. A/B
history: [DEVELOPMENT.md](DEVELOPMENT.md).

### Slow path (`--mode slow`, default)

Stack: hybrid_ppm3, two-level 16k-bank mixer, classifier-aware method bytes,
word model, cross-block decay, 32MB LZP window, BWT text trial,
JSON/CSV/XML stream splitting, DP-optimal LZP parse, Exec E8E9, global
XWRT-512 ESC, SSE/APM/APM2, adaptive DP thresholds, side-stream FSE-family,
classify-ahead, MixerAcc stretch/prefetch, wider BWT trials, mimalloc.

| file | orig (KB) | rcn ratio% | rcn cmp MB/s | rcn dec MB/s | zstd -1 ratio% | zstd -1 cmp MB/s | zstd -1 dec MB/s | zstd -19 ratio% | zstd -19 cmp MB/s | zstd -19 dec MB/s |
|------|----------:|-----------:|-------------:|-------------:|---------------:|-----------------:|-----------------:|---------------:|-----------------:|-----------------:|
| dickens | 9953.6 | 40.2 | 0.03 | 0.04 | 41.8 | 198.2 | 272.7 | 28.0 | 1.5 | 215.3 |
| webster | 40487.0 | 29.3 | 0.05 | 0.05 | 33.0 | 640.7 | 666.1 | 20.9 | 2.4 | 370.9 |
| nci | 32767.0 | 7.3 | 0.18 | 0.19 | 8.5 | 696.9 | 689.9 | 5.0 | 2.9 | 712.9 |
| mr | 9736.9 | 27.4 | 0.03 | 0.03 | 38.3 | 240.6 | 235.9 | 31.2 | 2.6 | 207.1 |
| json | 478.5 | 0.1 | 6.24 | 30.32 | 0.0 | 14.7 | 14.8 | 0.0 | 13.9 | 13.4 |

**Goal (ratio vs `zstd -19`):** slow wins `mr` (27.4% vs 31.2%), ties json
(~0.1% vs ~0.0%), loses dickens (40.2% vs 28.0%), webster (29.3% vs 20.9%), and
`nci` (7.3% vs 5.0%).

### Fast path (`--mode fast`)

Same corpus and peer ratio columns as above. Speed columns are informational.

| file | orig (KB) | rcn ratio% | cmp MB/s | dec MB/s | zstd -1 ratio% | zstd -19 ratio% |
|------|----------:|-----------:|---------:|---------:|---------------:|---------------:|
| dickens | 9953.6 | 26.9 | 1.55 | 3.11 | 41.8 | 28.0 |
| webster | 40487.0 | 20.3 | 1.92 | 4.09 | 33.0 | 20.9 |
| nci | 32767.0 | 5.1 | 4.25 | 6.74 | 8.5 | 5.0 |
| mr | 9736.9 | 35.0 | 2.34 | 3.67 | 38.3 | 31.2 |
| json | 478.5 | 0.1 | 6.04 | 53.54 | 0.0 | 0.0 |

**Goal (ratio vs `zstd -19`):** fast wins dickens (26.9% vs 28.0%) and webster
(20.3% vs 20.9%), ties json, loses `nci` (5.1% vs 5.0%) and `mr` (35.0% vs
31.2%). Those two losses keep fast from being the default.

### Other high-ratio peers (CLI ratio%)

Same 2026-09-11 peer pass (`brotli -q 11`). rcn columns from the slow/fast
benches above.

| file | rcn slow | rcn fast | zstd -19 | xz -9 | brotli -11 | gzip -9 | lz4 -9 | zstd -1 |
|------|---------:|---------:|---------:|------:|-----------:|--------:|-------:|--------:|
| dickens | 40.2 | 26.9 | 28.0 | 27.8 | 27.7 | 37.8 | 43.6 | 41.8 |
| webster | 29.3 | 20.3 | 20.9 | 20.2 | 20.3 | 29.1 | 33.8 | 33.0 |
| nci | 7.3 | 5.1 | 5.0 | 5.2 | 4.5 | 8.9 | 11.0 | 8.5 |
| mr | 27.4 | 35.0 | 31.2 | 27.6 | 28.3 | 36.7 | 42.6 | 38.3 |
| json | 0.1 | 0.1 | 0.0 | 0.1 | 0.0 | 0.4 | 0.4 | 0.0 |

**Ratio scorecard** (win = strictly smaller `ratio%`; tie = equal, or both
`< 0.5` for near-zero json scale). Throughput is not scored. crates.io names
above are CLI equivalents only.

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
| nci | lose | win | lose | win | win | win |
| mr | lose | lose | lose | win | win | win |
| json | tie | tie | tie | tie | tie | tie |

Notes (ratio only):

- **`nci`:** brotli -11 (4.5%) leads the peer set; then `zstd -19` (5.0%) ahead
  of rcn fast (5.1%).
- **`mr`:** rcn slow (27.4%) leads this peer set (ahead of xz -9 at 27.6%).
- crates.io counterparts: [`zstd`](https://crates.io/crates/zstd),
  [`xz2`](https://crates.io/crates/xz2), [`brotli`](https://crates.io/crates/brotli),
  [`flate2`](https://crates.io/crates/flate2),
  [`lz4_flex`](https://crates.io/crates/lz4_flex).

### Reading the tables

Four comparisons, kept separate on purpose:

1. **Goal — ratio vs `zstd -19`:** slow wins `mr`, ties json, loses
   dickens/webster/`nci`. Fast wins dickens/webster, ties json, loses `nci`/`mr`.
2. **Baseline — ratio vs `zstd -1`:** both modes beat `-1` on every headline
   file under the Goal scorecard’s strict compare (json `0.1` vs `0.0` counts
   as win there). The peer scorecard’s near-zero rule (`both < 0.5` → tie)
   treats json as a tie vs `-1`. Not the success criterion.
3. **Peers — ratio vs xz/brotli/gzip/lz4:** see scorecards under
   [Other high-ratio peers](#other-high-ratio-peers-cli-ratio).
4. **Mode choice — rcn fast vs slow:** fast better ratio on
   dickens/webster/`nci`; slow better on `mr`. Neither clears every file vs
   `-19` yet.

**Throughput** (context only, this machine): slow ~0.03–0.2 MB/s on large
files (json higher); fast ~1.5–6 cmp / ~3–54 dec MB/s. `zstd -1` remains far
faster — expected, and outside the success criterion.

## License

MIT — see [LICENSE](LICENSE).
