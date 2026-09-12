# Rcn

`rcn` — **R**ust **C**ompressor, **N**ew — is a leveled command-line compressor
written in Rust. It combines a hash-chain LZ wire engine, context mixing,
Burrows–Wheeler Transform (BWT), and DP-LZP matching.

Optimization tickets, A/B history, and CI notes:
[DEVELOPMENT.md](DEVELOPMENT.md) · [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md).

## Goal

**Primary success metric:** each numbered level beats its peer class on
**both** compression ratio (lower `ratio%`) **and** compress throughput
(higher cmp MB/s) on the headline 5-file set (dickens, webster, nci, mr,
json). A file **both**-wins when ratio is a win or tie **and** cmp speed
is a win. Decode MB/s is reported but is not the v1 gate.

| Level | Engine | Must beat on both axes |
|------:|--------|------------------------|
| `-1` | Wire: hash-chain LZ + optional order-0 rANS | `lz4 -9`, `zstd -1` |
| `-3` | General: byte CM + DP-LZP, no BWT trials | `gzip -9` |
| `-9` | Archive (default): hybrid Fast Text + Slow Binary | `zstd -19` (stretch: xz, brotli) |
| `-19` | Max: Slow bit CM on every block | xz / brotli (ratio stretch) |

CLI aliases: `--level 1` / `-1`, `3` / `-3`, `9` / `-9`, `19` / `-19`.
`--mode hybrid|slow|fast|wire|general` remains. Default is **`-9` / hybrid**.

One setting cannot beat `lz4`/`zstd -1` on speed **and** `xz`/`brotli` on
ratio. Peers themselves do not do that. Levels own a peer class each.

Scorecards below show **wins and losses**. Current snapshot: `-9` both-wins
`zstd -19` on text/`nci` and splits on `mr` (ratio win, speed lose) and json
(ratio tie, speed lose). `-1` both-wins `lz4 -9` on json/`mr` and otherwise
splits or loses; it does not yet beat `zstd -1` on cmp speed. `-3` does not
yet both-win `gzip -9` on the full set.

## Design methodology

`rcn` is a staged pipeline. Cheap levels skip expensive stages; archive
levels keep them:

- **Classify first.** Input is split into variable-size blocks by data type so
  each block gets the transform it deserves (see [The method](#the-method)).
  Level `-1` uses large independent chunks and the wire LZ engine.
- **Transform locally.** Archive/max Text blocks run BWT trials to turn
  long-range word repeats into local runs. Level `-3` skips those trials.
- **Match.** Wire uses a hash-chain LZ77. Archive/max/general use DP-LZP
  before entropy coding.
- **Mix or entropy-code.** Slow bit CM blends many context models; Fast/General
  use byte CM + interleaved rANS; Wire optionally wraps LZ tokens in order-0
  rANS.

Stages are modular: swap an engine, re-run `scripts/bench_vs_sota.sh`, keep or
discard based on dual-axis scorecards.

## Projected use cases

- **Wire / `-1`** — fast path for less-redundant or latency-sensitive data.
- **General / `-3`** — gzip-class work without BWT trial cost.
- **Archive / `-9`** — text-centric archival (dickens, webster, source trees,
  JSON) where hybrid ratio vs `zstd -19` matters.
- **Max / `-19`** — Slow bit CM everywhere; binary/`mr` ratio stretch.
- **Learning reference** — readable pipeline of LZ, BWT, LZP, and mixers.

## The method

| Flag | Level | Engine |
|------|------:|--------|
| `--level 1` / `--mode wire` | `-1` | Hash-chain LZ77 (64 KiB window, lazy match) + optional order-0 rANS. Copy if the block does not shrink. |
| `--level 3` / `--mode general` | `-3` | Byte CM + DP-LZP; no BWT/XWRT trials; Exec still uses E8E9. |
| `--level 9` / `--mode hybrid` (default) | `-9` | Fast byte CM on Text/Random; Slow bit CM + DP-LZP on Binary/Exec. |
| `--level 19` / `--mode slow` | `-19` | Bit-level CM with a two-level 16k-bank mixer on every block. |
| `--mode fast` | (not numbered) | Byte CM **with** BWT trials on every block. |

Input is split into variable-size blocks by data type. Each block is classified
by a cheap order-0 Shannon estimate into `Text` / `Binary` / `Exec` / `Random`:

- `Random` blocks are stored verbatim — no prediction cost (except level `-1`,
  which still tries wire LZ and copies if it expands).
- `Text` blocks can be up to 4 MB (enabling BWT trials that turn long-range
  word repeats into local runs).
- `Binary` / `Exec` blocks use **1 MiB** chunks (with up to 4 MiB of prior
  same-kind bytes as DP-LZP match history across block boundaries).
- Level `-1` uses 4 MiB independent chunks (no global XWRT dictionary).

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

# Compress (default level -9 / hybrid)
rcn compress input.bin output.rcn

# Numbered levels (1/3/9/19 or -1/-3/-9/-19)
rcn compress --level 1 input.bin output.rcn
rcn compress --level 3 input.bin output.rcn
rcn compress --level 19 input.bin output.rcn

# Per-block method/size on stderr
rcn compress --verbose input.bin output.rcn

# Engine aliases
rcn compress --mode wire input.bin output.rcn
rcn compress --mode general input.bin output.rcn
rcn compress --mode fast input.bin output.rcn
rcn compress --mode slow input.bin output.rcn

# Decompress
rcn decompress output.rcn restored.bin

# Benchmark one engine
rcn bench path/to/corpus
rcn bench --level 1 path/to/corpus

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

### Streaming / `--stdout`

The `compress` and `decompress` subcommands accept `-` as input/output path for
stdin/stdout piping. Compress accepts `--verbose` for per-block method and size.

## Stability (RCN1)

Magic `RCN1`, header `VERSION = 1`. Layout is documented in
[`src/container.rs`](src/container.rs):

`[MAGIC 4][Header 7][optional global dict][BlockEntry × N][payloads…]`

RCN1 `VERSION = 1` is the stable container. Unknown methods fail closed.
New engines add method bytes; they do not change the header or block-entry
layout.

## Peers (crates.io and CLI equivalents)

Headline benches use system CLIs. These crates.io packages are the usual Rust
counterparts for the same algorithms:

| Peer | Typical CLI / level | crates.io | Role vs rcn |
|------|---------------------|-----------|-------------|
| Zstandard | `zstd -19` (level `-9` target), `zstd -1` (level `-1` target) | [`zstd`](https://crates.io/crates/zstd) | Dual-axis peer |
| Brotli | `brotli -11` | [`brotli`](https://crates.io/crates/brotli) | High-ratio stretch for `-9`/`-19` |
| LZMA/XZ | `xz -9` | [`xz2`](https://crates.io/crates/xz2) | Archival stretch |
| LZ4 | `lz4 -9` | [`lz4_flex`](https://crates.io/crates/lz4_flex) | Level `-1` target |
| DEFLATE/gzip | `gzip -9` | [`flate2`](https://crates.io/crates/flate2) | Level `-3` target |
| Snappy | — | [`snap`](https://crates.io/crates/snap) | Speed-oriented (not in CLI harness) |

`scripts/bench_vs_sota.sh` prints per-file and corpus W-L-T for **ratio**,
**cmp speed**, and **both**.

- Full comparison: `scripts/bench_vs_sota.sh <corpus_dir>`
- Peers-only: `SKIP_RCN=1 scripts/bench_vs_sota.sh <corpus_dir>`

## Benchmarks

Host for the numbers below: release `rcn` 0.2.1, macOS, Apple Silicon (ARM64,
NEON `walk_dist`), 2026-09-12, `.work/bench5/`
(dickens/webster/nci/mr/json). Level `-9`/`fast`/`-19` ratios and speeds
match the 2026-09-11 W1–W7 refresh unless noted. Level `-1`/`-3` and peer
cmp MB/s from 2026-09-12. Peer ratios from the same-day 2026-09-11
`SKIP_RCN=1` pass.

- Default `-9`: `rcn bench .work/bench5`
- Wire `-1`: `rcn bench --level 1 .work/bench5`
- General `-3`: `rcn bench --level 3 .work/bench5`
- Dual scorecard: `scripts/bench_vs_sota.sh .work/bench5`

`ratio%` is compressed size as a percentage of the original (lower is
better). Speed is MB/s of original bytes (higher is better). Win = strictly
better; tie = equal ratio, or both `ratio%` `< 0.5` for near-zero json.
A/B history: [DEVELOPMENT.md](DEVELOPMENT.md).

### Level `-1` (`--mode wire`)

Hash-chain LZ + optional order-0 rANS.

| file | orig (KB) | rcn-1 ratio% | cmp MB/s | dec MB/s |
|------|----------:|-------------:|---------:|---------:|
| dickens | 9953.6 | 45.4 | 81.40 | 58.26 |
| webster | 40487.0 | 35.1 | 181.40 | 74.51 |
| nci | 32767.0 | 11.9 | 123.24 | 203.00 |
| mr | 9736.9 | 39.6 | 82.44 | 60.49 |
| json | 478.5 | 0.1 | 277.55 | 1185.48 |

### Level `-3` (`--mode general`)

Byte CM + DP-LZP, no BWT trials.

| file | orig (KB) | rcn-3 ratio% | cmp MB/s | dec MB/s |
|------|----------:|-------------:|---------:|---------:|
| dickens | 9953.6 | 39.5 | 12.18 | 3.92 |
| webster | 40487.0 | 33.9 | 37.09 | 5.55 |
| nci | 32767.0 | 14.2 | 53.68 | 17.44 |
| mr | 9736.9 | 31.4 | 4.36 | 5.22 |
| json | 478.5 | 0.5 | 2.77 | 168.37 |

### Level `-9` (`--mode hybrid`, default)

Text/Random → Fast byte CM (global XWRT + DP-LZP literal-skip); Binary/Exec →
Slow bit CM + DP-LZP.

| file | orig (KB) | rcn-9 ratio% | cmp MB/s | dec MB/s | vs zstd -19 ratio |
|------|----------:|-------------:|---------:|---------:|:-----------------:|
| dickens | 9953.6 | 26.5 | 3.23 | 5.00 | **win** (28.0) |
| webster | 40487.0 | 20.1 | 6.62 | 7.00 | **win** (20.9) |
| nci | 32767.0 | 5.0 | 8.29 | 12.25 | **win** (5.0 → 4.99) |
| mr | 9736.9 | 27.3 | 0.03 | 0.03 | **win** (31.2) |
| json | 478.5 | 0.1 | 14.33 | 113.70 | tie |

### Fast path (`--mode fast`, not a numbered level)

| file | orig (KB) | rcn ratio% | cmp MB/s | dec MB/s | zstd -19 ratio% |
|------|----------:|-----------:|---------:|---------:|---------------:|
| dickens | 9953.6 | 26.5 | 3.26 | 5.10 | 28.0 |
| webster | 40487.0 | 20.1 | 6.67 | 6.75 | 20.9 |
| nci | 32767.0 | 5.0 | 8.29 | 11.95 | 5.0 |
| mr | 9736.9 | 31.5 | 4.31 | 5.23 | 31.2 |
| json | 478.5 | 0.1 | 14.14 | 116.74 | 0.0 |

Fast wins dickens/webster/`nci` vs `-19`, ties json, loses `mr` (31.5% vs
31.2%). Hybrid covers `mr` via the Slow path.

### Level `-19` (`--mode slow`)

Stack: two-level 16k-bank mixer, classifier-aware method bytes, word model,
cross-block decay, 32MB LZP window, BWT text trial, JSON/CSV/XML stream
splitting, DP-optimal LZP parse, Exec E8E9, global XWRT-1024 ESC, SSE/APM/APM2,
adaptive DP thresholds, side-stream FSE-family, 1 MiB Binary/Exec blocks +
match hist, kind-specific DP costs, classify-ahead, MixerAcc stretch/prefetch,
wider BWT trials, mimalloc, enum `StackModel` (V3), default `bwt_libsais`.

Text/`nci` slow ratios are still far from `-19` (dickens 40.2%, webster 29.3%,
nci 7.3%, json 0.1%). Hybrid/Slow `mr` is **27.3%** (wins `-19` at 31.2%).

### Peer CLI ratio%

| file | rcn-1 | rcn-3 | rcn-9 | rcn-19 | rcn-fast | zstd -19 | xz -9 | brotli -11 | gzip -9 | lz4 -9 | zstd -1 |
|------|------:|------:|------:|-------:|---------:|---------:|------:|-----------:|--------:|-------:|--------:|
| dickens | 45.4 | 39.5 | 26.5 | 40.2 | 26.5 | 28.0 | 27.8 | 27.7 | 37.8 | 43.6 | 41.8 |
| webster | 35.1 | 33.9 | 20.1 | 29.3 | 20.1 | 20.9 | 20.2 | 20.3 | 29.1 | 33.8 | 33.0 |
| nci | 11.9 | 14.2 | 5.0 | 7.3 | 5.0 | 5.0 | 5.2 | 4.5 | 8.9 | 11.0 | 8.5 |
| mr | 39.6 | 31.4 | 27.3 | 27.3 | 31.5 | 31.2 | 27.6 | 28.3 | 36.7 | 42.6 | 38.3 |
| json | 0.1 | 0.5 | 0.1 | 0.1 | 0.1 | 0.0 | 0.1 | 0.0 | 0.4 | 0.4 | 0.0 |

### Peer CLI compress MB/s

From the 2026-09-11 `SKIP_RCN=1` refresh (same host family as the rcn
tables). Used for speed scorecards.

| file | zstd -19 | xz -9 | brotli -11 | gzip -9 | lz4 -9 | zstd -1 |
|------|---------:|------:|-----------:|--------:|-------:|--------:|
| dickens | 2.5 | 2.2 | 0.8 | 21.1 | 53.1 | 208.0 |
| webster | 3.2 | 2.1 | 0.8 | 31.0 | 241.8 | 693.8 |
| nci | 3.5 | 6.1 | 0.9 | 22.9 | 278.4 | 876.3 |
| mr | 4.3 | 3.7 | 0.7 | 11.5 | 46.6 | 304.2 |
| json | 19.6 | 18.3 | 16.1 | 20.8 | 21.3 | 20.6 |

### Dual scorecards

Verdict cells are `ratio / cmp-speed / both`. `both=win` only if ratio is
win or tie **and** cmp speed is win. json near-zero ratios are ties.

**rcn `-1` vs speed peers:**

| file | vs lz4-9 | vs zstd-1 |
|------|:--------:|:---------:|
| dickens | lose / win / split | lose / lose / lose |
| webster | lose / lose / lose | lose / lose / lose |
| nci | lose / lose / lose | lose / lose / lose |
| mr | win / win / **win** | lose / lose / lose |
| json | tie / win / **win** | tie / win / **win** |

**rcn `-3` vs gzip -9:**

| file | vs gzip-9 |
|------|:---------:|
| dickens | lose / lose / lose |
| webster | lose / win / split |
| nci | lose / win / split |
| mr | win / lose / split |
| json | lose / lose / lose |

**rcn `-9` vs `zstd -19` (default gate):**

| file | vs zstd-19 |
|------|:----------:|
| dickens | win / win / **win** |
| webster | win / win / **win** |
| nci | win / win / **win** |
| mr | win / lose / split |
| json | tie / lose / split |

**rcn `-9` vs other peers (ratio / cmp / both):**

| file | vs xz-9 | vs brotli-11 | vs gzip-9 | vs lz4-9 | vs zstd-1 |
|------|:-------:|:------------:|:---------:|:--------:|:---------:|
| dickens | win / win / **win** | win / win / **win** | win / lose / split | win / lose / split | win / lose / split |
| webster | lose / win / split | tie / win / **win** | win / lose / split | win / lose / split | win / lose / split |
| nci | win / win / **win** | lose / win / split | win / lose / split | win / lose / split | win / lose / split |
| mr | win / lose / split | win / lose / split | win / lose / split | win / lose / split | win / lose / split |
| json | tie / lose / split | tie / lose / split | tie / lose / split | tie / lose / split | tie / lose / split |

Notes:

- **`nci`:** brotli -11 (4.5%) still leads the peer set; rcn `-9`/`fast` at
  **5.0%** ties `zstd -19` on the printed tenth (strict 4.99%).
- **`mr`:** rcn `-9`/`-19` (**27.3%**) leads this peer set on ratio (ahead of
  xz -9 at 27.6%) but Slow encode is ~0.03 MB/s, so the dual-axis gate vs
  `zstd -19` is a **split**. Closing that split is the main `-9` follow-up
  (faster Binary path that stays ≤31.2%).
- **`-1` vs `zstd -1`:** wire is well behind on cmp MB/s (tens–low hundreds
  vs hundreds–high hundreds). Parallel LZ + lighter tokens are the next lever.
- crates.io counterparts: [`zstd`](https://crates.io/crates/zstd),
  [`xz2`](https://crates.io/crates/xz2), [`brotli`](https://crates.io/crates/brotli),
  [`flate2`](https://crates.io/crates/flate2),
  [`lz4_flex`](https://crates.io/crates/lz4_flex).

### Reading the tables

1. **Level `-9` vs `zstd -19`:** both-win on dickens/webster/`nci`; split on
   `mr` (ratio) and json (speed). Still the CLI default.
2. **Level `-1` vs `lz4 -9` / `zstd -1`:** both-win only on json/`mr` vs lz4
   (and json vs `zstd -1`). Open campaign.
3. **Level `-3` vs `gzip -9`:** no both-win on the headline set yet.
4. **`--mode fast`:** BWT byte path; better text ratio than `-3`, not a
   numbered level.

## License

MIT — see [LICENSE](LICENSE).
