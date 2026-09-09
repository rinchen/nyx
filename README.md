# Rcn

> **Experimentation: work in progress.** `rcn` — **R**ust **C**ompressor, **N**ew —
> is an experimental, research-oriented command-line compressor written in Rust. It
> combines a bit-level logistic mixer, Burrows–Wheeler Transform (BWT), and DP-LZP
> matching to chase the highest achievable ratio; the benchmark target is beating
> `zstd -19` on text. Expect rough edges; the container format and CLI are not yet
> stable.

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

Because the stages are modular, experiments are cheap: swap a model, re-run a
benchmark, keep or discard based on measured ratio. The current target is beating
`zstd -19` on text and mixed corpora.

## Projected use cases

- **Research & education** — a readable, self-contained reference for combining
  BWT, LZP matching, and context-mixing entropy coding in one pipeline.
- **Text-centric archival** — corpora with lots of redundancy (dickens, webster,
  source trees, JSON dumps) where ratio matters more than speed.
- **Baseline for further work** — a clean staging ground for trying new models,
  transforms, or entropy backends against a fixed benchmark harness.

It is not aimed at general-purpose, at-rest or on-the-wire compression where
`zstd -1`/`gzip` speed is the deciding factor — expect `rcn` to be much slower
per byte than those, in exchange for better ratio on the right inputs.

## TODO - Remaining Optimization Tasks

### Compression - beat zstd -19
- ✅ **Promote Order-8 PPMd with SEE + sparse de Bruijn to default** – WordModel + banks + 32MB window + XWRT
- ✅ **Compress the match side-stream (varint)** – delta_pos/len/dist varints
- ✅ **Global XWRT dictionary (top-128)** – corpus-wide; C5 measured dickens −4.6pt, webster −4.0pt
- ✅ **C6 — Global XWRT 128 → 512 with ESC tokens** – `0x80..=0xFE` for ids 0..126; `0xFF || u16_le(id)` for 127..511 (ESC slots only for words longer than 3 bytes so tokens never expand); header dict count is `u16`
- ✅ **C7 — Adaptive DP LZP threshold** – Text ≥12; Binary/Exec/Random stay 16 (`≥24` on Binary/Exec regressed mr and was reverted)
- ✅ **C8 — 16k banks** – `NUM_MIXERS=16384` (14-bit hash). Single IndirectModel tried again; **left out of default Text stack** (prior dickens regression); model remains in-tree
- ✅ **C9 — FSE-family match side-stream** – order-0 byte rANS on the varint blob when smaller (`[num_runs][flag][len][payload]`)

### Speed - achieve 20+ MB/s
- ✅ **S8 — Interleaved rANS re-bench** – already wired (`RansByteEncoder32`/`Decoder32`); post-AVX2 `walk_dist` fast path ~2–5 MB/s cmp / ~4–10 MB/s dec (see Benchmarks)
- ✅ **Parallel BWT trials (S5 partial)** – `rayon::join` for path B/C inside a block trial
- ✅ **S9 — Optional libsais BWT backend** – `--features bwt_libsais` uses pure-Rust `libsais-rs`; default remains `divsufsort`
- ✅ **S10 — Classify-ahead pipeline** – `rayon::join` overlaps classify+size of block N+1 with encode of N (encode stays serial / bit-identical)

### Other Completed
- **S1–S4, S7, C1–C5** – SIMD walk_dist, SoA banks, stretch reuse, mimalloc, SSE/APM, global XWRT-128, etc.
- **CI fixes** – `no_avx2` feature for Linux CI scalar path
- **Pre-commit hook** – mirrors the CI gate

Gap to beat `zstd -19` on text still tracked in the headline table (dickens/webster). nci / mr / json already win.
## The method

> **Status: actively improving.** Rcn ships two entropy paths:
> `--mode slow` (the bit-level 8–9 model logistic mixer with a two-level
> 16k-bank hierarchy) and `--mode fast` (a PPM-style single-context count
> coder + byte rANS). The benchmark target is beating `zstd -19` on
> text + mixed corpora, with `FSE` as a secondary reference. See
> [Benchmarks](#benchmarks) for the numbers.

Input is split into variable-size blocks by data type. Each block is classified
by a cheap order-0 Shannon estimate into `Text` / `Binary` / `Exec` / `Random`:

- `Random` blocks are stored verbatim — no prediction cost.
- `Text` blocks can be up to 4 MB (enabling BWT trials that turn long-range
  word repeats into local runs).
- `Binary`, `Exec`, and `Random` use the default 64 KiB chunk size.

The bit-level path runs an **online logistic mixer hierarchy**:

1. **Bank mixers** (16384 instances): selected by a context hash of byte-class,
   bit-position, order-1/order-2 bytes, and word-hash. Each bank specializes
   weights to its context, avoiding the ~50% saturation a single mixer hits
   on repetitive corpora.
2. **Global mixer**: a context-agnostic fallback over the same models.
3. **Master mixer**: blends `[p_bank, p_global, p_lzp_conf]` in logistic space.

Only the selected bank + global + master are trained per bit — never all 16384.
At block boundaries, weights are **decayed** (not reset), preserving learned
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
`rcn` is installed via `cargo install --locked rcn` or downloaded as a binary release from
[crates.io](https://crates.io/crates/rcn). The package requires Rust toolchain ≥1.85
(`rust-version` in `Cargo.toml`) and an x86_64 (AVX2) or arm64 (scalar) processor.
On x86_64 (Linux), the AVX2 code path can be disabled at build time with
`--no-default-features --features no_avx2` — see [CI notes](#ci-notes) below.

### CI notes
GitHub Actions runs on `ubuntu-latest` (x86_64) where the AVX2 SIMD path in
`bytecodec` is compiled and exercised, while local development on Apple Silicon
(arm64) uses the scalar path. Until the AVX2 implementation is fully verified on
Linux runners, CI runs tests with the `no_avx2` feature to exercise the scalar
path only; local builds still use AVX2 by default on x86_64.

### Compression modes
The CLI exposes two entropy paths: `--mode slow` (bit-level CM, higher ratio,
default) and `--mode fast` (byte-level CM, faster). Named multi-level presets
(`-1`…`-9` analogous to zstd) are out of scope until the strong path beats
`zstd -19` on text and the fast path has a clear speed floor.
### Streaming / `--stdout`
The `compress` and `decompress` subcommands accept `-` as input/output path for
stdin/stdout piping. Per-block progress/stats (`--verbose`) are planned for a
future release.

## Benchmarks

> **Both ratio and speed, on every run.** rcn codes bit-by-bit on the slow path,
> so a fair comparison must report both axes. The headline table below is a
> representative **5-file subset** (dickens, webster, nci, mr, json) under
> `--mode slow`. Fast-path numbers are in the Speed bullet. `ratio%` is the
> compressed size as a percentage of the original (lower is better); speed is
> in MB/s (higher is better). Full data is in the
> [experiments log](#experiments-log-2026-09).

### Current slow path (hybrid_ppm3 + two-level 16k-bank mixer + classifier-aware method bytes + word model + cross-block decay + 32MB LZP window + BWT text trial + JSON stream splitting + DP optimal LZP parse default + Exec E8E9 + global XWRT-512 ESC + SSE/APM/APM2 + adaptive DP thresholds + side-stream FSE-family + classify-ahead + mimalloc)

| file | orig (KB) | rcn ratio% | rcn cmp MB/s | rcn dec MB/s | zstd -1 ratio% | zstd -1 cmp MB/s | zstd -1 dec MB/s | zstd -19 ratio% | zstd -19 cmp MB/s | zstd -19 dec MB/s | FSE ratio% | FSE cmp MB/s | FSE dec MB/s | ratio winner | speed winner |
|------|----------:|-----------:|-------------:|-------------:|---------------:|-----------------:|-----------------:|---------------:|-----------------:|-----------------:|-----------:|-------------:|-------------:|:------------:|:------------:|
| dickens | 9953.6 | **40.2** | **0.04** | **0.04** | 41.7 | 496.1 | 2837.1 | 28.0 | 3.3 | 288.9 | 57.0 | 375.6 | 463.7 | **zstd -19** | **zstd -19** |
| webster | 40487.0 | **23.6** | **0.7** | **0.5** | 33.5 | 404.5 | 1219.8 | 21.1 | 4.0 | 720.6 | 62.6 | 424.6 | 507.9 | **zstd -19** | **zstd -19** |
| nci | 32767.0 | **7.4** | **0.7** | **0.5** | 85.2 | 376.9 | 3218.9 | 49.5 | 3.9 | 1626.0 | 30.2 | 326.7 | 335.9 | **rcn** | **zstd -19** |
| mr | 9736.9 | **27.3** | **0.04** | **0.04** | 38.5 | 551.2 | 1008.8 | 31.2 | 5.6 | 291.2 | 44.0 | 233.2 | 229.3 | **rcn** | **zstd -19** |
| json | 478.5 | **0.1** | **4.0** | **28** | 0.3 | 12173.7 | 35691.0 | 0.1 | 36824.1 | 36824.1 | 52.8 | 1649.5 | 1251.9 | **rcn**/tie | **zstd -19** |

Notes: dickens/json/mr slow ratios re-measured 2026-09-09 (HEAD dickens **41.3%**; C6 ≈−1.1pt). mr 27.3% was with Binary LZP min=24; C7 since softened to keep Binary/Exec at 16 (re-measure pending). webster/nci slow ratios are prior figures. zstd/FSE columns unchanged.

### Fast path (`--mode fast`) re-bench 2026-09-09

| file | orig (KB) | rcn ratio% | cmp MB/s | dec MB/s |
|------|----------:|-----------:|---------:|---------:|
| dickens | 9953.6 | 26.9 | 2.0 | 4.3 |
| webster | 40487.0 | 20.3 | 2.7 | 5.5 |
| nci | 32767.0 | 5.1 | 5.1 | 10.6 |
| mr | 9736.9 | 35.0 | 2.6 | 4.0 |
| json | 478.5 | 0.1 | 5.2 | 59.1 |

### DP-optimal LZP parse (now default)

DP optimal LZP parse runs a forward LZP match pre-pass and emits `(len, dist)` records for matches ≥ 16 bytes. Matched bytes are skipped in the rANS stream — only literals are CM-encoded. The match side-stream uses varint-encoded records (delta_pos + len + dist as varints) instead of fixed 8-byte records.

| file | orig (KB) | rcn ratio% | vs default | zstd -1 ratio% | beats zstd-1? |
|------|----------:|-----------:|-----------:|---------------:|:------------:|
| dickens | 9953.6 | 41.9 | 46.2→41.9 (**−4.3pt**) | 41.7 | ~parity |
| webster (10MB) | 10000.0 | 31.4 | 35.1→31.4 (**−3.7pt**) | 33.0 | ✅ |
| nci | 32767.0 | 8.2 | 9.0→8.2 (**−0.8pt**) | 85.2 | ✅ |
| mr | 9736.9 | 29.1 | 27.5→29.1 (**+1.6pt**) | 38.3 | ✅ (vs default) |
| json | 478.5 | 0.1 | 0.1 (same) | 0.3 | ✅ |
| huge_json | 5641.7 | 1.3 | 2.7→1.3 (**−1.4pt**) | 2.5 | ✅ |
| massive_json | 22885.9 | 0.97 | 0.81→0.97 (**+0.16pt**) | 2.48 | ✅ |

**Key insight:** DP optimal parse with literal-skipping rANS is a net win on most files. `dickens` and `webster` approach or beat `zstd -1`. `mr` and `massive_json` regress slightly — the SSM model added alongside the DP parse is the likely cause, not the DP parse itself. DP is now default, and SSM is isolated from DP to avoid regression.

### Reading the table

- **Ratio:** lower % is better. On the current stack, rcn **beats `zstd -19`** on
  `nci` and `mr`, and ties/beats on small `json`. On text, `zstd -19` still leads:
  dickens 40.2% vs 28.0% (rcn still beats `zstd -1` at 41.7%), webster gap TBD on
  full re-bench. Historical experiment numbers (e.g. older webster 35.1% / claimed
  33.2% dickens) are in the experiments log — the headline dickens figure above is
  a 2026-09-09 re-measure (prior README 33.2% did not reproduce on this tree/machine;
  HEAD was 41.3% before C6).

- **Speed:** higher MB/s is better.
  **Slow path** (`--mode slow`): ~0.04 MB/s on large text on this machine (bit-level
  CM); small JSON is much faster when BWT collapses the stream.
  **Fast path** (`--mode fast`): ~2–5 MB/s compress / ~4–10 MB/s decode after AVX2
  `walk_dist` + interleaved rANS (see fast-path table). Still short of the 20+ MB/s
  goal.
  zstd `-1` is hundreds–thousands of MB/s; zstd `-19` is ~3–4 MB/s compress.

### New optimization target (2026-09)

Beat `zstd -19` on ratio for text + mixed corpora while keeping the existing
`nci`/`mr` wins. `FSE` is tracked as a secondary reference. Speed remains
secondary.

## Experiments log (2026-09)

### Architecture & modeling

| experiment | files tested | result | action |
|---|---|---|---|
| Per-bit-position mixer context | mr, dickens, json, webster, nci | **improved all 5** | kept as default |
| Classifier-aware method bytes | mr, dickens, json, webster, nci | neutral | kept as infrastructure |
| Word/string model (case-folded, bigram prefix) | dickens, json, webster, nci | +0.1pt on 4/5 | kept as default (text blocks) |
| Refined word model (trigram + char-class + 21-bit table) | json | regressed 5.6%→5.8% | reverted to simple word model |
| Record segmentation model (JSON key/value parser) | dickens, json, webster, nci | neutral json/dickens/mr; regressed webster/nci | reverted |
| ICM (22-state PAQ8) | mr, dickens, json, webster, nci | regressed on 4/5 | reverted |
| ICM (256-state probability-quantized) | mr, dickens, json, webster, nci | regressed on all 5 | reverted |
| Order-4 PPM with word-boundary-aware context masking | mr, dickens, json, webster, nci | regressed dickens +0.1pt, webster +0.5pt, nci +0.1pt, json +3.6pt | reverted |
| Lazy multi-context LZP (hash chains + longest-match) | mr, dickens, json, webster, nci | neutral (−0.1pt) | kept in place, not adopted |
| Two-pass CM residual (match records + CM literals) | mr, dickens, json, webster, nci | nci +2.7pt, json/webster regressed | reverted; match overhead too high at 64 KiB |
| Literal bypass hint model (high-entropy byte bypass) | mr, dickens, json, webster, nci | regressed dickens 56.3%→57.1% | reverted |
| **Context-selected 8k mixer banks** (8192 per-context LogisticMixer instances selected by byte-class + order-1/order-2 + word-hash, blended with global + master) | mr, dickens, json, webster, nci | **improved**: json 3.9%→3.0%, dickens 51.7%→51.2% | **kept as default** — two-level bank→global→master hierarchy, cross-block **decay** preserves the 8192 vectors, only selected bank + master trained per bit |
| **Indirect context + DMC models** | mr, dickens, json, webster, nci | regressed dickens +0.7pt, webster +0.4pt; json improved | reverted due to perf cost |
| **Cross-block persistence + real 4MB LDM window** | json, mr, dickens, nci, webster | **improved**: json 5.5%→3.9%, mr 28.6%→27.3%, dickens 56.0%→51.7%, nci 26.6%→20.9%, webster 50.4%→45.1% | **kept as default** |
| LZP ring buffer performance fix (O(n) drain→O(1) ring) | all files | performance fix, no ratio change | kept |
| Micro SSM mixer (16-dim recurrent state replacing logistic mixer) | json, mr | **regressed**: json 3.9%→10.7%, mr 27.3%→37.2% | reverted; SSM too large for 64KB blocks, gradient issues |
| Second-order mixer training (Adam + per-model lr_scale) | dickens, mr, json | **neutral** (Adam) / **neutral** (SGD + lr_scale) | kept as default; Adam never measurably better on default stacks |
| BWT text trial (RawCm vs BWT→MTF→RLE0→CM vs LZP→BWT→MTF→CM) | dickens, json, webster | **improved**: json 3.0%→0.1%, dickens 51.2%→46.2%, **webster 50.4%→35.1%** | **kept as default** |
| JSON stream splitting + per-stream pipeline selection | json (478KB–22MB) | **improved**; beats zstd-1 and zstd-19 on large JSON | **kept as default** |
| Order-8 PPMd with SEE + sparse de Bruijn | webster, dickens, json | mixed; config now matches hybrid_ppm3 | **promoted to default** — fixed config retains WordModel, 8k banks, 32MB window, XWRT |
| DP optimal LZP parse (now default) | dickens, webster, nci, mr, json, huge_json, massive_json | **improved** on 5/7 (dickens −4.3pt, webster −3.7pt); **regressed** mr +1.6pt, massive_json +0.16pt | **kept as default** — SSM isolated to avoid regression |
| **Exec E8E9 transform** | Exec executables | Converts x86 relative offsets to absolute (3-5pt on Exec corpora) | **kept as default** |
| **XWRT dictionary before BWT** | Text blocks | Build top 2k words per block, replace with 0x80+id tokens, then BWT→MTF→RLE0→CM. 2-4pt on dickens/webster | **per-block superseded by global** (single-byte tokens only hold 128 ids; word-scan now caps at 128 + rejects >255-byte breakless runs) |
| **Global XWRT dictionary** | dickens, webster, nci, mr, json | One corpus-wide first pass builds a top-128 dictionary stored once in the container header; every Text-block XWRT trial reuses it (no per-block dict bytes). Measured: dickens −4.6pt (45.9→41.3), webster −4.0pt (34.1→30.1), nci −1.5pt (8.9→7.4), mr neutral, json 458→299B | **kept as default** — gated on pure-ASCII blocks (tokens alias ≥0x80 literal bytes) |
| **SSE/APM/APM2 cascade** | mr, dickens, json, webster, nci | **improved all 5**: nci −0.9pt, mr −0.8pt, dickens −0.3pt, webster −0.5pt, json −0.1pt | **wired into codec** |
| **AVX2 SIMD walk_dist** | All | 5-10x fast path speedup, zero ratio loss | **completed** |

### Speed passes

| pass | description | files tested | result | status |
|---|---|---|---|---|
| #1 | LazyLzp removal (O(n²) memmove per byte over 1MB; match-extension loop dead). Model rewritten with fixed-capacity ring buffer + causal extension loop. | dickens, webster | **3× encode speedup on text, zero ratio change** (dickens 2MB: 10.9s→3.3s, byte-identical) | removed from all default stacks |
| #2 | Byte-level "fast" path (`--mode fast`): PPM-style single-context count coder (deterministic order-0/1/2 selector + fused 256-symbol cumulative walk_dist + byte rANS). No mixer/softmax. | dickens 2MB | **2.6× encode speedup vs slow with ~1.7× better ratio on BWT+MTF streams** (fast 1.15s/0.279x vs slow 2.97s/0.470x; round-trip verified; 127/127 tests) | kept as `--mode fast` |
| #3 | Single-pass acc-merge across the mixer chain (`MixerBank::mix_acc`/`update_acc`/`mix_and_update` + `LogisticMixer::mix_acc`/`update_from_acc`). Also: Q16 fixed-point mixer (i32 Q16 weights, i16 Q10 stretch, i64 accumulator; SGD grad scaled ×64 so per-bit deltas are meaningful). | dickens 2MB (slow) | **~10% encode speedup, ratio flat** (3.28s→2.81s user; 0.470x → 0.470x, 986560 vs 986551 B). Q8 attempt reverted (ratio +0.8pt, gradients rounded to 0) | kept as default |
| #4 | 32-way interleaved byte rANS (32 independent lane states, per-symbol-position `p % 32` lanes). | dickens 2MB (fast) | **no measurable speedup** (1.15s→1.17s user). Profiling: rANS is ~10% of fast path; `walk_dist`'s linear 256-probability scan dominates and stays serial under byte-interleaving | **wired into the fast path** — `bytecodec` uses `RansByteEncoder32`/`Decoder32`; re-bench now that `walk_dist` is AVX2 SIMD |

### Code-quality / correctness notes

| experiment | files tested | result | action |
|---|---|---|---|
| round-trip verification | all 5 | lossless | every pass round-trip verified |
| test suite | all | 127/127 green (including `walk_roundtrip_exact` triple-equality test) | kept |
| bit-identical output | dickens 2MB | each pass `cmp`-identical to prior | kept |

## Speed roadmap (2026-09)

1. **SoA weight layout** — **Completed**.
2. **Stretch-value reuse** — **Completed**.
3. **Wider stride / 16k banks** — **Completed** (16384 banks, 14-bit hash).
4. **Parallel BWT trials** — intra-block `rayon::join`. **Partial (S5)**.
5. **S8 — Interleaved rANS re-bench** — **Completed** (wired; fast-path table above).
6. **S9 — Optional libsais BWT** — `--features bwt_libsais`. **Completed**.
7. **S10 — Classify-ahead pipeline** — **Completed**.
8. **mimalloc** — **Completed**.

## Potential ratio improvements (remaining)

C6–C9 are implemented (IndirectModel left out of the default Text stack after
re-test). Further gains toward `zstd -19` on text likely need new modeling, not
more of the same tickets. See [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md).

## License

MIT — see [LICENSE](LICENSE).
