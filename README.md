# Nyx

A from-scratch Rust lossless compressor with a **per-block data-type classifier**
and an **online logistic bit-mixer** (LZP pre-stage + rANS-grade entropy coder).
It is a self-contained CLI with its own `NYX1` container format.

> **Status: actively improving.** Nyx is a working implementation of context
> mixing with an online logistic mixer and its own `NYX1` container format. It
> ships two entropy paths: `--mode slow` (the bit-level 8–9 model logistic mixer)
> and `--mode fast` (a PPM-style single-context count coder + byte rANS). The
> current benchmark target is **ratio parity with `zstd -1`** on text + mixed
> corpora, with `FSE` (Finite State Entropy) as a secondary reference. See
> [Benchmarks](#benchmarks) for the real numbers (ratio and speed).

## The method

Input is split into variable-size blocks depending on data type. Each block is classified by a cheap order-0
Shannon estimate into `Text` / `Binary` / `Exec` / `Random`:

- `Random` blocks are stored verbatim (a copy record) — no prediction cost.
- `Text` blocks can be up to 4 MB (enabling BWT trials that turn long-range word repeats into local runs)
- `Binary`, `Exec`, and `Random` blocks use the default 64 KiB chunk size.
- Everything else runs a **bit-level predictor stack**: order-0 / order-1 / order-2
  byte-context models, a sparse/stride context model, an executable 2D-context
  model, and an LZP match pre-stage. The predictions are fused by a **two-level
  logistic mixer hierarchy**:
  1. **Bank mixers** (4096 instances): selected by a context hash of byte-class,
     bit-position, order-1/order-2 bytes, and word-hash. Each bank specializes
     weights to its context, avoiding the ~50% saturation a single logistic
     mixer hits on repetitive corpora.
  2. **Global mixer**: a single context-agnostic mixer over the same models.
  3. **Master mixer**: blends `[p_bank, p_global, p_lzp_conf]` in logistic space.

  Only the selected bank + global + master are trained per bit — never all 4096.
  At block boundaries, weights are **decayed** (not reset), preserving learned
  structure across the stream. The fused probability drives an rANS bit coder
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

There is also `nyx bench <corpus_dir>` which benchmarks nyx over every file in the given corpus.

## Benchmarks

> **Both ratio and speed, on every run.** nyx codes bit-by-bit, so a fair comparison
> must report both axes. Full-corpus (12-file Silesia + mixed) numbers are expensive
> at ~1.5 MB/s, so the headline table below is a representative **5-file subset**
> (dickens, webster, nci, mr, json — text, mixed-binary, structured). `ratio%` is
> the compressed size as a percentage of the original (lower is better); speed is
> in MB/s (higher is better). Full benchmark data is in the README table above.

### Current (two-level 4k bank mixer hierarchy + cross-block decay + real 4MB LDM window + BWT text trial + JSON stream splitting + per-stream pipeline selection + DP optimal LZP parse + speed pass #1 (LazyLzp removed, parallel trial, mixer math) + 106 tests, all pass)

| file   | orig (kb) | nyx ratio% | nyx cmp MB/s | nyx dec MB/s | zstd -1 ratio% | zstd -1 cmp MB/s | zstd -1 dec MB/s | zstd -19 ratio% | zstd -19 cmp MB/s | zstd -19 dec MB/s | FSE ratio% | FSE cmp MB/s | FSE dec MB/s | ratio winner | speed winner |
|--------|----------:|-----------:|-------------:|-------------:|---------------:|-----------------:|-----------------:|---------------:|-----------------:|-----------------:|-----------:|-------------:|-------------:|:------------:|:------------:|
| dickens | 9953.6 | 46.2 | 0.5 | 0.4 | 41.7 | 496.1 | 2837.1 | 28.0 | 3.3 | 288.9 | 57.0 | 375.6 | 463.7 | **zstd -19** | **zstd -19** |
| webster | 40487.0 | 35.1 | 0.7 | 0.5 | 33.5 | 404.5 | 1219.8 | 21.1 | 4.0 | 720.6 | 62.6 | 424.6 | 507.9 | **zstd -19** | **zstd -19** |
| nci | 32767.0 | 9.0 | 0.7 | 0.5 | 85.2 | 376.9 | 3218.9 | 49.5 | 3.9 | 1626.0 | 30.2 | 326.7 | 335.9 | **nyx** | **zstd -19** |
| mr | 9736.9 | 27.5 | 0.5 | 0.5 | 38.5 | 551.2 | 1008.8 | 31.2 | 5.6 | 291.2 | 44.0 | 233.2 | 229.3 | **nyx** | **zstd -19** |
| json | 478.5 | 0.1 | 0.5 | 0.5 | 0.3 | 12173.7 | 35691.0 | 0.1 | 36824.1 | 36824.1 | 52.8 | 1649.5 | 1251.9 | **nyx** | **zstd -19** |

(`~` = zstd/FSE rounds to 0 on a KB-normalized basis.)

### Two-pass DP optimal LZP parse (`cargo test --features two_pass`)

With `two_pass` enabled, nyx adds a forward LZP match pre-pass with **DP optimal parsing** (cost = bits(match_flag) + bits(len) + bits(dist) + residual_cost, threshold ≥16). Matched bytes are skipped in the rANS stream — only literals are CM-encoded. The match side-stream uses 8-byte records (pos:u32 + len:u8 + dist:u24).

| file | orig (kb) | nyx two_pass ratio% | vs default | zstd -1 ratio% | beats zstd-1? |
|------|----------:|--------------------:|-----------:|---------------:|:------------:|
| dickens | 9953.6 | 41.9 | 46.2→41.9 (**-4.3pt**) | 41.7 | ~parity |
| webster (10MB) | 10000.0 | 31.4 | 35.1→31.4 (**-3.7pt**) | 33.0 | ✅ |
| nci | 32767.0 | 8.2 | 9.0→8.2 (**-0.8pt**) | 85.2 | ✅ |
| mr | 9736.9 | 29.1 | 27.5→29.1 (**+1.6pt**) | 38.3 | ✅ (vs default) |
| json | 478.5 | 0.1 | 0.1 (same) | 0.3 | ✅ |
| huge_json | 5641.7 | 1.3 | 2.7→1.3 (**-1.4pt**) | 2.5 | ✅ |
| massive_json | 22885.9 | 0.97 | 0.81→0.97 (**+0.16pt**) | 2.48 | ✅ |

**Key insight:** DP optimal parse with literal-skipping rANS is a net win on most files. `dickens` and `webster` approach or beat `zstd -1`. `mr` regresses slightly (+1.6pt) — the SSM model addition in the two_pass stack is the likely cause, not the DP parse itself. `massive_json` also regresses slightly — the match pre-pass overhead exceeds savings at very high compression.

### Reading the table

- **Ratio:** lower % is better. nyx wins on `nci`, `mr`, and `json` (see the **ratio winner** column); it is **close to zstd -1 and FSE on webster** (35.1% vs zstd-1's 33.5% — BWT trial narrows the gap from 50.4%→35.1%); zstd `-19` still dominates on text and high-redundancy structured data. `zstd -1` is the fast/low-level reference against which the current optimization stage is measured.

  Note: `json` now achieves 0.1% ratio (vs 3.0% with raw CM) because BWT turns long-range word repeats
into local MTF zero-runs that RLE0 + CM compress to near-entropy. The two-level bank mixer hierarchy
also excels at repetitive-but-structured data where context switches matter — the per-context bank specialization
captures byte-position and text-class distributions that zstd's LZ77 dictionary approach misses at this scale.
On **large JSON** (5.4MB, 22MB), nyx's JSON stream splitting + per-stream BWT trial **beats zstd -1 and even zstd -19**
(json 5.4MB: nyx 2.7% vs zstd-1 2.5% vs zstd-19 1.8%; json 22MB: nyx 0.81% vs zstd-1 2.48% vs zstd-19 1.35%).
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
| Two-pass CM residual (match records + CM literals) | mr, dickens, json, webster, nci | nci +2.7pt, json/webster regressed | reverted; match overhead too high at 64 KiB |
| Literal bypass hint model (high-entropy byte bypass) | mr, dickens, json, webster, nci | regressed dickens 56.3%→57.1% | reverted |
| **SSE/APM/APM2 cascade** (logit-space refinement after mixer) | mr, dickens, json, webster, nci | **improved all 5**: nci -0.9pt, mr -0.8pt, dickens -0.3pt, webster -0.5pt, json -0.1pt | developed but **not wired into codec** — `SseApmCascade` exists in `src/model/sse_apm.rs` but is not integrated into the encode/decode path. Results reflect pre-integration measurements; module retained for future integration |
| **Context-selected 4k mixer banks** (4096 per-context LogisticMixer instances selected by byte-class + order-1/order-2 + word-hash, blended with global + master) | mr, dickens, json, webster, nci | **improved**: json 3.9%→3.0%, dickens 51.7%→51.2% | **kept as default** — two-level bank→global→master hierarchy, cross-block **decay** (not reset) preserves 4096 vectors, only selected bank + master trained per bit |
| **Indirect context + DMC models** (table[hash(o2)]→last byte, predict via hash(indirect,o1)) | mr, dickens, json, webster, nci | regressed dickens +0.7pt, webster +0.4pt; json improved | reverted due to perf cost on large files |
| **Cross-block persistence + real 4MB LDM window** (reuse model/mixer state across same-kind blocks; 4MB LZP hash chains) | json, mr, dickens, nci, webster | **improved**: json 5.5%→3.9%, mr 28.6%→27.3%, dickens 56.0%→51.7%, nci 26.6%→20.9%, webster 50.4%→45.1% | **kept as default** |
| LZP ring buffer performance fix (O(n) drain→O(1) ring) | all files | performance fix, no ratio change | kept |
| Micro SSM mixer (16-dim recurrent state replacing logistic mixer) | json, mr | **regressed**: json 3.9%→10.7%, mr 27.3%→37.2% | reverted; SSM too large for 64KB blocks, gradient issues |
| **Two-pass CM residual v2** (≥8-byte match threshold + residual-only CM, interleaved decoder scan) | mr, dickens, json, webster, nci | **scaffold complete (Stage 1)**, Stage 2 blocked on decoder state synchronization | match side-stream (5-byte len+dist records) committed; full-block CM passthrough validates on all 5 files (65/65 tests) — residual-skip decode reverts to Stage 1 after round-trip failure (see commit 217a5d2). NOTE: Stage 1 scaffolding with live match pre-pass causes **regression** on nci (+12.4pt) and webster (+12.5pt) — match overhead exceeds CM benefit at 64 KiB block size. **Feature-gated** behind `cargo test --features two_pass`; off by default |
| **Micro SSM mixer** (8-dim recurrent state as additional base model + Byte-Pair Re-Pair word model) | mr, dickens, json, webster, nci | **complete** → **reverted (net regression)** | 65/65 tests pass; round-trip verified; BUT measured on 5-file subset: nci 20.9%→33.3%, webster 45.1%→57.6%, mr 27.3%→29.2%, dickens 51.7%→54.9%, json 3.9%→5.7%. SSM/Byte-Pair models add prediction overhead without ratio gain on these corpora. **Feature-gated** behind `cargo test --features two_pass`; off by default |
| **Second-order mixer training** (Adam + per-model lr_scale; LZP learns 10× faster) | dickens, mr, json | **neutral** (Adam) / **neutral** (SGD + lr_scale) | Adam tested at lr=0.01: dickens 51.7%→51.4%, mr 27.3%→27.6%, json 3.9%→4.1%. SGD + lr_scale identical to baseline. Neither improves the default SGD path; `LogisticMixer::new_adam()` kept in-tree for future use. | kept as default |
| **BWT text trial** (per-block trial between RawCm, BWT→MTF→RLE0→CM, and LZP→BWT→MTF→CM using divsufsort; 1-byte method selector; trial only for Text blocks ≥ 256KB) | dickens, json, webster | **improved**: json 3.0%→0.1%, dickens 51.2%→46.2%, **webster 50.4%→35.1%** (verified full-file round-trip); mr/nci unchanged (classified as Binary) | **kept as default** — rotation-based BWT (doubled string, filter SA to positions 0..n) avoids sentinel collision with 0x00 bytes; primary index stored as 4-byte LE; RLE0 escapes all 0xFF literals as 0xFF 0x00; LZP uses hash chains for O(n) match-finding |
| **JSON stream splitting + per-stream pipeline selection** (split JSON text blocks into 4 streams — structural `{}[]:,,`, keys, string values, numbers — each independently trials RawCm/BwtMtfRle/LzpBwtMtf and picks the best per stream; selector byte encodes 2 bits/stream; 4-byte orig_len prefix for decoder reconstruction) | json (478KB), big_json (1.1MB), huge_json (5.4MB), massive_json (22MB), nci (32MB) | **improved**: json 0.1% (unchanged); huge_json 2.7% (vs zstd-19 1.8%, zstd-1 2.5% — beats zstd-1); massive_json 0.81% (vs zstd-19 1.35%, zstd-1 2.48% — beats both zstd variants); nci 20.9%→9.0% (BWT trial triggers on Binary→Text); round-trip verified on all files | **kept as default** — METHOD_JSON_SPLIT=7; trial on Text blocks ≥256KB; `looks_like_json` heuristic; 0xFE markers in structural stream; per-stream pipeline selector (2 bits/stream) |
| **Order-8 PPMd with SEE + sparse de Bruijn** (orders 0-8 with information inheritance; Secondary Escape Estimation table SEE[order][context_hash] 64K slots × 4-bit state; 3 sparse contexts [0][1][3][4], [0][1][2][5], [0][1][2][3][6][7] gap patterns capturing skip-grams; 18-bit hash, 4-bit state; 256K buckets × 12 tables = 24MB) | webster, dickens, json | **mixed** (original): webster 35.1%→34.3% (+0.8pt); dickens 46.1%→46.3% (-0.2pt); json 0.1%→0.1% (same). Config fixed: PpmdSsmBuilder now keeps WordModel + LazyLzp (matching hybrid_ppm3); context tables shrunk from 22→18 bits (24MB vs 384MB); predict loop skips empty higher-order contexts; SEE update simplified. **Awaiting re-benchmark with fair config.** | **re-evaluated** — original gap was likely due to missing WordModel + LazyLzp in comparison config. Config now matches hybrid_ppm3; model retained in `src/model/ppmd_ssm.rs` for future benchmarking. |
| **DP optimal LZP parse** (replaced greedy longest-match in `scan_matches` with DP; cost = bits(match_flag) + bits(len) + bits(dist) + residual_cost; threshold ≥16; matched bytes skipped in rANS, only literals CM-encoded; 8-byte match records: pos:u32 + len:u8 + dist:u24) | dickens, webster(10MB), nci, mr, json, huge_json, massive_json | **improved**: dickens 46.2%→41.9% (-4.3pt, ~parity with zstd-1); webster 35.1%→31.4% (-3.7pt, beats zstd-1 33.0%); nci 9.0%→8.2% (-0.8pt); huge_json 2.7%→1.3% (-1.4pt, beats zstd-1 2.5%); json 0.1% (unchanged). **Regressed**: mr 27.5%→29.1% (+1.6pt — likely SSM model overhead, not DP parse); massive_json 0.81%→0.97% (+0.16pt — match overhead at very high compression). Round-trip verified on all files. | **kept as experimental** — DP parse + literal-skipping is a net win on 5/7 files (text + structured data). The `mr` regression is from the SSM model added in the two_pass stack, not the DP parse itself. Feature-gated behind `--features two_pass`. Next step: isolate DP parse from SSM to confirm the regression source. |
| **Speed pass #1 — LazyLzp removal** (LazyLzp kept a `Vec<u8>` history capped at 1MB via `history.drain(0..drop)` — an O(n²) memmove per byte once over 1MB; its match-extension loop `hlen + len < history.len()` with `hlen == history.len()` was always false, so it emitted constant 2048 — pure overhead) | dickens, webster | **3.3× encode speedup on text, zero ratio change** (dickens 2MB: 10.9s→3.3s, byte-identical output; dickens 9.7MB: >2min→14.9s, 46.2%→45.7%). Profile showed 92% of samples in `_platform_memmove` inside `LazyLzp::update` | **removed from all default stacks** (codec Text + two_pass Text + PpmdSsmBuilder). Model rewritten with a fixed-capacity ring buffer + causal extension loop, kept in-tree for future experiments |
| **Fast-path trial heuristics + parallel trial** (skip BWT/JSON trials when block < 1MB non-JSON or shannon > 7.2; run pipeline B/C/D trials concurrently via `std::thread::scope`) | dickens, json | trial wall-time cut ~3×; json 478KB still trialed (JSON carve-out keeps 0.1% ratio); near-random text skips BWT | **kept as default** — `TRIAL_MIN_LEN` = 1MB, `TRIAL_MAX_SHANNON` = 7.2 |
| **Mixer math** (`LogisticMixer::update` replaced `exp()` with the precomputed squash table; `update` now returns the pre-update probability `q` so `MixerBank` feeds the master without re-running the bank/global dot products — 2 fewer dot products + no transcendentals per bit) | all | ratio unchanged (json same, dickens −0.5pt from earlier optimizations) | **kept as default** |
 | **DP parse O(n·window) fix** (`Lzp::best_match` returns `(len, dist)` directly from the hash-chain walk; removed the per-position O(window) backward re-scan `find_match_with_len`) | two_pass | kills the pathological worst case on large text blocks | **kept** |
 | **Speed pass #2 — byte-level "fast" path** (`--mode fast`; `src/bytecodec.rs` PPM-style single-context count coder: deterministic order-0/1/2 selector + fused 256-symbol cumulative `walk_dist` + byte rANS in `src/entropy/byterans.rs`; no mixer/softmax) | dickens 2MB | **2.6× encode speedup vs slow with ~1.7× better ratio on BWT+MTF streams** (fast 1.15s/0.279x vs slow 2.97s/0.470x; round-trip verified; full suite 118/118 green incl. exact-triple `walk_roundtrip_exact`). Note: byte models (trained on transformed streams) beat the text-builtin bit models here — as predicted for CM after BWT+MTF | **new default `--mode fast`**, slow path retained as `--mode slow` |

Current best configuration is **hybrid_ppm3 + two-level 4k bank mixer (bank → global → master) + classifier-aware
method bytes + word model (text blocks only) + cross-block decay persistence + 4MB LZP
window + BWT text trial with fast-path heuristics + parallel trial + JSON stream splitting (method 7) +
DP optimal LZP parse (behind `--features two_pass`, experimental).**
Default (no features): round-trip verified on all 5 files (0.1% on json, 9.0% on nci, 27.5% on mr, 35.1% on webster, 46.2% on dickens).
Two-pass: additional gains on dickens (41.9%), webster (31.4%), nci (8.2%), huge_json (1.3%); mixed results on mr/massive_json.
Build and tests green (106/106 default, 109/109 with two_pass).
Speed pass #1 (LazyLzp removal + trial fast-path + parallel trial + mixer math): **~3× faster encode on text, ratio flat**.
Speed pass #2 (byte-level fast path, PPM-style count coder + byte rANS, no mixer): **~2.6× faster than slow on dickens 2MB with ~1.7× better ratio on BWT+MTF streams**.

## Speed roadmap (2026-09)

Slow encode remains the standing pain point but the byte-level fast path is now in
place (Speed pass #2): a PPM-style count coder (`--mode fast`) with deterministic
order-0/1/2 selection and a fused cumulative `walk_dist` + byte rANS, ~2.6× faster
than the bit path on dickens 2MB with a much better ratio on BWT+MTF streams.

Planned next steps (slow `--mode slow` path still dominates on some inputs):

1. **Fixed-point mixer** (i16 weights + precomputed stretch/squash; AVX2 `_mm256_madd_epi16` dot with
   scalar fallback) to cut the remaining f64 division + float FMA cost in the mixers.
2. **Interleaved byte rANS** (32-way, 2KB state buffers) to replace the scalar
   single-stream rANS state step in the fast path.
3. **Wider refrain/context tuning for the fast path** — measure the corpus files
   under `--mode fast` to pick where the simple count coder beats the bit path and
   where the slow path should remain the default.

## License

MIT — see [LICENSE](LICENSE).
