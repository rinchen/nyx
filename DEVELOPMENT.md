# Rcn development notes

Optimization tickets, experiment log, CI/testing notes, and roadmaps.
For product overview and headline benchmarks, see [README.md](README.md).
Ticket tables also live in [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md).

Last updated: 2026-09-09. Test status: 143/143 passing (`cargo test --lib`).

---

## Completed optimization checklist

Numbered C/S backlog is empty; items below are the closed ticket record.

### Compression - beat zstd -19
- ✅ **Promote Order-8 PPMd with SEE + sparse de Bruijn to default** – WordModel + banks + 32MB window + XWRT
- ✅ **Compress the match side-stream (varint)** – delta_pos/len/dist varints
- ✅ **Global XWRT dictionary (top-128)** – corpus-wide; C5 measured dickens −4.6pt, webster −4.0pt
- ✅ **C6 — Global XWRT 128 → 512 with ESC tokens** – `0x80..=0xFE` for ids 0..126; `0xFF || u16_le(id)` for 127..511 (ESC slots only for words longer than 3 bytes so tokens never expand); header dict count is `u16`
- ✅ **C7 — Adaptive DP LZP threshold** – Text ≥12; Binary/Exec/Random stay 16 (`≥24` on Binary/Exec regressed mr and was reverted)
- ✅ **C8 — 16k banks** – `NUM_MIXERS=16384` (14-bit hash). Single IndirectModel tried again; **left out of default Text stack** (prior dickens regression); model remains in-tree
- ✅ **C9 — FSE-family match side-stream** – order-0 byte rANS on the varint blob when smaller (`[num_runs][flag][len][payload]`)

### Speed - achieve 20+ MB/s
- ✅ **S8 — Interleaved rANS re-bench** – already wired (`RansByteEncoder32`/`Decoder32`); post-AVX2 `walk_dist` fast path ~2–5 MB/s cmp / ~4–10 MB/s dec (see README Benchmarks)
- ✅ **Parallel BWT trials (S5 partial)** – `rayon::join` for path B/C inside a block trial
- ✅ **S9 — Optional libsais BWT backend** – `--features bwt_libsais` uses pure-Rust `libsais-rs`; default remains `divsufsort`
- ✅ **S10 — Classify-ahead pipeline** – `rayon::join` overlaps classify+size of block N+1 with encode of N (encode stays serial / bit-identical)

### Other Completed
- **S1–S4, S7, C1–C5** – SIMD walk_dist, SoA banks, stretch reuse, mimalloc, SSE/APM, global XWRT-128, etc.
- **CI fixes** – `no_avx2` feature for Linux CI scalar path
- **Pre-commit hook** – mirrors the CI gate

Gap to beat `zstd -19` on text is tracked in the README headline table (dickens/webster). nci / mr / json already win on ratio vs `-19` in recent measures.

---

## CI / testing

GitHub Actions runs on `ubuntu-latest` (x86_64). The AVX2 SIMD path in
`bytecodec` is compiled there, but until it is fully verified on Linux runners,
CI runs tests with the `no_avx2` feature (scalar path only). Local builds still
use AVX2 by default on x86_64. Apple Silicon (arm64) always uses the scalar path.

```bash
cargo test --lib
cargo test --lib --features bwt_libsais   # optional SA backend
cargo test --lib --features no_avx2       # CI-like scalar path
rcn self-test                             # wraps cargo test --lib
```

Pre-commit: `.pre-commit-config.yaml` runs `cargo build` + `cargo test` (mirrors CI).

---

## Speed roadmap (2026-09)

1. **SoA weight layout** — **Completed**.
2. **Stretch-value reuse** — **Completed**.
3. **Wider stride / 16k banks** — **Completed** (16384 banks, 14-bit hash).
4. **Parallel BWT trials** — intra-block `rayon::join`. **Partial (S5)**.
5. **S8 — Interleaved rANS re-bench** — **Completed** (wired; see README fast-path table).
6. **S9 — Optional libsais BWT** — `--features bwt_libsais`. **Completed**.
7. **S10 — Classify-ahead pipeline** — **Completed**.
8. **mimalloc** — **Completed**.

## Ratio research directions

**No open numbered ratio tickets** — C6–C9 are closed (IndirectModel stays
in-tree, not in the default Text stack). Ticket tables:
[OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md).

**North star:** beat `zstd -19` on text + mixed corpora while keeping `nci`/`mr`
wins. `FSE` is a secondary reference; speed remains secondary.

Concrete research angles (not scheduled tickets):

- New Text models beyond the current 8-model stack (full DMC previously hurt;
  Indirect shelved after dickens regression)
- Stronger text transforms / match modeling — not another XWRT vocab bump
- Finish the slow 5-file re-bench (webster/nci + mr post-C7 soften) so gaps vs
  `-19` are honest before claiming further wins

---

## DP-optimal LZP parse (historical measure)

DP optimal LZP parse runs a forward LZP match pre-pass and emits `(len, dist)`
records for matches ≥ 16 bytes (Text now gates at ≥12). Matched bytes are
skipped in the rANS stream — only literals are CM-encoded. The match side-stream
uses varint-encoded records (delta_pos + len + dist), optionally entropy-coded (C9).

| file | orig (KB) | rcn ratio% | vs default | zstd -1 ratio% | beats zstd-1? |
|------|----------:|-----------:|-----------:|---------------:|:------------:|
| dickens | 9953.6 | 41.9 | 46.2→41.9 (**−4.3pt**) | 41.7 | ~parity |
| webster (10MB) | 10000.0 | 31.4 | 35.1→31.4 (**−3.7pt**) | 33.0 | ✅ |
| nci | 32767.0 | 8.2 | 9.0→8.2 (**−0.8pt**) | 85.2 | ✅ |
| mr | 9736.9 | 29.1 | 27.5→29.1 (**+1.6pt**) | 38.3 | ✅ (vs default) |
| json | 478.5 | 0.1 | 0.1 (same) | 0.3 | ✅ |
| huge_json | 5641.7 | 1.3 | 2.7→1.3 (**−1.4pt**) | 2.5 | ✅ |
| massive_json | 22885.9 | 0.97 | 0.81→0.97 (**+0.16pt**) | 2.48 | ✅ |

**Key insight:** DP optimal parse with literal-skipping rANS is a net win on most
files. `mr` and `massive_json` regressed slightly when SSM was co-introduced; DP
is default and SSM is isolated from DP.

---

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
| **Context-selected 8k mixer banks** (8192 per-context LogisticMixer instances selected by byte-class + order-1/order-2 + word-hash, blended with global + master) | mr, dickens, json, webster, nci | **improved**: json 3.9%→3.0%, dickens 51.7%→51.2% | **kept as default** — later expanded to 16k banks (C8) |
| **Indirect context + DMC models** | mr, dickens, json, webster, nci | regressed dickens +0.7pt, webster +0.4pt; json improved | reverted; IndirectModel remains in-tree, not default |
| **Cross-block persistence + real 4MB LDM window** | json, mr, dickens, nci, webster | **improved**: json 5.5%→3.9%, mr 28.6%→27.3%, dickens 56.0%→51.7%, nci 26.6%→20.9%, webster 50.4%→45.1% | **kept as default** |
| LZP ring buffer performance fix (O(n) drain→O(1) ring) | all files | performance fix, no ratio change | kept |
| Micro SSM mixer (16-dim recurrent state replacing logistic mixer) | json, mr | **regressed**: json 3.9%→10.7%, mr 27.3%→37.2% | reverted; SSM too large for 64KB blocks, gradient issues |
| Second-order mixer training (Adam + per-model lr_scale) | dickens, mr, json | **neutral** (Adam) / **neutral** (SGD + lr_scale) | kept as default; Adam never measurably better on default stacks |
| BWT text trial (RawCm vs BWT→MTF→RLE0→CM vs LZP→BWT→MTF→CM) | dickens, json, webster | **improved**: json 3.0%→0.1%, dickens 51.2%→46.2%, **webster 50.4%→35.1%** | **kept as default** |
| JSON stream splitting + per-stream pipeline selection | json (478KB–22MB) | **improved**; beats zstd-1 and zstd-19 on large JSON | **kept as default** |
| Order-8 PPMd with SEE + sparse de Bruijn | webster, dickens, json | mixed; config now matches hybrid_ppm3 | **promoted to default** |
| DP optimal LZP parse (now default) | dickens, webster, nci, mr, json, huge_json, massive_json | **improved** on 5/7 (dickens −4.3pt, webster −3.7pt); **regressed** mr +1.6pt, massive_json +0.16pt | **kept as default** — SSM isolated to avoid regression |
| **Exec E8E9 transform** | Exec executables | Converts x86 relative offsets to absolute (3-5pt on Exec corpora) | **kept as default** |
| **XWRT dictionary before BWT** | Text blocks | Build top 2k words per block, replace with tokens, then BWT→MTF→RLE0→CM | **per-block superseded by global** |
| **Global XWRT dictionary** | dickens, webster, nci, mr, json | Top-128 then C6 top-512 with ESC; measured C5 dickens −4.6pt, webster −4.0pt; C6 ≈−1.1pt dickens vs HEAD | **kept as default** |
| **SSE/APM/APM2 cascade** | mr, dickens, json, webster, nci | **improved all 5** | **wired into codec** |
| **AVX2 SIMD walk_dist** | All | 5-10x fast path speedup, zero ratio loss | **completed** |

### Speed passes

| pass | description | files tested | result | status |
|---|---|---|---|---|
| #1 | LazyLzp removal (O(n²) memmove per byte over 1MB; match-extension loop dead). Model rewritten with fixed-capacity ring buffer + causal extension loop. | dickens, webster | **3× encode speedup on text, zero ratio change** (dickens 2MB: 10.9s→3.3s, byte-identical) | removed from all default stacks |
| #2 | Byte-level "fast" path (`--mode fast`): PPM-style single-context count coder (deterministic order-0/1/2 selector + fused 256-symbol cumulative walk_dist + byte rANS). No mixer/softmax. | dickens 2MB | **2.6× encode speedup vs slow with ~1.7× better ratio on BWT+MTF streams** | kept as `--mode fast` |
| #3 | Single-pass acc-merge across the mixer chain + Q16 fixed-point mixer | dickens 2MB (slow) | **~10% encode speedup, ratio flat** | kept as default |
| #4 | 32-way interleaved byte rANS | dickens 2MB (fast) | **no measurable speedup** while serial `walk_dist` dominated; later wired + re-benched after AVX2 `walk_dist` | **wired into the fast path** |

### Code-quality / correctness notes

| experiment | files tested | result | action |
|---|---|---|---|
| round-trip verification | all 5 | lossless | every pass round-trip verified |
| test suite | all | 143/143 green | kept |
| bit-identical output | dickens 2MB | each pass `cmp`-identical to prior where claimed | kept |
