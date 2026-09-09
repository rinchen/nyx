# Rcn Optimization Opportunity Log

Last updated: 2026-09-09
Test status: 143/143 passing

---

## Compression Big Tickets (Ratio Improvements)

| # | Ticket | Description | Expected Gain | Status |
|---|--------|-------------|---------------|--------|
| C1 | Wire SSE/APM/APM2 cascade | Integrated in slow-path encoder/decoder. | -2-4pt on text | Completed |
| C2 | Re-benchmark fixed PPMd config | Order-8 PPMd with SEE + sparse de Bruijn. | +1.5-2pt on webster | Completed |
| C3 | XWRT dictionary before BWT | Per-block XWRT before BWT→MTF→RLE0→CM. | 2-4pt on text | Completed |
| C4 | Exec/Binary transforms | DP-LZP default; E8E9; delta/stride. | 2-5pt on mr/nci | Completed |
| C5 | Corpus-wide global XWRT-128 | Top-128 dict once in header. | −4-5pt on text | Completed |
| C6 | Global XWRT 128 → 512 with ESC | `0x80..=0xFE` ids 0..126; `0xFF\|\|u16` ids 127..511; ESC only for words len>3; header count `u16`. Measured dickens ~41.3%→~40.2% (−1.1pt) vs HEAD on this machine. | −1 to −2pt on text | Completed |
| C7 | Adaptive DP LZP threshold | Text ≥12; Binary/Exec stay 16 (≥24 regressed mr). | Help text; avoid mr hit | Completed |
| C8 | 16k banks + indirect | `NUM_MIXERS=16384` (14-bit). IndirectModel tried; **not in default Text stack** (prior regression). | Banks kept; indirect shelved | Completed (banks); indirect in-tree only |
| C9 | FSE-family match side-stream | Order-0 byte rANS on varint blob when smaller. | ∼0.5–1pt | Completed |

---

## Speed Big Tickets (Throughput Improvements)

| # | Ticket | Description | Expected Gain | Status |
|---|--------|-------------|---------------|--------|
| S1 | SIMD `walk_dist` | AVX2 cumulative counts. | 5-10x fast path | Completed |
| S2 | SoA weight layout | Contiguous bank weights. | 10-20% slow | Completed |
| S3 | Stretch-value reuse | Precompute stretch in mix/update. | Small | Completed |
| S4 | Wider stride / context | More models / context bits. | Known gain | Completed |
| S5 | Parallel BWT trials | `rayon::join` path B/C. | Modest | Partial |
| S6 / S9 | libsais BWT backend | Optional `bwt_libsais` feature (`libsais-rs`); default `divsufsort`. | BWT trial speed | Completed (optional) |
| S7 | mimalloc | Allocator. | Small | Completed |
| S8 | Interleaved rANS re-bench | Already wired; re-benched after AVX2 walk_dist. | Confirm MB/s | Completed |
| S10 | Classify-ahead pipeline | `rayon::join` classify N+1 while encode N. | Overlap classify | Completed |

---

## Current Stack (What Works)

### Slow Path (`--mode slow`, default)
- 8 bit models (Text) + two-level **16k**-bank mixer hierarchy + master mixer
- SSE/APM/APM2 cascade; cross-block decay 0.995; 32MB LZP window
- BWT text trial + global XWRT-512 (ESC) + DP-LZP with adaptive thresholds
- Match side-stream: varint + optional order-0 rANS (C9)
- Classify-ahead Rayon overlap (S10)

### Fast Path (`--mode fast`)
- Orders 0-2 count models + **wired** 32-way interleaved byte rANS
- AVX2 SIMD walk_dist

---

## Recommended Next Step

Re-bench the full 5-file slow subset after each ratio ticket; chase remaining gap to `zstd -19` on webster/dickens. Optional: re-try a lighter indirect context now that SoA/16k banks are in place.

---

## Test Log

| Date | Notes |
|------|-------|
| 2026-09-08 | C5 global XWRT-128 |
| 2026-09-09 | Docs truth; C6–C9 / S8–S10; 143 tests; dickens HEAD 41.3% → C6 stack ~40.2% |
