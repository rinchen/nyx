# Rcn Optimization Opportunity Log

Last updated: 2026-09-09
Test status: 162/162 passing

Human-readable TODO, experiment log, and CI notes: [DEVELOPMENT.md](DEVELOPMENT.md).
Product overview and headline benches: [README.md](README.md).

---

## External proposal triage (2026-09)

Ideas from an external list that **collide with closed ticket IDs**. Status vs this tree:

| Proposal idea | Verdict |
|---------------|---------|
| S8 interleaved rANS after SIMD walk_dist | **Done** — already wired in `bytecodec` |
| S9 libsais BWT | **Done** — `bwt_libsais` → `libsais-rs` (default still `divsufsort` O(n)) |
| S10 prefetch + stretch-in-`MixerAcc` | **Done as S11** — stretch LUT on `MixerAcc`; `_mm_prefetch` next bank |
| S11 bumpalo / per-block bank arena | **Rejected** — banks are flat SoA + decay in place, not 8192 `Vec`s/block |
| S12 parallel BWT trials | **Done as S12** — `parallel_map_sizes` covers RawCm / XWRT / JSON / CSV / XML trials |
| C6 XWRT-512 ESC | **Done** |
| C7 classifier DP thresholds | **Done** (Text 12; Binary/Exec 16 — ≥24 hurt mr) |
| C8 16k + Indirect default | **Banks done; Indirect shelved** |
| C9 Order-12 PPMd Text | **Tried as C10 — reverted** (gain ≪ 0.3pt) |
| C10 FSE side-stream | **Done** (repo C9) |
| C11 CSV/XML stream split | **Done as C11** — detectors + BWT trial wiring |

---

## Compression Big Tickets (Ratio Improvements)

| # | Ticket | Description | Expected Gain | Status |
|---|--------|-------------|---------------|--------|
| C1 | Wire SSE/APM/APM2 cascade | Integrated in slow-path encoder/decoder. | -2-4pt on text | Completed |
| C2 | Re-benchmark fixed PPMd config | Order-8 PPMd with SEE + sparse de Bruijn. | +1.5-2pt on webster | Completed |
| C3 | XWRT dictionary before BWT | Per-block XWRT before BWT→MTF→RLE0→CM. | 2-4pt on text | Completed |
| C4 | Exec/Binary transforms | DP-LZP default; E8E9; delta/stride. | 2-5pt on mr/nci | Completed |
| C5 | Corpus-wide global XWRT-128 | Top-128 dict once in header. | −4-5pt on text | Completed |
| C6 | Global XWRT 128 → 512 with ESC | `0x80..=0xFE` ids 0..126; `0xFF\|\|u16` ids 127..511; ESC only for words len>3. | −1 to −2pt on text | Completed |
| C7 | Adaptive DP LZP threshold | Text ≥12; Binary/Exec stay 16 (≥24 regressed mr). | Help text; avoid mr hit | Completed |
| C8 | 16k banks + indirect | `NUM_MIXERS=16384`. Indirect **not** in default Text stack. | Banks kept; indirect shelved | Completed (banks) |
| C9 | FSE-family match side-stream | Order-0 byte rANS on varint blob when smaller. | ∼0.5–1pt | Completed |
| C10 | Order-12 PPMd for Text only | `with_max_order(12)` API kept; default Text stays order-8. | −0.5pt text | **Reverted** — 2MB dickens −0.003pt |
| C11 | CSV/XML stream splitting | Column / tag-attr-text splits like JSON; methods 15–18. | Structured-data ratio | Completed |

---

## Speed Big Tickets (Throughput Improvements)

| # | Ticket | Description | Expected Gain | Status |
|---|--------|-------------|---------------|--------|
| S1 | SIMD `walk_dist` | AVX2 cumulative counts. | 5-10x fast path | Completed |
| S2 | SoA weight layout | Contiguous bank weights. | 10-20% slow | Completed |
| S3 | Stretch-value reuse | Precompute stretch in mix/update (within-call). | Small | Completed |
| S4 | Wider stride / context | More models / context bits. | Known gain | Completed |
| S5 | Parallel BWT trials | `rayon::join` path B/C. | Modest | Partial → see S12 |
| S6 / S9 | libsais BWT backend | Optional `bwt_libsais` (`libsais-rs`); default `divsufsort`. | BWT trial speed | Completed (optional) |
| S7 | mimalloc | Allocator. | Small | Completed |
| S8 | Interleaved rANS re-bench | Wired; re-benched after AVX2 walk_dist. | Confirm MB/s | Completed |
| S10 | Classify-ahead pipeline | `rayon::join` classify N+1 while encode N. | Overlap classify | Completed |
| S11 | Prefetch + stretch on `MixerAcc` | Carry stretch LUT across mix→update; prefetch next bank weights. | Ratio-neutral speed | Completed |
| S12 | Wider BWT trial parallelism | `parallel_map_sizes` for remaining size trials. | Text trial wall time | Completed |

---

## Current Stack (What Works)

### Slow Path (`--mode slow`, default)
- 8 bit models (Text) + two-level **16k**-bank mixer hierarchy + master mixer
- SSE/APM/APM2 cascade; cross-block decay 0.995; 32MB LZP window
- BWT text trial (incl. CSV/XML split) + global XWRT-512 (ESC) + DP-LZP with adaptive thresholds
- Match side-stream: varint + optional order-0 rANS (C9)
- Classify-ahead Rayon overlap (S10); stretch-on-acc + bank prefetch (S11); wider BWT trial fan-out (S12)

### Fast Path (`--mode fast`)
- Orders 0-2 count models + **wired** 32-way interleaved byte rANS
- AVX2 SIMD walk_dist

---

## Recommended Next Step

1. Optional: re-bench dickens/json slow rows for a fully same-day 5-file set.
2. Profile slow path (flamegraph) if chasing 20+ MB/s; S11 prefetch is soft-hint only.
3. Real CSV/XML corpora A/B for C11 ratio claims (synthetic round-trips already green).

Do **not** re-default Indirect, re-open Binary DP ≥24, or re-bump Text PPMd order without a ≥0.3pt measure. Experiment log: [DEVELOPMENT.md](DEVELOPMENT.md).

---

## Test Log

| Date | Notes |
|------|-------|
| 2026-09-08 | C5 global XWRT-128 |
| 2026-09-09 | Docs truth; C6–C9 / S8–S10; triage → S11/S12/C11 done; C10 reverted (−0.003pt) |
