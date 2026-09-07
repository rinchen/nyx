# Nyx Optimization Opportunity Log

Last updated: 2026-09-06
Test status: 127/127 passing

---

## Compression Big Tickets (Ratio Improvements)

| # | Ticket | Description | Expected Gain | Status |
|---|--------|-------------|---------------|--------|
| C1 | Wire SSE/APM/APM2 cascade | `SseApmCascade` exists in `src/model/sse_apm.rs` but not integrated. Measured: nci -0.9pt, mr -0.8pt, dickens -0.3pt, webster -0.5pt, json -0.1pt | -2-4pt on text files | Not started |
| C2 | Re-benchmark fixed PPMd config | Order-8 PPMd with SEE + sparse de Bruijn had broken config (dropped WordModel + LazyLzp). Config now matches hybrid_ppm3, tables shrunk 22→18 bits. Previous +0.8pt webster was with broken config. | +1.5-2pt on webster | Not started |
| C3 | XWRT dictionary before BWT | Build top 2k words per Text block, replace with 0x80+id + cap bits, then BWT→MTF→RLE0→CM. How cmix gets text wins. | 2-4pt on dickens/webster | Not started |
| C4 | Exec/Binary transforms | E8E9 for Exec (3-5pt on Exec corpora), delta/stride for Binary (2-4pt) | 2-5pt on mr/nci | Not started |

---

## Speed Big Tickets (Throughput Improvements)

| # | Ticket | Description | Expected Gain | Status |
|---|--------|-------------|---------------|--------|
| S1 | SIMD-accelerated `walk_dist` | Linear 256-probability scan in `ByteCountModel::walk_dist` is fast-path bottleneck (~90% of encode+decode time). AVX2/AVX-512 cumulative counts. Also applicable to slow path. | 5-10x on fast path | Not started |
| S2 | SoA weight layout for 4096 banks | Contiguous weight arrays instead of per-bank `Vec`, single cache line fetch. Bit-identical, ratio-neutral. | 10-20% slow path | Not started |
| S3 | Stretch-value reuse | Carry stretch bucket lookups through `MixerAcc` to avoid ~11 table re-lookups per bit. Bit-identical, ratio-neutral. | Small | Not started |
| S4 | Wider stride / context model | Increase number of models or order-2 context size. | Unknown | Not started |
| S5 | Parallel blocks | Clone decayed state per rayon thread - near-linear speedup on webster 40MB. | Near-linear | Not started |
| S6 | Faster BWT (libsais SA-IS) | Replace rotation-based doubled string filter SA with SA-IS O(n). | 5-10x BWT trial | Not started |
| S7 | Allocator (mimalloc) | BWT trial does many Vec allocations. | Small | Not started |

---

## Current Stack (What Works)

### Slow Path (`--mode slow`, default)
- 8-9 bit models + two-level 4k-bank mixer hierarchy + master mixer
- Cross-block weight decay (0.995)
- 4MB LZP window for Text blocks
- BWT text trial (3 paths: RawCM / BWT+MTF+RLE0+CM / LZP+BWT+MTF+CM)
- JSON stream splitting (4 streams → 4× BWT → CM)
- rANS bit coder (ans crate)
- DP optimal LZP parse (behind `two_pass` feature)

### Fast Path (`--mode fast`)
- Orders 0-2 count models + byte rANS
- Deterministic order selection
- 32-way interleaved byte rANS (in-tree, not wired out)
- PPM-style cumulative walk_dist
- Classifier-aware method bytes

---

## Recommended Next Step (User Suggestion)

> "If you want one big win: do **#1 SIMD walk_dist + wire SSE/APM**. walk_dist fix makes interleaved rANS actually matter, and SSE/APM gets you webster 35.1%→~33% beating zstd -1 without touching slow path."

---

## Test Log

| Date | Commit | Tests | Notes |
|------|--------|-------|-------|
| 2026-09-06 | 740f11a | 127/127 pass | All tests green |

---