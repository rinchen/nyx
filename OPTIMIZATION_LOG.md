# Nyx Optimization Opportunity Log

Last updated: 2026-09-07
Test status: 135/135 passing

---

## Compression Big Tickets (Ratio Improvements)

| # | Ticket | Description | Expected Gain | Status |
|---|--------|-------------|---------------|--------|
| C1 | Wire SSE/APM/APM2 cascade | `SseApmCascade` exists in `src/model/sse_apm.rs` and is integrated in slow-path encoder/decoder. Measured: nci -0.9pt, mr -0.8pt, dickens -0.3pt, webster -0.5pt, json -0.1pt | -2-4pt on text files | Completed |
| C2 | Re-benchmark fixed PPMd config | Order-8 PPMd with SEE + sparse de Bruijn (CTX_BITS=18, matching hybrid_ppm3). Previous +0.8pt webster was with broken config. | +1.5-2pt on webster | Completed |
| C3 | XWRT dictionary before BWT | Build top 2k words per Text block, replace with 0x80+id + cap bits, then BWT→MTF→RLE0→CM. How cmix gets text wins. | 2-4pt on dickens/webster | Completed |
| C4 | Exec/Binary transforms | DP-optimal LZP parse promoted to default (isolated from SSM); E8E9 for Exec | 2-5pt on mr/nci | DP default + E8E9 done |

---

## Speed Big Tickets (Throughput Improvements)

| # | Ticket | Description | Expected Gain | Status |
|---|--------|-------------|---------------|--------|
| S1 | SIMD-accelerated `walk_dist` | Linear 256-probability scan in `ByteCountModel::walk_dist` is fast-path bottleneck (~90% of encode+decode time). AVX2/AVX-512 cumulative counts. Also applicable to slow path. | 5-10x on fast path | Completed |
| S2 | SoA weight layout for 8192 banks | Contiguous weight arrays instead of per-bank `Vec`, single cache line fetch. Bit-identical, ratio-neutral. | 10-20% slow path | Completed |
| S3 | Stretch-value reuse | Precompute stretch values in `mix_acc`/`update_from_acc` to avoid 11 table re-lookups per bit. | Small improvement | Completed |
| S4 | Wider stride / context model | Increase number of models or order-2 context size. | Known gain | Completed |
| S5 | Parallel blocks | Clone decayed state per rayon thread - near-linear speedup on webster 40MB. | Near-linear | Completed |
| S6 | Faster BWT (libsais SA-IS) | Replace rotation-based doubled string filter SA with SA-IS O(n). | 5-10x BWT trial | Deferred |
| S7 | Allocator (mimalloc) | BWT trial does many Vec allocations. | Small | Completed |

---

## Current Stack (What Works)

### Slow Path (`--mode slow`, default)
- 8-9 bit models + two-level 8k-bank mixer hierarchy + master mixer
- SSE/APM/APM2 cascade refinement
- Cross-block weight decay (0.995)
- 32MB LZP window for Text blocks
- BWT text trial (5 paths: RawCM / BWT+MTF+RLE0+CM / LZP+BWT+MTF+CM / JSON split / XWRT dict+BWT)
- Exec E8E9 transform (x86 relative → absolute offsets)
- rANS bit coder (ans crate)
- DP-optimal LZP parse (default) — runs forward LZP match pre-pass, emits (len, dist) records for matches ≥ 16 bytes, skips matched bytes in rANS stream

### Fast Path (`--mode fast`)
- Orders 0-2 count models + byte rANS
- Deterministic order selection
- 32-way interleaved byte rANS (in-tree, not wired out)
- PPM-style cumulative walk_dist (AVX2 SIMD-accelerated)
- Classifier-aware method bytes

---

## Recommended Next Step (User Suggestion)

> "If you want one big win: do **#1 SIMD walk_dist + wire SSE/APM**. walk_dist fix makes interleaved rANS actually matter, and SSE/APM gets you webster 35.1%→~33% beating zstd -1 without touching slow path."

---

## Test Log

| Date | Commit | Tests | Notes |
|------|--------|-------|-------|
| 2026-09-06 | 740f11a | 127/127 pass | All tests green |
| 2026-09-07 | 6b7bc3c | 132/132 pass | XWRT dict + E8E9 + SIMD walk_dist done |
| 2026-09-07 | (this session) | 135/135 pass | DP-optimal LZP promoted to default |

---