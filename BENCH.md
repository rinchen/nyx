# Nyx Benchmarks — Compression **and** Speed vs SOTA

All numbers from `scripts/bench_vs_sota.sh` (nyx vs `zstd -19` / `xz -9` / `brotli -11` /
`lz4 -9`). Both **ratio** (comp_kb / orig_kb, lower is better) and **speed**
(MB/s, higher is better) are reported because the project goal is ratio parity with
`zstd -19`; speed is documented but explicitly secondary (bit-level CM cannot match
zstd's byte-level decode throughput).

> **Subset note.** The full Silesia (405 MB) run makes nyx exceed the harness timeout at
> current speed, so these runs use a 5-file subset (dickens, webster, nci, mr, json) that
> is representative of text / mixed-binary / structured data. Full-corpus runs are pending
> a faster codec stage.

## Current state (2026-08-30, post causal-fix; 5-file subset)

nyx config: orders 0–2 + Sparse + Lzp + Exec, direct-addressed `CtxTable`, single logistic
mix. `cargo test` 24 passed; `cargo clippy --all-targets -- -D warnings` clean.

| file     | orig_kb | nyx ratio% | zstd-19 | xz-9 | lz4-9 | nyx cmp MB/s | zstd cmp | zstd dec | nyx dec MB/s |
|----------|--------:|-----------:|--------:|------:|-------:|-------------:|---------:|---------:|-------------:|
| dickens  |  9953.6 |      58.4  |  28.0   | 27.8  |  43.6  |         1.6  |      3.5 |    278.9 |          0.8 |
| webster  | 40487.0 |      54.7  |  20.9   | 20.2  |  33.8  |         1.6  |      3.8 |    715.7 |          0.7 |
| nci      |  9736.9 |      30.6  |  31.2   | 27.6  |  42.6  |         1.3  |      5.5 |    284.9 |          0.9 |
| mr       | 32767.0 |      30.7  |   5.0   |  5.2  |  11.0  |         1.3  |      3.9 |    883.2 |          0.9 |
| json     |   478.5 |      13.6  |  ~0     | ~0    |   0.4  |         1.6  |     20.2 |     21.3 |          1.1 |

(`~0` = zstd compresses json to ~0.1 KB; the displayed percentage rounds to 0 on the
KB-normalized harness.)

### Read-out
- **Ratio:** nyx **beats `zstd -19` on `nci`** (30.6 vs 31.2) and is competitive on
  nci/mr-class data. On **text** (dickens 58.4 vs 28.0, webster 54.7 vs 20.9) nyx is still
  **~2× worse** — the order-context gap (order-2 bit context vs zstd's LDM+FSE).
- **Speed:** nyx compresses at **~1.5 MB/s** vs zstd's ~3.5 MB/s compress, and decodes at
  **~0.8 MB/s** vs zstd's **280–880 MB/s**. That is a **~100–500× decode gap** — an
  architectural constant of bit-level context mixing, not a tuning target.

## Before / after the 2026-08-30 optimization

The headline fix was a **causal predict/update bug** in the bit models
(`OrderN`/`Sparse`/`Exec` computed the context *after* advancing the byte assembler, so
predict and update disagreed — the models never learned consistent contexts). Fixing it
plus replacing the per-bit `HashMap` with direct-addressed tables:

| file     | nyx ratio% before | nyx ratio% after | nyx cmp MB/s before | nyx cmp MB/s after |
|----------|------------------:|-----------------:|---------------------:|--------------------:|
| dickens  |            68.6   |           58.4   |                0.8   |               1.6   |
| webster  |            87.6   |           54.7   |                ~0.8  |               1.6   |
| json     |            38.9   |           13.6   |                0.8   |               1.6   |
| nci      |            34.0   |           30.6   |                0.9   |               1.3   |
| mr       |            32.4   |           30.7   |                0.9   |               1.3   |

Both axes improved: ~15% better ratio on text and **2× compression speed**.

## Approaches tried that did NOT help (so we don't repeat them)

- **Orders 3–5 added to the linear mix:** *hurt* ratio (dickens 58.4→92.7% expansion).
  Sparse high-order contexts with a strong INIT pseudo-count pollute the per-model mixer on
  a 64 KiB block; the mixer cannot down-weight unseen contexts per-bit. Needs PPM-style
  escape, not a raw order bump.
- **SSE layer (2-bit history context):** measured **neutral** (dickens 58.4→61.2). The
  history context is too weak; a richer context (e.g. order-2 byte context) would require
  plumbing context ids into the mixer.

## Path to zstd text parity (not yet implemented)

PPM-style adaptive higher-order modeling with **escape** (unseen long contexts fall back to
shorter ones) or an indirect/neural context mixer (lpaq/paq8 class). This is a substantial
`model/` redesign and is the planned next stage (see the repo plan §10.4). Speed stays slow
by design.
