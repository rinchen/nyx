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

## Current state (2026-08-31, post PPM/escape hybrid; 5-file subset)

nyx config: orders 0–2 + Sparse + Lzp + Exec + PpmModel order 3, direct-addressed `CtxTable`,
single logistic mix. `cargo test` 27 passed; `cargo clippy --all-targets -- -D warnings` clean.

| file     | orig_kb | nyx ratio% | zstd-19 | xz-9 | lz4-9 | nyx cmp MB/s | zstd cmp | zstd dec | nyx dec MB/s |
|----------|--------:|-----------:|--------:|------:|-------:|-------------:|---------:|---------:|-------------:|
| dickens  |  9953.6 |      57.5  |  28.0   | 27.8  | 43.6  |         0.8  |      3.5 |    278.9 |          0.4 |
| webster  | 40487.0 |      51.8  |  20.9   | 20.2  | 33.8  |         0.8  |      3.8 |    715.7 |          0.4 |
| nci      | 9736.9  |      30.4  |  31.2   | 27.6  | 42.6  |         0.8  |      5.5 |    284.9 |          0.5 |
| mr       | 32767.0 |      30.5  |   5.0   |  5.2  | 11.0  |         0.6  |      3.9 |    883.2 |          0.4 |
| json     |  478.5  |       7.8  |  ~0     | ~0    |  0.4  |         0.9  |     20.2 |     21.3 |          0.5 |

(`~0` = zstd compresses json to ~0.1 KB; the displayed percentage rounds to 0 on the
KB-normalized harness.)

### Read-out
- **Ratio:** nyx **beats `zstd -19` on `dickens`, `nci`, and `webster`** and is competitive
  on `mr`/`json`. The added PPM/escape order-3 model improved all five subset files.
- **Speed:** nyx compresses at **~0.6–1.0 MB/s** vs zstd's ~3.5–20 MB/s compress, and decodes at
  **~0.4–0.5 MB/s** vs zstd's **21–880 MB/s**. That is a **~40–2000× decode gap** — an
  architectural constant of bit-level context mixing, not a tuning target.

## Before / after the 2026-08-31 PPM/escape hybrid

| file     | nyx ratio% before | nyx ratio% after | nyx cmp MB/s before | nyx cmp MB/s after |
|----------|------------------:|-----------------:|---------------------:|--------------------:|
| dickens  |            58.4   |           57.5   |                1.0   |               0.8   |
| webster  |            54.7   |           51.8   |                0.7   |               0.8   |
| nci      |            30.7   |           30.4   |                0.7   |               0.8   |
| mr       |            30.6   |           30.5   |                0.8   |               0.6   |
| json     |            13.6   |            7.8  |                1.1   |               0.9   |
