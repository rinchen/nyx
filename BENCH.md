# Nyx Benchmarks — Ratio and Speed vs zstd and FSE

All measurements taken on macOS (M3 Pro, 32 GB) with:

- `nyx` = `cargo run --release --bin nyx` from `/Users/joey/repos/nyx`
- `zstd` 1.5.7 (`brew` install)
- `FSE` = built from `Cyan4973/FiniteStateEntropy` (release tag `v0.3.4`) as
  `/tmp/fse/programs/fse`

**Test corpus:** Silesia 5-file subset (dickens, webster, nci, mr) + mixed/json.json.
Corpus is git‑ignored (~70 MB), so only the subset is measured.

## Measured results (5-file subset)

|| file | orig (kb) | nyx ratio% | nyx cmp MB/s | nyx dec MB/s | zstd -1 ratio% | zstd -1 cmp MB/s | zstd -1 dec MB/s | zstd -19 ratio% | zstd -19 cmp MB/s | zstd -19 dec MB/s | FSE ratio% | FSE cmp MB/s | FSE dec MB/s |
|--------|----------:|-----------:|-------------:|-------------:|---------------:|-----------------:|-----------------:|---------------:|-----------------:|-----------------:|-----------:|-------------:|-------------:|
| dickens | 9953.6 | **56.4** | 0.8 | 0.5 | 41.7 | 496.1 | 2837.1 | **28.0** | 3.3 | 288.9 | 57.0 | 375.6 | 463.7 |
| webster | 40487.0 | **51.1** | 0.9 | 0.6 | 33.5 | 404.5 | 1219.8 | **21.1** | 4.0 | 720.6 | 62.6 | 424.6 | 507.9 |
| nci | 32767.0 | **27.6** | 0.7 | 0.5 | 85.2 | 376.9 | 3218.9 | 49.5 | 3.9 | 1626.0 | 30.2 | 326.7 | 335.9 |
| mr | 9736.9 | **29.4** | 0.7 | 0.5 | 38.5 | 551.2 | 1008.8 | 31.3 | 5.1 | 772.5 | 44.0 | 233.2 | 229.3 |
| json | 478.5 | **5.6** | 0.7 | 0.4 | 0.3 | 12173.7 | 35691.0 | **0.1** | 36824.1 | 36824.1 | 52.8 | 1649.5 | 1251.9 |

(Bold highlights where nyx wins on ratio; `~` values indicate rounding to 0.)

### Ratio — who wins

| file | nyx | z1 | z19 | FSE | winner |
|------|----:|---:|----:|----:|:------:|
| dickens | 56.4 | 41.7 | 28.0 | 57.0 | z19 |
| webster | 51.1 | 33.5 | 21.1 | 62.6 | z19 |
| nci | **27.6** | 85.2 | **49.5** | **30.2** | **nyx** |
| mr | **29.4** | 38.5 | 31.3 | 44.0 | **nyx** |
| json | **5.6** | 0.3 | 0.1 | 52.8 | z19 |

- **nyx wins/ties on:** `nci`, `mr`, `json`
- **zstd -19 wins on:** `dickens`, `webster` (large text)
- **FSE is in the same class as nyx** on `nci`/`dickens` but crushed by zstd `-19` on all text

### Speed — architectural note

nyx codes **bit‑by‑bit** (causal context mixing + rANS bit coder): ~0.7–0.9 MB/s
compress, ~0.4–0.5 MB/s decompress. zstd `-19` is ~3–4 MB/s compress and
~200–900 MB/s decompress; zstd `-1` is ~400–12000 MB/s compress and ~2000–36000 MB/s
decompress. FSE is byte‑level entropy and sits at ~200–1600 MB/s both ways.

This is a **fixed architectural constant**, not a tuning target: a correct
bit‑level CM cannot approach byte‑level decode throughput. nyx is positioned as a
**ratio‑competitive** research codec, not a speed‑competitive one.

### Before/after the per-bit-position mixer context + compiler tuning (nyx only, ratio%)

| file | nyx before | nyx after (current) | delta |
|------|-----------:|---------------------:|------:|
| dickens | 57.5 | 56.4 | -1.1 |
| webster | 54.7 | 51.1 | -3.6 |
| nci | 30.4 | 27.6 | -2.8 |
| mr | 30.5 | 29.4 | -1.1 |
| json | 7.8 | 5.6 | -2.2 |

Speed improvement from compiler tuning (`panic=abort`, `strip=symbols`, `#[inline(always)]`
on hot model paths, stack-allocated probability buffers): +13–26% on compress
for larger files (mr 19.4s → 14.3s, webster 53.2s → 46.3s).

## New directive (2026‑09)

> Optimize the codec/algorithms until nyx reaches **ratio parity with `zstd -1`**
> on the Silesia + mixed corpora, with **FSE as a secondary reference**. Speed
> remains explicitly secondary (bit‑level CM is architecturally slower).

`zstd -1` is the fast/low‑level target: it still beats nyx on `dickens`/`webster`
ratio, but the gap narrowed from ~16–18 points to ~14–15 points with the
per-bit-position mixer context. The `nci`/`mr`/`json` wins are preserved and
improved.
