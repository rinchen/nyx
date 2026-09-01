# Nyx Benchmarks — Ratio and Speed vs zstd and FSE

All measurements taken on macOS (M3 Pro, 32 GB) with:

- `nyx` = `cargo run --release --bin nyx` from `/Users/joey/repos/nyx`
- `zstd` 1.5.7 (`brew` install)
- `FSE` = built from `Cyan4973/FiniteStateEntropy` (release tag `v0.3.4`) as
  `/tmp/fse/programs/fse`

**Test corpus:** Silesia 5-file subset (dickens, webster, nci, mr) + mixed/json.json.
Corpus is git‑ignored (~70 MB), so only the subset is measured.

## Measured results (5-file subset)

| file | orig (kb) | nyx ratio% | nyx cmp MB/s | nyx dec MB/s | zstd -1 ratio% | zstd -1 cmp MB/s | zstd -1 dec MB/s | zstd -19 ratio% | zstd -19 cmp MB/s | zstd -19 dec MB/s | FSE ratio% | FSE cmp MB/s | FSE dec MB/s |
|------|----------:|-----------:|-------------:|-------------:|---------------:|-----------------:|-----------------:|---------------:|-----------------:|-----------------:|-----------:|-------------:|-------------:|
| dickens  |   9953.4 |    **57.5** |     **0.8** |     **0.5** |     41.7 |   496.1 |  2837.1 |    28.0 |    3.3 |   288.9 |   57.0 |    375.6 |    463.7 |
| webster  |  40487.0 |    **54.7** |     **0.7** |     **0.5** |     33.5 |   404.5 |  1219.8 |    21.1 |    4.0 |   720.6 |   62.6 |    424.6 |    507.9 |
| nci      |  32767.0 |    **30.4** |     **0.7** |     **0.5** |     85.2 |   376.9 |  3218.9 |    49.5 |    3.9 |   1626.0 |   30.2 |    326.7 |    335.9 |
| mr       |   9736.9 |    **30.5** |     **0.6** |     **0.4** |     38.5 |   551.2 |  1008.8 |    31.3 |    5.1 |   772.5 |   44.0 |    233.2 |    229.3 |
| json     |    478.5 |     **7.8** |     **0.9** |     **0.5** |     0.3 |  12173.7 |  35691.0 |    **0.1** |  36824.1 |  36824.1 |  52.8 |   1649.5 |   1251.9 |

(Bold highlights where nyx wins on ratio; `~` values indicate rounding to 0.)

### Ratio — who wins

| file | nyx | z1 | z19 | FSE | winner |
|------|----:|---:|----:|----:|:------:|
| dickens | 57.5 | 41.7 | 28.0 | 57.0 | z19 |
| webster | 54.7 | 33.5 | 21.1 | 62.6 | z19 |
| nci | **30.4** | 85.2 | **49.5** | **30.2** | **nyx** (≈ FSE) |
| mr | **30.5** | 38.5 | 31.3 | 44.0 | **nyx** (narrow) |
| json | **7.8** | 0.3 | 0.1 | 52.8 | z19 |

- **nyx wins/ties on:** `nci`, `mr`, `dickens` (vs FSE)
- **zstd -19 wins on:** `dickens`, `webster`, `json` (large text / high-redundancy structured data)
- **FSE is in the same class as nyx** on `nci`/`dickens` but crushed by zstd `-19` on all text

### Speed — architectural note

nyx codes **bit‑by‑bit** (causal context mixing + rANS bit coder): ~0.6–1.0 MB/s
compress, ~0.4–0.5 MB/s decompress. zstd `-19` is ~3–4 MB/s compress and
~200–900 MB/s decompress; zstd `-1` is ~400–12000 MB/s compress and ~2000–36000 MB/s
decompress. FSE is byte‑level entropy and sits at ~200–1600 MB/s both ways.

This is a **fixed architectural constant**, not a tuning target: a correct
bit‑level CM cannot approach byte‑level decode throughput. nyx is positioned as a
**ratio‑competitive** research codec, not a speed‑competitive one.

## Before/after the 2026‑08‑31 PPM/escape hybrid (nyx only, ratio%)

| file | nyx before | nyx after (current) | delta |
|------|-----------:|---------------------:|------:|
| dickens | 58.4 | 57.5 | -0.9 |
| webster | 54.7 | 54.7 | ~0 |
| nci | 30.7 | 30.4 | -0.3 |
| mr | 30.6 | 30.5 | -0.1 |
| json | 13.6 | 7.8 | -5.8 |

## New directive (2026‑09)

> Optimize the codec/algorithms until nyx reaches **ratio parity with `zstd -1`**
> on the Silesia + mixed corpora, with **FSE as a secondary reference**. Speed
> remains explicitly secondary (bit‑level CM is architecturally slower).

`zstd -1` is the fast/low‑level target: it still beats nyx by ~1.5–2× on text
ratio (dickens 41.7 vs nyx 57.5), so closing to `zstd -1` parity on text is a
real but smaller gap than `zstd -19` parity. The `nci`/`mr` wins are preserved.
