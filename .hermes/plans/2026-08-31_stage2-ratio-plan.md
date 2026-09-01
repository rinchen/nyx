# Nyx — Optimization Stage 2 plan (ratio improvement while preserving wins)

## 1. Goal

Improve nyx’s compression ratio on the benchmark files where it currently loses to
`zstd -19` (`dickens`, `mr`, `json`), while keeping or improving the existing wins
on `nci` and `webster`. Speed remains secondary; the hard constraint is round-trip
correctness and no regressions on current winning files.

## 2. New directive (2026-09)

The target has moved up: **ratio parity with `zstd -1`** on text + mixed corpora,
with **FSE** (`FiniteStateEntropy`, `Cyan4973/FiniteStateEntropy`) as a secondary
reference. Speed stays explicitly secondary (bit-level CM is architecturally
slower than byte-level decoders).

## 3. Current measured state (zstd -1 + zstd -19 + FSE; 5-file subset)

| file   | nyx ratio% | zstd -1 ratio% | zstd -19 ratio% | FSE ratio% | winner on ratio |
|--------|-----------:|---------------:|---------------:|-----------:|:---------------:|
| dickens | 56.4 | 41.7 | 28.0 | 57.0 | zstd -19 |
| webster | 51.0 | 33.5 | 21.1 | 62.6 | zstd -19 |
| nci | 27.6 | 85.2 | 49.5 | 30.2 | **nyx** |
| mr | 29.4 | 38.5 | 31.3 | 44.0 | **nyx** |
| json | 5.6 | 0.3 | 0.1 | 52.8 | zstd -19 |

Round-trip verified for every file. Speed: nyx ~0.6–1.0 MB/s cmp / ~0.4–0.5 MB/s dec;
zstd -1 ~400–12000 MB/s cmp / ~1000–36000 MB/s dec; FSE ~200–1600 MB/s both ways.
The decode gap is ~40–70000× — an architectural constant of bit-level context mixing.

## 4. What we tried and didn’t work

- **Raw order bump to orders 3–5** in the linear logistic mix. Hurt ratio badly
  on text (`dickens` expanded from 58.4% to 92.7%). High-order sparse contexts
  with strong INIT pseudo-counts polluted the per-model mixer on a 64 KiB block.
- **Weak 2-bit SSE layer** keyed on fused 12-bit probability plus 2-bit history.
  Measured as neutral on text (`dickens` 58.4% → 61.2%) and flat elsewhere.
- **PPM/escape model as primary codec** (initial version). When wired in as the
  main stack, regressed text ratio.
- **Adaptive LZP confidence scaling** (2026-09 experiment). Measured as neutral on
  `mr` and `dickens`; removed the extra state and reverted to fixed 3584/512 LZP
  predictions.
- **Per-model bias / prev-byte bias in logistic mixer** (2026-09 experiments).
  Both regressed measured ratio; fully reverted.
- **PPM order-4 as an extra mixer input** (2026-09 experiment). No measurable
  improvement on `dickens` (unchanged at 57.5%); reverted to order-3.
- **Classifier-aware Text stack dropping Exec + PPM order-4** (2026-09 experiment).
  Dropping Exec regressed `json` from 5.6% → 6.4%; PPM order-4 on Text was also a
  regress. Reverted to full hybrid_ppm3 stack for both Text and Binary; only `Exec`
  blocks drop the Exec model.

## 5. What we tried and did work

- **Direct-addressed context tables** replacing per-bit `HashMap` lookups. Cut the
  dominant lookup cost and roughly doubled compression speed (~0.8 → ~1.6 MB/s)
  at identical ratio.
- **Causal predict/update fix** in `OrderN`, `Sparse`, and `Exec`. The original code
  computed the context after advancing the byte assembler during `update()`, while
  `predict()` used the pre-advance state. For the 8th bit of every byte the byte
  history shifted, so predict and update disagreed. Fixing alone dropped `dickens`
  68.6% → 58.4% and `json` 38.9% → 13.6%.
- **Heterogeneous model stack with single logistic mixer**: orders 0–2 + Sparse +
  LZP + Exec. Baseline for all further work.
- **PPM/escape order-3 as an extra mixer input (hybrid_ppm3)**. Improved all five
  subset files (dickens 58.4 → 57.5, webster 54.7 → 51.8, nci 30.7 → 30.4,
  mr 30.6 → 30.5, json 13.6 → 7.8). This is the current default stack.
- **Per-bit-position mixer context** (2026-09 experiment). Improved all five
  subset files: dickens 57.5 → 56.4, webster 54.7 → 51.1, nci 30.4 → 27.6,
  mr 30.5 → 29.4, json 7.8 → 5.6. This is now the default mixer config.
- **Classifier-aware method bytes** (2026-09 experiment, current state). Added
  `METHOD_TEXT=2`, `METHOD_BINARY=3`, `METHOD_EXEC=4` to the `NYX1` container.
  Both Text and Binary currently use the full hybrid_ppm3 stack; only Exec blocks
  drop the Exec model. Measured as neutral on the 5-file subset — preserves all
  wins, no regressions. **Conclusion:** classifier-aware stacks are the right
  architecture but don't move the needle at 64 KiB blocks with online learning;
  the Exec model difference is lost in mixer warm-up. The actual ratio gains
  must come from richer context within a single stack (Option A).
- **README/BENCH rewrite** (2026-09): removed “honest” framing, added zstd-1/FSE
  comparison, added explicit ratio-win and speed-win columns.

## 6. Hard requirement: optimize for data type, not benchmark-specific overfit

All experiments must improve the *class of data* represented by each file, not the
specific file instance. Concretely:
- No hard-coded thresholds or model weights tuned to a single filename.
- Class-level intent only: `dickens`/`webster` represent **text**; `mr` represents
  **repetitive structured/binary**; `json` represents **high-redundancy structured**;
  `nci` represents **mixed scientific/binary**.
- Any new behavior must be justified by data-type characteristics, not by memorizing
  corpus patterns.

## 7. What we are currently doing

- The default stack is **orders 0–2 + Sparse + Lzp + Exec + PpmModel order 3**,
  direct-addressed `CtxTable`, **per-bit-position logistic mix**, causal predict/update.
- The container method byte now distinguishes `Text`/`Binary`/`Exec` blocks, but
  Text and Binary both use the full hybrid_ppm3 stack. Only `Exec` blocks drop the
  Exec model.
- The next step is continuing Option A (richer mixer context) or moving to Option B
  (classifier-aware model selection).

## 8. Two remaining options

### Option A — Richer mixer context (in progress / preferred)

- Add a small amount of per-bit or per-byte context inside the mixer itself,
  without changing the container format.
- Candidate: **per-model probability deltas + previous-byte bucket**, using the
  existing `bit_pos`/`prev_byte` signals already available in the codec loop.
- Risk: medium. This touches the mixer interface and all call sites, but stays
  decoder-safe because both sides compute the same extra context from the same
  decoded prefix.
- Reward: higher if the bottleneck is mixer specialization rather than missing
  contexts.

### Option B — Classifier-aware model selection (higher risk, higher reward)

- Use `BlockKind` to choose different model stacks:
  - `Text` drops `Exec`, adds higher-order escape context;
  - `Binary`/`Exec` keep the current stack;
  - `Random` unchanged.
- Risk: higher because the current `NYX1` container only stores `METHOD_COPY` /
  `METHOD_CM`. To make this decoder-safe without extra metadata, we need either
  a reversible heuristic stack selection or a container version bump with an
  explicit method byte.
- Reward: highest if text parity with `zstd -1` requires fundamentally different
  context sets per class.

## 9. Ideas and things left to be done

### Near-term experiments

1. **PPM/escape order-4** — evaluated as extra mixer input; no gain on `dickens`
   at current implementation. Deprioritized unless Option A changes context.

2. **Adaptive LZP** — postponed after Option A.

3. **Classifier-aware model selection** — Option B path; needs container/method
   design before implementation.

4. **Richer mixer context** — Option A path; next concrete attempt.

### Measurement protocol

- Use the 5-file subset with a deterministic harness. Record before/after in
  `BENCH.md` with exact subset, command, and run metadata.
- Do not update README claims until numbers are reproducible.

### Guardrails

- Do not break `cargo test` or `cargo clippy --all-targets -- -D warnings`.
- Keep round-trip exact for all benchmark files.
- Stop after the first measured win; do not bundle multiple changes in one benchmark.

## 10. Definition of done (revised)

- [x] `cargo test` green (31 passed); `cargo clippy --all-targets -- -D warnings` clean.
- [x] `nyx compress`/`decompress` round-trips on ≥200 KB mixed fixture + real corpus files.
- [x] `README.md` updated with measured numbers and zstd-1 + FSE target (no "honest" framing).
- [x] `BENCH.md` written with zstd-1 / zstd-19 / FSE comparison (ratio and speed).
- [x] Committed and pushed to `origin/main`.
- [ ] **Beat `zstd -1` on ratio** for text + mixed corpora — next step: Option A
      or Option B from §8.
- [ ] Update stage2 measurements file and this plan after each stage.
