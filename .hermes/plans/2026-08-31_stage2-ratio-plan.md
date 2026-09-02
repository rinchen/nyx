# Nyx — Optimization Stage 2 plan (ratio improvement while preserving wins)

## 1. Goal

Improve nyx’s compression ratio on the benchmark files where it currently loses to
`zstd -1` (`dickens`, `mr`, `json`), while keeping or improving the existing wins
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

Round-trip verified for every file. Speed: nyx ~0.7–0.9 MB/s cmp / ~0.4–0.5 MB/s dec;
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
- **Per-model reliability dampening** (2026-09 experiment). Used the mixer's
  previous per-model direction correctness as a dampening factor in both mix and
  update. Regressed both `dickens` (56.4% → 58.1%) and `json` (5.6% → 6.2%), so
  the logistic weights already encode per-model reliability; adding an explicit
  dampening term over-constrains the mix. Reverted.
- **Classifier-aware Text stack dropping Exec + PPM order-4** (2026-09 experiment).
  Dropping Exec regressed `json` from 5.6% → 6.4%; PPM order-4 on Text was also a
  regress. Reverted to full hybrid_ppm3 stack for both Text and Binary; only `Exec`
  blocks drop the Exec model.
- **Explicit match-copy records** (2026-09 experiment, REVERTED). Prototyped interleaved match/literal stream using existing LZP hash table. Control bit selects match vs literal; matches emit raw `(distance, length)` bits after control. Round-trips correctly, but regressed all 5 files:
  mr 29.4% → 37.2%, dickens 56.4% → 62.5%, json 5.6% → 9.2%, webster 51.0% → 59.8%, nci 27.6% → 32.3%.
  Likely cause: match overhead + model state divergence. Reverted.
- **Run-length-limited sparse contexts** (2026-09 experiment, REVERTED). Added `RunTracker` to `ByteAssembler`; `Sparse` stops updating after 8 identical bytes. Round-trips correctly; neutral on the 5-file subset (all within 1 byte, measurement noise). Likely gain ceiling is <1pt at 64 KiB; reverted to keep code path simple.
- **ICM — 22-state PAQ8 bit-history state machine** (2026-09 experiment, REVERTED).
  Measured as neutral-to-regressive: mr 29.4% → 30.1%, dickens 56.4% → 56.9%, webster 51.0% → 51.4%, nci 27.6% → 27.7%, json 5.6% → 5.5%. The 22-state design saturates too quickly at 64 KiB blocks. Reverted.
- **ICM — 256-state probability-quantized bit-history state machine** (2026-09 experiment, REVERTED).
  Measured as regressive on all 5 files: mr 29.4% → 29.9%, dickens 56.4% → 60.9%, webster 51.0% → 53.7%, nci 27.6% → 29.0%, json 5.6% → 5.8%. Despite finer probability resolution, the 2^21-bucket table aggressively aliases contexts, so the state machine loses precision against raw count tables. Reverted.

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
  subset files: dickens 57.5 → 56.4, webster 54.7 → 51.0, nci 30.4 → 27.6,
  mr 30.5 → 29.4, json 7.8 → 5.6. This is now the default mixer config.
- **Classifier-aware method bytes** (2026-09 experiment, current state). Added
  `METHOD_TEXT=2`, `METHOD_BINARY=3`, `METHOD_EXEC=4` to the `NYX1` container.
  Both Text and Binary currently use the full hybrid_ppm3 stack; only Exec blocks
  drop the Exec model. Measured as neutral on the 5-file subset — preserves all
  wins, no regressions.
- **README/BENCH rewrite** (2026-09): removed “honest” framing, added zstd-1/FSE
  comparison, added explicit ratio-win and speed-win columns.
- **Compiler tuning + inline hot paths** (2026-09): `panic=abort`, `strip=symbols`,
  `#[inline(always)]` on hot model paths, stack-allocated probability buffers.
  Ratios unchanged; compress speed improved ~13–26% on larger files.

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
- Classifier-aware stacks are architecturally sound but **neutral at 64 KiB blocks
  with online learning** — the Exec model difference is lost in mixer warm-up.
- Per-bit-position mixer context is the last confirmed win; all mixer reliability
  experiments (binary dampening, smoothed EMA accuracy) regressed ratio.
- The ~14–17 point text gap to zstd-1 is the core unsolved problem. Match-copy
  failed; run-limited sparse was neutral; two ICM variants were tested and rejected.
  The gap is structural: nyx's predictor stack is still shallow for long-range
  repetition and structured symbol boundaries.

## 8. Research summary (2026-09)

### Cutting-edge
- **StateSMix** (May 2025, arXiv 2605.02904): online-trained Mamba-style SSM +
  sparse n-gram context mixing + arithmetic coding. Trained from scratch per-file,
  no pre-trained weights, ~120K params. Directly comparable to nyx's architecture
  but replaces the logistic mixer with a state-space model. This validates the
  hybrid direction but requires fundamentally different runtime.
- **Nacrith** (Feb 2026, arXiv 2602.19626): 135M transformer + context mixing.
  Too heavy for local use but shows the current frontier.
- **NICOM / indirect context models (ZPAQ)**: ZPAQ's "Indirect Context Model"
  maps each context to a small bit-history state machine rather than a raw count.
  Tested here as 22-state and 256-state variants; both failed to improve on raw
  count tables at 64 KiB blocks.

### Old / underused
- **ZPAQ**: journaling + dedup + LZ77 + CM. Modular but encode.su consensus is
  subpar LZ and CM performance. Its ICM design was tested here and rejected.
- **PAQ / CMix** (Hutter Prize lineage): the original bit-level context mixing
  with hundreds of specialized models + neural-weight blending. CMix is the
  current text-compression champion; it's essentially "nyx with 100+ models and
  a neural mixer."
- **PPMD** (7-Zip): PPM with distance coding — exactly the "PPM + explicit match"
  hybrid architecture.
- **PAQ8 indirect context models / word models / record segmentation**: PAQ8
  variants include word models for English text, line/column models for tables,
  and "record segmentation" for structured data. These model the *structure* of
  data, not just bit patterns.

### Gap analysis for nyx
The ~14–17 point text gap to zstd-1 comes from two structural gaps:
1. **Long-range repetition handling** — zstd-1's LZ77 copy/distance coding.
2. **Symbol/word awareness** — PAQ/CMix models words and record fields; nyx
   only models raw bit contexts. On structured text (dickens/webster), word
   boundaries carry enormous predictive signal that nyx's byte-level models miss.

## 9. Plausible alternatives to attempt

### A. Indirect Context Model (ICM) — bit-history states (REJECTED 2026-09)
- **What:** Tested both 22-state PAQ8-style ICM and 256-state probability-quantized
  ICM. Both variants regressed or were neutral.
- **Measured outcomes:**
  - 22-state: mr 29.4% → 30.1%, dickens 56.4% → 56.9%, webster 51.0% → 51.4%, nci 27.6% → 27.7%, json 5.6% → 5.5%.
  - 256-state: mr 29.4% → 29.9%, dickens 56.4% → 60.9%, webster 51.0% → 53.7%, nci 27.6% → 29.0%, json 5.6% → 5.8%.
- **Conclusion:** State-machine bit histories are too lossy for 64 KiB blocks.
  The coarse state summary drops precision that the existing count-based tables
  capture. The 2 MB memory savings don't offset ratio loss.
- **Status:** REVERTED. Both variants removed. Do not re-attempt unless block size
  increases dramatically or table size is increased beyond practical limits.

### B. Word/string model for text (highest ROI remaining)
- **What:** Add a **word dictionary model** that tokenizes text into word symbols
  and maintains a `P(bit==1)` table keyed on the previous word + current bit.
  This is PAQ's "word model" — it turns natural language redundancy into direct
  symbol prediction instead of byte-level context chains.
- **Why plausible:** Dickents/webster are ~90% English prose. Word-level
  prediction on "the quick brown fox" is dramatically stronger than order-2 byte
  context because it skips 3–5 bytes of irrelevant context. CMix and PAQ8 both
  use word models for exactly this reason.
- **Risk:** medium. Needs a simple word break detector (ASCII space/punct), a
  rolling word buffer, and a new context table. Memory cost: ~2–4 MB per block.
- **Reward:** could close 3–5 points on dickens/webster, plus json (field names
  repeat exactly).
- **Implementation path:** only active for `Text` blocks; Binary/Exec keep the
  existing stack. Use the existing `ByteAssembler` word detection; add a small
  word-hash table and a new `BitModel` wrapper.

### C. Record segmentation model for structured data (old, rarely implemented)
- **What:** Detect record boundaries in semi-structured data (JSON objects, CSV
  rows, XML tags) and add a **field-position context model**. For JSON, key on
  the current JSON key string + bit position; for binary, key on offset modulo N.
- **Why plausible:** JSON files have extremely repetitive structure:
  `{"name":..., "level":..., "models":[...]}`. A model that recognizes "we're in
  the `models` array" predicts `[` and `"` with near-certainty. PAQ8's "record
  segmentation" does exactly this.
- **Risk:** medium. Needs lightweight per-class parsers or regex-like detectors.
  Must stay causal.
- **Reward:** could recover 2–4 points on json and other structured files.
- **Implementation path:** try after B. Only active for `Text` and `Binary` blocks
  that look structured. Use a simple JSON key detector; avoid full parsing.

### D. State-space mixer (long-shot, cutting-edge)
- **What:** Replace the logistic linear mix with a tiny Mamba-style SSM
  (StateSMix architecture). Train online per-block, no pre-trained weights.
- **Why plausible:** StateSMix published May 2025 shows this works with ~120K
  params. nyx's architecture is already close — swap the logistic mix for an SSM.
- **Risk:** very high. New runtime, new tuning, larger state per block, unknown
  behavior on small files.
- **Reward:** could be transformative if the SSM learns long-range dependencies
  the logistic mix can't represent. But this is multi-month research, not a
  weekend experiment.

### E. Dedicated dictionary / preprocessing pass (old, rarely used in modern codecs)
- **What:** Before CM coding, run a single-pass LZ77-style dictionary extraction
  that identifies all matches > L bytes, emits them as literals or copy records,
  and codes the rest with CM. This is what Brotli/Zstd do internally.
- **Why plausible:** It's the standard approach in production codecs. nyx lacks
  it entirely.
- **Risk:** high because it changes the fundamental "bit-by-bit" architecture
  that makes nyx decoder-safe. You'd need a new method in the container and
  careful state reconstruction.
- **Reward:** high but incremental — you'd be catching up to zstd-1's baseline
  rather than leapfrogging it.

## 10. Recommendation

Pursue **B** (word/string model) next. It addresses the structural word-boundary
blind spot for `dickens`/`webster`/`json`, fits nyx's existing `BitModel` trait,
and doesn't require changing the container format. It's the highest-ROI path
remaining after rejecting A (ICM).

If B plateaus, try **C** (record segmentation) for json and other structured files.
Save **D** (SSM mixer) and **E** (dictionary pass) for later phases.

## 11. Definition of done (revised)

- [x] `cargo test` green (36 passed); `cargo clippy --all-targets -- -D warnings` clean.
- [x] `nyx compress`/`decompress` round-trips on ≥200 KB mixed fixture + real corpus files.
- [x] `README.md` updated with measured numbers and zstd-1 + FSE target (no "honest" framing).
- [x] `BENCH.md` written with zstd-1 / zstd-19 / FSE comparison (ratio and speed).
- [x] Committed and pushed to `origin/main`.
- [x] Classifier-aware method bytes evaluated — architecturally sound, neutral on
      current subset. Documented in plan.
- [x] Explicit match-copy records — first prototype, measured, reverted (regressed).
- [x] Run-length-limited sparse contexts — prototype, measured, reverted (neutral).
- [x] ICM — 22-state PAQ8 variant — prototype, measured, reverted (regressed).
- [x] ICM — 256-state probability-quantized variant — prototype, measured, reverted (regressed).
- [x] Compiler tuning + inline hot paths — speed improved 13–26%, ratios unchanged.
- [ ] **Word/string model** — prototype, measure on 5-file subset.
- [ ] Beat `zstd -1` on ratio for text + mixed corpora.
- [ ] Update stage2 measurements file and this plan after each stage.
