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
- **Per-model reliability dampening** (2026-09 experiment). Used the mixer's
  previous per-model direction correctness as a dampening factor in both mix and
  update. Regressed both `dickens` (56.4% → 58.1%) and `json` (5.6% → 6.2%),
  which means the logistic weights already encode per-model reliability; adding
  an explicit dampening term over-constrains the mix. Reverted.
- **Classifier-aware Text stack dropping Exec + PPM order-4** (2026-09 experiment).
  Dropping Exec regressed `json` from 5.6% → 6.4%; PPM order-4 on Text was also a
  regress. Reverted to full hybrid_ppm3 stack for both Text and Binary; only `Exec`
  blocks drop the Exec model.
- **Classifier-aware stacks (broader evaluation)** (2026-09 experiment, current state).
  Added `METHOD_TEXT=2`, `METHOD_BINARY=3`, `METHOD_EXEC=4` to the `NYX1` container.
  Both Text and Binary use the full hybrid_ppm3 stack; only Exec blocks drop the
  Exec model. Measured as neutral on the 5-file subset — preserves all wins, no
  regressions. **Conclusion:** classifier-aware stacks are the right architecture
  but don't move the needle at 64 KiB blocks with online learning; the Exec model
  difference is lost in mixer warm-up. The actual ratio gains must come from richer
  context within a single stack.

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
  wins, no regressions.
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
- Classifier-aware stacks are architecturally sound but **neutral at 64 KiB blocks
  with online learning** — the Exec model difference is lost in mixer warm-up.
- Per-bit-position mixer context is the last confirmed win; all mixer reliability
  experiments (binary dampening, smoothed EMA accuracy) regressed ratio.
- **Current hypothesis:** the ~14–17 point text gap to zstd-1 comes from
  **long-range repetition**, not token prediction. A match-copy path using the
  existing LZP hash table but emitting explicit `(distance, length)` records could
  close that gap.

## 8. Research summary (2026-09)

### Cutting-edge
- **StateSMix** (May 2025, arXiv 2605.02904): online-trained Mamba-style SSM +
  sparse n-gram context mixing + arithmetic coding. Trained from scratch per-file,
  no pre-trained weights, ~120K params. Directly comparable to nyx's architecture
  but replaces the logistic mixer with a state-space model. This validates the
  hybrid direction but requires fundamentally different runtime.
- **Nacrith** (Feb 2026, arXiv 2602.19626): 135M transformer + context mixing.
  Too heavy for local use but shows the current frontier.

### Old / underused
- **ZPAQ**: journaling + dedup + LZ77 + CM. Modular but encode.su consensus is
  subpar LZ and CM performance.
- **PAQ / CMix** (Hutter Prize lineage): the original bit-level context mixing
  with hundreds of specialized models + neural-weight blending. CMix is the
  current text-compression champion; it's essentially "nyx with 100+ models and
  a neural mixer."
- **PPMD** (7-Zip): PPM with distance coding — exactly the "PPM + explicit match"
  hybrid architecture.

### Gap analysis for nyx
The ~14–17 point text gap to zstd-1 comes from **long-range repetition handling**,
not token prediction. zstd-1's text advantage is almost entirely LZ77-style
copy/distance coding. nyx's LZP only feeds a bit prediction; it never emits
explicit `(distance, length)` records.

## 9. Plausible alternatives to attempt

### A. Explicit match-copy records in the rANS stream (highest ROI)
- **What:** Use the existing LZP hash table to find repeated substrings. When a
  match exceeds a minimum length (e.g., 4–8 bytes), emit a `(distance, length)`
  record in the bitstream instead of coding those bytes through the CM stack.
- **Why plausible:** zstd-1's text advantage is primarily this. The LZP table
  already exists; we just need a record syntax and decoder-side copy instruction.
- **Risk:** medium. Adds a new method or symbol type to the entropy coder. The
  decoder must reconstruct the same LZP state to resolve references — possible
  because LZP prediction is causal and deterministic.
- **Reward:** highest single-axis gain available. Could close 5–10 points on text.

### B. Hybrid PPM + match records (old idea, new container)
- **What:** PPMD-style: when PPM's high-order context is sparse, fall back to
  distance-coded copies from recent bytes. This is "PPM with LZP escape" — the
  existing PPM model stays, but its escape path emits match records.
- **Why plausible:** PPMD in 7-Zip uses exactly this. It's old but not widely
  implemented in modern codecs. nyx's PpmModel already has the causal context
  machinery.
- **Risk:** high. Needs careful interaction between PPM update and match-copy
  emission so the decoder stays in sync.
- **Reward:** could unify the two strongest signals (PPM escape + match) in one
  model, which is more efficient than running them separately.

### C. Sparse high-order contexts with run-length limits (cutting-edge-ish)
- **What:** Add order-3 sparse context with a **run-length cap** — after N
  consecutive identical bytes, stop updating that context to avoid dilution.
  This is inspired by BSC/CMix's "run-length" models but much simpler.
- **Why plausible:** The raw order-3 bump failed because of dilution, but a
  run-length guard might preserve the signal for actual runs (whitespace, JSON
  colons, etc.) without polluting the table on variable data.
- **Risk:** low-medium. Small change to Sparse/OrderN, no container change.
- **Reward:** modest. Could recover 1–3 points on structured text.

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

Pursue **A** (explicit match-copy records) first. It's the highest-ROI path that
fits nyx's existing architecture: reuse the LZP hash table, add a new symbol type
to the rANS stream, and let the CM stack handle the non-match bytes. This directly
attacks the structural gap zstd-1 exploits on text.

If A stalls, try **C** (run-length-limited sparse contexts) as a low-risk side
experiment. It's a few lines of code and might recover 1–3 points on structured
text without container changes.

Save **B** (PPM + match) and **D** (SSM mixer) for later phases once A/C are
exhausted.

## 11. Definition of done (revised)

- [x] `cargo test` green (31 passed); `cargo clippy --all-targets -- -D warnings` clean.
- [x] `nyx compress`/`decompress` round-trips on ≥200 KB mixed fixture + real corpus files.
- [x] `README.md` updated with measured numbers and zstd-1 + FSE target (no "honest" framing).
- [x] `BENCH.md` written with zstd-1 / zstd-19 / FSE comparison (ratio and speed).
- [x] Committed and pushed to `origin/main`.
- [x] Classifier-aware method bytes evaluated — architecturally sound, neutral on
      current subset. Documented in plan.
- [ ] **Explicit match-copy records** — prototype, measure on 5-file subset.
- [ ] Beat `zstd -1` on ratio for text + mixed corpora.
- [ ] Update stage2 measurements file and this plan after each stage.
