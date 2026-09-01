# Nyx — Optimization Stage 2 plan (ratio improvement while preserving wins)

## 1. Goal

Improve nyx’s compression ratio on the benchmark files where it currently loses to
`zstd -19` (`dickens`, `mr`, `json`), while keeping or improving the existing wins
on `nci` and `webster`. Speed remains secondary; the hard constraint is round-trip
correctness and no regressions on current winning files.

## 2. What we tried and didn’t work

- **Raw order bump to orders 3–5** in the linear logistic mix. This hurt ratio badly
  on text (`dickens` expanded from 58.4% to 92.7%). High-order sparse contexts with
  strong INIT pseudo-counts polluted the per-model mixer on a 64 KiB block because
  the mixer weights models, not individual contexts.
- **Weak 2-bit SSE layer** keyed on the fused 12-bit probability plus 2-bit history.
  Measured as neutral on text (`dickens` 58.4% → 61.2%) and flat elsewhere. The
  history context was too small to justify its cost.
- **PPM/escape model as primary codec** (initial version). When wired in as the
  main stack, it regressed text ratio. The model itself was not yet tuned for the
  bit-level logistic mixer or causal update policy.

## 3. What we tried and did work

- **Direct-addressed context tables** replacing per-bit `HashMap` lookups. This cut
  the dominant lookup cost and roughly doubled compression speed (~0.8 → ~1.6 MB/s)
  at identical ratio.
- **Causal predict/update fix** in `OrderN`, `Sparse`, and `Exec`. The original code
  computed the context after advancing the byte assembler during `update()`, while
  `predict()` used the pre-advance state. For the 8th bit of every byte the byte
  history shifted, so predict and update disagreed and the models never learned
  consistent contexts. Fixing this alone dropped `dickens` 68.6% → 58.4% and
  `json` 38.9% → 13.6%.
- **Heterogeneous model stack with single logistic mixer**: orders 0–2 + Sparse +
  Lzp + Exec. This is the current best config and the baseline for all further work.

## 4. Hard requirement: optimize for data type, not benchmark-specific overfit

All experiments must improve the *class of data* represented by each file, not the
specific file instance. Concretely:
- No hard-coded thresholds or model weights tuned to a single filename.
- Class-level intent only: `dickens`/`webster` represent **text**; `mr` represents
  **repetitive structured/binary**; `json` represents **high-redundancy structured**;
  `nci` represents **mixed scientific/binary**.
- Any new behavior must be justified by data-type characteristics, not by memorizing
  corpus patterns.

## 5. What we are currently doing

- The current default stack is orders 0–2 + Sparse + Lzp + Exec with direct-addressed
  `CtxTable`, single logistic mix, and causal predict/update.
- A `PpmModel` exists in `src/model/ppm.rs` but is not in the default stack. It was
  measured as regressive when used as the primary codec, so it remains a test artifact.
- The next step is a controlled, isolated comparison of PPM variants and adaptive LZP
  through a dedicated harness, not by changing the default stack blindly.

## 6. Ideas and things left to be done

### Near-term experiments

1. **Isolated PPM comparison harness**
   - Compare baseline vs PPM primary (max_order 3, max_order 4) vs hybrid baseline
     plus PPM as an extra mixer input.
   - Must improve `dickens`, `mr`, or `json`; must not regress `nci` or `webster`
     by more than 0.5% ratio absolute.

2. **Adaptive LZP**
   - Replace fixed LZP probability with match-hit/miss or match-length statistics,
     keeping the same block/bit interface.

3. **Classifier-aware model selection**
   - Use the per-block classifier to choose model stacks by data type:
     `Text` drops `Exec` and adds PPM; `Binary`/`Exec` keep current stack;
     `Random` unchanged.
   - Must remain decoder-safe without extra metadata if possible.

4. **Richer mixer context**
   - Add per-model probability deltas or order-2 byte context as mixer inputs,
     only if the above experiments show the mixer is the bottleneck rather than
     missing contexts.

### Measurement protocol

- Use a fast representative subset with `scripts/bench_vs_sota.sh` or an equivalent
  temporary benchmark harness.
- Record before/after in `BENCH.md` with exact subset, command, and run metadata.
- Do not update README claims until numbers are reproducible.

### Guardrails

- Do not break `cargo test` or `cargo clippy --all-targets -- -D warnings`.
- Keep round-trip exact for all benchmark files.
- Stop after the first measured win; do not bundle multiple changes in one benchmark.
