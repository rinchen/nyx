# Nyx — Stage 2.2 measurement log (in progress)

## Experiments completed

### Adaptive LZP confidence scaling
- Files tested: `mr`, `dickens`
- Result: neutral. `mr` remained 30.5%, `dickens` remained 57.5%.
- Action: removed adaptive confidence path; LZP prediction reverted to fixed 3584/512.

### Per-model bias in logistic mixer
- Files tested: `mr`, `json`
- Result: regressed both. `mr` went from 30.5% to 31.3%; `json` from 7.8% to 8.0%.
- Action: fully reverted; mixer restored to weight-only SGD.

### Prev-byte bias feature in mixer interface
- Files tested: not measured directly; broke mixer test dynamics.
- Result: regressed learning; test `mixer_favors_correct_model` failed.
- Action: fully reverted; `mix()/update()` signatures restored.

### PPM order-4 as extra mixer input
- Files tested: `dickens`
- Result: neutral. `dickens` remained 57.5%, size unchanged.
- Action: reverted stack to PPM order-3 default.

### Explicit match-copy records (REVERTED)
- Files tested: mr, dickens, json, webster, nci
- Result: regressed all files. mr 29.4% → 37.2%, dickens 56.4% → 62.5%, json 5.6% → 9.2%, webster 51.0% → 59.8%, nci 27.6% → 32.3%.
- Round-trip: verified via md5 (all files match).
- Likely cause: match overhead + model state divergence + extra control bits.
- Action: fully reverted to baseline hybrid_ppm3 + per-bit-position mixer.

### Run-length-limited sparse contexts (REVERTED)
- Files tested: mr, dickens, json, webster, nci
- Result: neutral (all files within 1 byte; measurement noise). mr 29.4%, dickens 56.4%, json 5.6%, webster 51.0%, nci 27.6%.
- Round-trip: verified via md5 (all files match).
- Likely ceiling: <1pt at 64 KiB blocks; not worth added code path.
- Action: reverted to baseline Sparse implementation without RunTracker.

### Per-bit-position mixer context
- Files tested: mr, dickens, json, webster, nci
- Result: improved all 5 files. mr 30.5% → 29.4%, dickens 57.5% → 56.4%, json 7.8% → 5.6%, webster 54.7% → 51.0%, nci 30.4% → 27.6%.
- Round-trip: verified via md5 (all files match).
- Action: kept as default mixer config.

### Classifier-aware stacks (Text/Binary/Exec)
- Files tested: mr, dickens, json, webster, nci
- Result: neutral on current 5-file subset. Text/Binary use full hybrid_ppm3; Exec drops Exec model.
- Round-trip: verified via md5 (all files match).
- Action: kept container infrastructure; actual model stacks unchanged.

## Current best configuration (verified 2026-09-01)

- orders 0–2 + Sparse + Lzp + Exec + PpmModel order 3
- direct-addressed `CtxTable`
- single logistic mix with per-bit-position context
- causal predict/update
- 31 tests passing; build clean

## Measured ratios (baseline, no match-copy)

| file | ratio% | vs zstd-1 | vs zstd-19 |
|------|--------|-----------|------------|
| dickens | 56.4 | worse | worse |
| webster | 51.0 | worse | worse |
| nci | 27.6 | **better** | better |
| mr | 29.4 | **better** | worse |
| json | 5.6 | **better** | worse |

## Next candidates

1. Refined match-copy (learn from revert — reduce overhead, fix state divergence)
2. Dedicated dictionary / preprocessing pass (high effort, high reward)
3. State-space mixer (long-shot)
