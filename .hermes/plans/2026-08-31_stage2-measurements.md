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

## Current best configuration

- orders 0–2 + Sparse + Lzp + Exec + PpmModel order 3
- direct-addressed `CtxTable`
- single logistic mix, causal predict/update
- 27 tests passing; build clean

## Next candidates

1. Richer mixer context (Option A in plan)
2. Classifier-aware model selection (Option B in plan)
