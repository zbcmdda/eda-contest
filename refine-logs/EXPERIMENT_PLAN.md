# Generalization Experiment Plan

**Problem**: Improve hidden-set generalization without sacrificing the contest runtime budget.
**Method thesis**: Diagnose whether the estimator is data-limited, then add a small, targeted set of exact labels and learn a correction to the architecture-derived relaxed delay.
**Date**: 2026-08-17

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| C1: The current risk is distribution shift, not severe row memorization. | Random held-out accuracy alone can be optimistic. | Small train/validation gap plus grouped hold-out results. | B1, B2 |
| C2: Targeted exact labels or a residual target improve robust accuracy more efficiently than random scale. | Exact A* labels are expensive. | Better frozen and grouped scores at the same C++ inference cost. | B3, B4 |

## Experiment Blocks

### B1: Baseline audit

- Compute train and validation metrics with the production C++ model.
- Measure exact-pair overlap, endpoint coverage, port-pair coverage, and distance distributions.
- Priority: MUST-RUN.

### B2: Learning curve and grouped hold-outs

- Train on 100k, 250k, 500k, and 800k rows.
- Keep a development split and a deterministic final split separate.
- Evaluate spatial bands, distance bins, port groups, Line Gap crossings, and Block bands.
- Priority: MUST-RUN.

### B3: Exact-label acquisition

- Validate the exact oracle against a stratified public subset before generating labels.
- Generate a large unlabeled candidate pool cheaply.
- Select labels using coverage, architecture stress cases, and disagreement between retained models.
- Start with a small round; expand only when measured gains justify it.
- Priority: MUST-RUN after B2 shows a data limitation.

### B4: Target ablation

- Compare direct `log1p(delay)` regression with residual ratio regression around `relaxed_after_source`.
- Keep tree count and depth fixed so inference cost remains comparable.
- Priority: MUST-RUN.

### B5: Short-path specialist

- Train specialists on Manhattan distance thresholds 16, 32, and 64.
- Route only matching requests to the specialist and retain the production model elsewhere.
- Compare equal-capacity and higher-capacity specialists before acquiring new exact labels.
- Priority: MUST-RUN after the baseline audit identified short paths as the dominant error bucket.

## Run Order and Decision Gates

| Milestone | Runs | Decision gate | Cost |
|---|---|---|---|
| M0 sanity | Baseline audit, 10-row exact check | Zero oracle mismatches and parseable reports | Minutes |
| M1 diagnosis | Learning curve and grouped evaluation | Continue label generation only if data scaling or sparse groups show headroom | CPU-hours |
| M2 acquisition | 10k targeted exact labels first | Expand only if exact throughput and label validity are acceptable | Hours |
| M3 modeling | Direct vs residual target | Replace baseline only on frozen score, grouped robustness, and runtime | CPU-hours |
| M4 production | Retrain locked configuration, export C++, full tests | All regression and performance checks pass | Hours |

## Compute and Data Budget

- Training is CPU-based LightGBM; no GPU is required.
- Candidate generation is cheap and may scale to tens of millions of rows.
- Exact labels are the bottleneck. Label rounds must be checkpointed and resumable.
- The production executable must remain below 2 GB and deterministic across five runs.

## Stop Rules

- Do not generate 100 million exact labels blindly.
- Stop data expansion if an additional label round fails to improve the frozen public score or grouped worst-case metrics.
- Do not replace the production model for an accuracy gain that loses more weighted time score.
- Do not train on generated labels until the oracle matches a stratified public check set exactly.
