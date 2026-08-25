# Generalization Experiment Results

Date: 2026-08-17

## Decision

The production model is retained. Expanding the public set with exact labels is technically
possible, but the first strict train-only expansion did not produce a meaningful hidden-set proxy
gain. The limiting factor is short-path topology representation, not the raw number of rows.

## Evidence

The original production model has a small row-level train/validation gap:

```text
train score:       94.5582
validation score:  94.2958
```

There are no repeated exact endpoint pairs between the two parts. A stricter 80/10/10 split gives
the following final-test learning curve:

```text
training rows     final-test score
100,000           93.8623
250,000           94.0404
500,000           94.1589
800,000           94.2343
```

The marginal gain is already small at 800k. The main errors are concentrated in short queries:

```text
Manhattan 0-16:     score 64.30, mean relative error 11.35%
Manhattan 17-32:    score 76.22, mean relative error  6.44%
Block groups:       roughly 89.9-90.9
Long-path groups:   generally 95+
```

## Exact Expansion

Two exploratory 200k rounds were labeled exactly. The first recombined endpoint marginals and was
rejected because it changed the joint distribution. The second preserved source port, target port,
signed displacement, and translated the template to new legal SRBs.

For the second round, only the first 800k public rows supplied templates. A short specialist was
trained on the first 800k plus 200k exact labels. The middle 100k selected the extra-data weight;
the final 100k was evaluated once after that choice:

```text
baseline final test:                 94.2346
augmented specialist final test:     94.2433
development-selected weight:         0.1
```

The gain of 0.0087 is too small to justify further label generation. A residual target and several
short/low-delay loss weights were also tested; none beat uniform training on the development split.

## Operational Artifacts

- `scripts/analyze_generalization.py`: coverage and grouped error audit.
- `scripts/run_learning_curve.py`: fixed 80/10/10 learning curve.
- `scripts/generate_candidates.py`: deterministic architecture-focused candidates.
- `scripts/label_candidates.py`: parallel exact labeling with resumable shards.
- `scripts/run_short_specialist_threeway.py`: development-selected specialist experiment.
- `src/exact_csv.cpp`: offline exact-label CLI path.

The exact-label path is offline-only. The contest `-in/-out` production path remains the original
deterministic model until a candidate clears both accuracy and weighted runtime checks.

## First-Step Lookahead (2026-08-18)

The exact-first-step route lookahead was rechecked with three independent LightGBM seeds under the
strict 800k/100k/100k split. Each seed improved the untouched final test, but the gain varied:

```text
seed       baseline       lookahead       delta
20260815   94.2343        94.3043        +0.0700
20260816   94.2713        94.29395       +0.02265
20260817   94.2418        94.29928       +0.05748
```

The six features were implemented in C++ and their integer predictions matched the Python
lookahead model byte-for-byte on the final 100k rows. A fair 900k/100k candidate scored 94.3166 in
C++ versus 94.2958 for production, but increased the 1M-row runtime from the recorded 5.66s baseline
to 6.37-6.51s (about 13%). The weighted time-score loss is much larger than the 0.0208 accuracy gain,
so the candidate was rejected and the production 51-feature model was restored. The candidate model
and experiment artifacts remain under `models/candidates/` and `refine-logs/` for reference.

## Bounded medium-short residual (2026-08-24)

A new residual design succeeded where direct short specialists failed. Fixed two/three-hop route
descriptors are used only through Manhattan 128, and a compact conditional LightGBM predicts the
log-ratio correction to the rounded production output. The locked seed-20260824 C++ candidate
improved the 800k/100k development block by `+0.4048`; five seeds ranged from `+0.4005` to `+0.4299`.
The once-opened final 100k improved from `94.2958` to `94.7564` (`+0.4605`), with every bucket from
0 through 128 positive and paths above 128 byte-identical to production.

The candidate is exposed as `--short-residual`, with `--short-residual-threshold N` providing the
measured 32/64/96/128 Pareto. Five 1M outputs were identical; threshold 128 averaged 14.83 seconds in
the local Zig/Clang build and remained under the 18-second guideline. Full artifacts and the
multi-seed report are under `refine-logs/medium_short_search_v2/` and
`refine-logs/medium_short_qualification_v1/`. Production default remains unchanged pending a real
100M target-machine weighted-score decision.

## Architecture-driven weighted short route (2026-08-24)

The user set a hard target above 90 for both `0--16` and `17--32`, so the learned short residual was
replaced by direct local graph search. A port-compatible abstract shortest-path lower bound reduced
weighted-A* development expansions from 2.46 million to 0.72 million at weight 1.3. Compiling
terminal Arcs and exact Block-band Net transitions then reduced the route-loop time further.

The locked `--short-route` configuration uses weight 1.2 through Manhattan 32 and fixed train-only
calibration. Development scores are `97.1461` (`0--16`) and `97.0535` (`17--32`); all 25k blocks are
above `96.8`. The once-opened final block scores `96.8621` and `97.0510` in those segments, and the
complete score improves `94.2958 -> 95.0229` (`+0.7271`). No final-based adjustment was made.

Five full 1M runs average 16.16 seconds, peak at about 137 MiB, and share SHA-256
`63aace61548f432e166b60419f3f142fa2d6f451427d2bcc896b96b888cadbcf`. Qualification artifacts are
under `refine-logs/short_route_qualification_v1/`; the reproducible entry point is
`scripts/qualify_short_route.py`.
